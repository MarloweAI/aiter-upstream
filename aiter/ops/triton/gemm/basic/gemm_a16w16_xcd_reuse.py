# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton.gemm.basic.gemm_a16w16_small_m import (
    _accepts,
    _gluon_arch,
    _padded_layout_available,
    _row_major,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# The grid, tile placement and weight reuse are built for this one shape.
_M, _N, _K = 128, 2048, 2048


def gemm_a16w16_xcd_reuse_supported(
    M: int, N: int, K: int, bias: bool, dtype: torch.dtype, otype: torch.dtype
) -> bool:
    """Whether gemm_a16w16_xcd_reuse computes this GEMM here. Never raises.

    True only for M = 128, N = 2048, K = 2048, BF16 in and out, no bias, on
    gfx950. The tuned GEMM table checks it before taking a row that names this
    kernel, so every other call keeps its current kernel.
    """
    return (
        (M, N, K) == (_M, _N, _K)
        and not bias
        and dtype == otype == torch.bfloat16
        and _gluon_arch() == "gfx950"
        and _padded_layout_available()
    )


def gemm_a16w16_xcd_reuse_accepts(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether gemm_a16w16_xcd_reuse takes these operands. Never raises.

    Needs a supported shape and dtype (see gemm_a16w16_xcd_reuse_supported) on one
    GPU, and a contiguous, 16-byte aligned canonical weight. The named tuned row uses the ordinary
    Triton route for a call this rejects.
    """
    return _accepts(gemm_a16w16_xcd_reuse_supported, x, w, False)


def gemm_a16w16_xcd_reuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Computes Y = X @ W^T for X (128, 2048) and W (2048, 2048), BF16, on gfx950.

    256 workgroups each compute a 32x32 tile over the whole K, so every output
    element is one FP32 accumulation chain rounded once to BF16, with no split-K.
    The tiles are placed so that, when workgroup p runs on XCD p % 8, the four
    workgroups that read a slice of W can share one XCD's L2. Placement and cache
    reuse are performance assumptions; correctness does not depend on them.
    This schedule alone does not establish the amount of HBM traffic.

    Args:
        x (torch.Tensor): BF16 input with shape (128, 2048).
        w (torch.Tensor): BF16 weight with shape (2048, 2048), internally transposed.

    Returns:
        torch.Tensor: BF16 output with shape (128, 2048).

    Raises:
        ValueError: for operands gemm_a16w16_xcd_reuse_accepts rejects, which
            include a weight that is not contiguous and 16-byte aligned: copying
            the 8 MiB weight on every call, and into every captured graph, would
            cost more than the kernel saves. An x that is not contiguous or not
            aligned is copied first.
    """
    if not gemm_a16w16_xcd_reuse_accepts(x, w):
        raise ValueError(
            "gemm_a16w16_xcd_reuse: needs BF16 x (128, 2048) and a contiguous, "
            "16-byte aligned w (2048, 2048) on one gfx950 device, got "
            f"x {tuple(x.shape)} {x.dtype} on {x.device}, "
            f"w {tuple(w.shape)} {w.dtype} on {w.device}"
        )
    _LOGGER.info("GEMM_A16W16_XCD_REUSE: x=%s w=%s", tuple(x.shape), tuple(w.shape))

    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_xcd_reuse import (
        _gemm_a16w16_xcd_reuse_kernel,
    )

    if not _row_major(x):
        x = x.clone(memory_format=torch.contiguous_format)
    y = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    # The kernel's layouts fix 4 waves; these are not tuning parameters.
    _gemm_a16w16_xcd_reuse_kernel[(256,)](
        x,
        w,
        y,
        x.stride(0),
        w.stride(0),
        num_warps=4,
        num_stages=1,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
    )
    return y
