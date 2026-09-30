# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch

from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Each kernel is built for one M at this N and K.
_N, _K = 2048, 2048


@functools.cache
def _gluon_arch() -> str | None:
    """The GPU arch when Triton has Gluon with its AMD dialect, else None."""
    try:
        from triton.experimental.gluon import language as gl

        from aiter.ops.triton.utils._triton.arch_info import get_arch

        amd = getattr(gl, "amd", None)
        if not callable(getattr(amd, "AMDMFMALayout", None)) or not callable(
            getattr(getattr(amd, "cdna3", None), "mfma", None)
        ):
            return None
        return get_arch()
    except (ImportError, RuntimeError):
        return None


@functools.cache
def _padded_layout_available() -> bool:
    """Whether Gluon has the CDNA4 layout helper the M64 kernel uses."""
    try:
        from triton.experimental.gluon import language as gl

        cdna4 = getattr(getattr(gl, "amd", None), "cdna4", None)
        copy = getattr(cdna4, "async_copy", None)
        return callable(
            getattr(cdna4, "compute_efficient_padded_shared_layout", None)
        ) and all(
            callable(getattr(copy, name, None))
            for name in (
                "buffer_load_to_shared",
                "commit_group",
                "wait_group",
                "load_shared_relaxed",
            )
        )
    except (ImportError, RuntimeError):
        return False


def _supported(
    ms, M: int, N: int, K: int, bias: bool, dtype: torch.dtype, otype: torch.dtype
) -> bool:
    return (
        M in ms
        and (N, K) == (_N, _K)
        and not bias
        and dtype == otype == torch.bfloat16
        and _gluon_arch() == "gfx950"
    )


def gemm_a16w16_small_m_supported(
    M: int, N: int, K: int, bias: bool, dtype: torch.dtype, otype: torch.dtype
) -> bool:
    """Whether gemm_a16w16_small_m computes this GEMM here. Never raises.

    True only for M = 64, N = 2048, K = 2048, BF16 in and out, no bias, on gfx950.
    The tuned GEMM table checks it before taking a row that names this kernel, so
    every other call keeps its current kernel.
    """
    supported = _supported((64,), M, N, K, bias, dtype, otype)
    return supported and _padded_layout_available()


def _row_major(t: torch.Tensor) -> bool:
    # The kernels read contiguous rows in 16-byte vectors.
    return t.is_contiguous() and t.data_ptr() % 16 == 0


def _accepts(supported, x, w, shuffled) -> bool:
    if x.dim() != 2 or w.dim() != 2:
        return False
    M, K = x.shape
    N = w.shape[0]
    return (
        w.shape[1] == K
        and w.dtype == x.dtype
        and x.is_cuda
        and w.device == x.device
        and _row_major(w)
        and getattr(w, "is_shuffled_16x32", False) is shuffled
        and getattr(w, "is_shuffled", False) is False
        and supported(M, N, K, False, x.dtype, x.dtype)
    )


def gemm_a16w16_small_m_accepts(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether gemm_a16w16_small_m takes these operands. Never raises.

    Needs a supported shape and dtype (see gemm_a16w16_small_m_supported) on one
    GPU, and a contiguous, 16-byte aligned canonical weight. The named tuned row
    uses the ordinary Triton route for a call this rejects.
    """
    return _accepts(gemm_a16w16_small_m_supported, x, w, False)


def _check(name, accepts, x, w, shuffled):
    if not accepts(x, w):
        raise ValueError(
            f"{name}: unsupported operands x {tuple(x.shape)} {x.dtype} on "
            f"{x.device}, w {tuple(w.shape)} {w.dtype} on {w.device}, w "
            f"must be canonical, contiguous and 16-byte aligned; see {name}_accepts"
        )
    _LOGGER.info("%s: x=%s w=%s", name.upper(), tuple(x.shape), tuple(w.shape))


def gemm_a16w16_small_m(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Computes Y = X @ W^T for X (64, 2048) and W (2048, 2048), BF16, on gfx950.

    256 workgroups each compute a 32x16 tile over the whole K, so every output
    element is one FP32 accumulation chain rounded once to BF16, with no split-K.
    Operands are staged in LDS in K512 blocks, weights issued before activations.

    Args:
        x (torch.Tensor): BF16 input with shape (64, 2048).
        w (torch.Tensor): BF16 weight with shape (2048, 2048), internally transposed.

    Returns:
        torch.Tensor: BF16 output with shape (64, 2048).

    Raises:
        ValueError: for operands gemm_a16w16_small_m_accepts rejects, which
            include noncanonical, noncontiguous or misaligned weights: copying the
            8 MiB weight on every call,
            and into every captured graph, would cost more than the kernel saves.
            An x that is not contiguous or not aligned is copied first.
    """
    _check("gemm_a16w16_small_m", gemm_a16w16_small_m_accepts, x, w, False)
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_small_m import (
        _gemm_a16w16_small_m_m64_kernel,
    )

    if not _row_major(x):
        x = x.clone(memory_format=torch.contiguous_format)
    y = torch.empty((x.shape[0], _N), dtype=torch.bfloat16, device=x.device)
    # The kernel's layouts fix 2 waves; these are not tuning parameters.
    _gemm_a16w16_small_m_m64_kernel[(2, 128)](
        x,
        w,
        y,
        x.stride(0),
        w.stride(0),
        num_warps=2,
        num_stages=1,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
    )
    return y
