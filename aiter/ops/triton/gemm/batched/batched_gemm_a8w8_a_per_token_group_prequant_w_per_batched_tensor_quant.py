# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import inspect

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel,
    _get_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# (B, N, K) -> the M at which the gfx950 Gluon small-M kernel measured faster than
# the Triton kernel with its tuned config: the GLM-5 MLA value projection at TP8.
_GLUON_SMALL_M = {(8, 256, 512): (64,)}

# Set when the Gluon small-M kernel fails on a default-dispatch call; every later
# default-dispatch call then runs the Triton kernel.
_gluon_small_m_failed = False

# The gl.amd APIs the Gluon small-M kernel uses.
_GLUON_SMALL_M_APIS = (
    "cdna4.compute_efficient_padded_shared_layout",
    "cdna4.mfma",
    "cdna4.async_copy.buffer_load_to_shared",
    "cdna4.async_copy.load_shared_relaxed",
    "cdna4.async_copy.commit_group",
    "cdna4.async_copy.wait_group",
)


@functools.cache
def _gluon_small_m_available():
    """Whether this is gfx950 with a Triton whose Gluon has the APIs the kernel uses."""
    if get_arch() != "gfx950":
        return False
    try:
        from triton.experimental.gluon import language as gl
    except ImportError:
        return False
    amd = getattr(gl, "amd", None)
    for path in _GLUON_SMALL_M_APIS:
        api = amd
        for name in path.split("."):
            api = getattr(api, name, None)
        if api is None:
            return False
    layout = getattr(amd, "AMDMFMALayout", None)
    return (
        layout is not None and "tiles_per_warp" in inspect.signature(layout).parameters
    )


def _addressed_byte_span(tensor):
    """Bytes from the view's data pointer through its last addressed element."""
    if tensor.numel() == 0 or any(stride < 0 for stride in tensor.stride()):
        return 0
    return (
        1
        + sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
    ) * tensor.element_size()


def _overlaps_output(output, operand):
    """Conservative byte-range check, including holes in strided views."""
    output_start, operand_start = output.data_ptr(), operand.data_ptr()
    return output_start < operand_start + _addressed_byte_span(
        operand
    ) and operand_start < output_start + _addressed_byte_span(output)


def _nonoverlapping_strides(tensor):
    """Accept ordinary dense/transposed layouts and conservatively reject overlap."""
    span = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size > 1:
            if stride < span:
                return False
            span += (size - 1) * stride
    return True


def _gluon_small_m_supports(X, WQ, YQ, M, group_size, bias, dtype, w_scale):
    """Whether the gfx950 Gluon small-M kernel implements this call; WQ is (B, N, K)."""
    # 16-byte aligned rows keep the direct-to-LDS copies at their 16-byte width, and
    # the kernel's offsets are int32.
    return (
        _gluon_small_m_available()
        and X.is_cuda
        and WQ.is_cuda
        and w_scale.is_cuda
        and X.device == WQ.device == w_scale.device
        and w_scale.dtype == torch.float32
        and w_scale.numel() == 1
        and M > 0
        and M % 16 == 0
        and X.dtype == torch.bfloat16
        and X.stride(2) == 1
        and X.stride(0) % 16 == 0
        and X.stride(1) % 16 == 0
        and X.stride(0) > 0
        and X.stride(1) > 0
        and X.data_ptr() % 16 == 0
        and WQ.dtype == torch.float8_e4m3fn
        and WQ.shape[1] % 32 == 0
        and WQ.shape[2] == 512
        and WQ.is_contiguous()
        and WQ.storage_offset() % 2 == 0
        and WQ.data_ptr() % 16 == 0
        and group_size == 128
        and bias is None
        and dtype == torch.bfloat16
        and (
            YQ is None
            or (
                YQ.is_cuda
                and YQ.device == X.device
                and YQ.dtype == torch.bfloat16
                and YQ.stride(2) == 1
                and all(stride > 0 for stride in YQ.stride())
                and _nonoverlapping_strides(YQ)
                and _addressed_byte_span(YQ) < 2**31
                and not any(
                    _overlaps_output(YQ, operand) for operand in (X, WQ, w_scale)
                )
            )
        )
        and _addressed_byte_span(X) < 2**31
        and _addressed_byte_span(WQ) < 2**31
        and M * WQ.shape[0] * WQ.shape[1] * 2 < 2**31
    )


def _disable_gluon_small_m(error):
    """Routes every later default-dispatch call to the Triton kernel."""
    global _gluon_small_m_failed
    _gluon_small_m_failed = True
    _LOGGER.warning(
        "Gluon small-M batched_gemm_a8w8 kernel failed (%r); using the Triton kernel.",
        error,
    )


def _gluon_small_m(
    X, WQ, YQ, w_scale, M, transpose_bm_in, transpose_bm, **launch_options
):
    """Launches the gfx950 Gluon small-M kernel; WQ is (B, N, K)."""
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_small_m_kernel as kernel,
    )

    B, N = WQ.shape[0], WQ.shape[1]
    # The kernel reads each FP8 weight row as K // 2 16-bit words and computes one
    # 16x32 output tile per workgroup.
    w_words = WQ.view(torch.bfloat16)
    # Triton launches on the current device/stream. Select the operand's device so
    # same-device tensors remain correct when another device is current.
    with torch.cuda.device(X.device):
        kernel[(M // 16, N // 32, B)](
            X,
            w_words,
            YQ,
            w_scale,
            X.stride(0) if not transpose_bm_in else X.stride(1),
            X.stride(1) if not transpose_bm_in else X.stride(0),
            w_words.stride(0),
            w_words.stride(1),
            YQ.stride(0) if not transpose_bm else YQ.stride(1),
            YQ.stride(1) if not transpose_bm else YQ.stride(0),
            num_warps=2,
            **launch_options,
        )


def batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    splitK: int | None = None,
    YQ: torch.Tensor | None = None,
    transpose_bm: bool | None = False,
    transpose_bm_in: bool | None = False,
    config: dict | None = None,
    backend: str | None = None,
):
    """
    Computes batched 8 bit matrix multiplication Y[i] = X[i] @ W[i]^T with active activation quantization.
    X is quantized to INT8 during computation using per-token grouped quantization.
    W is pre-quantized INT8 with per-batch-element scaling.

    Args:
        X (torch.Tensor): Higher precision input batch with shape (B, M, K) or (M, B, K) if transpose_bm_in=True.
            Quantized to INT8 on-the-fly during GEMM.
        WQ (torch.Tensor): Pre-quantized INT8 weight batch with shape (B, N, K), internally transposed.
        w_scale (torch.Tensor): Per-batch scale for WQ with shape (1,).
        group_size (int): Group size for per-token grouped quantization of X. Must be power of 2.
        bias (Optional[torch.Tensor]): Bias batch with shape (B, 1, N).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        splitK (Optional[int]): Not supported. Must be None.
        YQ (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N) or (M, B, N) if transpose_bm=True.
        transpose_bm (Optional[bool]): Transpose batch and M dimensions in output.
        transpose_bm_in (Optional[bool]): Transpose batch and M dimensions in input.
        config (Optional[dict]): Triton kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M).
            Passing one selects the Triton kernel when backend is None.
        backend (Optional[str]): "triton", "gluon" or None. None picks the gfx950 Gluon small-M
            kernel at the (B, N, K) and M listed in _GLUON_SMALL_M when config is None, and the
            Triton kernel otherwise. "gluon" forces it and takes no config; it needs gfx950,
            M % 16 == 0, BF16 X and output, contiguous FP8 e4m3 WQ, N % 32 == 0, K == 512,
            group_size == 128, no bias, unit stride in K for X and in N for YQ, 16-byte aligned X,
            X rows and WQ, addressed input/weight/output spans under 2 GiB, same-device
            CUDA operands, a scalar FP32 weight scale and output not overlapping inputs.
            Warm up eagerly before graph capture. A default compile failure falls back to
            Triton; this does not guarantee recovery from a failed launch during capture.

    Returns:
        torch.Tensor: Output batch with shape (B, M, N) or (M, B, N) if transpose_bm=True.
    """

    # Check constraints.
    if not transpose_bm_in:
        B = X.shape[0]
        M = X.shape[1]
    else:
        M = X.shape[0]
        B = X.shape[1]
    K = X.shape[2]
    N = WQ.shape[1]

    assert B == WQ.shape[0], "Incompatible Batch dimensions!!!"
    assert K == WQ.shape[2], "Incompatible K dimensions!!!"
    assert (
        triton.next_power_of_2(group_size) == group_size
    ), "group_size mush be power of 2"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

    gluon_forced = backend == "gluon"
    if backend is None:
        backend = (
            "gluon"
            if config is None
            and M in _GLUON_SMALL_M.get((B, N, K), ())
            and not _gluon_small_m_failed
            and _gluon_small_m_supports(X, WQ, YQ, M, group_size, bias, dtype, w_scale)
            else "triton"
        )
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"
    if backend == "gluon":
        assert (
            config is None
        ), "config applies to the Triton kernel; pass backend='triton' to use it"
        assert _gluon_small_m_supports(
            X, WQ, YQ, M, group_size, bias, dtype, w_scale
        ), "Gluon backend: unsupported arch, Triton, dtype, shape, size, bias, alignment or layout (see docstring)"

    WQ = WQ.transpose(1, 2)

    has_bias = bias is not None
    if YQ is None:
        if transpose_bm:
            YQ = torch.empty((M, B, N), dtype=dtype, device=X.device)
        else:
            YQ = torch.empty((B, M, N), dtype=dtype, device=X.device)
    else:
        if transpose_bm:
            assert (
                YQ.shape[0] == M and YQ.shape[1] == B and YQ.shape[2] == N
            ), "Output dimension error"
        else:
            assert (
                YQ.shape[0] == B and YQ.shape[1] == M and YQ.shape[2] == N
            ), "Output dimension error"

    if backend == "gluon":
        # A default-dispatch failure, such as a compile error on an untested Triton,
        # falls back to the Triton kernel; a forced call raises.
        try:
            _gluon_small_m(
                X, WQ.transpose(1, 2), YQ, w_scale, M, transpose_bm_in, transpose_bm
            )
            return YQ
        except Exception as error:
            if gluon_forced:
                raise
            _disable_gluon_small_m(error)

    if config is None:
        config, _ = _get_config(M, N, K)
    config["BLOCK_SIZE_K"] = group_size

    grid = lambda META: (
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    DTYPE_MAX = (
        torch.finfo(WQ.dtype).max
        if torch.is_floating_point(WQ)
        else torch.iinfo(WQ.dtype).max
    )

    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel[
        grid
    ](
        X,
        WQ,
        YQ,
        w_scale,
        bias,
        M,
        N,
        K,
        X.stride(0) if not transpose_bm_in else X.stride(1),
        X.stride(1) if not transpose_bm_in else X.stride(0),
        X.stride(2),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        YQ.stride(0) if not transpose_bm else YQ.stride(1),
        YQ.stride(1) if not transpose_bm else YQ.stride(0),
        YQ.stride(2),
        bias.stride(0) if has_bias else 0,
        has_bias,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        **config,
    )

    return YQ
