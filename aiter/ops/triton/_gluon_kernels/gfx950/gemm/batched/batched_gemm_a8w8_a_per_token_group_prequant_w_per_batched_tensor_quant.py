# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon small-M batched FP8 GEMM with in-kernel activation quantization, gfx950."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_small_m_repr = make_kernel_repr(
    "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_small_m_kernel",
    [],
)


@gluon.jit(
    repr=_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_small_m_repr
)
def _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_small_m_kernel(
    x_ptr,  # bf16 [B, M, K], unit stride in K
    w_ptr,  # fp8 e4m3 weight [B, N, K] read as 16-bit words [B, N, K // 2]
    y_ptr,  # [B, M, N], unit stride in N
    w_scale_ptr,  # scalar weight scale
    stride_xb,
    stride_xm,
    stride_wb,
    stride_wn,
    stride_yb,
    stride_ym,
):
    """
    Computes Y[b] = X[b] @ W[b]^T for K = 512, quantizing each 128-element group of
    each X row to FP8 e4m3 inside the kernel, with the arithmetic of the Triton kernel
    `_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel`:
    scale = max(|x|, 1e-10) * (1 / 448), x * (1 / scale) clamped to [-448, 448], one
    FP32 dot product per group multiplied by its scale, then the weight scale, then
    one rounding to the output type.

    Each 2-warp workgroup computes a 16x32 tile of one batch entry. It copies the
    whole 16x512 X tile and 512x32 FP8 W tile to LDS with direct-to-LDS loads. The
    weight is moved as 16-bit words and split back into FP8 values in registers.
    Grid: (M // 16, N // 32, B); M must be a multiple of 16.
    """
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 2],
        tiles_per_warp=[1, 1],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma, 16)
    x_copy: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[2, 1],
        order=[1, 0],
    )
    x_read: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[2, 1],
        order=[1, 0],
    )
    w_copy: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[64, 1],
        warps_per_cta=[1, 2],
        order=[0, 1],
    )
    w_read: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 2],
        order=[0, 1],
    )
    x_shared: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        gl.DotOperandLayout(0, mfma, 8), [16, 512], gl.bfloat16
    )
    w_shared: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        gl.DotOperandLayout(1, mfma, 8), [512, 32], gl.bfloat16
    )

    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    pid_b = gl.program_id(2)

    offs_m = pid_m * 16 + gl.arange(0, 16, gl.SliceLayout(1, x_copy))
    offs_n = pid_n * 32 + gl.arange(0, 32, gl.SliceLayout(0, w_copy))
    offs_kx = gl.arange(0, 512, gl.SliceLayout(0, x_copy))
    offs_kw = gl.arange(0, 512, gl.SliceLayout(1, w_copy))

    # The W tile holds 512 rows of words to fit the padded shared layout; only the
    # first 256 rows, the 512 FP8 values of each column, are loaded and read.
    x_smem = gl.allocate_shared_memory(gl.bfloat16, [16, 512], x_shared)
    w_smem = gl.allocate_shared_memory(gl.bfloat16, [512, 32], w_shared)
    offs_x = offs_m[:, None] * stride_xm + pid_b * stride_xb + offs_kx[None, :]
    offs_w = pid_b * stride_wb + offs_kw[:, None] + offs_n[None, :] * stride_wn
    gl.amd.cdna4.async_copy.buffer_load_to_shared(x_smem, x_ptr, offs_x)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        w_smem, w_ptr, offs_w, mask=offs_kw[:, None] < 256
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(0)

    acc = gl.full((16, 32), 0, gl.float32, mfma)
    for group in gl.static_range(4):
        x = gl.amd.cdna4.async_copy.load_shared_relaxed(
            x_smem.slice(group * 128, 128, dim=1), x_read
        )
        words = gl.amd.cdna4.async_copy.load_shared_relaxed(
            w_smem.slice(group * 64, 64, dim=0), w_read
        ).to(gl.uint16, bitcast=True)
        low = (words & 255).to(gl.uint8)
        high = (words >> 8).to(gl.uint8)
        w_bytes = gl.join(low, high).permute(0, 2, 1).reshape(128, 32)
        w = gl.convert_layout(w_bytes.to(gl.float8e4nv, bitcast=True), dot_b)

        # 448 is the largest finite float8_e4m3fn value.
        x_max = gl.maximum(gl.max(gl.abs(x), 1), 1e-10)
        x_scale = x_max.to(gl.float32) * (1.0 / 448.0)
        x_fp8 = gl.clamp(x * (1.0 / x_scale[:, None]), -448.0, 448.0).to(gl.float8e4nv)
        dot = gl.amd.cdna4.mfma(
            gl.convert_layout(x_fp8, dot_a),
            w,
            gl.full((16, 32), 0, gl.float32, mfma),
        )
        acc += dot * gl.convert_layout(x_scale, gl.SliceLayout(1, mfma))[:, None]
    acc *= gl.load(w_scale_ptr)

    offs_ym = pid_m * 16 + gl.arange(0, 16, gl.SliceLayout(1, mfma))
    offs_yn = pid_n * 32 + gl.arange(0, 32, gl.SliceLayout(0, mfma))
    y_ptrs = y_ptr + offs_ym[:, None] * stride_ym + pid_b * stride_yb + offs_yn[None, :]
    gl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty))
