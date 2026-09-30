# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon BF16 GEMM for M64 at N2048, K2048, gfx950."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gemm_a16w16_small_m_m64_repr = make_kernel_repr("_gemm_a16w16_small_m_m64_kernel", [])


@gluon.jit(repr=_gemm_a16w16_small_m_m64_repr)
def _gemm_a16w16_small_m_m64_kernel(X, W, Y, XS: gl.constexpr, WS: gl.constexpr):
    """Y [64, 2048] = X [64, 2048] @ W [2048, 2048]^T in BF16, FP32 accumulation.

    Launched on a (2, 128) grid of 2-wave workgroups. Workgroup (i, j) computes the
    32x16 output tile at rows 32 * i and columns 16 * j over the whole K, one
    ascending-K32 MFMA chain per element and one BF16 rounding, so there is no
    split-K. Operands move to LDS by CDNA4 direct copies into two K512 buffer pairs,
    and each K step issues its weight copy before its activation copy. XS and WS
    are the row strides of X and W; Y is contiguous.
    """
    M: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 1],
        tiles_per_warp=[1, 1],
    )
    A: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=M, k_width=8)
    B: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=M, k_width=8)
    LX: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[2, 1],
        order=[1, 0],
    )
    LW: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[64, 1],
        warps_per_cta=[1, 2],
        order=[0, 1],
    )
    SX: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        A, [32, 512], gl.bfloat16
    )
    SW: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        B, [512, 16], gl.bfloat16
    )
    sx0 = gl.allocate_shared_memory(gl.bfloat16, [32, 512], SX)
    sx1 = gl.allocate_shared_memory(gl.bfloat16, [32, 512], SX)
    sw0 = gl.allocate_shared_memory(gl.bfloat16, [512, 16], SW)
    sw1 = gl.allocate_shared_memory(gl.bfloat16, [512, 16], SW)
    m = gl.program_id(0) * 32 + gl.arange(0, 32, gl.SliceLayout(1, LX))
    n = gl.program_id(1) * 16 + gl.arange(0, 16, gl.SliceLayout(0, LW))
    kx = gl.arange(0, 512, gl.SliceLayout(0, LX))
    kw = gl.arange(0, 512, gl.SliceLayout(1, LW))
    ox = m[:, None] * XS + kx[None, :]
    ow = n[None, :] * WS + kw[:, None]
    k0 = 0
    k1 = 512
    gl.amd.cdna4.async_copy.buffer_load_to_shared(sw0, gl.multiple_of(W + k0, 16), ow)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(sx0, gl.multiple_of(X + k0, 16), ox)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(sw1, gl.multiple_of(W + k1, 16), ow)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(sx1, gl.multiple_of(X + k1, 16), ox)
    gl.amd.cdna4.async_copy.commit_group()
    acc = gl.full((32, 16), 0, gl.float32, M)
    for block in gl.static_range(4):
        # The compiler inserts the producer barrier after this wait; an explicit
        # barrier here would also wait for the group still in flight.
        gl.amd.cdna4.async_copy.wait_group(1 if block < 3 else 0)
        if block % 2 == 0:
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(sx0, A)
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(sw0, B)
        else:
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(sx1, A)
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(sw1, B)
        acc = gl.amd.cdna3.mfma(x, w, acc)
        # Every wave's shared reads complete before its buffer is refilled. The
        # integer output only satisfies inline_asm_elementwise.
        gl.inline_asm_elementwise(
            "s_waitcnt lgkmcnt(0)\n\ts_barrier\n\tv_mov_b32_e32 $0, 0",
            constraints="=v,~{memory}",
            args=[],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        if block < 2:
            next_k = (block + 2) * 512
            if block % 2 == 0:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sw0, gl.multiple_of(W + next_k, 16), ow
                )
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sx0, gl.multiple_of(X + next_k, 16), ox
                )
            else:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sw1, gl.multiple_of(W + next_k, 16), ow
                )
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sx1, gl.multiple_of(X + next_k, 16), ox
                )
            gl.amd.cdna4.async_copy.commit_group()
    mo = gl.program_id(0) * 32 + gl.arange(0, 32, gl.SliceLayout(1, M))
    no = gl.program_id(1) * 16 + gl.arange(0, 16, gl.SliceLayout(0, M))
    gl.store(Y + mo[:, None] * 2048 + no[None, :], acc.to(gl.bfloat16))
