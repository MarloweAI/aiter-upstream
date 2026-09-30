# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon BF16 GEMM for M128, N2048, K2048 with XCD-local weight reuse, gfx950."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gemm_a16w16_xcd_reuse_repr = make_kernel_repr("_gemm_a16w16_xcd_reuse_kernel", [])


@gluon.jit(repr=_gemm_a16w16_xcd_reuse_repr)
def _gemm_a16w16_xcd_reuse_kernel(X, W, Y, XS: gl.constexpr, WS: gl.constexpr):
    """Y [128, 2048] = X [128, 2048] @ W [2048, 2048]^T in BF16, FP32 accumulation.

    Launched on 256 workgroups of 4 waves. Workgroup p computes one 32x32 output
    tile over the whole K, one ascending-K32 MFMA chain per element and one BF16
    rounding, so there is no split-K. Operands move to LDS by CDNA4 direct copies
    into two K512 buffer pairs. XS and WS are the row strides of X and W; Y is
    contiguous.

    Tile (p / 8) % 4 of rows and 8 * (p % 8) + (p / 8) / 4 of columns: if
    workgroup p runs on XCD p % 8, each XCD owns 8 column tiles and all 4 row
    tiles of each, so the four readers of a weight tile can share one L2.
    Actual placement and HBM traffic require measurement; results are unchanged.
    """
    M: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 2],
        tiles_per_warp=[1, 1],
    )
    A: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=M, k_width=8)
    B: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=M, k_width=8)
    LX: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    LW: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[64, 1],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    SX: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        A, [32, 512], gl.bfloat16
    )
    SW: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        B, [512, 32], gl.bfloat16
    )
    sx0 = gl.allocate_shared_memory(gl.bfloat16, [32, 512], SX)
    sx1 = gl.allocate_shared_memory(gl.bfloat16, [32, 512], SX)
    sw0 = gl.allocate_shared_memory(gl.bfloat16, [512, 32], SW)
    sw1 = gl.allocate_shared_memory(gl.bfloat16, [512, 32], SW)
    physical = gl.program_id(0)
    xcd = physical % 8
    local = physical // 8
    tile_m = local % 4
    tile_n = xcd * 8 + local // 4
    m = tile_m * 32 + gl.arange(0, 32, gl.SliceLayout(1, LX))
    n = tile_n * 32 + gl.arange(0, 32, gl.SliceLayout(0, LW))
    kx = gl.arange(0, 512, gl.SliceLayout(0, LX))
    kw = gl.arange(0, 512, gl.SliceLayout(1, LW))
    ox = m[:, None] * XS + kx[None, :]
    ow = n[None, :] * WS + kw[:, None]
    k0 = 0
    k1 = 512
    gl.amd.cdna4.async_copy.buffer_load_to_shared(sx0, gl.multiple_of(X + k0, 16), ox)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(sw0, gl.multiple_of(W + k0, 16), ow)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(sx1, gl.multiple_of(X + k1, 16), ox)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(sw1, gl.multiple_of(W + k1, 16), ow)
    gl.amd.cdna4.async_copy.commit_group()
    acc = gl.full((32, 32), 0, gl.float32, M)
    for block in gl.static_range(4):
        # The compiler inserts the producer barrier after this wait.
        gl.amd.cdna4.async_copy.wait_group(1 if block < 3 else 0)
        if block % 2 == 0:
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(sx0, A)
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(sw0, B)
        else:
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(sx1, A)
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(sw1, B)
        acc = gl.amd.cdna3.mfma(x, w, acc)
        # Every wave's shared reads complete before its buffer is refilled.
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
                    sx0, gl.multiple_of(X + next_k, 16), ox
                )
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sw0, gl.multiple_of(W + next_k, 16), ow
                )
            else:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sx1, gl.multiple_of(X + next_k, 16), ox
                )
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sw1, gl.multiple_of(W + next_k, 16), ow
                )
            gl.amd.cdna4.async_copy.commit_group()
    mo = tile_m * 32 + gl.arange(0, 32, gl.SliceLayout(1, M))
    no = tile_n * 32 + gl.arange(0, 32, gl.SliceLayout(0, M))
    gl.store(Y + mo[:, None] * 2048 + no[None, :], acc.to(gl.bfloat16))
