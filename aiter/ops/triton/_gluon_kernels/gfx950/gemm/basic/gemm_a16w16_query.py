# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experimental full-K BF16 family; no production dispatch row enables it."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _issue(
    sx, sw, X, W, k, ox, ow, mask, WEIGHT_FIRST: gl.constexpr, MASKED: gl.constexpr
):
    if WEIGHT_FIRST:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(sw, gl.multiple_of(W + k, 16), ow)
    if MASKED:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            sx, gl.multiple_of(X + k, 16), ox, mask=mask, other=0
        )
    else:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(sx, gl.multiple_of(X + k, 16), ox)
    if not WEIGHT_FIRST:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(sw, gl.multiple_of(W + k, 16), ow)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit(repr=make_kernel_repr("_gemm_a16w16_query_kernel", []))
def _gemm_a16w16_query_kernel(
    X,
    W,
    Y,
    ROWS: gl.constexpr,
    XS: gl.constexpr,
    WS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    GROUPED: gl.constexpr,
):
    """One FP32 ascending full-K chain; one BF16 store, with no split-K.

    GROUPED partitions column tiles by program_id % 8 and iterates all row
    tiles in each partition. It is a tile permutation, not an XCD placement
    guarantee. The contiguous full-tile variants retain the saved launch shapes.
    """
    WAVES: gl.constexpr = BLOCK_N // 8
    MFMA: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, BLOCK_N // 16],
        tiles_per_warp=[1, 1],
    )
    A: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=8)
    B: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=8)
    LX: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[WAVES, 1],
        order=[1, 0],
    )
    LW: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[64, 1],
        warps_per_cta=[1, WAVES],
        order=[0, 1],
    )
    SX: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        A, [32, 512], gl.bfloat16
    )
    SW: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        B, [512, BLOCK_N], gl.bfloat16
    )
    sx0 = gl.allocate_shared_memory(gl.bfloat16, [32, 512], SX)
    sx1 = gl.allocate_shared_memory(gl.bfloat16, [32, 512], SX)
    sw0 = gl.allocate_shared_memory(gl.bfloat16, [512, BLOCK_N], SW)
    sw1 = gl.allocate_shared_memory(gl.bfloat16, [512, BLOCK_N], SW)
    if GROUPED:
        row_tiles: gl.constexpr = (ROWS + 31) // 32
        physical = gl.program_id(0)
        tile_m = (physical // 8) % row_tiles
        tile_n = (physical % 8) * (2048 // BLOCK_N // 8) + (physical // 8) // row_tiles
    else:
        tile_m, tile_n = gl.program_id(0), gl.program_id(1)
    m = tile_m * 32 + gl.arange(0, 32, gl.SliceLayout(1, LX))
    n = tile_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, LW))
    kx = gl.arange(0, 512, gl.SliceLayout(0, LX))
    kw = gl.arange(0, 512, gl.SliceLayout(1, LW))
    ox = m[:, None] * XS + kx[None, :]
    ow = n[None, :] * WS + kw[:, None]
    if ROWS % 32:
        mask = m[:, None] < ROWS
    else:
        mask = None
    _issue(sx0, sw0, X, W, 0, ox, ow, mask, BLOCK_N == 16, ROWS % 32 != 0)
    _issue(sx1, sw1, X, W, 512, ox, ow, mask, BLOCK_N == 16, ROWS % 32 != 0)
    acc = gl.full((32, BLOCK_N), 0, gl.float32, MFMA)
    for block in gl.static_range(4):
        gl.amd.cdna4.async_copy.wait_group(1 if block < 3 else 0)
        if block % 2 == 0:
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(sx0, A)
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(sw0, B)
        else:
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(sx1, A)
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(sw1, B)
        acc = gl.amd.cdna3.mfma(x, w, acc)
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
                _issue(
                    sx0, sw0, X, W, next_k, ox, ow, mask, BLOCK_N == 16, ROWS % 32 != 0
                )
            else:
                _issue(
                    sx1, sw1, X, W, next_k, ox, ow, mask, BLOCK_N == 16, ROWS % 32 != 0
                )
    mo = tile_m * 32 + gl.arange(0, 32, gl.SliceLayout(1, MFMA))
    no = tile_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, MFMA))
    if ROWS % 32:
        gl.store(
            Y + mo[:, None] * 2048 + no[None, :],
            acc.to(gl.bfloat16),
            mask=mo[:, None] < ROWS,
        )
    else:
        gl.store(Y + mo[:, None] * 2048 + no[None, :], acc.to(gl.bfloat16))
