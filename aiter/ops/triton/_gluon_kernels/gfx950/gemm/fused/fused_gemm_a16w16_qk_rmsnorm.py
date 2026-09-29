# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon split-K producer of the MLA input projection for gfx950."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gemm_a16w16_splitk_planes_repr = make_kernel_repr(
    "_gemm_a16w16_splitk_planes_kernel", ["BLOCK_M"]
)


@gluon.jit(repr=_gemm_a16w16_splitk_planes_repr)
def _gemm_a16w16_splitk_planes_kernel(x_ptr, w_ptr, partial_ptr, BLOCK_M: gl.constexpr):
    """Six fp32 split-K planes of x [BLOCK_M, 6144] @ w [2624, 6144]^T.

    Launched on 246 workgroups: 41 column tiles of 64 times 6 K ranges of 1024,
    one wave on a 256-CU gfx950. Workgroup (feature, plane) computes the
    [BLOCK_M, 64] tile of columns 64 * feature over K range plane and stores it
    unrounded to partial [plane, :, 64 * feature : 64 * (feature + 1)], so every
    weight element is read by exactly one workgroup. Each K range accumulates
    16 K64 steps, each two ascending K32 MFMAs, in fp32.

    x and w are row-major and contiguous; BLOCK_M is 128 or 256. Operands move to
    LDS by direct async copies, NUM_BUFFERS K64 steps ahead of the MFMAs.
    """
    # Four K64 steps in flight; at BLOCK_M = 256 four buffer pairs would exceed
    # the 160 KiB of LDS, so three.
    NUM_BUFFERS: gl.constexpr = 4 if BLOCK_M == 128 else 3
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=False,
        warps_per_cta=[2, 2],
        tiles_per_warp=[BLOCK_M // 32, 2],
    )
    x_operand: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    w_operand: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    x_copy: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    w_copy: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    x_lds: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [BLOCK_M, 64], [1, 0]
    )
    w_lds: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [64, 64], [0, 1]
    )
    sx0 = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, 64], x_lds)
    sx1 = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, 64], x_lds)
    sx2 = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, 64], x_lds)
    if NUM_BUFFERS == 4:
        sx3 = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, 64], x_lds)
    sw0 = gl.allocate_shared_memory(gl.bfloat16, [64, 64], w_lds)
    sw1 = gl.allocate_shared_memory(gl.bfloat16, [64, 64], w_lds)
    sw2 = gl.allocate_shared_memory(gl.bfloat16, [64, 64], w_lds)
    if NUM_BUFFERS == 4:
        sw3 = gl.allocate_shared_memory(gl.bfloat16, [64, 64], w_lds)

    # Workgroup p runs on XCD p % 8. XCDs 0-5 take 31 consecutive (feature, plane)
    # tiles each and XCDs 6-7 take 30, so the planes of a feature share an XCD.
    pid = gl.program_id(0)
    xcd = pid % 8
    slot = pid // 8
    tile = slot + xcd * gl.where(xcd < 6, 31, 30) + gl.where(xcd < 6, 0, 6)
    feature = tile // 6
    plane = tile % 6

    rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, x_copy))
    x_k = gl.arange(0, 64, gl.SliceLayout(0, x_copy))
    w_k = gl.arange(0, 64, gl.SliceLayout(1, w_copy))
    cols = gl.arange(0, 64, gl.SliceLayout(0, w_copy))
    x_offsets = rows[:, None] * 6144 + x_k[None, :]
    w_offsets = cols[None, :] * 6144 + w_k[:, None]
    x_base = plane * 1024
    w_base = feature * 64 * 6144 + plane * 1024

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        sx0, gl.multiple_of(x_ptr + x_base, 16), x_offsets
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        sw0, gl.multiple_of(w_ptr + w_base, 16), w_offsets, cache_modifier=".cg"
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        sx1, gl.multiple_of(x_ptr + x_base + 64, 16), x_offsets
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        sw1, gl.multiple_of(w_ptr + w_base + 64, 16), w_offsets, cache_modifier=".cg"
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        sx2, gl.multiple_of(x_ptr + x_base + 128, 16), x_offsets
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        sw2, gl.multiple_of(w_ptr + w_base + 128, 16), w_offsets, cache_modifier=".cg"
    )
    gl.amd.cdna4.async_copy.commit_group()
    if NUM_BUFFERS == 4:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            sx3, gl.multiple_of(x_ptr + x_base + 192, 16), x_offsets
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            sw3,
            gl.multiple_of(w_ptr + w_base + 192, 16),
            w_offsets,
            cache_modifier=".cg",
        )
        gl.amd.cdna4.async_copy.commit_group()

    acc = gl.full((BLOCK_M, 64), 0, gl.float32, mfma_layout)
    for block in gl.static_range(16):
        gl.amd.cdna4.async_copy.wait_group(
            NUM_BUFFERS - 1 if block < 17 - NUM_BUFFERS else 15 - block
        )
        if block % NUM_BUFFERS == 0:
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(sx0, x_operand)
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(sw0, w_operand)
        elif block % NUM_BUFFERS == 1:
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(sx1, x_operand)
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(sw1, w_operand)
        elif block % NUM_BUFFERS == 2:
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(sx2, x_operand)
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(sw2, w_operand)
        else:
            x = gl.amd.cdna4.async_copy.load_shared_relaxed(sx3, x_operand)
            w = gl.amd.cdna4.async_copy.load_shared_relaxed(sw3, w_operand)
        acc = gl.amd.cdna3.mfma(x, w, acc)
        if block < 16 - NUM_BUFFERS:
            # Wait for this workgroup's LDS reads, not for its outstanding copies,
            # before the buffer just read is refilled.
            gl.inline_asm_elementwise(
                "s_waitcnt lgkmcnt(0)\n\ts_barrier\n\tv_mov_b32_e32 $0, 0",
                constraints="=v,~{memory}",
                args=[],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            x_next = gl.multiple_of(x_ptr + x_base + (block + NUM_BUFFERS) * 64, 16)
            w_next = gl.multiple_of(w_ptr + w_base + (block + NUM_BUFFERS) * 64, 16)
            if block % NUM_BUFFERS == 0:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(sx0, x_next, x_offsets)
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sw0, w_next, w_offsets, cache_modifier=".cg"
                )
            elif block % NUM_BUFFERS == 1:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(sx1, x_next, x_offsets)
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sw1, w_next, w_offsets, cache_modifier=".cg"
                )
            elif block % NUM_BUFFERS == 2:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(sx2, x_next, x_offsets)
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sw2, w_next, w_offsets, cache_modifier=".cg"
                )
            else:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(sx3, x_next, x_offsets)
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    sw3, w_next, w_offsets, cache_modifier=".cg"
                )
            gl.amd.cdna4.async_copy.commit_group()

    out_rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma_layout))
    out_cols = feature * 64 + gl.arange(0, 64, gl.SliceLayout(0, mfma_layout))
    gl.store(
        partial_ptr
        + plane * BLOCK_M * 2624
        + out_rows[:, None] * 2624
        + out_cols[None, :],
        acc,
    )
