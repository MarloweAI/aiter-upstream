# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experimental canonical BF16 query GEMM, with no stock-dispatch changes.

The structural domain is M1..1024, N=K2048 on gfx950. That is not a measured
performance or numerical qualification claim for an untested M. Only independently
qualified useful shapes may later receive production routing.
"""

import functools

import torch


@functools.cache
def _available():
    try:
        from triton.experimental.gluon import language as gl

        from aiter.ops.triton.utils._triton.arch_info import get_arch

        cdna4 = gl.amd.cdna4
        return (
            get_arch() == "gfx950"
            and callable(cdna4.compute_efficient_padded_shared_layout)
            and all(
                callable(getattr(cdna4.async_copy, name, None))
                for name in (
                    "buffer_load_to_shared",
                    "commit_group",
                    "wait_group",
                    "load_shared_relaxed",
                )
            )
        )
    except (AttributeError, ImportError, RuntimeError):
        return False


def gemm_a16w16_query_accepts(x, w):
    """Structural capability only; no implicit copy or packed-weight reinterpretation."""
    return (
        x.dim() == w.dim() == 2
        and 1 <= x.shape[0] <= 1024
        and x.shape[1] == 2048
        and tuple(w.shape) == (2048, 2048)
        and x.dtype == w.dtype == torch.bfloat16
        and x.is_cuda
        and w.device == x.device
        and x.is_contiguous()
        and w.is_contiguous()
        and x.data_ptr() % 16 == w.data_ptr() % 16 == 0
        and getattr(w, "is_shuffled", False) is False
        and getattr(w, "is_shuffled_16x32", False) is False
        and _available()
    )


def _geometry(rows, block_n=None, grouped=None):
    if block_n is None:
        block_n = 16 if rows <= 64 else 32
    if block_n not in (16, 32):
        raise ValueError("block_n must be 16 or32")
    if grouped is None:
        grouped = block_n == 32
    if type(grouped) is not bool:
        raise TypeError("grouped must be bool")
    tiles = (rows + 31) // 32
    grid = (tiles * (2048 // block_n),) if grouped else (tiles, 2048 // block_n)
    return block_n, grouped, grid


def gemm_a16w16_query(x, w, *, block_n=None, grouped=None):
    """Direct experimental call. Rejected operands remain a caller fallback decision.

    The two saved compile-time geometries share one device body. Defaults retain
    the M64 32x16/2-wave and M128 32x32/4-wave launch choices; masking extends the
    arithmetic domain, without a separate tail kernel or a new backend.
    """
    if not gemm_a16w16_query_accepts(x, w):
        raise ValueError(
            "requires canonical contiguous/aligned BF16 X[M1..1024,2048]/W[2048,2048] on gfx950"
        )
    block_n, grouped, grid = _geometry(x.shape[0], block_n, grouped)
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_query import (
        _gemm_a16w16_query_kernel,
    )

    rows = x.shape[0]
    out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=x.device)
    _gemm_a16w16_query_kernel[grid](
        x,
        w,
        out,
        rows,
        x.stride(0),
        w.stride(0),
        block_n,
        grouped,
        num_warps=block_n // 8,
        num_stages=1,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
    )
    return out
