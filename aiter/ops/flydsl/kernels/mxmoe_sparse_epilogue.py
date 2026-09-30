# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Marlowe AI. All rights reserved.
"""Explicit, graph-stable opt-in for the guarded GLM layout-v2 G2 epilogue."""

from functools import partial
from typing import Any, Callable


def g2_sparse_epilogue_supported(
    *,
    gfx: str,
    M: int,
    hidden: int,
    intermediate: int,
    experts: int,
    topk: int,
    BM: int,
    BN: int,
    BK: int,
    SBM: int,
    a_dtype: str,
    b_dtype: str,
    epilog: str,
    out_dtype: str,
    bf16_lds: bool,
    kstatic: bool,
    persist: bool,
    bias: bool,
    is_ep: bool,
) -> bool:
    """Use actual route count before the atomic launcher's normalized cache key."""
    return (
        gfx == "gfx950"
        and M in (32, 64, 128)
        and (hidden, intermediate, experts, topk) == (6144, 256, 257, 9)
        and (BM, BN, BK, SBM) == (16, 128, 128, 16)
        and a_dtype == b_dtype == "fp4"
        and epilog == "atomic"
        and out_dtype == "bf16"
        and bf16_lds
        and kstatic
        and not (persist or bias or is_ep)
    )


def _sparse_epilogue_stage2(
    *,
    ordinary_stage2: Callable[..., Any],
    stage2_args: tuple[Any, ...],
    stage2_kwargs: dict[str, Any],
    expert_mask: Any,
    enabled: bool,
) -> Any:
    target = ordinary_stage2
    while isinstance(target, partial):
        target = target.func
    if (
        enabled
        and expert_mask is None
        and getattr(target, "__name__", "")
        in (
            "_mxfp4_a4w4_stage2_fw",
            "_flydsl_v2_stage2_wrapper",
        )
    ):
        kwargs = dict(stage2_kwargs, g2_skip_padded_lds=True)
        return ordinary_stage2(*stage2_args, **kwargs)
    return ordinary_stage2(*stage2_args, **stage2_kwargs)


def make_g2_sparse_epilogue_override(enabled: bool = True) -> Callable[..., Any]:
    """Bind once, then pass as ``fused_moe(..., _stage2_override=...)``.

    Native sorting, output zeroing and G1 run normally. Unsupported stage-two
    families and shapes retain their original dispatch. This callable changes no
    global dispatch or environment state during graph capture/replay. The full
    native caller passes its actual expert mask to this callback; an EP mask
    disables the opt-in even when the front stage-two wrapper would drop it.
    """
    override = partial(_sparse_epilogue_stage2, enabled=bool(enabled))
    override._uses_sparse_g2_epilogue = True
    return override
