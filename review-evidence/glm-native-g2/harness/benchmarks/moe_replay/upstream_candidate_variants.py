"""Matched, supplied-route removals from the selected Port-EW backend.

These are benchmark factories, not changes to serving policy. Every pair uses
the same caller-supplied int32 IDs and FP32 weights. Ablations are not additive
AITER PR increments: the measured delta is selected versus selected-minus-X.
All GPU/PyHIP imports stay lazy so contracts can be inspected on a CPU host.
"""

from dataclasses import asdict, dataclass, fields
from functools import cache
import hashlib
import json
from pathlib import Path
from typing import Any


PRIMARY_ROWS = (4, 8, 16, 32, 64, 128, 256)
SUPPORTED_ROWS = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class GroupSpec:
    name: str
    label: str
    rows: tuple[int, ...]
    baseline: str
    candidate: str = "selected_supplied"
    dependencies: tuple[str, ...] = ("native_mxfp4_backend",)
    contract_change: bool = False
    limitation: str = ""


GROUPS = {
    "g2_coalesced_sparse": GroupSpec(
        "g2_coalesced_sparse", "LDS coalesced and sparse G2 output", (8, 16, 32, 64, 128),
        "g2_wave", limitation="Wave coalescing remains in the baseline; this does not price the entire historical scattered-to-coalesced change.",
    ),
    "g2_sparse_epilogue": GroupSpec(
        "g2_sparse_epilogue", "Skip empty LDS output chunks", (8, 16, 32, 64, 128),
        "g2_lds_dense", dependencies=("native_mxfp4_backend", "g2_lds_coalescing"),
        limitation="Subset of g2_coalesced_sparse; never sum both deltas as independent PR gains.",
    ),
    "paired_load_scheduling": GroupSpec(
        "paired_load_scheduling", "Paired G1/G2 scales and packed input-scale loads", (8, 16, 32, 64, 128),
        "unpaired_loads", limitation="Same matrix tiles/split/route grid. Scheduling and scale-load width change together; required ISA waits remain enabled.",
    ),
    "small_m_fusion": GroupSpec(
        "small_m_fusion", "Direct small-row preparation and consumer fusion", (1, 2, 4),
        "small_separate", dependencies=("native_mxfp4_backend", "g2_wave_coalescing"),
        limitation="One bundle removes inline G1 route/input preparation and G2 local split reduction/packing. The output combination stays BF16 atomic.",
    ),
    "small_m_g1_overlap": GroupSpec(
        "small_m_g1_overlap", "Overlap G1 weight requests with input packing", (1, 2, 4),
        "small_serial_pack", dependencies=("small_m_fusion",),
        limitation="Subset of small_m_fusion; not an independent additive delta.",
    ),
    "small_m_scale_predecode": GroupSpec(
        "small_m_scale_predecode", "Predecode independent G1 scale operands", (1, 2, 4),
        "small_serial_scales", dependencies=("small_m_fusion",),
        limitation="Baseline retains required explicit MFMA scale-input NOPs. Subset, not an additive delta.",
    ),
    "m8_ballot_merge": GroupSpec(
        "m8_ballot_merge", "Consumer-side eight-row route merge", (8,),
        "m8_direct", dependencies=("native_mxfp4_backend", "paired_load_scheduling", "g2_coalesced_sparse"),
        limitation="Direct grid72 versus ballot grid65; grouped row layout and compatible output epilogue necessarily change together.",
    ),
    "fixed_expert_dispatch": GroupSpec(
        "fixed_expert_dispatch", "Fixed expert segments and double-buffered counts", (16, 32, 64, 128),
        "chunk_map_dispatch", dependencies=("native_mxfp4_backend", "paired_load_scheduling", "g2_coalesced_sparse"),
        limitation="The chunk-map baseline reuses existing emitters with a new launch adapter and needs fresh GPU qualification. Scratch layout/state reset and route resolution are inseparable.",
    ),
    "supplied_direct_combine": GroupSpec(
        "supplied_direct_combine", "Direct BF16 supplied-route accumulation", (1, 2, 4, 8, 16, 32, 64, 128),
        "ordered_supplied", contract_change=True,
        limitation="BF16-rounded contributions are unchanged, but ordered FP32 sum becomes BF16 atomic accumulation. Both require explicit own-contract references; exclude from arithmetic-preserving ranking.",
    ),
}


def group_manifest() -> dict[str, Any]:
    """Serializable descriptions without importing Torch or GPU runtimes."""
    return {
        "schema": 1,
        "comparison": "selected backend minus one group versus selected backend",
        "boundary": "supplied routes through complete local MoE output",
        "row_contract": "EAGLE 3-1-4 target verification: M=4*C",
        "primary_rows": list(PRIMARY_ROWS),
        "unsupported": {"256": "No qualified custom target-C64 program; do not extend it here."},
        "groups": {name: asdict(spec) for name, spec in GROUPS.items()},
        "reference": {
            "input": "BF16 [M,6144]",
            "routes": "int32 [M,8], distinct within each token",
            "route_weights": "FP32 [M,8], finite, native supplied values",
            "shared": "Append expert256/weight1 exactly once",
            "intermediate": "FP32 SiLU/up before MXFP4 at M1/4/8/32; BF16 storage rounding before MXFP4 at M2/16/64/128",
            "output": "BF16 expert operands and direct BF16 atomic sum, except declared ordered baseline",
            "gates": "Independent eager/graph references, exact IDs/weights, zero-input clearing, changing-input replay, state and source identity; thresholds unchanged",
        },
    }


def supported_groups(rows: int, *, include_contract_changes: bool = False) -> tuple[str, ...]:
    return tuple(name for name, spec in GROUPS.items() if rows in spec.rows and
                 (include_contract_changes or not spec.contract_change))


def _selected_config(rows: int) -> dict[str, Any]:
    from marlowe_kernels._kernels.amd.glm52_moe.bandwidth_program import BANDWIDTH_CONFIGS

    return dict(BANDWIDTH_CONFIGS[rows]["config"])


def _coalesced_config(config: dict[str, Any]) -> dict[str, Any]:
    from marlowe_kernels._kernels.amd.glm52_moe.coalesced_program import CoalescedConfig

    return {f.name: config[f.name] for f in fields(CoalescedConfig) if f.name in config}


def _emitted_policy(rows: int, variant: str) -> dict[str, Any]:
    """Describe overrides applied by methods, not only inherited dataclasses."""
    config = _selected_config(rows)
    policy = dict(config)
    policy["supplied_combine"] = "bf16_atomic"
    if variant == "ordered_supplied":
        policy["supplied_combine"] = "ordered_fp32"
    elif variant == "g2_wave":
        policy["combine"] = "wave"
    elif variant == "g2_lds_dense":
        policy["combine"] = "lds"
    elif variant == "unpaired_loads":
        policy.update(g1_paired=False, g2_paired=False, packed_input_scales=False,
                      g1_scalar_prefetch_depth=config["depth"],
                      g2_scalar_prefetch_depth=config["down_depth"] or 2)
    elif variant == "small_separate":
        policy.update(inline_input_prepare=False, consumer_middle_fusion=False,
                      g1_paired=False, g2_weight_middle_overlap=False)
    elif variant == "small_serial_pack":
        policy["pack_overlap"] = False
    elif variant == "small_serial_scales":
        policy["predecode_scales"] = False
    elif variant == "m8_direct":
        policy.update(direct=True, route_grid=rows * 9,
                      output_epilogue="wave, because one direct route owns a tile")
    elif variant == "chunk_map_dispatch":
        policy["dispatch_layout"] = "chunk-map publication, banked compact rows"
    return policy


def _fixed_unpaired(owner: Any, rows: int):
    """Retain fixed routing, tiles and resets while using existing scalar loads."""
    from marlowe_kernels._kernels.amd.glm52_moe.fixed_bins import (
        FixedBinConfig, FixedBinProgram, fixed_projection,
    )

    config = _selected_config(rows)
    config.pop("ordered_supplied", None)
    config.pop("ordered_all", None)

    class UnpairedFixed(FixedBinProgram):
        def _stage(self, inp, scale, out, first):
            c, p, b = self.fc, self.p, self.b
            n, k, bn, nw = ((512, 6144, c.gate_n, c.gate_waves) if first else
                            (6144, 256, c.down_n, c.down_waves))
            weights, scales = ((self.owner.w13, self.owner.weights.w13_scale) if first else
                               (self.owner.w2, self.owner.weights.w2_scale))
            depth = c.depth if first else c.down_depth or 2
            fixed_projection([n // bn, self.grid, c.split_k if first else 1], [64 * nw],
                k, n, bn, first, c.split_k if first else 1, self.fp32_middle, nw,
                c.non_temporal, depth, False, c.combine, b, inp.data_ptr(), scale,
                weights.data_ptr(), scales.data_ptr(), out.data_ptr(), self.mid_scale.data_ptr(),
                p["sorted_ids"].data_ptr(), p["sorted_weights"].data_ptr(),
                p["experts"].data_ptr(), self.state.data_ptr(), b)

    return UnpairedFixed(owner, rows, FixedBinConfig(**config))


@cache
def _chunk_projection():
    """Reuse current paired/coalesced bodies with historical chunk-map routing."""
    from marlowe_kernels._runtime import jit

    @jit()
    def variant_chunk_projection(J, K, N, BN, FIRST, SK, FP32, NW, NT, DEPTH,
            PAIR, PACKED_SCALES, COMBINE, BATCH, MAXIMUM,
            X: "void*", XS: "void*", W: "void*", WS: "void*", Y: "void*", YS: "void*",
            IDS: "void*", RW: "void*", EXP: "void*", STATE: "void*", B: "int"):
        from marlowe_kernels._kernels.amd.glm52_moe.block_fused import pointer
        from marlowe_kernels._kernels.amd.glm52_moe.fused_dispatch import valid_pointer, clear_next
        from marlowe_kernels._kernels.amd.glm52_moe.native_pairprefetch import paired_projection, packed_input_projection
        from marlowe_kernels._kernels.amd.glm52_moe.native_prefetch import prefetch_projection
        from marlowe_kernels._kernels.amd.glm52_moe.native_coalesced import coalesced_projection

        valid, bank = valid_pointer(J, STATE)
        ids = pointer(J, IDS, bank[0] * (MAXIMUM * 4))
        weights = pointer(J, RW, bank[0] * (MAXIMUM * 4))
        args = (K, N, BN, FIRST, SK, FP32, NW, False, True, NT)
        operands = (X, XS, W, WS, Y, YS, ids, weights, EXP, valid, B)
        if PAIR:
            emit = packed_input_projection if PACKED_SCALES else paired_projection
            emit.gen_func(J, *args, DEPTH, COMBINE, *operands)
        elif FIRST:
            prefetch_projection.gen_func(J, *args, DEPTH, *operands)
        else:
            coalesced_projection.gen_func(J, *args, DEPTH, COMBINE, *operands)
        if not FIRST:
            clear_next(J, STATE, IDS, RW, BATCH, 64 * NW, N // BN, MAXIMUM)

    # The package JIT digest covers marlowe_kernels/, not this benchmark module.
    # Bind this wrapper's source to its compiled-cache name as well as receipts.
    variant_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    variant_chunk_projection.func_name += "_" + variant_digest
    variant_chunk_projection.gen_func_unique_id += "_" + variant_digest
    return variant_chunk_projection


def _chunk_map_program(owner: Any, rows: int):
    from marlowe_kernels._kernels.amd.glm52_moe.front_program import FrontConfig, FrontProgram

    selected = _selected_config(rows)
    config = dict(_coalesced_config(selected), route_preload=True, grid_cap=True)

    class PairedChunkMap(FrontProgram):
        def _stage(self, inp, scale, out, first):
            c, p, b = self.fc, self.p, self.b
            n, k, bn, nw = ((512, 6144, c.gate_n, c.gate_waves) if first else
                            (6144, 256, c.down_n, c.down_waves))
            pair = bool(selected["pair_depth"]) if first else selected["pair_g2"]
            depth = (selected["pair_depth"] if first and pair else c.depth if first else
                     1 if pair else c.down_depth or 2)
            weights, scales = ((self.owner.w13, self.owner.weights.w13_scale) if first else
                               (self.owner.w2, self.owner.weights.w2_scale))
            _chunk_projection()([n // bn, self.grid, c.split_k if first else 1], [64 * nw],
                k, n, bn, first, c.split_k if first else 1, self.fp32_middle, nw,
                c.non_temporal, depth, pair, selected.get("packed_input_scales", False),
                c.combine, b, self.maximum, inp.data_ptr(), scale, weights.data_ptr(),
                scales.data_ptr(), out.data_ptr(), self.mid_scale.data_ptr(),
                p["sorted_ids"].data_ptr(), p["sorted_weights"].data_ptr(),
                p["experts"].data_ptr(), self.state.data_ptr(), b)

    return PairedChunkMap(owner, rows, FrontConfig(**config))


def _make_program(owner: Any, rows: int, variant: str):
    from marlowe_kernels._kernels.amd.glm52_moe.native_combine import create_native_combine_program

    if variant == "selected_supplied":
        return create_native_combine_program(owner, rows)
    if variant == "ordered_supplied":
        from marlowe_kernels._kernels.amd.glm52_moe.bandwidth_program import create_bandwidth_program
        return create_bandwidth_program(owner, rows)
    if variant in ("g2_wave", "g2_lds_dense"):
        return create_native_combine_program(owner, rows,
            {"combine": "wave" if variant == "g2_wave" else "lds"})
    if variant == "unpaired_loads":
        if rows >= 16:
            return _fixed_unpaired(owner, rows)
        from marlowe_kernels._kernels.amd.glm52_moe.pair_program import PairConfig, PairProgram
        config = _selected_config(rows)
        config.update(pair_depth=0, pair_g2=False, packed_input_scales=False)
        return PairProgram(owner, rows, PairConfig(**config))
    if variant == "small_separate":
        from marlowe_kernels._kernels.amd.glm52_moe.coalesced_program import CoalescedConfig, CoalescedProgram
        return CoalescedProgram(owner, rows, CoalescedConfig(**_coalesced_config(_selected_config(rows))))
    if variant == "small_serial_pack":
        return create_native_combine_program(owner, rows, {"pack_overlap": False})
    if variant == "small_serial_scales":
        return create_native_combine_program(owner, rows, {"predecode_scales": False})
    if variant == "m8_direct":
        from marlowe_kernels._kernels.amd.glm52_moe.pair_program import PairConfig, PairProgram
        config = _selected_config(rows)
        config["direct"] = True
        return PairProgram(owner, rows, PairConfig(**config))
    if variant == "chunk_map_dispatch":
        return _chunk_map_program(owner, rows)
    raise ValueError(f"Unknown variant {variant!r}")


class PreparedVariant:
    """Uniform supplied-route interface; caller owns correctness and timing."""

    def __init__(self, owner: Any, rows: int, group: str, enabled: bool):
        if group not in GROUPS:
            raise ValueError(f"Unknown group {group!r}")
        self.spec = GROUPS[group]
        if type(rows) is not int or rows not in self.spec.rows:
            raise ValueError(f"{group} does not support M={rows}; no implicit padding or fallback")
        self.rows, self.enabled = rows, enabled
        self.variant = self.spec.candidate if enabled else self.spec.baseline
        self.program = _make_program(owner, rows, self.variant)
        self.fp32_middle = self.program.fp32_middle
        if self.fp32_middle != (rows in (1, 4, 8, 32)):
            raise ValueError("Factory changed the established intermediate rounding contract")
        self.contract = {
            "routing": "supplied_native_distinct",
            "fp32_middle": self.fp32_middle,
            "combination": "ordered_fp32" if self.variant == "ordered_supplied" else "bf16_atomic",
            "expert_contributions": "BF16 rounded",
            "reference": "independent MXFP4 algebra for identical routes and declared intermediate/output contracts",
            "contract_change_pair": self.spec.contract_change,
        }
        self.info = {
            "group": group, "variant": self.variant, "rows": rows, "enabled": enabled,
            "candidate_config": _selected_config(rows),
            "emitted_policy": _emitted_policy(rows, self.variant),
            "grid": self.program.grid,
            "program_type": type(self.program).__name__,
            "program_info": getattr(self.program, "info", {}),
            "contract": self.contract, "limitation": self.spec.limitation,
            "library_source_sha256": self._library_source(),
            "variant_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        encoded = json.dumps(self.info, sort_keys=True, separators=(",", ":")).encode()
        self.info["configuration_sha256"] = hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _library_source() -> str:
        from marlowe_kernels._runtime import source_digest
        return source_digest()

    def run(self, x: Any, ids: Any, weights: Any, out: Any = None):
        # No host tensor-value reads in replay. The executor independently
        # validates ID distinctness/range, finite weights, aliasing and devices.
        return self.program.run(x, x, x, out=out, supplied_ids=ids, supplied_weights=weights)

    def state_check(self) -> dict[str, Any]:
        if hasattr(self.program, "state_check"):
            return self.program.state_check()
        if not hasattr(self.program, "state"):
            return {"passed": True, "stateless": True}
        # Historical chunk-map state shares one count/map region. G2 resets
        # that region after preparation's last consumer; row storage is banked.
        state = self.program.state.cpu()
        next_bank, current_bank = int(state[0]), int(state[1])
        if next_bank not in (0, 1) or current_bank not in (0, 1):
            return {"passed": False, "next_bank": next_bank,
                    "current_bank": current_bank, "reason": "Invalid bank selector"}
        next_rows = int(state[2 + next_bank])
        result = {
            "next_bank": next_bank, "current_bank": current_bank,
            "counts_and_maps_zero": not bool(state[4:-1].any()),
            "next_valid_rows": next_rows, "error": int(state[-1]),
        }
        result["passed"] = (next_bank in (0, 1) and current_bank in (0, 1) and
            next_bank != current_bank and result["counts_and_maps_zero"] and
            next_rows == ((self.rows + 15) // 16) * 16 and result["error"] == 0)
        return result


def make_variant(owner: Any, rows: int, group: str, enabled: bool = True) -> PreparedVariant:
    return PreparedVariant(owner, rows, group, enabled)
