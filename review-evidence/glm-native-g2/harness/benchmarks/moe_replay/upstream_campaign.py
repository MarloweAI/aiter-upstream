"""Strict, single-GPU supplied-route MoE removal-ablation campaign.

Every arm qualifies before any clean timing. Conditional group deltas are not
additive upstream PR gains. Captures and current AITER provenance are mandatory.
"""

import argparse
import importlib
import json
import os
import random
import socket
import subprocess
import time
import traceback
from pathlib import Path

from .upstream_candidate_variants import GROUPS, group_manifest, make_variant
from .upstream_contract import (
    CONCURRENCIES,
    QualificationFailure,
    file_digest,
    marginal_summary,
    per_case_estimator,
    require_gate,
    select_cases,
)
from .upstream_profile import write_dump

PRIMARY_GROUPS = (
    "g2_coalesced_sparse",
    "paired_load_scheduling",
    "small_m_fusion",
    "m8_ballot_merge",
    "fixed_expert_dispatch",
)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def callable_info(function):
    base = getattr(function, "func", function)
    return {
        "module": getattr(base, "__module__", ""),
        "name": getattr(base, "__name__", ""),
        "kwargs": {
            k: str(v) for k, v in (getattr(function, "keywords", {}) or {}).items()
        },
    }


def metadata_info(metadata):
    return {
        "stage1": callable_info(metadata.stage1),
        "stage2": callable_info(metadata.stage2),
        **{
            key: str(getattr(metadata, key, None))
            for key in (
                "block_m",
                "ksplit",
                "fuse_quant",
                "output_aux",
                "full_impl",
                "prequant",
            )
        },
    }


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aiter-source", type=Path, required=True)
    parser.add_argument("--aiter-sha", required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--sglang-sha", required=True)
    parser.add_argument("--library-digest", required=True)
    parser.add_argument("--variant-digest", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=list(CONCURRENCIES)
    )
    parser.add_argument(
        "--groups", choices=PRIMARY_GROUPS, nargs="+", default=list(PRIMARY_GROUPS)
    )
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--steps", type=int, nargs="*")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=355052)
    parser.add_argument("--profile-cases", type=int, default=2)
    parser.add_argument(
        "--comparison-mode",
        choices=("original", "native-only", "ordered-recovery", "native-port"),
        default="original",
    )
    parser.add_argument("--ordered-digest")
    parser.add_argument("--native-port-manifest", type=Path)
    return parser.parse_args()


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=False)
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run in an owned single-GPU Slurm allocation")
    if any(c not in CONCURRENCIES for c in args.concurrencies):
        raise SystemExit("Unsupported concurrency")
    if args.samples < 1 or args.warmups < 0 or args.profile_cases < 1:
        raise SystemExit("Invalid repetition/profile counts")
    if args.comparison_mode == "ordered-recovery" and args.concurrencies != [32]:
        raise SystemExit("Versioned ordered recovery is M128/C32 only")
    port_manifest = None
    if args.comparison_mode == "native-port":
        from .upstream_native_port import (
            read_manifest,
            validate_harness,
            validate_source,
        )

        if args.native_port_manifest is None:
            raise SystemExit("Native-port mode requires its versioned manifest")
        port_manifest = read_manifest(args.native_port_manifest)
        validate_source(port_manifest, args.aiter_source)
        validate_harness(port_manifest, Path(__file__).resolve().parents[2])
        if not set(args.concurrencies) <= set(port_manifest["concurrencies"]):
            raise SystemExit(
                "Run only affected native-port Cs; reducer reports fallback"
            )
    import aiter
    import torch
    import triton

    from marlowe_kernels import MoEConfig, MoEWeights, create_moe
    from marlowe_kernels._runtime import source_digest

    from .events import Event
    from .kernels import touch_input
    from .reference import compare
    from .upstream_reference import (
        candidate_contract,
        expert_reference,
        native_contract,
        unpack_weights,
    )

    if torch.cuda.device_count() != 1 or torch.version.hip is None:
        raise SystemExit("Requires exactly one visible AMD GPU")
    module = importlib.import_module("aiter.fused_moe")
    variant_file = Path(__file__).with_name("upstream_candidate_variants.py")
    ordered_file = Path(__file__).with_name("upstream_ordered_variant.py")
    if args.comparison_mode == "ordered-recovery":
        if not args.ordered_digest or file_digest(ordered_file) != args.ordered_digest:
            raise SystemExit("Ordered recovery source identity mismatch")
        from .upstream_ordered_variant import make_ordered_variant

    def source_valid():
        return (
            (
                port_manifest is None
                or (
                    validate_source(port_manifest, args.aiter_source)
                    and validate_harness(
                        port_manifest, Path(__file__).resolve().parents[2]
                    )
                )
            )
            and (
                port_manifest is None
                or file_digest(sorting_module.__file__)
                == port_manifest["sorting_adapter_sha256"]
            )
            and source_digest() == args.library_digest
            and file_digest(variant_file) == args.variant_digest
            and (
                args.comparison_mode != "ordered-recovery"
                or file_digest(ordered_file) == args.ordered_digest
            )
        )

    if (
        source_digest() != args.library_digest
        or file_digest(variant_file) != args.variant_digest
    ):
        raise SystemExit("Candidate source identity mismatch")
    if revision(args.aiter_source) != args.aiter_sha:
        raise SystemExit("AITER source revision mismatch")
    imported = Path(module.__file__).resolve()
    if not imported.is_relative_to(args.aiter_source.resolve()):
        raise SystemExit(f"Wrong imported AITER source: {imported}")
    changes = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.aiter_source),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    )
    if changes:
        raise SystemExit("Current AITER source has tracked edits")
    if revision(args.sglang_source) != args.sglang_sha:
        raise SystemExit("SGLang source revision mismatch")
    sorting_module = importlib.import_module("sglang.kernels.ops.moe.moe_sorting_small")
    if (
        not Path(sorting_module.__file__)
        .resolve()
        .is_relative_to(args.sglang_source.resolve())
    ):
        raise SystemExit("Wrong imported SGLang sorting adapter")
    if (
        port_manifest
        and file_digest(sorting_module.__file__)
        != port_manifest["sorting_adapter_sha256"]
    ):
        raise SystemExit("Historical/current SGLang sorting adapter byte proof failed")
    sorting_module.apply_aiter_small_moe_sort_patch()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    identity = {
        "schema": 1,
        "job": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "aiter_sha": args.aiter_sha,
        "aiter_source": str(imported),
        "sglang_sha": args.sglang_sha,
        "native_sglang_sorting_adapter": str(sorting_module.__file__),
        "native_sorting_callable": callable_info(module._moe_sorting_impl),
        "library_digest": source_digest(),
        "variant_digest": file_digest(variant_file),
        "comparison_mode": args.comparison_mode,
        "ordered_digest": args.ordered_digest,
        "native_port_manifest": port_manifest,
        "native_port_manifest_sha256": file_digest(args.native_port_manifest)
        if port_manifest
        else None,
        "runner_sha256": file_digest(__file__),
        "capture": str(args.capture),
        "runtime": json.loads(args.runtime_manifest.read_text()),
        "samples": args.samples,
        "warmups": args.warmups,
        "boundary": "supplied native routes through complete local combination",
        "estimator": "mean of per-case medians, randomized paired graph samples",
        "scope": "one TP8/EP1 rank; captured sampled layers only; no collectives or router",
        "group_meaning": "matched native baseline-versus-source-patch comparison"
        if port_manifest
        else "conditional selected-minus-group contribution; not additive upstream PR gains",
    }
    save(args.output / "identity.json", identity)
    save(args.output / "groups.json", group_manifest())
    owners, weights, weight_snapshots, route_views, receipts = {}, {}, {}, {}, []
    if port_manifest:
        from .upstream_native_port import coverage, make_callback

        receipts.extend(coverage(port_manifest))
        port_callback = make_callback(port_manifest, args.aiter_source)

    def load_layer(layer):
        if layer not in weights:
            path = args.capture / "weights" / f"layer-{layer}.pt"
            payload = torch.load(path, weights_only=True, map_location="cuda")
            w = {key: payload[key] for key in ("w1", "w2", "w1_scale", "w2_scale")}
            for key in ("w1", "w2"):
                w[key] = w[key].view(torch.float4_e2m1fn_x2)
                w[key].is_shuffled = True
            for key in ("w1_scale", "w2_scale"):
                w[key] = w[key].view(torch.float8_e8m0fnu)
            if tuple(w["w1"].shape) != (257, 512, 3072) or tuple(w["w2"].shape) != (
                257,
                6144,
                128,
            ):
                raise QualificationFailure("Captured TP8 packed weight layout mismatch")
            weights[layer] = w
            weight_snapshots[layer] = {
                key: tensor.view(torch.uint8).clone() for key, tensor in w.items()
            }
            if port_manifest is None:
                owners[layer] = create_moe(
                    MoEConfig(),
                    MoEWeights(w["w1"], w["w2"], w["w1_scale"], w["w2_scale"]),
                    implementation="port_ew",
                )
            return file_digest(path)
        return None

    class Native:
        def __init__(self, layer, x, ids, route_weights, *, enabled=False):
            self.layer = layer
            self.override = port_callback if enabled else None
            self.enabled = enabled
            records = []
            old = module.get_2stage_cfgs

            def observe(*a, **kw):
                value = old(*a, **kw)
                records.append(value)
                return value

            module.get_2stage_cfgs = observe
            try:
                self.run(x, ids, route_weights)
                torch.cuda.synchronize()
            finally:
                module.get_2stage_cfgs = old
            if not records:
                raise QualificationFailure("Native dispatch metadata was not observed")
            self.metadata = records[-1]
            self.reference_contract = native_contract(self.metadata)
            self.info = {
                "backend": "current AITER native port"
                if self.enabled
                else "current AITER",
                "native_override": port_manifest["stage2_factory"]
                if self.enabled
                else None,
                "static_kwargs": port_manifest["stage2_kwargs"] if self.enabled else {},
                "metadata": metadata_info(self.metadata),
                "resolution_sequence": [metadata_info(value) for value in records],
                "reference_contract": self.reference_contract,
            }

        def run(self, x, ids, route_weights, *, capture_override=None):
            override = (
                capture_override if capture_override is not None else self.override
            )
            return module.fused_moe(
                x,
                **weights[self.layer],
                topk_ids=ids,
                topk_weight=route_weights,
                activation=aiter.ActivationType.Silu,
                quant_type=aiter.QuantType.per_1x32,
                gate_mode=module.GateMode.SEPARATED.value,
                **({"_stage2_override": override} if override is not None else {}),
            )

        def state_check(self):
            return {
                "passed": True,
                "scratch_contract": "native API checked through alternating-input and zero-input replay",
            }

    def invoke(arm, x, ids, route_weights):
        return (
            arm.run(x, ids, route_weights)
            if isinstance(arm, Native)
            else arm.run(x, *route_views[ids.data_ptr()])
        )

    def immutable(bundle, snapshots):
        inputs = all(
            torch.equal(tensor, saved)
            for item, saved_item in zip(bundle, snapshots)
            for tensor, saved in zip(item[1:], saved_item)
        )
        packed = all(
            torch.equal(
                tensor.view(torch.uint8), weight_snapshots[layer][key].view(torch.uint8)
            )
            for layer in {item[0].layer for item in bundle}
            for key, tensor in weights[layer].items()
        )
        routes8 = all(
            torch.equal(route_views[ids.data_ptr()][0], ids[:, :8])
            and torch.equal(route_views[ids.data_ptr()][1], rw[:, :8])
            for _, _, ids, rw in bundle
        )
        return inputs and packed and routes8

    def qualify(arms, bundle, snapshots, references, root):
        """No timing graph may be constructed before this function passes."""
        all_checks, graph_checks, states = {}, {}, {}
        for name, arm_by_layer in arms.items():
            checks = []
            for index, (case, x, ids, rw) in enumerate(bundle):
                arm = arm_by_layer[case.layer]
                output = invoke(arm, x, ids, rw).clone()
                check = compare(output, references[name][index])
                checks.append({"layer": case.layer, "step": case.step, **check})
                if not check["passed"]:
                    torch.save(
                        {
                            "input": x.cpu(),
                            "ids": ids.cpu(),
                            "weights": rw.cpu(),
                            "actual": output.cpu(),
                            "expected": references[name][index].cpu(),
                        },
                        root / f"failure-{name}-l{case.layer}-s{case.step}.pt",
                    )
            all_checks[name] = checks
            graph = torch.cuda.CUDAGraph()
            outputs = []
            with torch.cuda.graph(graph):
                for case, x, ids, rw in bundle:
                    outputs.append(invoke(arm_by_layer[case.layer], x, ids, rw).clone())
            values = []
            for replay in range(3):
                graph.replay()
                torch.cuda.synchronize()
                values.extend(
                    {"replay": replay, "case": i, **compare(output, reference)}
                    for i, (output, reference) in enumerate(
                        zip(outputs, references[name])
                    )
                )
            graph_checks[name] = values
            del graph, outputs
            # Alternate inputs of a real layer, then zero input, then restore:
            # detects persistent atomic output, stale routing and bank defects.
            states[name] = []
            for layer, arm in arm_by_layer.items():
                indexes = [i for i, item in enumerate(bundle) if item[0].layer == layer]
                for index in (indexes[-1], indexes[0]):
                    _, x, ids, rw = bundle[index]
                    check = compare(invoke(arm, x, ids, rw), references[name][index])
                    state = arm.state_check()
                    states[name].append(
                        {
                            "layer": layer,
                            "case": index,
                            "output": check,
                            "state": state,
                            "passed": check["passed"] and state["passed"],
                        }
                    )
                _, x, ids, rw = bundle[indexes[0]]
                zero = torch.zeros_like(x)
                zero_check = compare(invoke(arm, zero, ids, rw), zero)
                restored = compare(
                    invoke(arm, x, ids, rw), references[name][indexes[0]]
                )
                states[name].append(
                    {
                        "layer": layer,
                        "zero": zero_check,
                        "restored": restored,
                        "passed": zero_check["passed"] and restored["passed"],
                    }
                )
        parity = (
            [
                compare(references["candidate"][i], references["baseline"][i])
                for i in range(len(bundle))
            ]
            if "candidate" in arms
            else []
        )
        dispatch = True
        if port_manifest:
            dispatch = False
            ready = (
                all(c["passed"] for values in all_checks.values() for c in values)
                and all(c["passed"] for values in graph_checks.values() for c in values)
                and all(c["passed"] for values in states.values() for c in values)
                and source_valid()
                and immutable(bundle, snapshots)
            )
            if ready:
                from .upstream_native_port import check_engagement

                case, x, ids, rw = bundle[0]
                engagement_profiles = {}
                for name, values in arms.items():
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        invoke(values[case.layer], x, ids, rw)
                    graph.replay()
                    torch.cuda.synchronize()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        graph.replay()
                        torch.cuda.synchronize()
                    prefix = root / f"engagement-{name}-l{case.layer}-s{case.step}"
                    trace = str(prefix) + ".trace.json"
                    prof.export_chrome_trace(trace)
                    engagement_profiles[name] = write_dump(trace, prefix)
                engagement = check_engagement(port_manifest, engagement_profiles)
                save(root / "engagement.json", engagement)
                dispatch = engagement["passed"]
        receipt = {
            "source": source_valid(),
            "routes": immutable(bundle, snapshots),
            "dispatch": dispatch,
            "eager": all(c["passed"] for values in all_checks.values() for c in values),
            "graph": all(
                c["passed"] for values in graph_checks.values() for c in values
            ),
            "state": all(c["passed"] for values in states.values() for c in values),
        }
        save(
            root / "qualification.json",
            {
                "gate": receipt,
                "eager": all_checks,
                "graph": graph_checks,
                "state": states,
                "cross_reference_parity": parity,
                "cross_reference_parity_passed": all(c["passed"] for c in parity)
                if parity
                else None,
            },
        )
        require_gate(receipt)
        return receipt

    def time_pair(arms, bundle, snapshots, root):
        graphs, marker_free = {}, {}
        sink = torch.empty((triton.cdiv(256 * 6144, 1024),), device="cuda")
        for name, arm_by_layer in arms.items():
            markers = [(Event(), Event()) for _ in bundle]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for (case, x, ids, rw), (start, end) in zip(bundle, markers):
                    touch_input[(triton.cdiv(x.numel(), 1024),)](
                        x, sink, x.numel(), 1024
                    )
                    start.record()
                    invoke(arm_by_layer[case.layer], x, ids, rw)
                    end.record()
            graphs[name] = (graph, markers)
            control = torch.cuda.CUDAGraph()
            with torch.cuda.graph(control):
                for case, x, ids, rw in bundle:
                    touch_input[(triton.cdiv(x.numel(), 1024),)](
                        x, sink, x.numel(), 1024
                    )
                    invoke(arm_by_layer[case.layer], x, ids, rw)
            marker_free[name] = control
        for _ in range(args.warmups):
            for graph, _ in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        rng, samples, controls = (
            random.Random(args.seed),
            {name: [] for name in arms},
            {name: [] for name in arms},
        )
        for _ in range(args.samples):
            order = list(arms)
            rng.shuffle(order)
            for name in order:
                graph, markers = graphs[name]
                graph.replay()
                torch.cuda.synchronize()
                samples[name].append([start.elapsed_us(end) for start, end in markers])
        # Total marker-free spans include input touches; explicitly a control,
        # never substituted for the clean per-block latency estimator.
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(5):
            for name, graph in marker_free.items():
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                controls[name].append(start.elapsed_time(end) * 1000)
        result = {name: per_case_estimator(values) for name, values in samples.items()}
        save(
            root / "clean.json",
            {
                "timing": result,
                "raw_samples_us": samples,
                "marker_free_graph_span_us_including_preloads": controls,
            },
        )
        for arm_by_layer in arms.values():
            for arm in arm_by_layer.values():
                if not arm.state_check()["passed"]:
                    raise QualificationFailure(
                        "State failed after clean timing; discard timing admission"
                    )
        if not immutable(bundle, snapshots):
            raise QualificationFailure(
                "Immutable captured inputs/routes/weights changed during clean replay"
            )
        # Profile representative real cases separately. Produce both dump
        # formats even if the profiler cannot expose captured HIP kernel names.
        for name, arm_by_layer in arms.items():
            for index, (case, x, ids, rw) in enumerate(bundle[: args.profile_cases]):
                prefix = root / f"profile-{name}-l{case.layer}-s{case.step}"
                profile_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(profile_graph):
                    invoke(arm_by_layer[case.layer], x, ids, rw)
                profile_graph.replay()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    profile_graph.replay()
                    torch.cuda.synchronize()
                trace = str(prefix) + ".trace.json"
                prof.export_chrome_trace(trace)
                write_dump(
                    trace, prefix, clean_us=result[name]["per_case_median_us"][index]
                )
        if (
            not source_valid()
            or not immutable(bundle, snapshots)
            or any(
                not arm.state_check()["passed"]
                for arm_by_layer in arms.values()
                for arm in arm_by_layer.values()
            )
        ):
            raise QualificationFailure(
                "State or immutable tensors changed during profiling"
            )
        return result

    for c in args.concurrencies:
        source = args.capture / f"c{c}-d4-s355052"
        metadata = json.loads((source / "capture.json").read_text())
        cases, excluded = select_cases(
            metadata, c, layers=args.layers, steps=args.steps
        )
        data = torch.load(source / "tensors.pt", map_location="cuda", weights_only=True)
        hashes = {
            "metadata": file_digest(source / "capture.json"),
            "tensors": file_digest(source / "tensors.pt"),
        }
        for layer in sorted({case.layer for case in cases}):
            digest = load_layer(layer)
            if digest:
                hashes[f"weights-{layer}"] = digest
        bundle = []
        for case in cases:
            x = data["input"][case.slot, case.layer_index, : case.rows].contiguous()
            ids = data["ids"][case.slot, case.layer_index, : case.rows].contiguous()
            rw = data["routing_weights"][
                case.slot, case.layer_index, : case.rows
            ].contiguous()
            if (
                ids.dtype != torch.int32
                or rw.dtype != torch.float32
                or x.dtype != torch.bfloat16
            ):
                raise QualificationFailure("Captured dtype contract mismatch")
            if not (ids[:, 8] == 256).all() or not (rw[:, 8] == 1).all():
                raise QualificationFailure(
                    "Shared expert must appear exactly once with weight1"
                )
            if not ((ids[:, :8] >= 0) & (ids[:, :8] < 256)).all():
                raise QualificationFailure("Invalid routed expert IDs")
            if (
                ids[:, :8].sort(1).values[:, 1:] == ids[:, :8].sort(1).values[:, :-1]
            ).any():
                raise QualificationFailure("Duplicate supplied routed experts")
            if not torch.isfinite(rw).all() or not torch.isfinite(x).all():
                raise QualificationFailure("Nonfinite captured inputs/routes")
            bundle.append((case, x, ids, rw))
            route_views[ids.data_ptr()] = (
                ids[:, :8].contiguous(),
                rw[:, :8].contiguous(),
            )
        snapshots = [tuple(tensor.clone() for tensor in item[1:]) for item in bundle]
        reference_cache = {}
        save(
            args.output / f"c{c}-capture.json",
            {
                "source": str(source),
                "hashes": hashes,
                "cases": len(cases),
                "excluded_steps": excluded,
                "metadata": metadata,
            },
        )
        selected_groups = args.groups if args.comparison_mode == "original" else []
        groups = [group for group in selected_groups if 4 * c in GROUPS[group].rows]
        for group in selected_groups:
            if group not in groups:
                receipts.append(
                    {
                        "group": group,
                        "concurrency": c,
                        "status": "unsupported" if c == 64 else "not_applicable",
                        "identical_dispatch": c != 64,
                        "reason": "No qualified M256 candidate"
                        if c == 64
                        else "Group does not alter this shape",
                    }
                )
        control_group = {
            "original": "current_native_to_selected",
            "native-only": "current_native_only",
            "ordered-recovery": "current_native_to_ordered_m128_v1",
            "native-port": port_manifest["unit"] if port_manifest else None,
        }[args.comparison_mode]
        groups.insert(0, control_group)
        for group in groups:
            root = args.output / f"{group}-c{c}"
            root.mkdir()
            started = time.monotonic()
            phase_seconds = {}

            def progress(
                phase, group=group, c=c, started=started, root=root, **details
            ):
                value = {
                    "group": group,
                    "concurrency": c,
                    "phase": phase,
                    "elapsed_seconds": time.monotonic() - started,
                    **details,
                }
                save(root / "progress.json", value)
                print(json.dumps(value), flush=True)

            try:
                progress("prepare_dispatch", cases=len(bundle))
                arms = {"baseline": {}, "candidate": {}}
                for case, x, ids, rw in bundle:
                    if case.layer in arms["baseline"]:
                        continue
                    if group == control_group:
                        arms["baseline"][case.layer] = Native(case.layer, x, ids, rw)
                        if args.comparison_mode == "native-port":
                            arms["candidate"][case.layer] = Native(
                                case.layer, x, ids, rw, enabled=True
                            )
                            if (
                                arms["baseline"][case.layer].reference_contract
                                != arms["candidate"][case.layer].reference_contract
                            ):
                                raise QualificationFailure(
                                    "Native-port reference contract differs across arms"
                                )
                            continue
                        if c == 64 or args.comparison_mode == "native-only":
                            continue
                        if args.comparison_mode == "ordered-recovery":
                            arms["candidate"][case.layer] = make_ordered_variant(
                                owners[case.layer], 4 * c
                            )
                        else:
                            supported = next(
                                name
                                for name in PRIMARY_GROUPS
                                if 4 * c in GROUPS[name].rows
                            )
                            arms["candidate"][case.layer] = make_variant(
                                owners[case.layer], 4 * c, supported, True
                            )
                    else:
                        for name, enabled in (("baseline", False), ("candidate", True)):
                            arms[name][case.layer] = make_variant(
                                owners[case.layer], 4 * c, group, enabled
                            )
                    for arm in (
                        values[case.layer]
                        for values in arms.values()
                        if case.layer in values
                    ):
                        invoke(arm, x, ids, rw)
                if not arms["candidate"]:
                    del arms["candidate"]
                torch.cuda.synchronize()
                save(
                    root / "dispatch.json",
                    {
                        name: {str(layer): arm.info for layer, arm in values.items()}
                        for name, values in arms.items()
                    },
                )
                phase_seconds["prepare_dispatch"] = time.monotonic() - started
                reference_started = time.monotonic()
                progress("independent_references")
                references = {name: [None] * len(bundle) for name in arms}
                for layer in arms["baseline"]:
                    decoded = None
                    for name, values in arms.items():
                        arm = values[layer]
                        contract = (
                            arm.reference_contract
                            if isinstance(arm, Native)
                            else candidate_contract(arm)
                        )
                        key = (layer, json.dumps(contract, sort_keys=True))
                        if key not in reference_cache:
                            if decoded is None:
                                decoded = unpack_weights(weights[layer])
                            reference_cache[key] = {
                                i: expert_reference(decoded, x, ids, rw, contract)
                                for i, (case, x, ids, rw) in enumerate(bundle)
                                if case.layer == layer
                            }
                        for index, expected in reference_cache[key].items():
                            references[name][index] = expected
                    del decoded
                    progress("independent_references", layer_complete=layer)
                phase_seconds["independent_references"] = (
                    time.monotonic() - reference_started
                )
                gate_started = time.monotonic()
                progress("eager_graph_state_gates")
                gate = qualify(arms, bundle, snapshots, references, root)
                phase_seconds["eager_graph_state_gates"] = (
                    time.monotonic() - gate_started
                )
                if group == control_group:
                    # Same saved routes, independently qualified API; compare
                    # against production output without treating old captures
                    # or different native versions as the oracle.
                    saved_output_checks = []
                    for case, x, ids, rw in bundle:
                        saved_output_checks.append(
                            {
                                "layer": case.layer,
                                "step": case.step,
                                **compare(
                                    invoke(arms["baseline"][case.layer], x, ids, rw),
                                    data["output"][
                                        case.slot, case.layer_index, : case.rows
                                    ],
                                ),
                            }
                        )
                    save(
                        root / "captured-native-output-diagnostics.json",
                        {
                            "capture_sglang_sha": metadata.get("sglang_sha"),
                            "capture_aiter_sha": metadata.get("aiter_sha"),
                            "all_passed": all(
                                check["passed"] for check in saved_output_checks
                            ),
                            "checks": saved_output_checks,
                        },
                    )
                if port_manifest:
                    from .upstream_native_stage2 import (
                        measure_stage2_pair,
                        snapshot_stage2,
                    )

                    progress("same_native_g1_stage2_comparison")
                    fixtures = [
                        snapshot_stage2(
                            lambda callback, case=case, x=x, ids=ids, rw=rw, arms=arms: (
                                arms["baseline"][case.layer].run(
                                    x, ids, rw, capture_override=callback
                                )
                            )
                        )
                        for case, x, ids, rw in bundle
                    ]
                    measure_stage2_pair(
                        fixtures=fixtures,
                        references=references["baseline"],
                        cases=[case for case, *_ in bundle],
                        callback=port_callback,
                        args=args,
                        root=root / "g2-secondary",
                        source_valid=source_valid,
                        parent_immutable=lambda bundle=bundle, snapshots=snapshots: (
                            immutable(bundle, snapshots)
                        ),
                        save=save,
                    )
                    del fixtures
                timing_started = time.monotonic()
                progress("clean_then_profile")
                timing = time_pair(arms, bundle, snapshots, root)
                phase_seconds["clean_and_profiles"] = time.monotonic() - timing_started
                row = {
                    "group": group,
                    "concurrency": c,
                    "rows": 4 * c,
                    "status": "qualified" if "candidate" in arms else "native_only",
                    "gate": gate,
                    "baseline_us": timing["baseline"]["mean_per_case_median_us"],
                    "candidate_us": timing["candidate"]["mean_per_case_median_us"]
                    if "candidate" in arms
                    else None,
                    "cases": len(bundle),
                    "independent_reference_cases_per_arm": len(bundle),
                    "phase_seconds": phase_seconds,
                    "comparison_semantics": "matched native source patch, unchanged quantizer/route/output contract"
                    if port_manifest
                    else "different declared quantizer/reduction contracts"
                    if group == control_group and "candidate" in arms
                    else "independently qualified native-only control"
                    if group == control_group
                    else "matched supplied-route removal ablation",
                    "excluded_from_primary_ranking": group == control_group
                    and not port_manifest,
                }
                if group == control_group:
                    row["cross_reference_parity_passed"] = json.loads(
                        (root / "qualification.json").read_text()
                    )["cross_reference_parity_passed"]
            except Exception:  # noqa: BLE001 - retain every cell's failure receipt
                row = {
                    "group": group,
                    "concurrency": c,
                    "rows": 4 * c,
                    "status": "failed",
                    "error": traceback.format_exc(),
                }
                row["fatal_state_corruption"] = not immutable(bundle, snapshots)
            receipts.append(row)
            row["total_wall_seconds"] = time.monotonic() - started
            save(root / "receipt.json", row)
            save(
                args.output / "summary.json",
                {
                    "identity": identity,
                    "receipts": receipts,
                    "marginal_savings": marginal_summary(
                        [
                            r
                            for r in receipts
                            if r["group"] in PRIMARY_GROUPS
                            or (port_manifest and r["group"] == port_manifest["unit"])
                        ],
                        CONCURRENCIES if port_manifest else args.concurrencies,
                    ),
                },
            )
            print(json.dumps(row), flush=True)
            if row.get("fatal_state_corruption"):
                raise QualificationFailure(
                    "Saved inputs or weights changed; campaign cannot continue"
                )
            torch.cuda.empty_cache()
        del bundle, data, snapshots, reference_cache
        route_views.clear()
    save(
        args.output / "completed.json",
        {
            "state": "completed",
            "cells": len(receipts),
            "qualified": sum(row["status"] == "qualified" for row in receipts),
        },
    )


if __name__ == "__main__":
    main()
