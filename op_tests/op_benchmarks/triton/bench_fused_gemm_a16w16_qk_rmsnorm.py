# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 input projection + Q/KV RMSNorm against configured AITER GEMM + norm.

Select the incumbent CSV before importing AITER. Correctness gates precede
three interleaved clean median measurements. Profiles use a separate invocation
bound to an existing clean receipt and never provide the reported latency.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

BASELINE_PROJECTION_POLICY = {
    "name": "aiter_assertAllclose_bf16_projection_v1",
    "reference": "independent FP32 GEMM rounded once to BF16, then upcast",
    "argument_order": "actual, rounded_reference",
    "atol": 0.01,
    "rtol": 0.01,
    "tol_err_ratio": 0.05,
    "catastrophic_check": True,
    "finite_required": True,
    "unrounded_fp32_nrmse_limit": 0.01,
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nrmse(actual, expected):
    error = (actual.float() - expected.float()).square().mean().sqrt().item()
    scale = expected.float().square().mean().sqrt().item()
    return error / scale if scale else (0.0 if error == 0.0 else math.inf)


def reference_outputs(x, weight, q_weight, k_weight):
    projection = F.linear(x.float(), weight.float())
    # The operator normalizes the once-BF16-rounded projection.
    rounded = projection.bfloat16().float()
    refs = [projection]
    for values, norm_weight in (
        (rounded[:, :2048], q_weight),
        (rounded[:, 2048:2560], k_weight),
    ):
        refs.append(
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-5)
            * norm_weight.float()
        )
    return refs


def operand_hashes(operands):
    return {
        name: hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        for name, t in zip(("x", "weight", "q_weight", "k_weight"), operands)
    }


class GateFailure(RuntimeError):
    def __init__(self, m, arm, field, metadata):
        self.metadata = {"m": m, "arm": arm, "field": field, **metadata}
        super().__init__(f"M{m} {arm} {field}: documented correctness gate failed")


def check_baseline_projection(m, draw, actual, expected):
    """Use AITER's BF16 outlier policy, plus the existing aggregate bound.

    AITER's native BF16 tuner uses a cast-back FP32 reference and explicitly
    filters mismatch fractions. This public-helper .01/.01 policy is stricter
    than the tuner's .05/.05 BF16 tolerance. It does not require the incumbent
    split-K BF16 atomics to reproduce the candidate's FP32 reduction order.
    """
    from aiter.test_common import assertAllclose

    metadata = {"draw": draw, "policy": BASELINE_PROJECTION_POLICY}

    def reject(reason):
        raise GateFailure(m, "separate", "projection", {**metadata, "reason": reason})

    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        reject("Expected equal projection shapes and BF16 incumbent output")
    if not bool(torch.isfinite(actual).all()) or not bool(
        torch.isfinite(expected).all()
    ):
        reject("Nonfinite projection or independent reference")
    ratio = nrmse(actual, expected)
    difference = (actual.float() - expected).abs()
    original_failed = difference > (0.02 + 0.02 * expected.abs())
    metadata.update(
        nrmse=ratio if math.isfinite(ratio) else None,
        nrmse_limit=0.01,
        original_all_element_02_02_diagnostic={
            "passed": not bool(original_failed.any()),
            "failed_elements": int(original_failed.sum()),
            "element_count": actual.numel(),
            "max_abs_error": float(difference.max()),
            "timing_admission": False,
        },
    )
    try:
        mismatch = assertAllclose(
            actual.float(),
            expected.bfloat16().float(),
            rtol=0.01,
            atol=0.01,
            tol_err_ratio=0.05,
            catastrophic_check=True,
            msg=f"M{m} separate projection ({draw})",
        )
    except AssertionError as exc:
        reject(str(exc))
    metadata["upstream_mismatch_ratio"] = float(mismatch)
    if not math.isfinite(ratio) or ratio > 0.01:
        reject("Unrounded independent FP32 NRMSE exceeds unchanged 1% limit")
    return ratio


def check_outputs(m, arm, draw, outputs, refs):
    ratios = []
    for field, actual, expected in zip(
        ("projection", "q_norm", "kv_norm"), outputs, refs
    ):
        if arm == "separate" and field == "projection":
            ratios.append(check_baseline_projection(m, draw, actual, expected))
            continue
        ratio = nrmse(actual, expected)
        try:
            torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
            assert math.isfinite(ratio) and ratio <= 0.01
        except AssertionError as exc:
            difference = (actual.float() - expected).abs()
            allowance = 0.02 + 0.02 * expected.abs()
            failed = (~torch.isfinite(actual.float())) | (difference > allowance)
            coordinates = torch.nonzero(failed, as_tuple=False).cpu().tolist()

            def finite(value):
                result = float(value)
                return result if math.isfinite(result) else None

            raise GateFailure(
                m,
                arm,
                field,
                {
                    "draw": draw,
                    "nrmse": finite(ratio),
                    "nrmse_limit": 0.01,
                    "atol": 0.02,
                    "rtol": 0.02,
                    "failed_elements": len(coordinates),
                    "element_count": actual.numel(),
                    "outliers": [
                        {
                            "coordinate": c,
                            "actual": finite(actual[tuple(c)]),
                            "expected": finite(expected[tuple(c)]),
                            "abs_error": finite(difference[tuple(c)]),
                            "allowance": finite(allowance[tuple(c)]),
                        }
                        for c in coordinates[:32]
                    ],
                },
            ) from exc
        ratios.append(ratio)
    if arm == "fused":
        rounded_ratio = nrmse(outputs[0], refs[0].bfloat16())
        if not math.isfinite(rounded_ratio) or rounded_ratio >= 5e-4:
            raise GateFailure(
                m,
                arm,
                "projection_vs_once_bf16_rounded_reference",
                {
                    "draw": draw,
                    "nrmse": rounded_ratio if math.isfinite(rounded_ratio) else None,
                    "strict_nrmse_limit": 5e-4,
                },
            )
    return ratios


def make_case(m, require_backend, return_context=False, candidate_only=False):
    from aiter.ops.enum import QuantType
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm
    from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_qk_rmsnorm import (
        fused_gemm_a16w16_qk_rmsnorm,
    )
    from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm

    gen = torch.Generator(device="cuda").manual_seed(2624 + m)
    x = torch.randn((m, 6144), device="cuda", generator=gen).bfloat16()
    weight = (torch.randn((2624, 6144), device="cuda", generator=gen) * 0.02).bfloat16()
    q_weight = (torch.rand(2048, device="cuda", generator=gen) + 0.5).bfloat16()
    k_weight = (torch.rand(512, device="cuda", generator=gen) + 0.5).bfloat16()
    config = get_GEMM_A16W16_config(
        M=m,
        N=2624,
        K=6144,
        bias=False,
        dtype=str(x.dtype),
        otype=str(x.dtype),
        scaleAB=False,
        bpreshuffle=False,
    )
    config = json.loads(json.dumps(config, default=str))
    if require_backend and config["libtype"] != require_backend:
        raise ValueError(f"Expected {require_backend}, selected {config}")

    def separate():
        out = tgemm.mm(x, weight)
        q_out = torch.empty((m, 2048), device=x.device, dtype=x.dtype)
        k_out = torch.empty((m, 512), device=x.device, dtype=x.dtype)
        fused_qk_rmsnorm(
            q_out_quantized=q_out,
            k_out=k_out,
            q=out[:, :2048],
            q_weight=q_weight,
            q_epsilon=1e-5,
            k=out[:, 2048:2560],
            k_weight=k_weight,
            k_epsilon=1e-5,
            quant_type=QuantType.No,
        )
        return out, q_out, k_out

    def fused():
        return fused_gemm_a16w16_qk_rmsnorm(x, weight, q_weight, 1e-5, k_weight, 1e-5)

    arms = {"separate": separate, "fused": fused}
    if candidate_only:
        arms = {"fused": fused}
    operands = (x, weight, q_weight, k_weight)
    saved = tuple(t.clone() for t in operands) if return_context else None
    refs = reference_outputs(x, weight, q_weight, k_weight)
    metrics = {}
    for arm, fn in arms.items():
        try:
            metrics[arm] = check_outputs(m, arm, "eager_random", fn(), refs)
        except GateFailure as exc:
            exc.metadata["selected_config"] = config
            exc.metadata["operand_sha256"] = operand_hashes(operands)
            raise
    if return_context:
        return arms, metrics, config, (operands, saved)
    return arms, metrics, config


def clean_median_ms(fn, warmup, rep):
    value = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5])
    return float(value[0] if isinstance(value, (tuple, list)) else value)


def graph_mean_us(graph, replays, warmup):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed_us = float(start.elapsed_time(end)) * 1000
    assert math.isfinite(elapsed_us) and elapsed_us > 0
    return {
        "batch_elapsed_us": elapsed_us,
        "replays": replays,
        "mean_us": elapsed_us / replays,
    }


def capture_graphs(m, arms, context, config, case):
    operands, saved = context
    graphs, captured, streams = {}, {}, {}
    for arm, fn in arms.items():
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        # Stream-keyed native scratch was initialized on this exact stream.
        with torch.cuda.graph(graph, stream=warm):
            captured[arm] = fn()
        graphs[arm], streams[arm] = graph, warm

    def scratch_check():
        if config["libtype"] == "flydsl" and "separate" in streams:
            from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import get_split_k_buffers

            semaphore, signal = get_split_k_buffers(
                streams["separate"], operands[0].device
            )
            torch.cuda.synchronize()
            state = {
                "semaphore_nonzero": int(torch.count_nonzero(semaphore)),
                "signal_nonzero": int(torch.count_nonzero(signal)),
            }
            case.setdefault("scratch_after_sync", []).append(state)
            assert not any(state.values()), "FlyDSL synchronization scratch not reset"

    scratch_check()
    for draw in ("random", "zero", "restored_random"):
        operands[0].copy_(saved[0])
        if draw == "zero":
            operands[0].zero_()
        refs = reference_outputs(*operands)
        for arm, graph in graphs.items():
            for output in captured[arm]:
                output.fill_(float("nan"))
            graph.replay()
            case.setdefault("graph_gates", []).append(
                {
                    "draw": draw,
                    "arm": arm,
                    "nrmse_projection_q_k": check_outputs(
                        m, arm, draw, captured[arm], refs
                    ),
                }
            )
    assert all(torch.equal(a, b) for a, b in zip(operands, saved)), "Mutated operands"
    scratch_check()
    return graphs, captured, scratch_check


def profile_case(m, arms, args, clean_case, identity):
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
    for arm, fn in arms.items():
        fn()
        torch.cuda.synchronize()
        prefix = args.profile_dir / f"m{m}-{arm}"
        trace = prefix.with_suffix(".trace.json")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace))
        receipt = {
            "status": "profiled",
            "case": {"case_id": f"input-projection-M{m}-{arm}"},
            "profile_sha256": file_sha256(trace),
            "identity": identity,
            "profile_binding": {
                "clean_result_sha256": file_sha256(args.clean_input),
                "clean_median_us": clean_case["summary"][arm].get(
                    "median_us",
                    clean_case["summary"][arm].get("median_of_three_means_us"),
                ),
                "clean_timing_kind": (
                    "median of three batch-mean graph replays"
                    if args.timing_mode == "graph"
                    else "median of three eager median quantiles"
                ),
            },
        }
        prefix.with_suffix(".profile.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        receipts.append(str(prefix.with_suffix(".profile.json")))
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-M", nargs="+", type=int, default=[128, 256], choices=[128, 256]
    )
    parser.add_argument("--gemm-config", type=Path, required=True)
    parser.add_argument(
        "--require-backend", choices=["flydsl", "hipblaslt", "asm", "triton"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--timing-mode", choices=["eager", "graph"], default="eager")
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=25)
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Absolute candidate latency only; no incumbent execution or paired gain claim",
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--clean-input", type=Path)
    args = parser.parse_args()
    if bool(args.profile_dir) != bool(args.clean_input):
        parser.error("Profiles require both --profile-dir and --clean-input")
    if min(args.rep, args.warmup, args.replays, args.warmup_replays) <= 0:
        parser.error("Timing budgets and replay counts must be positive")
    args.gemm_config = args.gemm_config.resolve(strict=True)
    # AITER reads the config at import, so this must precede all AITER imports.
    os.environ["AITER_CONFIG_GEMM_BF16"] = str(args.gemm_config)
    from aiter.jit.core import AITER_CONFIGS

    hip_module = importlib.import_module("aiter.ops.splitk_reduce_qk_rmsnorm")
    api_module = importlib.import_module(
        "aiter.ops.triton.gemm.fused.fused_gemm_a16w16_qk_rmsnorm"
    )
    producer_module = importlib.import_module(
        "aiter.ops.triton._gluon_kernels.gfx950.gemm.fused.fused_gemm_a16w16_qk_rmsnorm"
    )
    incumbent_module = importlib.import_module("aiter.tuned_gemm")
    norm_module = importlib.import_module("aiter.ops.fused_qk_rmsnorm_group_quant")
    flydsl_module = importlib.import_module(
        "aiter.ops.flydsl.kernels.gemm_a16w16_gfx950"
    )
    flydsl_parser_module = importlib.import_module("aiter.ops.flydsl.gemm_kernels")
    numerical_helper_module = importlib.import_module("aiter.test_common")

    actual_config = Path(AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE).resolve(strict=True)
    if actual_config != args.gemm_config:
        raise ValueError("Imported AITER uses a different GEMM config")
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    float32_matmul_precision = torch.get_float32_matmul_precision()
    if allow_tf32 or float32_matmul_precision != "highest":
        raise ValueError(
            "Independent FP32 references require TF32 off and highest precision"
        )
    identity = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "gemm_config_sha256": file_sha256(actual_config),
        "operator_sha256": file_sha256(api_module.__file__),
        "producer_sha256": file_sha256(producer_module.__file__),
        "hip_wrapper_sha256": file_sha256(hip_module.__file__),
        "hip_kernel_sha256": file_sha256(
            Path(hip_module.__file__).parents[2]
            / "csrc/kernels/splitk_reduce_qk_rmsnorm.cu"
        ),
        "incumbent_sha256": file_sha256(incumbent_module.__file__),
        "norm_wrapper_sha256": file_sha256(norm_module.__file__),
        "flydsl_kernel_sha256": file_sha256(flydsl_module.__file__),
        "flydsl_parser_sha256": file_sha256(flydsl_parser_module.__file__),
        "benchmark_sha256": file_sha256(__file__),
        "baseline_projection_policy": BASELINE_PROJECTION_POLICY,
        "numerical_helper_sha256": file_sha256(numerical_helper_module.__file__),
        "reference_allow_tf32": allow_tf32,
        "reference_float32_matmul_precision": float32_matmul_precision,
        "warmup_ms": args.warmup,
        "rep_ms": args.rep,
        "timing_mode": args.timing_mode,
        "replays": args.replays,
        "warmup_replays": args.warmup_replays,
        "candidate_only": args.candidate_only,
    }
    clean = json.loads(args.clean_input.read_text()) if args.clean_input else None
    if clean and (clean["run_kind"] != "clean" or clean["identity"] != identity):
        raise ValueError("Profile run does not match clean source/config/runtime")
    report = {
        "run_kind": "profile" if clean else "clean",
        "identity": identity,
        "gemm_config": str(actual_config),
        "scope": "BF16 input projection and Q/KV RMSNorm, including native scratch/outputs",
        "cache_condition": (
            "warm graph replay, no cache flush"
            if args.timing_mode == "graph"
            else "do_bench cache flush per timing iteration"
        ),
        "timer": (
            "CUDA event batch span / replay count, median of three interleaved means"
            if args.timing_mode == "graph"
            else "median quantile, three separate/fused interleaved repetitions"
        ),
        "allocation_contract": (
            "one native operation per captured graph, native scratch/outputs fixed at capture and reused during replay"
            if args.timing_mode == "graph"
            else "native scratch/outputs included"
        ),
        "comparison_kind": (
            "unpaired candidate absolute latency; incumbent not executed; no gain claim"
            if args.candidate_only
            else "paired incumbent/candidate comparison"
        ),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for m in args.M:
        case = {"m": m, "admitted": False}
        report["cases"].append(case)
        try:
            arms, metrics, config, context = make_case(
                m,
                args.require_backend,
                return_context=True,
                candidate_only=args.candidate_only,
            )
            case.update(
                selected_config=config,
                nrmse_projection_q_k=metrics,
                selected_config_execution=(
                    "incumbent row observed only, not executed"
                    if args.candidate_only
                    else "incumbent executed"
                ),
            )
            case["operand_sha256"] = operand_hashes(context[0])
            captured = None
            scratch_check = None
            if args.timing_mode == "graph":
                graphs, captured, scratch_check = capture_graphs(
                    m, arms, context, config, case
                )
                arms = {arm: graph.replay for arm, graph in graphs.items()}
            if clean:
                parent = next(row for row in clean["cases"] if row["m"] == m)
                if parent["selected_config"] != config or not parent["admitted"]:
                    raise ValueError(
                        "Profile incumbent row differs or parent is unadmitted"
                    )
                case["profiles"] = profile_case(m, arms, args, parent, identity)
            else:
                case["clean"] = []
                for repetition in range(3):
                    order = list(arms)
                    if args.timing_mode == "graph" and repetition % 2:
                        order.reverse()
                    for arm in order:
                        if args.timing_mode == "graph":
                            measured = graph_mean_us(
                                graphs[arm], args.replays, args.warmup_replays
                            )
                        else:
                            measured = {
                                "median_us": clean_median_ms(
                                    arms[arm], args.warmup, args.rep
                                )
                                * 1000
                            }
                        case["clean"].append(
                            {"repetition": repetition, "arm": arm, **measured}
                        )
                case["summary"] = {}
                for arm in arms:
                    key = "mean_us" if args.timing_mode == "graph" else "median_us"
                    values = [row[key] for row in case["clean"] if row["arm"] == arm]
                    case["summary"][arm] = (
                        {
                            "median_of_three_means_us": statistics.median(values),
                            "range_of_batch_means_us": [min(values), max(values)],
                        }
                        if args.timing_mode == "graph"
                        else {
                            "median_us": statistics.median(values),
                            "range_us": [min(values), max(values)],
                        }
                    )
            operands, saved = context
            if captured is not None:
                refs = reference_outputs(*operands)
                case["final_graph_nrmse"] = {
                    arm: check_outputs(m, arm, "final_random", outputs, refs)
                    for arm, outputs in captured.items()
                }
                scratch_check()
            assert all(
                torch.equal(a, b) for a, b in zip(operands, saved)
            ), "Mutated operands"
            case["admitted"] = True
        except GateFailure as exc:
            case["failed_gate"] = exc.metadata
            save()
            raise
        save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
