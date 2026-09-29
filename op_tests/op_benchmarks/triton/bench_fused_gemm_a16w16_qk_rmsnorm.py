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


def make_case(m, require_backend):
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
    refs = reference_outputs(x, weight, q_weight, k_weight)
    metrics = {}
    for arm, fn in arms.items():
        metrics[arm] = []
        outputs = fn()
        for actual, expected in zip(outputs, refs):
            torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
            ratio = nrmse(actual, expected)
            assert math.isfinite(ratio) and ratio <= 0.01, (arm, ratio)
            metrics[arm].append(ratio)
        if arm == "fused":
            rounded_ratio = nrmse(outputs[0], refs[0].bfloat16())
            assert math.isfinite(rounded_ratio) and rounded_ratio < 5e-4
    return arms, metrics, config


def clean_median_ms(fn, warmup, rep):
    value = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5])
    return float(value[0] if isinstance(value, (tuple, list)) else value)


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
                "clean_median_us": clean_case["summary"][arm]["median_us"],
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
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--clean-input", type=Path)
    args = parser.parse_args()
    if bool(args.profile_dir) != bool(args.clean_input):
        parser.error("Profiles require both --profile-dir and --clean-input")
    if args.rep <= 0 or args.warmup <= 0:
        parser.error("--rep and --warmup must be positive")
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

    actual_config = Path(AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE).resolve(strict=True)
    if actual_config != args.gemm_config:
        raise ValueError("Imported AITER uses a different GEMM config")
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
        "benchmark_sha256": file_sha256(__file__),
        "warmup_ms": args.warmup,
        "rep_ms": args.rep,
    }
    clean = json.loads(args.clean_input.read_text()) if args.clean_input else None
    if clean and (clean["run_kind"] != "clean" or clean["identity"] != identity):
        raise ValueError("Profile run does not match clean source/config/runtime")
    report = {
        "run_kind": "profile" if clean else "clean",
        "identity": identity,
        "gemm_config": str(actual_config),
        "scope": "BF16 input projection and Q/KV RMSNorm, including native scratch/outputs",
        "cache_condition": "do_bench cache flush per timing iteration",
        "timer": "median quantile, three separate/fused interleaved repetitions",
        "cases": [],
    }
    for m in args.M:
        arms, metrics, config = make_case(m, args.require_backend)
        case = {"m": m, "selected_config": config, "nrmse_projection_q_k": metrics}
        if clean:
            parent = next(row for row in clean["cases"] if row["m"] == m)
            if parent["selected_config"] != config:
                raise ValueError("Profile incumbent row differs from clean run")
            case["profiles"] = profile_case(m, arms, args, parent, identity)
        else:
            case["clean"] = []
            for repetition in range(3):
                for arm, fn in arms.items():
                    value = clean_median_ms(fn, args.warmup, args.rep) * 1000
                    case["clean"].append(
                        {"repetition": repetition, "arm": arm, "median_us": value}
                    )
            case["summary"] = {}
            for arm in arms:
                values = [r["median_us"] for r in case["clean"] if r["arm"] == arm]
                case["summary"][arm] = {
                    "median_us": statistics.median(values),
                    "range_us": [min(values), max(values)],
                }
        report["cases"].append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
