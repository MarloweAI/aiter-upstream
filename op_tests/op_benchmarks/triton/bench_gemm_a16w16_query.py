# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Clean BF16 2048² query-GEMM comparisons; collect profiles in a separate process.

Supply pristine native CSV files explicitly. No tuning occurs here. Saved legal
FlyDSL rows can be supplied separately; missing/invalid providers stay visible.
Cold timings use Triton's L2-cleared event timer; warm timings use its unrolled
graph timer. Neither is an attention-block or full-model measurement.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

N = K = 2048
BF16 = torch.bfloat16
SIZES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
CANDIDATES = (64, 128)
NAMES = ("gemm_a16w16_small_m", "gemm_a16w16_xcd_reuse")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_catalogue(paths):
    """Preserve each supplied configuration, including same-shape alternatives."""
    for path in paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = (
                    row["gfx"],
                    int(row["cu_num"]),
                    int(row["M"]),
                    int(row["N"]),
                    int(row["K"]),
                    row["bias"] == "True",
                    row["dtype"],
                    row["outdtype"],
                    row["scaleAB"] == "True",
                    row["bpreshuffle"] == "True",
                )
                row = dict(row)
                row["solidx"] = int(float(row["solidx"] or 0))
                if row.get("splitK"):
                    row["splitK"] = int(float(row["splitK"]))
                yield key, row


def read_rows(paths):
    """Native dispatch table semantics: last row wins for each shape key."""
    return dict(read_catalogue(paths))


@contextlib.contextmanager
def native_table(rows):
    """Select the provider table outside all timed/captured calls."""
    from aiter import tuned_gemm as tuned

    original = tuned.get_GEMM_A16W16_config_
    tuned.get_GEMM_A16W16_config_ = lambda: rows
    tuned.get_GEMM_A16W16_config.cache_clear()
    try:
        yield
    finally:
        tuned.get_GEMM_A16W16_config_ = original
        tuned.get_GEMM_A16W16_config.cache_clear()


def errors(out, reference):
    delta = out.float() - reference.float()
    rms = delta.square().mean().sqrt().item()
    norm = reference.float().square().mean().sqrt().item()
    return {
        "rms_error": rms if math.isfinite(rms) else None,
        "max_abs_error": (
            delta.abs().max().item() if bool(torch.isfinite(delta).all()) else None
        ),
        "reference_rms": norm,
        "nrmse": (
            (rms / norm if norm else (0.0 if not rms else None))
            if math.isfinite(rms)
            else None
        ),
    }


def aggregate(runs, incumbent_runs):
    """Pair each run with its same-round incumbent bookends, outside timers."""
    medians = [r["median_us"] for r in runs]
    paired = []
    for run in runs:
        base = statistics.median(
            r["median_us"] for r in incumbent_runs if r["round"] == run["round"]
        )
        gain = base - run["median_us"]
        paired.append(
            {
                "round": run["round"],
                "incumbent_us": base,
                "gain_us": gain,
                "gain_percent": 100 * gain / base,
            }
        )
    return {
        "median_us": statistics.median(medians),
        "run_median_range_us": [min(medians), max(medians)],
        "runs": runs,
        "paired_gains": paired,
        "paired_gain_median_us": statistics.median(p["gain_us"] for p in paired),
        "paired_gain_range_us": [
            min(p["gain_us"] for p in paired),
            max(p["gain_us"] for p in paired),
        ],
    }


def qualify(fn, x, w, contract):
    """Report identical references/metrics, retaining each provider's own gate."""
    from aiter.test_common import checkAllclose

    out = fn()
    if out.shape != (x.shape[0], N) or out.dtype != BF16:
        raise ValueError(f"wrong output shape/dtype: {out.shape} {out.dtype}")
    exact = F.linear(x.float(), w.float())
    rounded = exact.to(BF16)
    metrics = {"fp32": errors(out, exact), "bf16_rounded_fp32": errors(out, rounded)}
    finite = bool(torch.isfinite(out).all())
    if contract == "candidate":
        nrmse = metrics["bf16_rounded_fp32"]["nrmse"]
        passed = (
            finite
            and nrmse is not None
            and nrmse < 5e-4
            and torch.allclose(out.float(), exact, atol=0.02, rtol=0.02)
        )
        gate = {
            "reference": "FP32 product / BF16-rounded FP32 product",
            "atol": 0.02,
            "rtol": 0.02,
            "rounded_nrmse_lt": 5e-4,
        }
    elif contract in ("triton", "persistent"):
        reference = F.linear(x, w)
        rtol = 0.1 if contract == "persistent" else 0.01
        passed = finite and torch.allclose(out, reference, atol=0.1, rtol=rtol)
        gate = {
            "reference": "F.linear BF16",
            "atol": 0.1,
            "rtol": rtol,
            "source": "op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py",
        }
    else:
        ratio = checkAllclose(out, rounded, atol=0.05, rtol=0.05, printLog=False)
        passed = finite and ratio <= 0.05
        gate = {
            "reference": "BF16-rounded FP32 product",
            "atol": 0.05,
            "rtol": 0.05,
            "mismatch_ratio_le": 0.05,
            "observed_mismatch_ratio": ratio,
            "source": "csrc/gemm_a16w16/gemm_a16w16_tune.py BF16 defaults",
        }
    return {"passed": bool(passed), "finite": finite, "metrics": metrics, "gate": gate}


def providers(m, x, w, stock, args):
    from aiter import tuned_gemm as tuned
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    from aiter.ops.triton.utils.gemm_config_utils import (
        compute_splitk_params,
        get_gemm_config,
    )

    # tuple: name, call, table (None = stock), configuration, numerical contract
    with native_table(stock):
        incumbent = dict(
            tuned.get_GEMM_A16W16_config(m, N, K, False, str(BF16), str(BF16))
        )
    arms = [
        (
            "incumbent",
            lambda: tuned.tgemm.mm(x, w),
            stock,
            incumbent,
            "triton" if incumbent["libtype"] == "triton" else "native",
        )
    ]
    arms.append(
        (
            "torch",
            lambda: F.linear(x, w),
            None,
            {"implementation": "F.linear; backend identified by separate profile"},
            "native",
        )
    )
    arms.append(
        (
            "hipblaslt",
            lambda: tuned.hipb_gemm(x, w, -1, otype=BF16),
            None,
            {"implementation": "hipb_gemm", "solidx": -1},
            "native",
        )
    )
    for persistent in (False, True):
        family = "GEMM-A16W16-PERSISTENT" if persistent else "GEMM-A16W16"
        try:
            has_persistent = "persistent" in inspect.signature(gemm_a16w16).parameters
            if persistent and not has_persistent:
                raise TypeError("native source has no persistent GEMM API")
            config, is_tuned = get_gemm_config(family, m, N, K, backend="triton")
            if not persistent:
                # The stock plain _get_config derives these launch fields too.
                config = compute_splitk_params(config, K)
            kwargs = {"backend": "triton", "config": config}
            if has_persistent:
                kwargs["persistent"] = persistent

            def call(kwargs=kwargs):
                return gemm_a16w16(x, w, **kwargs)

            arms.append(
                (
                    "persistent" if persistent else "triton",
                    call,
                    None,
                    {"family": family, "configuration": config, "is_tuned": is_tuned},
                    "persistent" if persistent else "triton",
                )
            )
        except (KeyError, AssertionError, TypeError) as error:
            arms.append(
                (
                    "persistent" if persistent else "triton",
                    None,
                    None,
                    {"unavailable": str(error)},
                    "native",
                )
            )
    if m in CANDIDATES and not args.screen:
        rows = dict(stock)
        named = {
            key: row
            for key, row in read_rows([args.candidate_csv]).items()
            if row.get("kernelName") in NAMES
        }
        rows.update(named)
        with native_table(rows):
            selected = dict(
                tuned.get_GEMM_A16W16_config(m, N, K, False, str(BF16), str(BF16))
            )
        expected_name = NAMES[0] if m == 64 else NAMES[1]
        if selected.get("kernelName") != expected_name:
            raise RuntimeError(f"M{m}: candidate named row did not engage: {selected}")
        arms.append(
            ("candidate", lambda: tuned.tgemm.mm(x, w), rows, selected, "candidate")
        )
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    wanted = (
        get_gfx(),
        get_cu_num(),
        m,
        N,
        K,
        False,
        str(BF16),
        str(BF16),
        False,
        False,
    )
    found_flydsl = False
    seen_flydsl = set()
    for key, row in read_catalogue(args.flydsl_rows):
        if key == wanted and row["libtype"] == "flydsl":
            identity = row["kernelName"]
            if identity in seen_flydsl:
                continue
            seen_flydsl.add(identity)
            found_flydsl = True
            name = (
                "flydsl-" + hashlib.sha256(row["kernelName"].encode()).hexdigest()[:12]
            )

            def call(row=row):
                return tuned.flydsl_gemm(x, w, row["solidx"], otype=BF16, config=row)

            config = dict(row)
            try:
                params = (
                    tuned._get_flydsl_gemm_kernels().get_flydsl_hgemm_kernel_params(
                        row["kernelName"]
                    )
                )
                if params is None:
                    raise ValueError("selected row absent from native FlyDSL catalogue")
            except (ImportError, KeyError, ValueError) as error:
                call, config["unavailable"] = None, str(error)
            arms.append((name, call, None, config, "native"))
    if not found_flydsl:
        arms.append(
            (
                "flydsl",
                None,
                None,
                {"unavailable": "no applicable saved native FlyDSL row supplied"},
                "native",
            )
        )
    return arms


def timer(fn, mode, args):
    import triton.testing

    if mode == "cold":
        ms = triton.testing.do_bench(
            fn, warmup=args.warmup, rep=args.rep, return_mode="all"
        )
    else:
        ms = triton.testing.do_bench_cudagraph(fn, rep=args.rep, return_mode="all")
    samples = [float(v) * 1000 for v in ms]
    return {
        "median_us": statistics.median(samples),
        "samples_us": samples,
        "min_us": min(samples),
        "max_us": max(samples),
    }


def profile(fn, mode, prefix, clean):
    # Flush/warm outside profiling, so the trace contains only the selected call.
    graph = None
    if mode == "warm":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        for _ in range(10):
            graph.replay()
    else:
        torch.empty(256 * 1024**2, dtype=torch.int8, device="cuda").zero_()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as trace:
        graph.replay() if graph is not None else fn()
        torch.cuda.synchronize()
    path = prefix.with_suffix(".trace.json")
    trace.export_chrome_trace(str(path))
    receipt = {
        key: clean[key] for key in ("case", "code_commit", "config_sha256", "platform")
    }
    receipt.update(
        {
            "profile_sha256": digest(path),
            "profile_binding": {
                "clean_median_us": clean["timing"]["median_us"],
                "clean_result_sha256": digest(prefix.with_suffix(".clean.json")),
            },
        }
    )
    write(prefix.with_suffix(".profile.json"), receipt)


def run(args):
    instrumentation = (
        "ROCP_TOOL_LIBRARIES",
        "ROCPROFILER_SDK_TOOL_LIBRARIES",
        "CUDA_INJECTION64_PATH",
    )
    if not args.profile_only and any(os.environ.get(key) for key in instrumentation):
        raise RuntimeError(
            "Clean measurements require an environment without profiler injection"
        )
    import triton
    import triton.testing

    from aiter import tuned_gemm as tuned

    if not torch.cuda.is_available() or not triton.__version__.startswith("3.7."):
        raise RuntimeError(
            "Clean timing/profiles require GPU and qualified Triton3.7; use unit tests for3.8"
        )
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    stock = read_rows(args.incumbent_csv)
    if any(row.get("kernelName") in NAMES for row in stock.values()):
        raise ValueError(
            "Incumbent CSV must be pristine, without this change's candidate rows"
        )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    provenance = {
        "incumbent_csv": {str(p.resolve()): digest(p) for p in args.incumbent_csv},
        "flydsl_csv": {str(p.resolve()): digest(p) for p in args.flydsl_rows},
        "candidate_csv_sha256": digest(args.candidate_csv),
        "benchmark_sha256": digest(__file__),
        "native_dispatch_sha256": digest(tuned.__file__),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "hip": torch.version.hip,
        "device": properties.name,
        "cu_num": properties.multi_processor_count,
        "seed": args.seed,
        "screen": args.screen,
        "protocol": {
            "rounds": 3,
            "cold_warmup_ms": args.warmup,
            "repetition_ms": args.rep,
            "cold": "triton.testing.do_bench, default L2 flush, raw events",
            "warm": "triton.testing.do_bench_cudagraph, unrolled replay events",
            "summary": "median of run medians; same-round incumbent bookends",
        },
    }
    try:
        provenance["flydsl_version"] = importlib.metadata.version("flydsl")
    except importlib.metadata.PackageNotFoundError:
        provenance["flydsl_version"] = "package metadata unavailable"
    native_root = Path(tuned.__file__).parent
    source_paths = [
        Path(triton.testing.__file__),
        native_root / "ops/triton/gemm/basic/gemm_a16w16.py",
        native_root / "ops/triton/utils/gemm_config_utils.py",
        native_root / "ops/triton/utils/config_utils.py",
        native_root / "test_common.py",
        native_root / "ops/triton/gemm/basic/gemm_a16w16_small_m.py",
        native_root / "ops/triton/gemm/basic/gemm_a16w16_xcd_reuse.py",
        native_root
        / "ops/triton/_gluon_kernels/gfx950/gemm/basic/gemm_a16w16_small_m.py",
        native_root
        / "ops/triton/_gluon_kernels/gfx950/gemm/basic/gemm_a16w16_xcd_reuse.py",
        native_root / "ops/flydsl/gemm_kernels.py",
        native_root.parent / "csrc/gemm_a16w16/gemm_a16w16_tune.py",
        *native_root.glob("ops/triton/_triton_kernels/gemm/basic/gemm_a16w16*.py"),
        *native_root.glob("ops/flydsl/kernels/gemm_a16w16_gfx950*.py"),
    ]
    provenance["provider_source_sha256"] = {
        str(p.resolve()): digest(p) for p in source_paths
    }
    summary = {"schema": "bf16-query-clean-v1", "provenance": provenance, "results": []}
    for m in args.M:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        x = torch.randn((m, K), device="cuda", generator=generator).to(BF16)
        w = (torch.randn((N, K), device="cuda", generator=generator) * 0.02).to(BF16)
        before_x, before_w = x.clone(), w.clone()
        storage = {
            "canonical_weight_bytes": w.numel() * w.element_size(),
            "validation_snapshot_bytes": before_x.numel() * before_x.element_size()
            + before_w.numel() * before_w.element_size(),
        }
        arms = providers(m, x, w, stock, args)
        valid = []
        for name, fn, table, config, contract in arms:
            record = {
                "M": m,
                "provider": name,
                "configuration": config,
                "contract": contract,
            }
            try:
                if fn is None:
                    raise RuntimeError(config["unavailable"])
                with native_table(stock if table is None else table):
                    check = qualify(fn, x, w, contract)
                record["correctness"] = check
                record["status"] = (
                    "eligible" if check["passed"] else "numerical rejection"
                )
                if check["passed"]:
                    valid.append((name, fn, table, config, contract, record))
            except Exception as error:  # noqa: BLE001 -- retain failed providers
                record.update(
                    status="unavailable or failed",
                    error=f"{type(error).__name__}: {error}",
                )
            summary["results"].append(record)
            # Mutation or a poisoned device stops this process: later arms must
            # not use corrupted shared inputs, and no device-fault recovery is promised.
            write(args.output / "summary.json", summary)
            assert torch.equal(x, before_x) and torch.equal(
                w, before_w
            ), "provider mutated canonical operands"
        write(args.output / "summary.json", summary)
        if any(
            record["M"] == m
            and record["provider"] == "candidate"
            and record["status"] != "eligible"
            for record in summary["results"]
        ):
            raise RuntimeError(
                f"M{m}: candidate failed correctness; no timing permitted"
            )
        for mode in args.mode:
            timings = {name: [] for name, *_ in valid}
            for repeat in range(3):
                if args.profile_only:
                    break
                # Incumbent bookends plus rotated alternatives; shared operands, no profiles.
                incumbent = next((arm for arm in valid if arm[0] == "incumbent"), None)
                if incumbent is None:
                    raise RuntimeError(
                        f"M{m}: incumbent did not qualify; no relative claim"
                    )
                other = [arm for arm in valid if arm[0] != "incumbent"]
                ordered = (
                    [incumbent]
                    + other[repeat % len(other) :]
                    + other[: repeat % len(other)]
                    + [incumbent]
                    if other
                    else [incumbent]
                )
                for name, fn, table, _, _, _ in ordered:
                    with native_table(stock if table is None else table):
                        fn()  # resolve/JIT outside clean events and graph capture
                        torch.cuda.synchronize()
                        result = timer(fn, mode, args)
                    timings[name].append({"round": repeat, **result})
            for name, fn, table, config, contract, record in valid:
                case = {
                    "case_id": f"m{m}-{name}-{mode}",
                    "M": m,
                    "N": N,
                    "K": K,
                    "provider": name,
                    "cache_mode": mode,
                    "boundary": "single GEMM complete public call",
                }
                prefix = args.output / case["case_id"]
                gate = {
                    k: v
                    for k, v in record["correctness"]["gate"].items()
                    if not k.startswith("observed_")
                }
                binding = {
                    "provenance": provenance,
                    "configuration": config,
                    "contract": gate,
                    "case": case,
                }
                config_hash = hashlib.sha256(
                    json.dumps(binding, sort_keys=True).encode()
                ).hexdigest()
                if args.profile_only:
                    clean_path = args.clean_root / (case["case_id"] + ".clean.json")
                    clean = json.loads(clean_path.read_text())
                    if (
                        clean["config_sha256"] != config_hash
                        or clean["code_commit"] != args.code_commit
                    ):
                        raise ValueError(f"Clean source/config mismatch: {clean_path}")
                    # Formatter binding uses the adjacent, byte-identical clean receipt.
                    prefix.with_suffix(".clean.json").write_bytes(
                        clean_path.read_bytes()
                    )
                    with native_table(stock if table is None else table):
                        fn()
                        torch.cuda.synchronize()
                        profile(fn, mode, prefix, clean)
                    continue
                receipt = {
                    "case": case,
                    "code_commit": args.code_commit,
                    "config_sha256": config_hash,
                    "platform": "MI355X",
                    "binding": binding,
                    "correctness": record["correctness"],
                    "timing": aggregate(timings[name], timings["incumbent"]),
                    "resident_storage": storage,
                }
                write(prefix.with_suffix(".clean.json"), receipt)
                record[mode] = receipt["timing"]
                print(
                    f"M{m:4d} {name:22s} {mode:4s} {receipt['timing']['median_us']:.3f} us",
                    flush=True,
                )
        assert torch.equal(x, before_x) and torch.equal(
            w, before_w
        ), "provider mutated canonical operands"
        write(args.output / "summary.json", summary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "-M", nargs="+", type=int, choices=SIZES, default=list(CANDIDATES)
    )
    parser.add_argument("--incumbent-csv", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv"),
    )
    parser.add_argument("--flydsl-rows", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--mode", nargs="+", choices=("cold", "warm"), default=["cold", "warm"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument(
        "--warmup", type=int, default=25, help="cold timer warmup milliseconds"
    )
    parser.add_argument(
        "--rep", type=int, default=100, help="timer repetition milliseconds"
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="stock providers only; no specialized calls or profiles",
    )
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--clean-root", type=Path)
    args = parser.parse_args(argv)
    if args.profile_only and (args.clean_root is None or args.screen):
        parser.error(
            "profiles require --clean-root and cannot be combined with --screen"
        )
    if args.rep <= 0 or args.warmup < 0:
        parser.error("--rep must be positive and --warmup nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
