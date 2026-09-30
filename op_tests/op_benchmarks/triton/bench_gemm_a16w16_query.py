# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare named M64/M128 query GEMMs with one baseline on shared operands.

Warm mode uses Triton's unrolled graph timer; cold mode uses its L2-cleared
event timer. This example does not tune kernels, profile, or measure a block.
"""

import argparse
import contextlib
import os
import statistics

import torch
import torch.nn.functional as F


@contextlib.contextmanager
def native_table(tuned, rows):
    """Switch native lookup tables outside all timed/captured calls."""
    original = tuned.get_GEMM_A16W16_config_
    tuned.get_GEMM_A16W16_config_ = lambda: rows
    tuned.get_GEMM_A16W16_config.cache_clear()
    try:
        yield
    finally:
        tuned.get_GEMM_A16W16_config_ = original
        tuned.get_GEMM_A16W16_config.cache_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-M", type=int, nargs="+", choices=(64, 128), default=[64, 128])
    parser.add_argument("--mode", choices=("warm", "cold"), default="warm")
    parser.add_argument("--baseline", choices=("triton", "torch"), default="triton")
    parser.add_argument(
        "--baseline-csv",
        help="Complete pristine native dispatch snapshot; overrides --baseline",
    )
    parser.add_argument(
        "--rep", type=int, default=100, help="Milliseconds per timer run"
    )
    args = parser.parse_args()
    if args.rep <= 0:
        parser.error("--rep must be positive")
    if any(
        os.environ.get(name)
        for name in (
            "ROCP_TOOL_LIBRARIES",
            "ROCPROFILER_SDK_TOOL_LIBRARIES",
            "CUDA_INJECTION64_PATH",
        )
    ):
        parser.error("Clean timings require a process without profiler injection")

    import pandas as pd
    import triton.testing

    from aiter import tuned_gemm as tuned
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    from aiter.ops.triton.gemm.basic.gemm_a16w16_small_m import (
        gemm_a16w16_small_m_accepts,
    )
    from aiter.ops.triton.gemm.basic.gemm_a16w16_xcd_reuse import (
        gemm_a16w16_xcd_reuse_accepts,
    )
    from aiter.test_common import checkAllclose
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_small_m import (
        _assert_product,
        generate_inputs,
    )

    stock = None
    if args.baseline_csv:
        # Use the installed loader's exact shape-key and duplicate policy.
        stock = (
            pd.read_csv(args.baseline_csv)
            .drop_duplicates()
            .set_index(
                [
                    "gfx",
                    "cu_num",
                    "M",
                    "N",
                    "K",
                    "bias",
                    "dtype",
                    "outdtype",
                    "scaleAB",
                    "bpreshuffle",
                ]
            )
            .to_dict("index")
        )
    names = {64: "gemm_a16w16_small_m", 128: "gemm_a16w16_xcd_reuse"}
    accepts = {64: gemm_a16w16_small_m_accepts, 128: gemm_a16w16_xcd_reuse_accepts}
    torch.set_float32_matmul_precision("highest")
    print(f"{args.mode}: three alternating runs; median and range of run medians (us)")
    print("M     baseline [range]             candidate [range]            gain %")
    with torch.inference_mode():
        for m in args.M:
            x, w = generate_inputs(m)
            before_x, before_w = x.clone(), w.clone()
            config = tuned.get_GEMM_A16W16_config(
                m, 2048, 2048, False, str(x.dtype), str(x.dtype)
            )
            assert config["kernelName"] == names[m] and accepts[m](
                x, w
            ), "Candidate row is not selected or this runtime cannot run it"

            def candidate(x=x, w=w):
                return tuned.tgemm.mm(x, w)

            if stock is not None:
                with native_table(tuned, stock):
                    baseline_config = tuned.get_GEMM_A16W16_config(
                        m, 2048, 2048, False, str(x.dtype), str(x.dtype)
                    )
                assert (
                    baseline_config.get("kernelName") not in names.values()
                ), "Baseline snapshot contains the candidate"
                baseline = candidate
                baseline_label = str(baseline_config)
                libtype = baseline_config["libtype"]
            elif args.baseline == "triton":

                def baseline(x=x, w=w):
                    return gemm_a16w16(x, w, backend="triton")

                baseline_label, libtype = "configured generic Triton", "triton"
            else:

                def baseline(x=x, w=w):
                    return F.linear(x, w)

                baseline_label, libtype = "F.linear (backend unspecified)", "torch"

            _assert_product(candidate(), x, w)
            with (
                native_table(tuned, stock)
                if stock is not None
                else contextlib.nullcontext()
            ):
                out = baseline()
                assert out.shape == (m, 2048) and out.dtype == x.dtype
                assert torch.isfinite(out).all()
                if libtype == "triton":
                    torch.testing.assert_close(out, F.linear(x, w), atol=0.1, rtol=0.01)
                else:
                    reference = F.linear(x.float(), w.float()).to(x.dtype)
                    assert checkAllclose(out, reference, atol=0.05, rtol=0.05) <= 0.05
            samples = {"baseline": [], "candidate": []}
            for round_id in range(3):
                order = (
                    ("baseline", "candidate")
                    if round_id % 2 == 0
                    else ("candidate", "baseline")
                )
                for arm in order:
                    table = stock if arm == "baseline" else None
                    with (
                        native_table(tuned, table)
                        if table is not None
                        else contextlib.nullcontext()
                    ):
                        fn = baseline if arm == "baseline" else candidate
                        if args.mode == "warm":
                            ms = triton.testing.do_bench_cudagraph(
                                fn, rep=args.rep, return_mode="all"
                            )
                        else:
                            ms = triton.testing.do_bench(
                                fn, rep=args.rep, return_mode="all"
                            )
                    samples[arm].append(statistics.median(ms) * 1000)
            assert torch.equal(x, before_x) and torch.equal(w, before_w)
            b, c = samples["baseline"], samples["candidate"]
            baseline_us, candidate_us = statistics.median(b), statistics.median(c)
            gain = 100 * (baseline_us - candidate_us) / baseline_us
            print(
                f"{m:<5} {baseline_us:7.3f} [{min(b):.3f},{max(b):.3f}]     {candidate_us:7.3f} [{min(c):.3f},{max(c):.3f}]     {gain:+7.2f}"
            )
            print(f"      baseline: {baseline_label}")


if __name__ == "__main__":
    main()
