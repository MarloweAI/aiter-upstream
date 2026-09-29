# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark the fused MLA input projection and q/kv RMSNorm against the path it
replaces: the tuned gemm_a16w16 (tgemm.mm) for x [M, 6144] @ w [2624, 6144]^T,
then fused_qk_rmsnorm on the q (2048) and kv (512) column slices of its output.
On a 256-CU gfx950 the tuned table picks FlyDSL split-K hgemm kernels for this
shape at M = 128 and M = 256.

Times come from triton.testing.do_bench, which clears the caches before every
repetition, so the weight is read from HBM each time.
"""

import argparse
import sys

import torch
import triton

from aiter.ops.enum import QuantType
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm
from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_qk_rmsnorm import (
    fused_gemm_a16w16_qk_rmsnorm,
)
from aiter.tuned_gemm import tgemm
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
from op_tests.triton_tests.gemm.fused.test_fused_gemm_a16w16_qk_rmsnorm import (
    generate_inputs,
)

_PROVIDERS = ("fused", "unfused")
_EPS = 1e-5
_Q_DIM = 2048
_KV_DIM = 512


def _unfused(x, weight, q_weight, k_weight):
    out = tgemm.mm(x, weight)
    q_out = torch.empty((x.shape[0], _Q_DIM), device=x.device, dtype=x.dtype)
    k_out = torch.empty((x.shape[0], _KV_DIM), device=x.device, dtype=x.dtype)
    fused_qk_rmsnorm(
        q_out_quantized=q_out,
        k_out=k_out,
        q=out[:, :_Q_DIM],
        q_weight=q_weight,
        q_epsilon=_EPS,
        k=out[:, _Q_DIM : _Q_DIM + _KV_DIM],
        k_weight=k_weight,
        k_epsilon=_EPS,
        quant_type=QuantType.No,
    )
    return out, q_out, k_out


def bench_fn(M, provider, args):
    x, weight, q_weight, k_weight = generate_inputs(M)
    if provider == "fused":

        def fn():
            return fused_gemm_a16w16_qk_rmsnorm(
                x, weight, q_weight, _EPS, k_weight, _EPS
            )

    else:

        def fn():
            return _unfused(x, weight, q_weight, k_weight)

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    return ms * 1000  # us


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    benchmark = triton.testing.Benchmark(
        x_names=["M"],
        x_vals=args.M,
        line_arg="provider",
        line_vals=list(providers),
        line_names=[f"{p} (us)" for p in providers],
        styles=[("red", "-"), ("blue", "-")][: len(providers)],
        ylabel="us",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench(M, provider):
        return bench_fn(M, provider, args)

    bench.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark fused_gemm_a16w16_qk_rmsnorm",
        description="Fused MLA input projection + q/kv RMSNorm vs tgemm.mm + "
        "fused_qk_rmsnorm",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-M",
        type=int,
        nargs="+",
        default=[128, 256],
        choices=[128, 256],
        help="Rows of x (the fused op supports 128 and 256)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="fused op, the unfused tgemm.mm + fused_qk_rmsnorm path, or both",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to a CSV file"
    )
    return parser.parse_args()


def main():
    run_benchmark(parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
