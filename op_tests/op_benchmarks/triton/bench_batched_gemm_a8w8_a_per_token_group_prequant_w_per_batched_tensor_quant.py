import functools
import json
import math
from pathlib import Path

import torch
import triton

from aiter.ops.triton.gemm.batched import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as op_module,
)
from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    add_argparse_ff,
    get_ff_args,
    get_parser,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    batched_model_benchmark_shapes,
    get_caller_name_no_ext,
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
)
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    generate_batched_gemm_a16w8_inputs as generate_batched_gemm_a8w8_per_token_group_inputs,
)


def bench_gemm_fn(
    batch: int,
    M: int,
    N: int,
    K: int,
    metric: str,
    backend: str | None,
    layout: str,
    group_size: int,
    has_bias: bool,
    transpose_bm: bool,
    transpose_bm_in: bool,
    config: dict | None = None,
    gluon_launch_config: dict | None = None,
    profile_path: str | None = None,
):
    c_dtype = torch.bfloat16
    x, weight, w_scale, bias, y = generate_batched_gemm_a8w8_per_token_group_inputs(
        batch,
        M,
        N,
        K,
        c_dtype,
        has_bias=has_bias,
        output=True,
        layout=layout,
        transpose_bm=transpose_bm,
    )
    if transpose_bm_in:
        x = x.transpose(0, 1).contiguous()
    # flops
    flops = 2.0 * batch * M * N * K
    # memory transfer
    mem_read = (
        x.numel() * x.element_size()
        + weight.numel() * weight.element_size()
        + w_scale.numel() * w_scale.element_size()
        + (bias.numel() * bias.element_size() if bias is not None else 0)
    )
    mem_write = y.numel() * y.element_size()
    mem = mem_read + mem_write

    def fn():
        return batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x,
            weight,
            w_scale,
            group_size=group_size,
            bias=bias,
            backend=backend,
            config=config,
            dtype=c_dtype,
            YQ=y,
            transpose_bm=transpose_bm,
            transpose_bm_in=transpose_bm_in,
        )

    original_launch = op_module._gluon_small_m
    if profile_path is not None and Path(profile_path).exists():
        raise FileExistsError(f"Refusing to overwrite profile: {profile_path}")
    if gluon_launch_config is not None:
        assert backend == "gluon", "Launch diagnostics require --backend gluon"
        assert set(gluon_launch_config) <= {"num_stages", "waves_per_eu"}
        op_module._gluon_small_m = functools.partial(
            original_launch, **gluon_launch_config
        )
    try:
        if profile_path is not None:
            # Eager warmup/JIT is outside capture and outside the profiler.
            fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                fn()
                torch.cuda.synchronize()
            Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(profile_path)
            print(f"Profile only: {profile_path}; no clean timing reported.")
            return float("nan")
        # Native do_bench uses GPU events with a cache flush before each sample.
        # The default return_mode is a mean, so request the median explicitly.
        ms = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
    finally:
        op_module._gluon_small_m = original_launch

    # Return exactly one scalar depending on which metric is active
    if metric == "time":
        return ms
    elif metric == "throughput":
        return flops / ms * 1e-9
    elif metric == "bandwidth":
        return mem / ms * 1e-6
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def run_model_benchmark(args):
    plot_name = get_caller_name_no_ext()
    x_names = ["M", "hidden_dim", "intermediate_dim", "batch", "model_name"]
    benchmark = get_model_benchmark_object(
        plot_name,
        args,
        x_names=x_names,
        model_benchmark_shapes_fn=batched_model_benchmark_shapes,
    )

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_a8w8_per_token_group_prequant_w_per_batched_tensor_quant(
        M, hidden_dim, intermediate_dim, batch, metric, layer, **kwargs
    ):
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            K = math.ceil(K / args.tp)
        else:
            raise ValueError(f"Unsupported layer: {layer}")

        return bench_gemm_fn(
            batch,
            M,
            N,
            K,
            metric,
            args.backend,
            args.layout,
            args.group_size,
            not args.no_bias,
            args.transpose_bm,
            args.transpose_bm_in,
            args.config,
            args.gluon_launch_config,
            args.profile,
        )

    bench_batched_gemm_a8w8_per_token_group_prequant_w_per_batched_tensor_quant.run(
        save_path="." if args.o else None, print_data=True
    )


def run_shape_benchmark(args):
    plot_name = get_caller_name_no_ext()
    x_names = ["batch", "M", "N", "K"]
    benchmark = get_shape_benchmark_object(plot_name, args, x_names=x_names)

    @triton.testing.perf_report([benchmark])
    def bench_batched_gemm_a8w8_per_token_group_prequant_w_per_batched_tensor_quant(
        batch, M, N, K, metric, **kwargs
    ):
        return bench_gemm_fn(
            batch,
            M,
            N,
            K,
            metric,
            args.backend,
            args.layout,
            args.group_size,
            not args.no_bias,
            args.transpose_bm,
            args.transpose_bm_in,
            args.config,
            args.gluon_launch_config,
            args.profile,
        )

    bench_batched_gemm_a8w8_per_token_group_prequant_w_per_batched_tensor_quant.run(
        save_path="." if args.o else None, print_data=True
    )


def run_benchmark(args, defaults):
    if args.model:
        run_model_benchmark(args)
    else:
        run_shape_benchmark(args)


def parse_args(args: list[str] | None = None):
    parser = get_parser(
        "Batched A8W8 GEMM (A per-token-group pre-quant, W per-batched-tensor quant)"
    )
    parser = add_argparse_ff(parser)
    parser.add_argument("-B", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["triton", "gluon"],
        default=None,
        help="Kernel backend; default: the wrapper's own choice by arch and shape.",
    )
    parser.add_argument(
        "--config",
        type=json.loads,
        default=None,
        help="Explicit Triton config JSON; use --backend triton.",
    )
    parser.add_argument(
        "--gluon-launch-config",
        type=json.loads,
        default=None,
        help="Gluon launch diagnostic JSON (num_stages and waves_per_eu only).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Write one warmed-up invocation's Chrome trace; no clean timing.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        dest="group_size",
        help="Per-token group size for X quantization (default: 128).",
    )
    parser.add_argument(
        "--no-bias",
        action="store_true",
        default=False,
        help="Disable bias.",
    )
    parser.add_argument(
        "--transpose-bm",
        action="store_true",
        default=False,
        dest="transpose_bm",
        help="Transpose batch and M dimensions in the output tensor.",
    )
    parser.add_argument(
        "--transpose-bm-in",
        action="store_true",
        default=False,
        dest="transpose_bm_in",
        help="Transpose batch and M dimensions in the input tensor.",
    )
    parsed, defaults = get_ff_args(parser, args=args)
    if parsed.config is not None and not isinstance(parsed.config, dict):
        parser.error("--config must be a JSON object")
    if parsed.gluon_launch_config is not None:
        if not isinstance(parsed.gluon_launch_config, dict) or not set(
            parsed.gluon_launch_config
        ) <= {"num_stages", "waves_per_eu"}:
            parser.error(
                "--gluon-launch-config accepts num_stages and waves_per_eu only"
            )
        if parsed.backend != "gluon":
            parser.error("--gluon-launch-config requires --backend gluon")
    if parsed.profile and (parsed.model or not parsed.shape or len(parsed.shape) != 4):
        parser.error("--profile requires one --shape B M N K and no --model")
    return parsed, defaults


def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print_vgpr(lambda: run_benchmark(parsed_args, defaults))
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
