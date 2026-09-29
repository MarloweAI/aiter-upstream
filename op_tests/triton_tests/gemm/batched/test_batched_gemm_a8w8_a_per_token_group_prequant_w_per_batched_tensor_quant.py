# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import types

import pytest
import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    _get_config,
)
from aiter.ops.triton.gemm.batched import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as op_module,
)
from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype

e5m2_type, e4m3_type = get_fp8_dtypes()
DEVICE_ARCH = arch_info.get_arch()


def generate_batched_gemm_a16w8_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype | str,
    has_bias: bool,
    output: bool,
    layout: str = "TN",
    transpose_bm: bool = False,
):
    """
    Returns:
        - x: shape (B, M, K)
        - weight: shape (B, N, K)
        - x_scale: shape (B, M, 1)
        - w_scale: shape (B, 1, N)
    """
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]
    if layout[0] == "T":
        x = (torch.rand((B, M, K), dtype=torch.float16, device="cuda") / 10).to(
            torch.bfloat16
        )
    else:
        x = (
            (torch.rand((B, K, M), dtype=torch.float16, device="cuda") / 10)
            .to(torch.bfloat16)
            .permute(0, 2, 1)
        )

    if layout[1] == "N":
        weight = (torch.rand((B, N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        weight = (
            (torch.rand((B, N, K), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .permute(0, 2, 1)
        )

    w_scale = torch.rand([1], dtype=torch.float32, device="cuda")[0]
    if has_bias:
        bias = torch.rand([B, 1, N], dtype=dtype).cuda() * 10
    else:
        bias = None

    y = None
    if output:
        if transpose_bm:
            y = torch.empty((M, B, N), dtype=dtype, device=x.device)
        else:
            y = torch.empty((B, M, N), dtype=dtype, device=x.device)

    return x, weight, w_scale, bias, y


def run_torch(x, weight, w_scale, bias=None, dtype=torch.bfloat16, transpose_bm=True):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    w_bf16 = weight.to(torch.bfloat16) * w_scale.to(torch.bfloat16)
    out = torch.bmm(x, w_bf16.transpose(1, 2))
    if bias is not None:
        out = out + bias
    if transpose_bm:
        out = out.transpose(0, 1)
    return out.to(dtype)


def run_group_quantized_fp32(x, weight, w_scale, transpose_bm=True):
    """Independent oracle: exact FP8 group quantization, FP32 GEMM/accumulation."""
    out = torch.zeros(
        (x.shape[0], x.shape[1], weight.shape[1]), device=x.device, dtype=torch.float32
    )
    for group in range(4):
        first, last = group * 128, (group + 1) * 128
        a = x[..., first:last].float()
        scale = a.abs().amax(-1, keepdim=True).clamp_min(1e-10) * (1.0 / 448.0)
        quantized = (a * scale.reciprocal()).clamp(-448, 448).to(torch.float8_e4m3fn)
        out += (
            torch.bmm(
                quantized.float(), weight[..., first:last].float().transpose(1, 2)
            )
            * scale
        )
    out *= w_scale
    return out.transpose(0, 1) if transpose_bm else out


def run_full_precision_fp32(x, weight, w_scale, transpose_bm=True):
    """Diagnostic only: no activation quantization or BF16 intermediate rounding."""
    out = torch.bmm(x.float(), weight.float().transpose(1, 2)) * w_scale
    return out.transpose(0, 1) if transpose_bm else out


def _nrmse(actual, reference):
    return (
        actual.float() - reference
    ).square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12)


def check_group_quantized_fp32(actual, x, weight, scale, transpose_bm=True):
    """Reusable untimed gate; full-precision error is a separate diagnostic."""
    reference = run_group_quantized_fp32(x, weight, scale, transpose_bm)
    torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    error = _nrmse(actual, reference).item()
    assert error <= 0.01, f"Quantized reference NRMSE {error} exceeds 1%"
    full_precision = run_full_precision_fp32(x, weight, scale, transpose_bm)
    return {
        "quantized_nrmse": error,
        "quantized_max_abs": (actual.float() - reference).abs().max().item(),
        "full_precision_diagnostic_nrmse": _nrmse(actual, full_precision).item(),
    }


def run_triton(
    x,
    weight,
    w_scale,
    group_size=128,
    bias=None,
    dtype=torch.bfloat16,
    y=None,
    transpose_bm=False,
):
    return batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x,
        weight,
        w_scale,
        group_size=group_size,
        bias=bias,
        dtype=dtype,
        YQ=y,
        transpose_bm=transpose_bm,
    )


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    x_vals += [(v**2, 128, 512) for v in range(7)]
    x_vals += [(v**2, 512, 128) for v in range(7)]
    x_vals += [(1, 128, 1)]  # minimal case
    return x_vals


@pytest.mark.parametrize(
    "dtype, b, m, n, k, group_size, has_bias, output, transpose_bm",
    [
        (dtype, b, *shape, group_size, has_bias, output, transpose_bm)
        for output in [True, False]
        for dtype in ["bf16"]
        for b in [16]
        for shape in get_x_vals()
        for group_size in [128]
        for has_bias in [True, False]
        for transpose_bm in [True, False]
    ],
)
def test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
    dtype, b, m, n, k, group_size, has_bias, output, transpose_bm
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dtype = str_to_torch_dtype[dtype]
    x, weight, w_scale, bias, y = generate_batched_gemm_a16w8_inputs(
        b, m, n, k, dtype, has_bias, output, transpose_bm=transpose_bm
    )
    a = run_torch(x, weight, w_scale, bias, dtype, transpose_bm)
    b = run_triton(
        x,
        weight,
        w_scale,
        group_size=group_size,
        bias=bias,
        dtype=dtype,
        y=y,
        transpose_bm=transpose_bm,
    )

    triton.testing.assert_close(a, b, atol=0.1, rtol=0.1)


# (B, N, K) of the gfx950 Gluon small-M kernel's dispatch entry, and its M.
SMALL_M_SHAPE = (8, 256, 512)
SMALL_M_DISPATCH = (64,)


class _TensorMetadata:
    """CPU-only metadata stand-in; no kernel, CUDA query or backing allocation."""

    def __init__(self, shape, dtype, pointer, device="cuda:0", strides=None):
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.pointer = pointer
        if strides is None:
            reversed_strides, span = [], 1
            for size in reversed(shape):
                reversed_strides.append(span)
                span *= size
            strides = tuple(reversed(reversed_strides))
        self.strides = strides

    def stride(self, axis=None):
        return self.strides if axis is None else self.strides[axis]

    def data_ptr(self):
        return self.pointer

    def numel(self):
        product = 1
        for size in self.shape:
            product *= size
        return product

    def element_size(self):
        return {
            torch.bfloat16: 2,
            torch.float16: 2,
            torch.float32: 4,
            torch.float8_e4m3fn: 1,
        }[self.dtype]

    def is_contiguous(self):
        expected = _TensorMetadata(self.shape, self.dtype, self.pointer).stride()
        return self.stride() == expected

    def storage_offset(self):
        return 0


def _metadata_inputs(b=8):
    return (
        _TensorMetadata((64, b, 512), torch.bfloat16, 0x10000),
        _TensorMetadata((b, 256, 512), torch.float8_e4m3fn, 0x100000000),
        _TensorMetadata((), torch.float32, 0x10000000000),
    )


def _supports(x, weight, output, scale):
    return op_module._gluon_small_m_supports(
        x, weight, output, 64, 128, None, torch.bfloat16, scale
    )


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "x_span",
        "y_span",
        "x_alignment",
        "weight_alignment",
        "output_dtype",
        "output_device",
        "weight_device",
        "scale_device",
        "cpu_scale",
        "scale_dtype",
        "scale_vector",
        "alias_x",
        "alias_weight",
        "alias_scale",
        "output_overlap",
    ],
)
def test_gluon_small_m_buffer_contract(case, monkeypatch):
    monkeypatch.setattr(op_module, "_gluon_small_m_available", lambda: True)
    x, weight, scale = _metadata_inputs()
    output = _TensorMetadata((64, 8, 256), torch.bfloat16, 0x200000)
    if case == "x_span":
        x.strides = (2**25, 512, 1)
    elif case == "y_span":
        output.strides = (2**25, 256, 1)
    elif case == "x_alignment":
        x.pointer += 2
    elif case == "weight_alignment":
        weight.pointer += 1
    elif case == "output_dtype":
        output.dtype = torch.float32
    elif case.endswith("device"):
        {"output_device": output, "weight_device": weight, "scale_device": scale}[
            case
        ].device = torch.device("cuda:1")
    elif case == "cpu_scale":
        scale.is_cuda = False
        scale.device = torch.device("cpu")
    elif case == "scale_dtype":
        scale.dtype = torch.bfloat16
    elif case == "scale_vector":
        scale.shape, scale.strides = (8,), (1,)
    elif case.startswith("alias_"):
        output.pointer = {"alias_x": x, "alias_weight": weight, "alias_scale": scale}[
            case
        ].pointer
    elif case == "output_overlap":
        output.strides = (256, 256, 1)
    assert _supports(x, weight, output, scale) == (case == "valid")


def test_gluon_small_m_strided_span_boundary(monkeypatch):
    monkeypatch.setattr(op_module, "_gluon_small_m_available", lambda: True)
    x, weight, scale = _metadata_inputs()
    x.strides = (2**24, 512, 1)
    assert x.numel() * x.element_size() < 2**31
    assert _supports(x, weight, None, scale)
    x.strides = (2**25, 512, 1)
    assert x.numel() * x.element_size() < 2**31
    assert op_module._addressed_byte_span(x) > 2**31
    assert not _supports(x, weight, None, scale)


def test_gluon_small_m_quantized_oracle_cpu():
    """An exactly representable fixture proves scaling and group boundaries."""
    x = torch.zeros((1, 16, 512), dtype=torch.bfloat16)
    for group, magnitude in enumerate((1.0, 2.0, 4.0, 8.0)):
        x[..., group * 128 : (group + 1) * 128] = magnitude
    weight = torch.ones((1, 32, 512), dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.25)
    reference = run_group_quantized_fp32(x, weight, scale, transpose_bm=False)
    # Each constant group quantizes to 448 exactly and dequantizes to its input.
    expected = torch.full((1, 16, 32), 128 * (1 + 2 + 4 + 8) * 0.25)
    torch.testing.assert_close(reference, expected, atol=1e-4, rtol=1e-6)
    x.zero_()
    assert torch.count_nonzero(run_group_quantized_fp32(x, weight, scale)) == 0


def test_gluon_small_m_benchmark_median(monkeypatch):
    """Exercise the native benchmark's actual clean path without a GPU timer."""
    from op_tests.op_benchmarks.triton import (
        bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as benchmark,
    )

    x = torch.zeros((8, 64, 512), dtype=torch.bfloat16)
    weight = torch.zeros((8, 256, 512), dtype=torch.float8_e4m3fn)
    output = torch.empty((64, 8, 256), dtype=torch.bfloat16)
    monkeypatch.setattr(
        benchmark,
        "generate_batched_gemm_a8w8_per_token_group_inputs",
        lambda *args, **kwargs: (x, weight, torch.ones(()), None, output),
    )
    calls = []

    def operation(*args, **kwargs):
        calls.append(kwargs)
        return output

    def do_bench(fn, **kwargs):
        assert kwargs == {"warmup": 25, "rep": 100, "return_mode": "median"}
        fn()
        return 0.0123

    monkeypatch.setattr(
        benchmark,
        "batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant",
        operation,
    )
    monkeypatch.setattr(triton.testing, "do_bench", do_bench)
    launch = op_module._gluon_small_m
    result = benchmark.bench_gemm_fn(
        8,
        64,
        256,
        512,
        "time",
        "gluon",
        "TN",
        128,
        False,
        True,
        True,
        gluon_launch_config={"num_stages": 1, "waves_per_eu": 0},
    )
    assert result == 0.0123
    assert calls[0]["backend"] == "gluon"
    assert calls[0]["transpose_bm_in"] is True
    assert op_module._gluon_small_m is launch


def _require_gfx950():
    if DEVICE_ARCH != "gfx950":
        pytest.skip("The Gluon small-M kernel requires gfx950.")


def _small_m_inputs(m, b=SMALL_M_SHAPE[0], has_bias=False, transpose_bm=True):
    _, n, k = SMALL_M_SHAPE
    return generate_batched_gemm_a16w8_inputs(
        b,
        m,
        n,
        k,
        torch.bfloat16,
        has_bias=has_bias,
        output=True,
        transpose_bm=transpose_bm,
    )


def _misaligned_copy(x):
    """A copy of x whose data starts two bytes past a 16-byte boundary."""
    buffer = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)
    return buffer[1:].view(x.shape).copy_(x)


def _count_launches(monkeypatch):
    """Counts, by backend, the kernels the op launches from here on."""
    counts = {"gluon": 0, "triton": 0}
    gluon_launch = op_module._gluon_small_m
    triton_kernel = (
        op_module._batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel
    )

    def counted_gluon_launch(*args, **kwargs):
        counts["gluon"] += 1
        return gluon_launch(*args, **kwargs)

    class CountedTritonKernel:
        def __getitem__(self, grid):
            counts["triton"] += 1
            return triton_kernel[grid]

    monkeypatch.setattr(op_module, "_gluon_small_m", counted_gluon_launch)
    monkeypatch.setattr(
        op_module,
        "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel",
        CountedTritonKernel(),
    )
    return counts


@pytest.mark.parametrize("m", [16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("transpose_bm_in", [True, False])
@pytest.mark.parametrize("transpose_bm", [True, False])
@pytest.mark.parametrize("output", [True, False])
def test_gluon_small_m(m, transpose_bm_in, transpose_bm, output):
    _require_gfx950()
    x, weight, w_scale, _, y = _small_m_inputs(m, transpose_bm=transpose_bm)
    # Signed operands and independently scaled groups, including the zero-row floor.
    x = torch.randn_like(x)
    for group, magnitude in enumerate((0.001, 0.1, 1.0, 10.0)):
        x[..., group * 128 : (group + 1) * 128] *= magnitude
    x[:, 0] = 0
    weight = (
        torch.randn(weight.shape, device=weight.device)
        .mul_(40)
        .clamp_(-448, 448)
        .to(weight.dtype)
    )
    w_scale.fill_(0.0003792898787651211)
    x_in = x.transpose(0, 1).contiguous() if transpose_bm_in else x
    originals = (x_in.clone(), weight.view(torch.uint8).clone(), w_scale.clone())
    kwargs = {"transpose_bm": transpose_bm, "transpose_bm_in": transpose_bm_in}
    actual = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x_in, weight, w_scale, YQ=y if output else None, backend="gluon", **kwargs
    )
    expected = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x_in, weight, w_scale, backend="triton", **kwargs
    )
    reference = run_group_quantized_fp32(x, weight, w_scale, transpose_bm=transpose_bm)
    full_precision = run_full_precision_fp32(
        x, weight, w_scale, transpose_bm=transpose_bm
    )

    if output:
        assert actual is y
    # Same quantization as the Triton kernel; only FP32 summation order differs.
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    assert _nrmse(actual, reference) <= 0.01
    torch.testing.assert_close(expected.float(), reference, atol=0.02, rtol=0.02)
    assert _nrmse(expected, reference) <= 0.01
    assert torch.isfinite(full_precision).all()
    print(
        f"M={m}: quantized NRMSE={_nrmse(actual, reference).item():.6g}; full-precision diagnostic NRMSE={_nrmse(actual, full_precision).item():.6g}"
    )
    for original, current in zip(originals, (x_in, weight.view(torch.uint8), w_scale)):
        torch.testing.assert_close(current, original, atol=0, rtol=0)


@pytest.mark.parametrize("b, n", [(1, 32), (3, 64)])
def test_gluon_small_m_forced_general_domain(b, n):
    _require_gfx950()
    x, weight, scale, _, _ = generate_batched_gemm_a16w8_inputs(
        b, 16, n, 512, torch.bfloat16, has_bias=False, output=False
    )
    actual = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x, weight, scale, backend="gluon", transpose_bm=False
    )
    check_group_quantized_fp32(actual, x, weight, scale, transpose_bm=False)


@pytest.mark.parametrize("arm", ["native", "tile16x64", "tile16x32", "gluon_donor"])
def test_gluon_small_m_comparator_accuracy(arm, monkeypatch):
    _require_gfx950()
    x, weight, scale, _, _ = _small_m_inputs(64)
    x = (x - 0.05).contiguous()
    weight = (weight.float() - 0.05).to(weight.dtype)
    scale.fill_(0.125)
    config = None
    if arm.startswith("tile"):
        config = dict(_get_config(64, 256, 512)[0])
        config.update(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=64 if arm == "tile16x64" else 32,
            GROUP_SIZE_M=1,
            num_warps=4,
            num_stages=2,
            waves_per_eu=2 if arm == "tile16x64" else 1,
            matrix_instr_nonkdim=16,
            cache_modifier=".cg",
        )
    if arm == "gluon_donor":
        monkeypatch.setattr(
            op_module,
            "_gluon_small_m",
            functools.partial(op_module._gluon_small_m, num_stages=1, waves_per_eu=0),
        )
    actual = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x.transpose(0, 1).contiguous(),
        weight,
        scale,
        backend="gluon" if arm == "gluon_donor" else "triton",
        config=config,
        transpose_bm=True,
        transpose_bm_in=True,
    )
    check_group_quantized_fp32(actual, x, weight, scale)


@pytest.mark.parametrize("m", [3, 4, 5, 8, 16, 32, 63, 64, 65, 128])
@pytest.mark.parametrize("case", ["default", "config", "bias", "batch16", "misaligned"])
def test_gluon_small_m_dispatch(m, case, monkeypatch):
    _require_gfx950()
    b = 16 if case == "batch16" else SMALL_M_SHAPE[0]
    x, weight, w_scale, bias, y = _small_m_inputs(m, b=b, has_bias=case == "bias")
    x = x.transpose(0, 1).contiguous()
    if case == "misaligned":
        x = _misaligned_copy(x)
    kwargs = {"bias": bias, "transpose_bm": True, "transpose_bm_in": True}
    backend = "gluon" if case == "default" and m in SMALL_M_DISPATCH else "triton"
    expected = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x, weight, w_scale, backend=backend, **kwargs
    )
    config = _get_config(m, *SMALL_M_SHAPE[1:])[0] if case == "config" else None

    launches = _count_launches(monkeypatch)
    actual = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x, weight, w_scale, YQ=y, config=config, **kwargs
    )

    assert launches == {
        "gluon": int(backend == "gluon"),
        "triton": int(backend == "triton"),
    }
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "case",
    [
        "config",
        "misaligned",
        "m8",
        "output_dtype",
        "scale_dtype",
        "scale_vector",
        "alias_x",
    ],
)
def test_gluon_small_m_forced_invalid(case):
    _require_gfx950()
    m = 8 if case == "m8" else 64
    x, weight, w_scale, _, y = _small_m_inputs(m)
    x = x.transpose(0, 1).contiguous()
    kwargs = {"transpose_bm": True, "transpose_bm_in": True, "backend": "gluon"}
    if case == "config":
        kwargs["config"] = _get_config(m, *SMALL_M_SHAPE[1:])[0]
    elif case == "misaligned":
        x = _misaligned_copy(x)
    elif case == "output_dtype":
        kwargs["YQ"] = y.float()
    elif case == "scale_dtype":
        w_scale = w_scale.to(torch.bfloat16)
    elif case == "scale_vector":
        w_scale = w_scale.expand(8).contiguous()
    elif case == "alias_x":
        kwargs["YQ"] = x[..., :256]

    with pytest.raises(AssertionError):
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x, weight, w_scale, **kwargs
        )


@pytest.mark.parametrize("m", SMALL_M_DISPATCH)
def test_gluon_small_m_graph_replay(m, monkeypatch):
    _require_gfx950()
    x, weight, w_scale, _, y = _small_m_inputs(m)
    x = x.transpose(0, 1).contiguous()
    kwargs = {"transpose_bm": True, "transpose_bm_in": True}
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x, weight, w_scale, YQ=y, **kwargs
    )
    torch.cuda.synchronize()
    launches = _count_launches(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x, weight, w_scale, YQ=y, **kwargs
        )
    assert launches == {"gluon": 1, "triton": 0}
    x.copy_((x.float() * 0.5).to(x.dtype))
    for _ in range(3):
        graph.replay()
    expected = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x, weight, w_scale, backend="gluon", **kwargs
    )

    torch.testing.assert_close(y, expected, atol=0, rtol=0)


def test_gluon_small_m_nondefault_stream():
    _require_gfx950()
    x, weight, scale, _, y = _small_m_inputs(64)
    x = x.transpose(0, 1).contiguous()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x.fill_(0.125)
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x,
            weight,
            scale,
            YQ=y,
            backend="gluon",
            transpose_bm=True,
            transpose_bm_in=True,
        )
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    reference = run_group_quantized_fp32(x.transpose(0, 1), weight, scale)
    torch.testing.assert_close(y.float(), reference, atol=0.02, rtol=0.02)
    assert _nrmse(y, reference) <= 0.01


def test_gluon_small_m_operand_device():
    _require_gfx950()
    if torch.cuda.device_count() < 2:
        pytest.skip("Current-device mismatch check needs two visible GPUs.")
    original_device = torch.cuda.current_device()
    target_device = 1 if original_device == 0 else 0
    with torch.cuda.device(target_device):
        x, weight, scale, _, y = _small_m_inputs(64)
        x = x.transpose(0, 1).contiguous()
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x, weight, scale, YQ=y, backend="gluon", transpose_bm=True, transpose_bm_in=True
    )
    assert torch.cuda.current_device() == original_device
    with torch.cuda.device(target_device):
        torch.cuda.synchronize()
        reference = run_group_quantized_fp32(x.transpose(0, 1), weight, scale)
        torch.testing.assert_close(y.float(), reference, atol=0.02, rtol=0.02)
    with pytest.raises(AssertionError):
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x,
            weight,
            scale.to(f"cuda:{original_device}"),
            backend="gluon",
            transpose_bm_in=True,
        )


def test_gluon_small_m_fallback_graph_replay(monkeypatch):
    """An eager failure disables Gluon before capture; the fallback graph replays."""
    _require_gfx950()
    x, weight, scale, _, y = _small_m_inputs(64)
    x = x.transpose(0, 1).contiguous()
    kwargs = {"transpose_bm": True, "transpose_bm_in": True, "YQ": y}

    def failing_launch(*args, **kwargs):
        raise RuntimeError("eager compile failure")

    monkeypatch.setattr(op_module, "_gluon_small_m_failed", False)
    monkeypatch.setattr(op_module, "_gluon_small_m", failing_launch)
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x, weight, scale, **kwargs
    )
    assert op_module._gluon_small_m_failed
    launches = _count_launches(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x, weight, scale, **kwargs
        )
    assert launches == {"gluon": 0, "triton": 1}
    x.mul_(0.5)
    for _ in range(3):
        graph.replay()
    expected = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x, weight, scale, backend="triton", transpose_bm=True, transpose_bm_in=True
    )
    torch.testing.assert_close(y, expected, atol=0, rtol=0)


@pytest.mark.parametrize("b, supported", [(16383, True), (16384, False)])
def test_gluon_small_m_weight_size_bound(b, supported, monkeypatch):
    """Metadata tensors exercise actual support logic without large allocations."""
    monkeypatch.setattr(op_module, "_gluon_small_m_available", lambda: True)
    x, weight, scale = _metadata_inputs(b=b)
    assert _supports(x, weight, None, scale) == supported


@pytest.mark.parametrize(
    "missing", [None, "amd", "cdna4", "mfma", "load_shared_relaxed", "tiles_per_warp"]
)
def test_gluon_small_m_api_probe(missing, monkeypatch):
    """A Gluon without one of the APIs the kernel uses selects the Triton kernel."""
    gluon = pytest.importorskip("triton.experimental.gluon")

    def api(*args, **kwargs):
        pass

    def AMDMFMALayout(version, instr_shape, transposed, warps_per_cta, tiles_per_warp):
        pass

    def AMDMFMALayoutWithoutTiles(version, instr_shape, transposed, warps_per_cta):
        pass

    async_copy = types.SimpleNamespace(
        buffer_load_to_shared=api,
        load_shared_relaxed=api,
        commit_group=api,
        wait_group=api,
    )
    cdna4 = types.SimpleNamespace(
        compute_efficient_padded_shared_layout=api, mfma=api, async_copy=async_copy
    )
    amd = types.SimpleNamespace(cdna4=cdna4, AMDMFMALayout=AMDMFMALayout)
    language = types.SimpleNamespace(amd=amd)
    if missing == "amd":
        del language.amd
    elif missing == "cdna4":
        del amd.cdna4
    elif missing == "mfma":
        del cdna4.mfma
    elif missing == "load_shared_relaxed":
        del async_copy.load_shared_relaxed
    elif missing == "tiles_per_warp":
        amd.AMDMFMALayout = AMDMFMALayoutWithoutTiles
    monkeypatch.setattr(gluon, "language", language)
    monkeypatch.setattr(op_module, "get_arch", lambda: "gfx950")

    assert op_module._gluon_small_m_available.__wrapped__() == (missing is None)


def test_gluon_small_m_failure_falls_back(monkeypatch):
    """A Gluon failure on a default call runs the Triton kernel from then on; a
    forced call, before or after, tries the Gluon kernel and raises its error.
    Stub kernels on CPU tensors; no kernel runs."""
    calls = {"gluon": 0, "triton": 0, "warnings": 0}

    def failing_gluon_launch(*args, **kwargs):
        calls["gluon"] += 1
        raise RuntimeError("stub compile error")

    class TritonKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls["triton"] += 1

            return launch

    def warning(*args):
        calls["warnings"] += 1

    monkeypatch.setattr(op_module, "_gluon_small_m_failed", False)
    monkeypatch.setattr(op_module, "_gluon_small_m_available", lambda: True)
    monkeypatch.setattr(op_module, "_gluon_small_m_supports", lambda *args: True)
    monkeypatch.setattr(op_module, "_gluon_small_m", failing_gluon_launch)
    monkeypatch.setattr(
        op_module,
        "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel",
        TritonKernel(),
    )
    monkeypatch.setattr(op_module, "_get_config", lambda M, N, K: ({}, False))
    monkeypatch.setattr(op_module._LOGGER, "warning", warning)
    b, n, k = SMALL_M_SHAPE
    x = torch.zeros((64, b, k), dtype=torch.bfloat16)
    weight = torch.zeros((b, n, k), dtype=torch.float8_e4m3fn)
    w_scale = torch.ones((), dtype=torch.float32)
    kwargs = {"transpose_bm": True, "transpose_bm_in": True}

    with pytest.raises(RuntimeError):
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x, weight, w_scale, backend="gluon", **kwargs
        )
    assert not op_module._gluon_small_m_failed
    for _ in range(2):
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x, weight, w_scale, **kwargs
        )
    assert op_module._gluon_small_m_failed
    # A forced call still tries the kernel, and raises its error.
    with pytest.raises(RuntimeError):
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            x, weight, w_scale, backend="gluon", **kwargs
        )

    assert calls == {"gluon": 3, "triton": 2, "warnings": 1}
