# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

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


@pytest.mark.parametrize("m", [16, 32, 64, 128])
@pytest.mark.parametrize("transpose_bm_in", [True, False])
@pytest.mark.parametrize("transpose_bm", [True, False])
def test_gluon_small_m(m, transpose_bm_in, transpose_bm):
    _require_gfx950()
    x, weight, w_scale, _, y = _small_m_inputs(m, transpose_bm=transpose_bm)
    # Signed values, one large value in each 128-element group of the first batch
    # entry, and an all-zero first row, which takes the 1e-10 scale floor.
    x = x - 0.05
    x[0, :, ::128] = 4.0
    x[:, 0] = 0
    x_in = x.transpose(0, 1).contiguous() if transpose_bm_in else x
    kwargs = {"transpose_bm": transpose_bm, "transpose_bm_in": transpose_bm_in}
    actual = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x_in, weight, w_scale, YQ=y, backend="gluon", **kwargs
    )
    expected = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x_in, weight, w_scale, backend="triton", **kwargs
    )
    reference = run_torch(x, weight, w_scale, transpose_bm=transpose_bm)

    assert actual is y
    # Same quantization as the Triton kernel; only FP32 summation order differs.
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    triton.testing.assert_close(reference, actual, atol=0.1, rtol=0.1)


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


@pytest.mark.parametrize("case", ["config", "misaligned", "m8"])
def test_gluon_small_m_forced_invalid(case):
    _require_gfx950()
    m = 8 if case == "m8" else 64
    x, weight, w_scale, _, _ = _small_m_inputs(m)
    x = x.transpose(0, 1).contiguous()
    kwargs = {"transpose_bm": True, "transpose_bm_in": True, "backend": "gluon"}
    if case == "config":
        kwargs["config"] = _get_config(m, *SMALL_M_SHAPE[1:])[0]
    elif case == "misaligned":
        x = _misaligned_copy(x)

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
    graph.replay()
    expected = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x, weight, w_scale, backend="gluon", **kwargs
    )

    torch.testing.assert_close(y, expected, atol=0, rtol=0)


@pytest.mark.parametrize("b, supported", [(16383, True), (16384, False)])
def test_gluon_small_m_weight_size_bound(b, supported, monkeypatch):
    """The kernel's int32 offsets need the weight under 2 GiB; no kernel runs."""
    monkeypatch.setattr(op_module, "_gluon_small_m_available", lambda: True)
    _, n, k = SMALL_M_SHAPE
    x = torch.empty((64, b, k), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((b, n, k), dtype=torch.float8_e4m3fn, device="meta")

    assert (
        op_module._gluon_small_m_supports(
            x, weight, None, 64, 128, None, torch.bfloat16
        )
        == supported
    )


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
