# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16_xcd_reuse import (
    gemm_a16w16_xcd_reuse,
    gemm_a16w16_xcd_reuse_accepts,
    gemm_a16w16_xcd_reuse_supported,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_small_m import _assert_product

M, N, K = 128, 2048, 2048
BF16 = torch.bfloat16


def generate_inputs(m: int = M, device: str = "cuda", seed: int = 2048):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, K), device=device, generator=generator).to(BF16)
    w = (torch.randn((N, K), device=device, generator=generator) * 0.02).to(BF16)
    return x, w


def _device_supported() -> bool:
    return torch.cuda.is_available() and gemm_a16w16_xcd_reuse_supported(
        M, N, K, False, BF16, BF16
    )


requires_device = pytest.mark.skipif(
    not _device_supported(), reason="needs gfx950 with Gluon"
)


@requires_device
@pytest.mark.parametrize("seed", [17, 2048, 4096])
def test_matches_fp32_reference(seed):
    x, w = generate_inputs(seed=seed)
    out = gemm_a16w16_xcd_reuse(x, w)
    assert out.shape == (M, N) and out.dtype == BF16

    _assert_product(out, x, w)


@requires_device
def test_copies_x_it_cannot_read_in_place_and_rejects_such_a_weight():
    x, w = generate_inputs()
    expected = gemm_a16w16_xcd_reuse(x, w)
    strided_x = torch.zeros((M, 2 * K), device=x.device, dtype=BF16)[:, ::2]
    strided_x.copy_(x)
    misaligned_x = torch.zeros(M * K + 1, device=x.device, dtype=BF16)[1:].view(M, K)
    misaligned_x.copy_(x)
    for a in (strided_x, misaligned_x):
        out = gemm_a16w16_xcd_reuse(a, w)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    misaligned_w = torch.zeros(N * K + 1, device=w.device, dtype=BF16)[1:].view(N, K)
    misaligned_w.copy_(w)
    assert gemm_a16w16_xcd_reuse_accepts(x, w)
    for b in (w.t().contiguous().t(), misaligned_w):
        assert not gemm_a16w16_xcd_reuse_accepts(x, b)
        with pytest.raises(ValueError):
            gemm_a16w16_xcd_reuse(x, b)


@requires_device
@pytest.mark.parametrize("kind", ["zero", "cancellation"])
def test_zero_and_cancellation(kind):
    x, w = generate_inputs()
    if kind == "zero":
        x.zero_()
    else:
        x.fill_(1)
        w[:, ::2] = 0.125
        w[:, 1::2] = -0.125
    _assert_product(gemm_a16w16_xcd_reuse(x, w), x, w)


@requires_device
def test_graph_replay_is_bitwise_equal_to_eager():
    x, w = generate_inputs()
    original_w = w.clone()
    gemm_a16w16_xcd_reuse(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gemm_a16w16_xcd_reuse(x, w)
    for _ in range(5):
        x.copy_(torch.randn_like(x))
        before = x.clone()
        eager = gemm_a16w16_xcd_reuse(x, w)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))
        _assert_product(captured, x, w)
        assert torch.equal(x, before) and torch.equal(w, original_w)


@requires_device
@pytest.mark.parametrize("m", [100, 127, 128, 129])
def test_tuned_gemm_takes_the_kernel_at_m128_only(m: int):
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm

    if get_cu_num() != 256:
        pytest.skip("the tuned row is for a 256-CU gfx950")
    x, w = generate_inputs(m)
    config = get_GEMM_A16W16_config(m, N, K, False, str(BF16), str(BF16))
    if m == M:
        assert (config["libtype"], config["kernelName"]) == (
            "triton",
            "gemm_a16w16_xcd_reuse",
        )
    else:
        assert config.get("kernelName") != "gemm_a16w16_xcd_reuse"
    out = tgemm.mm(x, w)
    torch.testing.assert_close(
        out.float(), F.linear(x.float(), w.float()), atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize(
    "call",
    [
        (64, N, K, False, BF16, BF16),
        (127, N, K, False, BF16, BF16),
        (129, N, K, False, BF16, BF16),
        (256, N, K, False, BF16, BF16),
        (M, 4096, K, False, BF16, BF16),
        (M, N, 6144, False, BF16, BF16),
        (M, N, K, True, BF16, BF16),
        (M, N, K, False, torch.float16, torch.float16),
        (M, N, K, False, BF16, torch.float32),
    ],
)
def test_supported_rejects_every_other_call(call, monkeypatch):
    from aiter.ops.triton.gemm.basic import gemm_a16w16_xcd_reuse as module

    monkeypatch.setattr(module, "_gluon_arch", lambda: "gfx950")
    monkeypatch.setattr(module, "_padded_layout_available", lambda: True)
    assert not gemm_a16w16_xcd_reuse_supported(*call)


def test_supported_is_false_on_a_gluon_without_amd(monkeypatch):
    from aiter.ops.triton.gemm.basic import gemm_a16w16_small_m as module

    language = types.ModuleType("triton.experimental.gluon.language")
    gluon = types.ModuleType("triton.experimental.gluon")
    gluon.language = language
    arch_info = types.ModuleType("aiter.ops.triton.utils._triton.arch_info")
    arch_info.get_arch = lambda: "gfx950"
    monkeypatch.setitem(sys.modules, "triton.experimental.gluon", gluon)
    monkeypatch.setitem(sys.modules, language.__name__, language)
    monkeypatch.setitem(sys.modules, arch_info.__name__, arch_info)
    module._gluon_arch.cache_clear()
    module._padded_layout_available.cache_clear()
    try:
        assert not gemm_a16w16_xcd_reuse_supported(M, N, K, False, BF16, BF16)
    finally:
        module._gluon_arch.cache_clear()
        module._padded_layout_available.cache_clear()


@requires_device
@pytest.mark.parametrize(
    "case", ["m64", "m127", "m129", "fp16", "mixed dtypes", "wrong weight"]
)
def test_rejects_other_calls(case: str):
    x, w = generate_inputs()
    a, b = {
        "m64": (x[:64], w),
        "m127": (x[:127], w),
        "m129": (torch.cat([x, x[:1]]), w),
        "fp16": (x.half(), w.half()),
        "mixed dtypes": (x, w.float()),
        "wrong weight": (x, w[:1024]),
    }[case]
    with pytest.raises(ValueError):
        gemm_a16w16_xcd_reuse(a, b)


def test_rejects_cpu_tensors():
    x, w = torch.zeros((M, K), dtype=BF16), torch.zeros((N, K), dtype=BF16)
    assert not gemm_a16w16_xcd_reuse_accepts(x, w)
    assert not gemm_a16w16_xcd_reuse_accepts(x.flatten(), w)
    with pytest.raises(ValueError):
        gemm_a16w16_xcd_reuse(x, w)
