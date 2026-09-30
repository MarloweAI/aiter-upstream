# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16_small_m import (
    gemm_a16w16_small_m,
    gemm_a16w16_small_m_supported,
)

N = K = 2048
BF16 = torch.bfloat16


def generate_inputs(m: int, device: str = "cuda", seed: int = 2048):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, K), device=device, generator=generator).to(BF16)
    w = (torch.randn((N, K), device=device, generator=generator) * 0.02).to(BF16)
    return x, w


def _nrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = actual.float(), expected.float()
    error, norm = (a - b).square().mean().sqrt(), b.square().mean().sqrt()
    return (
        (error / norm).item()
        if norm.item()
        else (0.0 if not error.item() else float("inf"))
    )


def _assert_product(out, x, w):
    assert out.shape == (x.shape[0], N) and out.dtype == BF16
    exact = F.linear(x.float(), w.float())
    assert _nrmse(out, exact.to(BF16)) < 5e-4
    torch.testing.assert_close(out.float(), exact, atol=2e-2, rtol=2e-2)


def _misaligned(t: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(t.numel() + 1, device=t.device, dtype=t.dtype)[1:].view_as(t)
    return out.copy_(t)


requires_device = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not gemm_a16w16_small_m_supported(64, N, K, False, BF16, BF16),
    reason="needs gfx950 with Gluon",
)


@requires_device
@pytest.mark.parametrize("seed", [17, 2048, 4096])
def test_matches_fp32_reference(seed):
    x, w = generate_inputs(64, seed=seed)
    _assert_product(gemm_a16w16_small_m(x, w), x, w)


@requires_device
@pytest.mark.parametrize("case", ["zero", "cancellation"])
def test_zero_and_cancellation(case):
    x, w = generate_inputs(64)
    if case == "zero":
        x.zero_()
    else:
        x.fill_(1)
        w[:, ::2], w[:, 1::2] = 0.125, -0.125
    out = gemm_a16w16_small_m(x, w)
    assert not out.count_nonzero().item()
    _assert_product(out, x, w)


@requires_device
def test_copies_x_it_cannot_read_in_place():
    x, w = generate_inputs(64)
    expected = gemm_a16w16_small_m(x, w)
    strided = torch.zeros((64, 2 * K), device=x.device, dtype=BF16)[:, ::2]
    for a in (strided.copy_(x), _misaligned(x)):
        before = a.clone()
        out = gemm_a16w16_small_m(a, w)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(a, before)


@requires_device
def test_rejects_weights_it_would_have_to_copy():
    x, w = generate_inputs(64)
    for weight in (w.t().contiguous().t(), _misaligned(w)):
        with pytest.raises(ValueError):
            gemm_a16w16_small_m(x, weight)


@requires_device
def test_graph_replay_with_changed_input_and_poisoned_output():
    x, w = generate_inputs(64)
    original = w.clone()
    gemm_a16w16_small_m(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gemm_a16w16_small_m(x, w)
    for _ in range(5):
        x.copy_(torch.randn_like(x))
        before = x.clone()
        eager = gemm_a16w16_small_m(x, w)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))
        _assert_product(captured, x, w)
        assert torch.equal(x, before) and torch.equal(w, original)


@pytest.mark.parametrize(
    "call",
    [(m, N, K, False, BF16, BF16) for m in (16, 32, 48, 63, 65, 128)]
    + [
        (64, 4096, K, False, BF16, BF16),
        (64, N, 6144, False, BF16, BF16),
        (64, N, K, True, BF16, BF16),
        (64, N, K, False, torch.float16, torch.float16),
        (64, N, K, False, BF16, torch.float32),
    ],
)
def test_supported_rejects_every_other_call(call, monkeypatch):
    from aiter.ops.triton.gemm.basic import gemm_a16w16_small_m as module

    monkeypatch.setattr(module, "_gluon_arch", lambda: "gfx950")
    monkeypatch.setattr(module, "_padded_layout_available", lambda: True)
    assert not gemm_a16w16_small_m_supported(*call)


def test_rejects_cpu_tensors():
    with pytest.raises(ValueError):
        gemm_a16w16_small_m(
            torch.zeros((64, K), dtype=BF16), torch.zeros((N, K), dtype=BF16)
        )


def test_probes_are_false_on_a_gluon_without_amd(monkeypatch):
    from aiter.ops.triton.gemm.basic import gemm_a16w16_small_m as module

    language = types.ModuleType("triton.experimental.gluon.language")
    gluon = types.ModuleType("triton.experimental.gluon")
    gluon.language = language
    arch_info = types.ModuleType("aiter.ops.triton.utils._triton.arch_info")
    arch_info.get_arch = lambda: "gfx950"
    monkeypatch.setitem(sys.modules, "triton.experimental.gluon", gluon)
    monkeypatch.setitem(sys.modules, language.__name__, language)
    monkeypatch.setitem(sys.modules, arch_info.__name__, arch_info)
    probes = (module._gluon_arch, module._padded_layout_available)
    for probe in probes:
        probe.cache_clear()
    try:
        assert module._gluon_arch() is None
        assert not module._padded_layout_available()
        assert not gemm_a16w16_small_m_supported(64, N, K, False, BF16, BF16)
    finally:
        for probe in probes:
            probe.cache_clear()
