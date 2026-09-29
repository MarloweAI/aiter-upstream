# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.enum import QuantType
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm
from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_qk_rmsnorm import (
    fused_gemm_a16w16_qk_rmsnorm,
    fused_gemm_a16w16_qk_rmsnorm_supported,
)

HIDDEN = 6144
OUT_DIM = 2624
Q_DIM = 2048
KV_DIM = 512
EPS = 1e-5


def generate_inputs(m: int, device: str = "cuda", seed: int = 2624):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, HIDDEN), device=device, generator=generator).to(torch.bfloat16)
    weight = (
        torch.randn((OUT_DIM, HIDDEN), device=device, generator=generator) * 0.02
    ).to(torch.bfloat16)
    q_weight = (torch.rand(Q_DIM, device=device, generator=generator) + 0.5).to(
        torch.bfloat16
    )
    k_weight = (torch.rand(KV_DIM, device=device, generator=generator) + 0.5).to(
        torch.bfloat16
    )
    return x, weight, q_weight, k_weight


def _device_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    return fused_gemm_a16w16_qk_rmsnorm_supported(*generate_inputs(128, seed=0))[0]


requires_device = pytest.mark.skipif(
    not _device_supported(), reason="needs a 256-CU gfx950 with Gluon"
)


def _nrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = actual.float(), expected.float()
    error = (a - b).square().mean().sqrt().item()
    scale = b.square().mean().sqrt().item()
    return error / scale if scale else (0.0 if error == 0.0 else math.inf)


def _rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    scale = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + EPS)
    return (xf * scale * weight.float()).to(torch.bfloat16)


@requires_device
@pytest.mark.parametrize("m", [128, 256])
def test_matches_fp32_reference(m: int):
    x, weight, q_weight, k_weight = generate_inputs(m)

    out, q_out, k_out = fused_gemm_a16w16_qk_rmsnorm(
        x, weight, q_weight, EPS, k_weight, EPS
    )

    # One bf16 rounding of an fp32 sum: out differs from the correctly rounded
    # fp32 product only where the summation order flips a rounding.
    exact = F.linear(x.float(), weight.float())
    assert _nrmse(out, exact.to(torch.bfloat16)) < 5e-4
    torch.testing.assert_close(out.float(), exact, atol=2e-2, rtol=2e-2)

    q_ref = torch.empty_like(q_out)
    k_ref = torch.empty_like(k_out)
    fused_qk_rmsnorm(
        q_out_quantized=q_ref,
        k_out=k_ref,
        q=out[:, :Q_DIM],
        q_weight=q_weight,
        q_epsilon=EPS,
        k=out[:, Q_DIM : Q_DIM + KV_DIM],
        k_weight=k_weight,
        k_epsilon=EPS,
        quant_type=QuantType.No,
    )
    assert torch.equal(q_out.view(torch.int16), q_ref.view(torch.int16))
    assert torch.equal(k_out.view(torch.int16), k_ref.view(torch.int16))
    torch.testing.assert_close(
        q_out, _rmsnorm_fp32(exact[:, :Q_DIM], q_weight), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        k_out,
        _rmsnorm_fp32(exact[:, Q_DIM : Q_DIM + KV_DIM], k_weight),
        atol=2e-2,
        rtol=2e-2,
    )


@requires_device
@pytest.mark.parametrize("m", [128, 256])
def test_graph_replay_is_bitwise_equal_to_eager(m: int):
    x, weight, q_weight, k_weight = generate_inputs(m)
    eager = fused_gemm_a16w16_qk_rmsnorm(x, weight, q_weight, EPS, k_weight, EPS)
    again = fused_gemm_a16w16_qk_rmsnorm(x, weight, q_weight, EPS, k_weight, EPS)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_gemm_a16w16_qk_rmsnorm(x, weight, q_weight, EPS, k_weight, EPS)
    graph.replay()
    torch.cuda.synchronize()

    for a, b, c in zip(eager, again, captured):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
        assert torch.equal(a.view(torch.int16), c.view(torch.int16))


def _cpu_operands(m: int):
    return (
        torch.zeros((m, HIDDEN), dtype=torch.bfloat16),
        torch.zeros((OUT_DIM, HIDDEN), dtype=torch.bfloat16),
        torch.ones(Q_DIM, dtype=torch.bfloat16),
        torch.ones(KV_DIM, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("m", [1, 4, 64, 127, 129, 255, 257, 512])
def test_rejects_other_m(m: int):
    x, weight, q_weight, k_weight = _cpu_operands(m)
    supported, reason = fused_gemm_a16w16_qk_rmsnorm_supported(
        x, weight, q_weight, k_weight
    )
    assert not supported and "128 or 256" in reason
    with pytest.raises(ValueError):
        fused_gemm_a16w16_qk_rmsnorm(x, weight, q_weight, EPS, k_weight, EPS)


def test_rejects_other_operands():
    x, weight, q_weight, k_weight = _cpu_operands(128)
    cases = {
        "fp16 x": (x.half(), weight, q_weight, k_weight),
        "strided x": (
            torch.zeros((128, 2 * HIDDEN), dtype=torch.bfloat16)[:, ::2],
            weight,
            q_weight,
            k_weight,
        ),
        "wrong weight": (x, weight[:2560], q_weight, k_weight),
        "fp32 norm weight": (x, weight, q_weight.float(), k_weight),
        "cpu tensors": (x, weight, q_weight, k_weight),
    }
    for name, operands in cases.items():
        supported, _ = fused_gemm_a16w16_qk_rmsnorm_supported(*operands)
        assert not supported, name
        with pytest.raises(ValueError):
            fused_gemm_a16w16_qk_rmsnorm(
                operands[0], operands[1], operands[2], EPS, operands[3], EPS
            )


def test_nrmse_zero_reference():
    assert _nrmse(torch.zeros(4), torch.zeros(4)) == 0.0
    assert math.isinf(_nrmse(torch.ones(4), torch.zeros(4)))


def test_dispatch_selects_operand_device_and_restores_it(monkeypatch):
    """Exercise the public wrapper's actual dispatch without a GPU launch."""
    module = importlib.import_module(
        "aiter.ops.triton.gemm.fused.fused_gemm_a16w16_qk_rmsnorm"
    )
    state = {"device": "caller", "fail": False}
    calls = []

    class DeviceContext:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            self.previous = state["device"]
            state["device"] = self.device

        def __exit__(self, *args):
            state["device"] = self.previous

    class Producer:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                assert state["device"] == "operand"
                calls.append("producer")
                if state["fail"]:
                    raise RuntimeError("launch failure")

            return launch

    def empty(*args, **kwargs):
        assert state["device"] == "operand"
        return SimpleNamespace(device="operand")

    outputs = (object(), object(), object())

    def reduce(*args):
        assert state["device"] == "operand"
        calls.append("reduce")
        return outputs

    producer_path = (
        "aiter.ops.triton._gluon_kernels.gfx950.gemm.fused."
        "fused_gemm_a16w16_qk_rmsnorm"
    )
    fake_producer = ModuleType(producer_path)
    fake_producer._gemm_a16w16_splitk_planes_kernel = Producer()
    hip_path = "aiter.ops.splitk_reduce_qk_rmsnorm"
    fake_hip = ModuleType(hip_path)
    fake_hip.splitk_reduce_qk_rmsnorm = reduce
    monkeypatch.setitem(sys.modules, producer_path, fake_producer)
    monkeypatch.setitem(sys.modules, hip_path, fake_hip)
    monkeypatch.setattr(
        module, "fused_gemm_a16w16_qk_rmsnorm_supported", lambda *a: (True, "")
    )
    monkeypatch.setattr(module.torch.cuda, "device", DeviceContext)
    monkeypatch.setattr(module.torch, "empty", empty)
    x = SimpleNamespace(shape=(128, 6144), device="operand")
    weight = SimpleNamespace(shape=(2624, 6144))
    assert (
        module.fused_gemm_a16w16_qk_rmsnorm(x, weight, object(), EPS, object(), EPS)
        == outputs
    )
    assert calls == ["producer", "reduce"]
    assert state["device"] == "caller"
    state["fail"] = True
    with pytest.raises(RuntimeError, match="launch failure"):
        module.fused_gemm_a16w16_qk_rmsnorm(x, weight, object(), EPS, object(), EPS)
    assert state["device"] == "caller"


@requires_device
@pytest.mark.parametrize("m", [128, 256])
def test_changed_inputs_and_poisoned_graph_workspace(m, monkeypatch):
    module = importlib.import_module(
        "aiter.ops.triton.gemm.fused.fused_gemm_a16w16_qk_rmsnorm"
    )
    inputs = generate_inputs(m)
    partials = []
    original_empty = torch.empty

    def record_empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if tuple(tensor.shape) == (6, m, OUT_DIM) and tensor.dtype == torch.float32:
            tensor.fill_(float("nan"))
            partials.append(tensor)
        return tensor

    monkeypatch.setattr(module.torch, "empty", record_empty)
    fused_gemm_a16w16_qk_rmsnorm(inputs[0], inputs[1], inputs[2], EPS, inputs[3], EPS)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_gemm_a16w16_qk_rmsnorm(
            inputs[0], inputs[1], inputs[2], EPS, inputs[3], EPS
        )
    graph_partial = partials[-1]
    for pattern, seed in (("random", 101), ("zero", 102), ("cancellation", 103)):
        changed = generate_inputs(m, seed=seed)
        for actual, replacement in zip(inputs, changed):
            actual.copy_(replacement)
        if pattern == "zero":
            inputs[0].zero_()
        elif pattern == "cancellation":
            inputs[0][:, 1::2].copy_(inputs[0][:, ::2])
            inputs[1][:, 1::2].copy_(-inputs[1][:, ::2])
            inputs[1][:, 0].add_(0.03125)
        snapshots = [tensor.clone() for tensor in inputs]
        exact = F.linear(inputs[0].float(), inputs[1].float())
        for _ in range(3):
            graph_partial.fill_(float("nan"))
            for output in captured:
                output.fill_(float("nan"))
            graph.replay()
            eager = fused_gemm_a16w16_qk_rmsnorm(
                inputs[0], inputs[1], inputs[2], EPS, inputs[3], EPS
            )
            torch.cuda.synchronize()
            for actual, expected in zip(captured, eager):
                assert torch.isfinite(actual).all()
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert _nrmse(captured[0], exact.bfloat16()) < 5e-4
            torch.testing.assert_close(captured[0].float(), exact, atol=0.02, rtol=0.02)
            for actual, reference in zip(
                captured[1:],
                (
                    _rmsnorm_fp32(exact[:, :Q_DIM], inputs[2]),
                    _rmsnorm_fp32(exact[:, Q_DIM : Q_DIM + KV_DIM], inputs[3]),
                ),
            ):
                torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
                assert _nrmse(actual, reference) <= 0.01
            for actual, snapshot in zip(inputs, snapshots):
                assert torch.equal(actual.view(torch.int16), snapshot.view(torch.int16))
        assert torch.isfinite(graph_partial).all()
        for plane in range(6):
            start = plane * 1024
            reference = F.linear(
                inputs[0][:, start : start + 1024].float(),
                inputs[1][:, start : start + 1024].float(),
            )
            torch.testing.assert_close(
                graph_partial[plane], reference, atol=0.02, rtol=0.02
            )


@requires_device
def test_rejects_invalid_gpu_operands():
    x, weight, q_weight, k_weight = generate_inputs(128)
    misaligned = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)[1:].view_as(
        x
    )
    cases = (
        (misaligned, weight, q_weight, k_weight),
        (x[:, :-1], weight, q_weight, k_weight),
        (x, weight[:2560], q_weight, k_weight),
        (x, weight.float(), q_weight, k_weight),
        (x, weight, q_weight.float(), k_weight),
        (x, weight, q_weight[:-1], k_weight),
        (x, weight, q_weight, k_weight[:-1]),
    )
    for operands in cases:
        assert not fused_gemm_a16w16_qk_rmsnorm_supported(*operands)[0]
        with pytest.raises(ValueError):
            fused_gemm_a16w16_qk_rmsnorm(
                operands[0], operands[1], operands[2], EPS, operands[3], EPS
            )


@requires_device
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two visible GPUs")
def test_operand_device_stream_and_mixed_device_rejection():
    with torch.cuda.device(1):
        original_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream(device=1)
        with torch.cuda.stream(stream):
            inputs = generate_inputs(128, device="cuda:1")
            with torch.cuda.device(0):
                caller_stream = torch.cuda.current_stream()
                outputs = fused_gemm_a16w16_qk_rmsnorm(
                    inputs[0], inputs[1], inputs[2], EPS, inputs[3], EPS
                )
                assert torch.cuda.current_device() == 0
                assert torch.cuda.current_stream() == caller_stream
            assert torch.cuda.current_stream() == stream
            exact = F.linear(inputs[0].float(), inputs[1].float())
            stream.synchronize()
            assert _nrmse(outputs[0], exact.bfloat16()) < 5e-4
            torch.testing.assert_close(outputs[0].float(), exact, atol=0.02, rtol=0.02)
        assert torch.cuda.current_stream() == original_stream
    bad_weight = inputs[1].to("cuda:0")
    assert not fused_gemm_a16w16_qk_rmsnorm_supported(
        inputs[0], bad_weight, inputs[2], inputs[3]
    )[0]
    with pytest.raises(ValueError, match="one device"):
        fused_gemm_a16w16_qk_rmsnorm(
            inputs[0], bad_weight, inputs[2], EPS, inputs[3], EPS
        )
