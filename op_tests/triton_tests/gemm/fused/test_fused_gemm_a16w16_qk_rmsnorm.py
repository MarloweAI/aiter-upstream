# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

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
    return ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()


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
