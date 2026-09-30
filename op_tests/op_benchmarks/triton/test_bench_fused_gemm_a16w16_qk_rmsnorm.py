# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU tests of the benchmark gates, without AITER GPU/JIT initialization."""

import ast
import logging
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).with_name("bench_fused_gemm_a16w16_qk_rmsnorm.py")


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def gates(monkeypatch):
    # Execute the real public helper and benchmark decision code on CPU tensors.
    # Extract only these definitions so importing the tests does not probe GPUs.
    native = load_definitions(
        ROOT / "aiter/test_common.py",
        {
            "_CATASTROPHIC_REL_THRESHOLD",
            "_relmag_catastrophic",
            "_check_catastrophic",
            "_catastrophic_check_silent",
            "checkAllclose",
            "assertAllclose",
        },
        {"torch": torch, "logger": logging.getLogger(__name__)},
    )
    helper_module = ModuleType("aiter.test_common")
    calls = []

    def observed_helper(actual, reference, **kwargs):
        calls.append((actual.clone(), reference.clone(), dict(kwargs)))
        return native["assertAllclose"](actual, reference, **kwargs)

    helper_module.assertAllclose = observed_helper
    monkeypatch.setitem(sys.modules, "aiter.test_common", helper_module)
    bench = load_definitions(
        BENCH,
        {
            "BASELINE_PROJECTION_POLICY",
            "nrmse",
            "GateFailure",
            "check_baseline_projection",
            "check_outputs",
        },
        {"torch": torch, "math": math},
    )
    return SimpleNamespace(**bench, calls=calls)


def projection_pair():
    reference = torch.ones(10000, dtype=torch.float32)
    return reference.bfloat16(), reference


def whole_outputs(projection, reference):
    norm_reference = torch.ones((2, 16), dtype=torch.float32)
    return (
        [projection, norm_reference.bfloat16(), norm_reference.bfloat16()],
        [reference, norm_reference, norm_reference],
    )


def test_retained_sparse_near_zero_mechanism_is_not_a_candidate_relaxation(gates):
    actual, reference = projection_pair()
    reference[0] = -0.03184366226196289
    actual[0] = -0.052734375
    outputs, refs = whole_outputs(actual, reference)
    assert gates.check_outputs(256, "separate", "random", outputs, refs)[0] < 0.01
    with pytest.raises(gates.GateFailure) as failure:
        gates.check_outputs(256, "fused", "random", outputs, refs)
    assert failure.value.metadata["field"] == "projection"
    assert failure.value.metadata["failed_elements"] == 1


def test_native_argument_order_and_cast_back_reference(gates):
    actual, reference = projection_pair()
    actual[0] = 0
    reference[0] = 0.010009765625
    gates.check_baseline_projection(256, "random", actual, reference)
    native_actual, native_reference, kwargs = gates.calls[-1]
    assert torch.equal(native_actual, actual.float())
    assert torch.equal(native_reference, reference.bfloat16().float())
    assert kwargs["catastrophic_check"] is True
    assert kwargs["atol"] == kwargs["rtol"] == 0.01
    assert kwargs["tol_err_ratio"] == 0.05
    # This position passes only with actual first / reference second.
    assert torch.isclose(native_actual, native_reference, atol=0.01, rtol=0.01).all()
    assert not torch.isclose(
        native_reference, native_actual, atol=0.01, rtol=0.01
    ).all()


@pytest.mark.parametrize("count, accepted", [(500, True), (501, False)])
def test_native_mismatch_ratio_boundary_with_low_aggregate_error(
    gates, count, accepted
):
    actual, reference = projection_pair()
    actual[:count] = 1.0234375
    assert gates.nrmse(actual, reference) < 0.01
    if accepted:
        gates.check_baseline_projection(256, "random", actual, reference)
    else:
        with pytest.raises(gates.GateFailure):
            gates.check_baseline_projection(256, "random", actual, reference)


def test_catastrophic_sparse_error_is_rejected_even_with_low_nrmse(gates):
    actual = torch.ones(100000, dtype=torch.bfloat16)
    reference = actual.float()
    actual[0] = 1.75
    assert gates.nrmse(actual, reference) < 0.01
    with pytest.raises(gates.GateFailure, match="documented correctness"):
        gates.check_baseline_projection(256, "random", actual, reference)


def test_aggregate_limit_remains_when_native_isclose_passes(gates):
    actual, reference = projection_pair()
    actual.fill_(1.015625)
    assert torch.isclose(actual.float(), reference, atol=0.01, rtol=0.01).all()
    with pytest.raises(gates.GateFailure) as failure:
        gates.check_baseline_projection(256, "random", actual, reference)
    assert "NRMSE" in failure.value.metadata["reason"]
    assert failure.value.metadata["original_all_element_02_02_diagnostic"]["passed"]
    assert (
        failure.value.metadata["policy"]["argument_order"]
        == "actual, rounded_reference"
    )


@pytest.mark.parametrize("side", ["actual", "reference"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_fail_closed(gates, side, value):
    actual, reference = projection_pair()
    (actual if side == "actual" else reference)[0] = value
    with pytest.raises(gates.GateFailure) as failure:
        gates.check_baseline_projection(256, "random", actual, reference)
    assert "Nonfinite" in failure.value.metadata["reason"]
    assert not gates.calls


@pytest.mark.parametrize("defect", ["broadcast_shape", "wrong_dtype"])
def test_projection_contract_cannot_be_broadcast_or_upcast(gates, defect):
    actual, reference = projection_pair()
    if defect == "broadcast_shape":
        actual, reference = actual.reshape(100, 100), reference[:100].reshape(1, 100)
    else:
        actual = actual.float()
    with pytest.raises(gates.GateFailure):
        gates.check_baseline_projection(256, "random", actual, reference)
    assert not gates.calls


@pytest.mark.parametrize("arm", ["separate", "fused"])
@pytest.mark.parametrize("norm_field", [1, 2])
def test_common_ideal_norm_gates_are_not_relaxed(gates, arm, norm_field):
    actual, reference = projection_pair()
    outputs, refs = whole_outputs(actual, reference)
    outputs[norm_field][0, 0] = 2
    with pytest.raises(gates.GateFailure) as failure:
        gates.check_outputs(256, arm, "random", outputs, refs)
    assert failure.value.metadata["field"] in ("q_norm", "kv_norm")
    assert failure.value.metadata["atol"] == failure.value.metadata["rtol"] == 0.02


def test_candidate_strict_once_rounded_bound_is_preserved(gates):
    actual, reference = projection_pair()
    actual.fill_(1.0078125)
    outputs, refs = whole_outputs(actual, reference)
    assert torch.isclose(actual.float(), reference, atol=0.02, rtol=0.02).all()
    with pytest.raises(gates.GateFailure) as failure:
        gates.check_outputs(256, "fused", "random", outputs, refs)
    assert (
        failure.value.metadata["field"] == "projection_vs_once_bf16_rounded_reference"
    )
    assert failure.value.metadata["strict_nrmse_limit"] == 5e-4


@pytest.mark.parametrize("factor", [0.75, 2.0])
def test_missing_split_or_wrong_scale_is_rejected(gates, factor):
    actual, reference = projection_pair()
    actual.mul_(factor)
    with pytest.raises(gates.GateFailure):
        gates.check_baseline_projection(256, "random", actual, reference)


def test_zero_reference_remains_well_defined(gates):
    actual = torch.zeros(10000, dtype=torch.bfloat16)
    assert gates.check_baseline_projection(256, "zero", actual, actual.float()) == 0
