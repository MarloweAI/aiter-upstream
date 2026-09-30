# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Marlowe AI. All rights reserved.
"""CPU guard and graph-stable callback tests; no ROCm runtime is required."""

import importlib.util
import ast
import inspect
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_MODULE = (
    Path(__file__).resolve().parents[1]
    / "aiter/ops/flydsl/kernels/mxmoe_sparse_epilogue.py"
)
_SPEC = importlib.util.spec_from_file_location("sparse_epilogue_policy", _MODULE)
policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(policy)

SUPPORTED = dict(
    gfx="gfx950",
    M=32,
    hidden=6144,
    intermediate=256,
    experts=257,
    topk=9,
    BM=16,
    BN=128,
    BK=128,
    SBM=16,
    a_dtype="fp4",
    b_dtype="fp4",
    epilog="atomic",
    out_dtype="bf16",
    bf16_lds=True,
    kstatic=True,
    persist=False,
    bias=False,
    is_ep=False,
)


@pytest.mark.parametrize("M", [32, 64, 128])
def test_supported_actual_graph_rows(M):
    assert policy.g2_sparse_epilogue_supported(**dict(SUPPORTED, M=M))


@pytest.mark.parametrize(
    "difference",
    [
        dict(gfx="gfx942"),
        dict(gfx="gfx950:sramecc+"),
        dict(M=16),
        dict(M=31),
        dict(M=256),
        dict(hidden=4096),
        dict(intermediate=512),
        dict(experts=256),
        dict(topk=1),  # atomic cache-normalized topk is not the real domain
        dict(topk=8),
        dict(BM=32),
        dict(BN=256),
        dict(BK=256),
        dict(SBM=32),
        dict(a_dtype="fp8"),
        dict(b_dtype="fp8"),
        dict(epilog="reduce"),
        dict(out_dtype="fp8"),
        dict(bf16_lds=False),
        dict(kstatic=False),
        dict(persist=True),
        dict(bias=True),
        dict(is_ep=True),
    ],
)
def test_unsupported_domain_preserves_original_dispatch(difference):
    assert not policy.g2_sparse_epilogue_supported(**dict(SUPPORTED, **difference))


def test_frozen_callback_preserves_arguments_output_and_mutable_inputs():
    calls = []
    output = object()

    def _mxfp4_a4w4_stage2_fw(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    ordinary = partial(_mxfp4_a4w4_stage2_fw, kernelName2="native-layout-v2")
    kwargs = dict(sorted_weights=object(), block_m=16)
    before = dict(kwargs)
    args = (object(), object())
    baseline = policy.make_g2_sparse_epilogue_override(False)
    candidate = policy.make_g2_sparse_epilogue_override(True)
    for arm in (candidate, baseline, candidate):
        assert (
            arm(
                ordinary_stage2=ordinary,
                stage2_args=args,
                stage2_kwargs=kwargs,
                expert_mask=None,
            )
            is output
        )
    assert kwargs == before
    assert [call[1].get("g2_skip_padded_lds", False) for call in calls] == [
        True,
        False,
        True,
    ]
    assert all(call[0] == args for call in calls)
    assert all(call[1]["sorted_weights"] is kwargs["sorted_weights"] for call in calls)


def test_other_stage_two_family_receives_no_new_keyword():
    def original_stage_two(x, *, sorted_weights):
        return x, sorted_weights

    x, weights = object(), object()
    result = policy.make_g2_sparse_epilogue_override()(
        ordinary_stage2=original_stage_two,
        stage2_args=(x,),
        stage2_kwargs=dict(sorted_weights=weights),
        expert_mask=None,
    )
    assert result == (x, weights)


def test_wave_vote_retains_valid_rows_for_every_bm16_hole_pattern():
    # Independent epilogue ownership: 256 threads, 32 columns/row, 64 lanes/wave.
    # Each of the two MR groups owns eight rows; every wave covers two rows.
    # Model memory reads, rather than numerical math which this patch cannot alter.
    for pattern in range(1 << 16):
        valid = [bool(pattern & (1 << row)) for row in range(16)]
        loaded_rows = set()
        for mr in range(2):
            for wave in range(4):
                rows = [mr * 8 + wave * 2, mr * 8 + wave * 2 + 1]
                if any(valid[row] for row in rows):
                    loaded_rows.update(rows)
        assert all(row in loaded_rows for row in range(16) if valid[row])
        assert len(loaded_rows) == sum(
            2 for row in range(0, 16, 2) if valid[row] or valid[row + 1]
        )


@pytest.mark.parametrize("valid_row", [0, 1, 7, 8, 14, 15])
def test_single_valid_row_does_not_assume_prefix_or_one_lane(valid_row):
    loaded_rows = {
        valid_row // 2 * 2,
        valid_row // 2 * 2 + 1,
    }
    assert valid_row in loaded_rows
    assert len(loaded_rows) == 2


@pytest.mark.parametrize("token_ids", [(31, 32), (32, 31), (32, 32), (0, 0)])
def test_token_boundary_and_duplicate_routes(token_ids):
    loaded = any(token < 32 for token in token_ids)
    stores = [token for token in token_ids if token < 32]
    assert loaded == bool(stores)
    if token_ids == (0, 0):
        assert stores == [0, 0]  # repeated expert/routes are never deduplicated


def test_real_outer_ep_mask_disables_even_mxmoe_front_wrapper():
    def _mxfp4_a4w4_stage2_fw(x, *, sorted_weights):
        return x, sorted_weights

    x, weights = object(), object()
    override = policy.make_g2_sparse_epilogue_override()
    assert override._uses_sparse_g2_epilogue
    assert override(
        ordinary_stage2=_mxfp4_a4w4_stage2_fw,
        stage2_args=(x,),
        stage2_kwargs=dict(sorted_weights=weights),
        expert_mask=object(),
    ) == (x, weights)


def _public_api_with_stub_backends():
    # Execute the actual public function body without importing Torch/ROCm.
    # This catches missing public keywords and the schema-wrapped forwarding
    # mistake that lower-helper tests cannot detect.
    source = Path(__file__).resolve().parents[1] / "aiter/fused_moe.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "fused_moe"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, function], type_ignores=[])
    )
    calls = []
    result = object()

    def backend(name):
        def record(**kwargs):
            calls.append((name, kwargs))
            return result

        return record

    namespace = dict(
        ActivationType=SimpleNamespace(Silu=SimpleNamespace(value=1)),
        QuantType=SimpleNamespace(No=SimpleNamespace(value=0)),
        GateMode=SimpleNamespace(SEPARATED=SimpleNamespace(value="separated")),
        fused_moe_=backend("custom_op"),
        _fused_moe_impl=backend("native_impl"),
    )
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["fused_moe"], calls, result


def test_public_default_preserves_custom_op_path_and_schema():
    public, calls, result = _public_api_with_stub_backends()
    parameter = inspect.signature(public).parameters["_stage2_override"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    assert public(*[object() for _ in range(5)]) is result
    assert calls[0][0] == "custom_op"
    assert "_stage2_override" not in calls[0][1]
    assert calls[0][1]["block_size_M"] == -1
    assert calls[0][1]["ep_world_size"] == 0


def test_public_override_reaches_native_impl_with_actual_ep_and_output():
    public, calls, result = _public_api_with_stub_backends()
    args = [object() for _ in range(5)]
    callback, mask, scatter, output = object(), object(), object(), object()
    assert (
        public(
            *args,
            expert_mask=mask,
            stage2_scatter=scatter,
            output=output,
            block_size_M=16,
            _stage2_override=callback,
        )
        is result
    )
    assert calls[0][0] == "native_impl"
    forwarded = calls[0][1]
    assert forwarded["_stage2_override"] is callback
    assert forwarded["expert_mask"] is mask
    assert forwarded["stage2_scatter"] is scatter
    assert forwarded["output"] is output
    assert forwarded["topk_ids"] is args[4]
    assert forwarded["topk_weight"] is args[3]
    assert forwarded["block_size_M"] == 16
    assert forwarded["activation"] == 1
    assert forwarded["quant_type"] == 0
    assert "ep_world_size" not in forwarded


def test_public_shared_specialization_retains_original_fallback(monkeypatch):
    public, calls, _ = _public_api_with_stub_backends()
    shared_calls = []
    shared_result = object()
    shared_module = ModuleType("aiter.fhmoe")
    shared_module._fhmoe = lambda **kwargs: (
        shared_calls.append(kwargs),
        shared_result,
    )[1]
    monkeypatch.setitem(__import__("sys").modules, "aiter.fhmoe", shared_module)
    shared_weight = object()
    assert (
        public(
            *[object() for _ in range(5)],
            shared_w1=shared_weight,
            _stage2_override=object(),
        )
        is shared_result
    )
    assert not calls
    assert shared_calls[0]["shared_w1"] is shared_weight
    assert "_stage2_override" not in shared_calls[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
