# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from aiter import tuned_gemm as tuned

BF16 = torch.bfloat16
N = K = 2048


def _key(m, bias=False, dtype=BF16, otype=BF16, scale=False, shuffled=False):
    return ("gfx950", 256, m, N, K, bias, str(dtype), str(otype), scale, shuffled)


@pytest.fixture(params=[(64, "small_m"), (128, "xcd_reuse")])
def variant(request, monkeypatch):
    m, suffix = request.param
    name = "gemm_a16w16_" + suffix
    module = importlib.import_module("aiter.ops.triton.gemm.basic." + name)
    monkeypatch.setattr(module, "_gluon_arch", lambda: "gfx950")
    monkeypatch.setattr(module, "_padded_layout_available", lambda: True)
    row = {"libtype": "triton", "kernelName": name, "solidx": 0}
    rows = {_key(m): row}
    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config_", lambda: rows)
    monkeypatch.setattr(tuned, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(tuned, "get_cu_num", lambda: 256)
    monkeypatch.setattr(
        tuned,
        "get_padded_m",
        lambda m, n, k, gl: (
            (m + 15) // 16 * 16 if gl == 0 else 1 << (m - 1).bit_length()
        ),
    )
    seen = []

    def ordinary(x, w, bias=None, dtype=None):
        seen.append(("ordinary", bias, dtype))
        return F.linear(x, w, bias).to(dtype or x.dtype)

    # Instrument only the unchanged ordinary provider; the real selector is tested.
    generic = types.ModuleType("aiter.ops.triton.gemm.basic.gemm_a16w16")
    generic.gemm_a16w16 = ordinary
    monkeypatch.setitem(sys.modules, generic.__name__, generic)
    tuned.get_GEMM_A16W16_config.cache_clear()
    yield m, name, module, rows, seen
    tuned.get_GEMM_A16W16_config.cache_clear()


def _config(m, **kw):
    return tuned.get_GEMM_A16W16_config(
        m,
        N,
        K,
        kw.get("bias", False),
        str(kw.get("dtype", BF16)),
        str(kw.get("otype", BF16)),
        kw.get("scale", False),
        kw.get("shuffled", False),
    )


def test_exact_row_and_padded_near_match(variant):
    m, name, _, rows, _ = variant
    assert _config(m)["kernelName"] == name
    for actual in (16, 32, m - 1, 100 if m == 128 else m + 1):
        assert _config(actual).get("kernelName") != name
    # Rejected actual-M row must continue to an explicitly valid next candidate.
    rows[_key(256)] = {"libtype": "torch", "kernelName": "next", "solidx": 0}
    rows[_key(129)] = rows[_key(m)]
    tuned.get_GEMM_A16W16_config.cache_clear()
    assert _config(129)["kernelName"] == "next"


@pytest.mark.parametrize(
    "change", ["bias", "dtype", "otype", "scale", "shuffled", "capability"]
)
def test_incompatible_named_row_is_ignored(variant, change, monkeypatch):
    m, name, module, rows, _ = variant
    kw = {change: True}
    if change in ("dtype", "otype"):
        kw[change] = torch.float16
    if change == "capability":
        kw = {}
        monkeypatch.setattr(module, "_padded_layout_available", lambda: False)
    rows[_key(m, **kw)] = {"libtype": "triton", "kernelName": name, "solidx": 0}
    assert _config(m, **kw).get("kernelName") != name


def test_named_public_dispatch_and_call_local_fallback(variant, monkeypatch):
    m, name, module, _, seen = variant
    x, w = torch.ones((m, K), dtype=BF16), torch.ones((N, K), dtype=BF16)
    monkeypatch.setattr(module, name + "_accepts", lambda x, w: True)

    def selected(x, w):
        seen.append((name, None, BF16))
        return F.linear(x, w)

    monkeypatch.setattr(module, name, selected)
    out = tuned.tgemm.mm(x.view(2, m // 2, K), w)
    assert out.shape == (2, m // 2, N) and out.dtype == BF16
    assert seen[-1][0] == name
    monkeypatch.setattr(module, name + "_accepts", lambda x, w: False)
    tuned.tgemm.mm(x, w)
    assert seen[-1] == ("ordinary", None, BF16)
    assert _config(m)["kernelName"] == name  # rejection is not a disabled row


def test_selected_kernel_failure_propagates(variant, monkeypatch):
    m, name, module, _, seen = variant
    monkeypatch.setattr(module, name + "_accepts", lambda x, w: True)

    def failed(x, w):
        raise RuntimeError("selected kernel compile failure")

    monkeypatch.setattr(module, name, failed)
    with pytest.raises(RuntimeError, match="compile failure"):
        tuned.tgemm.mm(torch.ones((m, K), dtype=BF16), torch.ones((N, K), dtype=BF16))
    assert not seen and _config(m)["kernelName"] == name


def test_ordinary_triton_rows_and_optional_arguments_are_unchanged(variant):
    m, name, _, _, seen = variant
    x, w = torch.zeros((m, K), dtype=BF16), torch.zeros((N, K), dtype=BF16)
    bias = torch.ones(N, dtype=BF16)
    for config in (None, {}, {"kernelName": "auto"}, {"kernelName": name}):
        out = tuned.triton_gemm(x, w, 0, bias=bias, otype=torch.float32, config=config)
        assert out.dtype == torch.float32 and out.eq(1).all()
        assert seen[-1] == ("ordinary", bias, torch.float32)
    with pytest.raises(AssertionError, match="scaling"):
        tuned.triton_gemm(x, w, 0, scale_c=torch.ones(1), config={"kernelName": name})
    with pytest.raises(AssertionError, match="bpreshuffle"):
        tuned.triton_gemm(x, w, 0, bpreshuffle=True, config={"kernelName": name})


def test_existing_packed_format_keeps_its_route(variant, monkeypatch):
    m, _, _, rows, _ = variant
    rows[_key(m, shuffled=True)] = {
        "libtype": "asm",
        "kernelName": "existing",
        "solidx": 0,
    }
    seen = []
    monkeypatch.setitem(
        tuned.solMap,
        "asm",
        lambda *args, **kwargs: seen.append(args[8]) or F.linear(args[0], args[1]),
    )
    w = torch.zeros((N, K), dtype=BF16)
    w.is_shuffled = True
    tuned.tgemm.mm(torch.zeros((m, K), dtype=BF16), w)
    assert seen == [True]
