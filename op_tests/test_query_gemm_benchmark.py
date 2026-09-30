# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import csv

import pytest
import torch

from op_tests.op_benchmarks.triton import bench_gemm_a16w16_query as bench


def test_zero_reference_metrics():
    zero = torch.zeros(2, 2)
    assert bench.errors(zero, zero)["nrmse"] == 0
    assert bench.errors(torch.ones_like(zero), zero)["nrmse"] is None
    assert bench.errors(torch.full_like(zero, float("nan")), zero)["rms_error"] is None


def test_bookends_pair_with_their_own_round():
    runs = [{"round": 0, "median_us": 7}, {"round": 1, "median_us": 17}]
    bookends = [
        {"round": 0, "median_us": 8},
        {"round": 0, "median_us": 12},
        {"round": 1, "median_us": 18},
        {"round": 1, "median_us": 22},
    ]
    result = bench.aggregate(runs, bookends)
    assert result["median_us"] == 12
    assert result["run_median_range_us"] == [7, 17]
    assert [p["incumbent_us"] for p in result["paired_gains"]] == [10, 20]
    assert result["paired_gain_median_us"] == 3
    assert [p["gain_percent"] for p in result["paired_gains"]] == [30, 15]


def test_csv_preserves_native_selection_fields(tmp_path):
    path = tmp_path / "selected.csv"
    row = {
        "gfx": "gfx950",
        "cu_num": "256",
        "M": "64",
        "N": "2048",
        "K": "2048",
        "bias": "False",
        "dtype": "torch.bfloat16",
        "outdtype": "torch.bfloat16",
        "scaleAB": "False",
        "bpreshuffle": "False",
        "libtype": "flydsl",
        "solidx": "17.0",
        "splitK": "4.0",
        "kernelName": "native-name",
    }
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        writer.writerow({**row, "kernelName": "second-native-name"})
    key, selected = next(iter(bench.read_rows([path]).items()))
    assert key == (
        "gfx950",
        256,
        64,
        2048,
        2048,
        False,
        "torch.bfloat16",
        "torch.bfloat16",
        False,
        False,
    )
    assert selected["solidx"] == 17 and selected["splitK"] == 4
    assert selected["kernelName"] == "second-native-name"
    assert [item[1]["kernelName"] for item in bench.read_catalogue([path])] == [
        "native-name",
        "second-native-name",
    ]


def test_profile_mode_requires_clean_binding():
    args = [
        "--incumbent-csv",
        "stock.csv",
        "--output",
        "out",
        "--code-commit",
        "source",
    ]
    with pytest.raises(SystemExit):
        bench.parse_args(args + ["--profile-only"])
    with pytest.raises(SystemExit):
        bench.parse_args(args + ["--profile-only", "--clean-root", "clean", "--screen"])
    parsed = bench.parse_args(args + ["--profile-only", "--clean-root", "clean"])
    assert parsed.profile_only and parsed.clean_root.name == "clean"
