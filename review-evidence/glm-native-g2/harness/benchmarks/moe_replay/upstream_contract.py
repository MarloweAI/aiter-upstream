"""CPU contracts for strict, fixed-route one-GPU MoE qualification."""

import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
TOLERANCES = {"rtol": 0.02, "atol": 0.02, "nrmse": 0.01}


class QualificationFailure(RuntimeError):
    """A preserved correctness failure must never enter timing admission."""


@dataclass(frozen=True)
class Case:
    slot: int
    step: int
    layer_index: int
    layer: int
    rows: int


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_cases(metadata, concurrency, *, layers=None, steps=None):
    """Keep actual full-M cases; never invent native routes for padded rows.

    Saved MTP captures may contain shrinking request batches. They do not hold
    valid caller routes for every unused row in the larger storage buffer.
    Those steps are explicitly excluded rather than replacing their work.
    """
    if concurrency not in CONCURRENCIES:
        raise ValueError("Unsupported requested concurrency")
    if metadata.get("forward_mode") != "target_verify":
        raise ValueError("A target-verification capture is required")
    rows = 4 * concurrency
    selected_layers = set(metadata["layers"] if layers is None else layers)
    if not selected_layers or not selected_layers <= set(metadata["layers"]):
        raise ValueError("Requested layer is absent from capture")
    selected_steps = None if steps is None else set(steps)
    cases, excluded = [], []
    for slot, step in enumerate(metadata["steps"]):
        step_id = step["decode_step"]
        if selected_steps is not None and step_id not in selected_steps:
            continue
        if step.get("draft_token_num") != 4:
            raise ValueError("Capture changed the four-token verification contract")
        actual, graph = step["actual_token_count"], step["graph_rows"]
        if not 0 <= actual <= graph <= rows:
            raise ValueError("Invalid actual/graph row metadata")
        if actual != rows or graph != rows:
            excluded.append(
                {
                    "slot": slot,
                    "step": step_id,
                    "actual_rows": actual,
                    "graph_rows": graph,
                    "reason": "not a full-M fixed-route case",
                }
            )
            continue
        for j, layer in enumerate(metadata["layers"]):
            if layer in selected_layers:
                cases.append(Case(slot, step_id, j, layer, rows))
    if not cases:
        raise ValueError("No full-M real captured cases are available")
    return cases, excluded


def require_gate(receipt):
    required = ("source", "routes", "eager", "graph", "state", "dispatch")
    if any(receipt.get(key) is not True for key in required):
        raise QualificationFailure("Correctness/identity gate failed before timing")
    return True


def qualify_then_measure(qualify, measure):
    receipt = qualify()
    require_gate(receipt)
    return receipt, measure()


def per_case_estimator(samples):
    if not samples or not samples[0]:
        raise ValueError("No complete timing samples")
    width = len(samples[0])
    if any(len(row) != width for row in samples):
        raise ValueError("Partial timing sample")
    medians = [statistics.median(row[j] for row in samples) for j in range(width)]
    return {
        "per_case_median_us": medians,
        "mean_per_case_median_us": statistics.mean(medians),
        "samples": len(samples),
        "cases": width,
    }


def marginal_summary(receipts, concurrencies=CONCURRENCIES):
    """Sum only same-cell paired absolute savings, retaining missing cells."""
    result = {}
    for receipt in receipts:
        group, c = receipt["group"], receipt["concurrency"]
        if c not in concurrencies:
            raise ValueError("Receipt contains an unexpected concurrency")
        row = result.setdefault(
            group,
            {
                "cells": {},
                "saved_us_sum": 0.0,
                "missing": [],
                "measured_concurrencies": [],
            },
        )
        if str(c) in row["cells"]:
            raise ValueError("Duplicate measurement cell")
        state = receipt["status"]
        cell = {"status": state}
        if state == "qualified":
            require_gate(receipt["gate"])
            old, new = receipt["baseline_us"], receipt["candidate_us"]
            saved = old - new
            cell.update(
                baseline_us=old,
                candidate_us=new,
                saved_us=saved,
                reduction_pct=100 * saved / old,
            )
            row["saved_us_sum"] += saved
            row["measured_concurrencies"].append(c)
        elif state == "not_applicable":
            if receipt.get("identical_dispatch") is not True:
                raise ValueError("A structural zero requires identical dispatch")
            cell["saved_us"] = 0.0
        row["cells"][str(c)] = cell
    for row in result.values():
        row["missing"] = [
            c
            for c in concurrencies
            if row["cells"].get(str(c), {}).get("status")
            not in ("qualified", "not_applicable")
        ]
        row["complete"] = not row["missing"]
    return result
