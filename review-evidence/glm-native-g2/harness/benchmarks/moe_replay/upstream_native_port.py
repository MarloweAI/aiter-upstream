"""Versioned native-port identities and immutable per-arm stage2 selection.

The callback is supplied by a reviewed, committed AITER patch. Baseline never
receives it. No global AITER dispatch or environment switch changes between arms.
"""

import hashlib
import importlib
import json
import subprocess
from pathlib import Path

from .upstream_contract import CONCURRENCIES, QualificationFailure, file_digest


def read_manifest(path):
    value = json.loads(Path(path).read_text())
    required = (
        "schema",
        "unit",
        "aiter_base_sha",
        "aiter_feature_sha",
        "patch_sha256",
        "source_files",
        "stage2_factory",
        "stage2_kwargs",
        "concurrencies",
        "fallback",
        "expected_kernel_marker",
        "unchanged_source_files",
        "harness_files",
    )
    if any(key not in value for key in required) or value["schema"] != 1:
        raise ValueError("Incomplete native-port manifest")
    for key in ("aiter_base_sha", "aiter_feature_sha"):
        if len(value[key]) != 40 or any(
            c not in "0123456789abcdef" for c in value[key]
        ):
            raise ValueError("Source pins must be full commit SHAs")
    if not value["source_files"] or not value["expected_kernel_marker"]:
        raise ValueError("Source/JIT identity cannot be empty")
    if not value["concurrencies"] or any(
        c not in CONCURRENCIES for c in value["concurrencies"]
    ):
        raise ValueError("Invalid native-port domain")
    if len(set(value["concurrencies"])) != len(value["concurrencies"]):
        raise ValueError("Duplicate native-port shape")
    if set(map(int, value["fallback"])) != set(CONCURRENCIES) - set(
        value["concurrencies"]
    ):
        raise ValueError("Every ineligible shape needs explicit fallback evidence")
    mandatory = {
        f"benchmarks/moe_replay/{name}"
        for name in (
            "upstream_campaign.py",
            "upstream_native_port.py",
            "upstream_native_stage2.py",
            "reference.py",
            "upstream_reference.py",
        )
    }
    if not mandatory <= set(value["harness_files"]):
        raise ValueError("Native runner, helpers and oracle must be frozen")
    for name in value["source_files"]:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Source paths must stay inside AITER")
    if not value["stage2_factory"].startswith("aiter."):
        raise ValueError("Native callback must come from pinned AITER")
    return value


def validate_source(manifest, root):
    root = Path(root).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    if git("rev-parse", "HEAD").decode().strip() != manifest["aiter_feature_sha"]:
        raise QualificationFailure("Native feature revision mismatch")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise QualificationFailure("Native AITER source has tracked edits")
    patch = git(
        "diff", "--binary", manifest["aiter_base_sha"], manifest["aiter_feature_sha"]
    )
    if hashlib.sha256(patch).hexdigest() != manifest["patch_sha256"]:
        raise QualificationFailure("Native patch identity mismatch")
    for name, expected in manifest["source_files"].items():
        if file_digest(root / name) != expected:
            raise QualificationFailure(f"Native source mismatch: {name}")
    for name in manifest["unchanged_source_files"]:
        pristine = git("show", f"{manifest['aiter_base_sha']}:{name}")
        if (root / name).read_bytes() != pristine:
            raise QualificationFailure(f"Pristine native helper changed: {name}")
    return True


def make_callback(manifest, root):
    module_name, name = manifest["stage2_factory"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(Path(root).resolve()):
        raise QualificationFailure("Native override imported outside pinned source")
    relative = str(source.relative_to(Path(root).resolve()))
    if relative not in manifest["source_files"]:
        raise QualificationFailure("Native override source is absent from manifest")
    callback = getattr(module, name)(**manifest["stage2_kwargs"])
    if not callable(callback):
        raise QualificationFailure("Native factory did not return a callable")
    return callback


def coverage(manifest):
    rows = []
    for c in CONCURRENCIES:
        if c in manifest["concurrencies"]:
            continue
        reason = manifest["fallback"][str(c)]
        failed_native = c == 64
        rows.append(
            {
                "group": manifest["unit"],
                "concurrency": c,
                "status": "historical_native_gate_failed"
                if failed_native
                else "not_applicable",
                "identical_dispatch": not failed_native,
                "reason": reason,
                "timing": "not rerun; native dispatch unchanged",
            }
        )
    return rows


def check_engagement(manifest, profiles):
    """Actual full-operator kernels, not canonical metadata, establish execution."""
    marker = manifest["expected_kernel_marker"]
    baseline = [k["name"] for k in profiles["baseline"]["kernels"]]
    candidate = [k["name"] for k in profiles["candidate"]["kernels"]]
    if not baseline or not candidate:
        raise QualificationFailure("No visible GPU kernels for dispatch proof")
    if any(marker in name for name in baseline) or not any(
        marker in name for name in candidate
    ):
        raise QualificationFailure(
            "Native candidate did not engage its distinct kernel"
        )
    return {
        "passed": True,
        "marker": marker,
        "baseline_kernels": baseline,
        "candidate_kernels": candidate,
    }


def validate_harness(manifest, root):
    root = Path(root).resolve()
    for name, expected in manifest["harness_files"].items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or file_digest(root / path) != expected
        ):
            raise QualificationFailure(f"Native harness source mismatch: {name}")
    return True
