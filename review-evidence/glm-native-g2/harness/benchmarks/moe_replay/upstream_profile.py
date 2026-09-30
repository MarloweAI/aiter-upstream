"""Readable dumps from a separate replay trace; never retime clean receipts."""

import argparse
import json
from pathlib import Path


def dump_trace(trace, *, clean_us=None):
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    kernels = []
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "").lower()
        if "kernel" not in category:
            continue
        args = event.get("args", {})
        kernels.append(
            {
                "name": event["name"],
                "start_us": event["ts"],
                "duration_us": event["dur"],
                "stream": args.get("stream", event.get("tid")),
                "device": args.get("device", event.get("pid")),
            }
        )
    kernels.sort(key=lambda row: (row["start_us"], str(row["stream"])))
    origin = min((k["start_us"] for k in kernels), default=0)
    end = max((k["start_us"] + k["duration_us"] for k in kernels), default=origin)
    for order, kernel in enumerate(kernels):
        kernel.update(order=order, relative_start_us=kernel["start_us"] - origin)
    span = end - origin if kernels else None
    kernel_sum = sum(k["duration_us"] for k in kernels) if kernels else None
    result = {
        "schema": 1,
        "clean_mean_per_case_median_us": clean_us,
        "profiled_device_span_us": span,
        "profiled_kernel_sum_us": kernel_sum,
        "profile_minus_clean_us": None
        if span is None or clean_us is None
        else span - clean_us,
        "visible_kernels": len(kernels),
        "usable_kernel_breakdown": bool(kernels),
        "warning": None
        if kernels
        else "Trace has no visible GPU kernel events; do not infer a breakdown.",
        "kernels": kernels,
    }
    lines = [
        "Separate MoE replay profile (profiled durations are not clean latency)",
        f"Clean mean of per-case medians: {clean_us} us",
        f"Profiled device span: {span} us",
        f"Profiled kernel sum: {kernel_sum} us",
        f"Profile minus clean: {result['profile_minus_clean_us']} us",
        f"Visible kernels: {len(kernels)}",
        "",
        "Order Stream Start(us) Duration(us) Kernel",
    ]
    lines.extend(
        f"{k['order']:4} {k['stream']!s:>6} {k['relative_start_us']:12.3f} "
        f"{k['duration_us']:12.3f} {k['name']}"
        for k in kernels
    )
    if result["warning"]:
        lines.append(result["warning"])
    return result, "\n".join(lines) + "\n"


def write_dump(trace_path, output_prefix, *, clean_us=None):
    result, text = dump_trace(
        json.loads(Path(trace_path).read_text()), clean_us=clean_us
    )
    prefix = str(output_prefix)
    Path(prefix + ".json").write_text(json.dumps(result, indent=2) + "\n")
    Path(prefix + ".txt").write_text(text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-us", type=float)
    args = parser.parse_args()
    write_dump(args.trace, args.output, clean_us=args.clean_us)


if __name__ == "__main__":
    main()
