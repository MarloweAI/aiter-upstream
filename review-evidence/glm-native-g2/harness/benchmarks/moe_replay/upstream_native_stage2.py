"""Secondary G2 comparisons on identical, real native G1 intermediates.

Output reset is included for both isolated arms; sorting/G1 are excluded. Full
native fused_moe measurements alone determine adoption, never these stage deltas.
"""

import random
from pathlib import Path

from .upstream_contract import QualificationFailure, per_case_estimator, require_gate
from .upstream_profile import write_dump


def snapshot_stage2(run_with_override):
    import torch

    record = {}

    def clone(tensor):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        return tensor.view(torch.uint8).clone().view(tensor.dtype).reshape(tensor.shape)

    def observe(*, ordinary_stage2, stage2_args, stage2_kwargs):
        if record:
            raise QualificationFailure("Expected one native G2 call")
        record["ordinary"] = ordinary_stage2
        record["args"] = tuple(
            value if i in (1, 2) else clone(value)
            for i, value in enumerate(stage2_args)
        )
        record["kwargs"] = {
            key: value if key == "w2_scale" else clone(value)
            for key, value in stage2_kwargs.items()
        }
        return ordinary_stage2(*stage2_args, **stage2_kwargs)

    run_with_override(observe)
    torch.cuda.synchronize()
    if not record:
        raise QualificationFailure("Native G1 intermediate was not captured")
    return record


def measure_stage2_pair(
    *,
    fixtures,
    references,
    cases,
    callback,
    args,
    root,
    source_valid,
    parent_immutable,
    save,
):
    import torch
    import triton

    from .events import Event
    from .kernels import touch_input
    from .reference import compare

    root = Path(root)
    root.mkdir()
    outputs = {
        name: [torch.zeros_like(f["args"][6]) for f in fixtures]
        for name in ("baseline", "candidate")
    }
    snapshots = []
    for fixture in fixtures:
        values = [
            v
            for i, v in enumerate(fixture["args"][:6])
            if i not in (1, 2) and isinstance(v, torch.Tensor)
        ]
        values += [
            v
            for key, v in fixture["kwargs"].items()
            if key != "w2_scale" and isinstance(v, torch.Tensor)
        ]
        snapshots.append([(v, v.view(torch.uint8).clone()) for v in values])

    def immutable():
        return parent_immutable() and all(
            torch.equal(value.view(torch.uint8), saved)
            for pairs in snapshots
            for value, saved in pairs
        )

    def run(name, index):
        fixture = fixtures[index]
        out = outputs[name][index]
        out.zero_()
        stage_args = (*fixture["args"][:6], out, *fixture["args"][7:])
        if name == "baseline":
            fixture["ordinary"](*stage_args, **fixture["kwargs"])
        else:
            callback(
                ordinary_stage2=fixture["ordinary"],
                stage2_args=stage_args,
                stage2_kwargs=fixture["kwargs"],
                expert_mask=None,
            )
        return out

    qualification = {"eager": {}, "graph": {}, "state": {}}
    for name in outputs:
        checks = []
        for index, case in enumerate(cases):
            actual = run(name, index).clone()
            check = {
                "layer": case.layer,
                "step": case.step,
                **compare(actual, references[index]),
            }
            checks.append(check)
            if not check["passed"]:
                torch.save(
                    {"actual": actual.cpu(), "expected": references[index].cpu()},
                    root / f"failure-{name}-l{case.layer}-s{case.step}.pt",
                )
        qualification["eager"][name] = checks
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = [run(name, index).clone() for index in range(len(fixtures))]
        checks = []
        for replay in range(3):
            graph.replay()
            torch.cuda.synchronize()
            checks.extend(
                {"case": i, "replay": replay, **compare(actual, expected)}
                for i, (actual, expected) in enumerate(zip(graph_outputs, references))
            )
        qualification["graph"][name] = checks
        state = []
        # Alternate fixtures exercise route/output reuse; zero packed middle,
        # restore exact bytes, then check the original reference again.
        for index, fixture in enumerate(fixtures):
            run(name, (index + 1) % len(fixtures))
            middle = fixture["args"][0].view(torch.uint8)
            before = middle.clone()
            middle.zero_()
            zero = compare(run(name, index), torch.zeros_like(outputs[name][index]))
            middle.copy_(before)
            restored = compare(run(name, index), references[index])
            state.append(
                {
                    "case": index,
                    "zero": zero,
                    "restored": restored,
                    "passed": zero["passed"] and restored["passed"],
                }
            )
        qualification["state"][name] = state
    gate = {
        "source": source_valid(),
        "routes": immutable(),
        "dispatch": True,
        "eager": all(
            v["passed"] for rows in qualification["eager"].values() for v in rows
        ),
        "graph": all(
            v["passed"] for rows in qualification["graph"].values() for v in rows
        ),
        "state": all(
            v["passed"] for rows in qualification["state"].values() for v in rows
        ),
    }
    qualification["gate"] = gate
    save(root / "qualification.json", qualification)
    require_gate(gate)
    graphs = {}
    max_words = max(triton.cdiv(f["args"][0].numel(), 1024) for f in fixtures)
    sink = torch.empty((max_words,), device="cuda")
    for name in outputs:
        markers = [(Event(), Event()) for _ in fixtures]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for index, (start, end) in enumerate(markers):
                middle = fixtures[index]["args"][0].view(torch.uint8)
                touch_input[(triton.cdiv(middle.numel(), 1024),)](
                    middle, sink, middle.numel(), 1024
                )
                start.record()
                run(name, index)
                end.record()
        graphs[name] = (graph, markers)
    for _ in range(args.warmups):
        for graph, _ in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in graphs}
    rng = random.Random(args.seed)
    for _ in range(args.samples):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            graph, markers = graphs[name]
            graph.replay()
            torch.cuda.synchronize()
            samples[name].append([start.elapsed_us(end) for start, end in markers])
    timing = {name: per_case_estimator(values) for name, values in samples.items()}
    save(
        root / "clean.json",
        {
            "boundary": "G2 projection/combine plus explicit output reset; excludes native sorting/G1",
            "timing": timing,
            "raw_samples_us": samples,
        },
    )
    for name in outputs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(name, 0)
        graph.replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            graph.replay()
            torch.cuda.synchronize()
        prefix = root / f"profile-{name}-l{cases[0].layer}-s{cases[0].step}"
        trace = str(prefix) + ".trace.json"
        prof.export_chrome_trace(trace)
        write_dump(trace, prefix, clean_us=timing[name]["per_case_median_us"][0])
    if not immutable() or not source_valid():
        raise QualificationFailure(
            "G2 snapshot/provenance changed after timing/profile"
        )
    save(
        root / "receipt.json",
        {
            "status": "qualified",
            "gate": gate,
            "cases": len(cases),
            "same_g1_intermediates": True,
            "baseline_us": timing["baseline"]["mean_per_case_median_us"],
            "candidate_us": timing["candidate"]["mean_per_case_median_us"],
            "secondary_only": True,
        },
    )
    return timing
