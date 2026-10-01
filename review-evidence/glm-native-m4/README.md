# GLM native exact-M4 split-six evidence

This opt-in prepared native FlyDSL pipeline preserves native MXFP4 input and FP32-middle quantization, changes GEMM1's reduction tree, and consumes supplied FP32 routes with packed BF16 output atomics. Measurements cover one MI355X TP8/EP1 local MoE operation, including the shared expert once. Router, collectives and attention are outside the boundary; serving ITL and full-model throughput have not been measured for this candidate.

| Mean of 480 per-case clean medians | C1 / M4 |
| --- | ---: |
| Official AITER `569d981` native control | 24.3607968655 µs |
| Candidate `e3f517eb` split-six pipeline | 17.7902500300 µs |
| Saved | 6.5705468354 µs / 26.9718058556% |

Each case uses 40 randomized paired samples and 10 warmups. The bank contains 30 captured layers × 16 verification frames. All 480 independent references, 480 eager checks per arm, 1,440 graph checks per arm and 90 state checks per arm passed unchanged policy: finite output, NRMSE ≤1%, and zero elementwise violations with rtol/atol 0.02. Native source, weights, inputs and routes were checked as immutable, and both candidate kernel markers were observed.

[qualification-rollup.json](qualification-rollup.json) indexes the qualification. [full/](full/) retains exact raw samples, every checker result, dispatch/capture hashes and separate trace/JSON/TXT profile dumps. Profiles do not supply the table timing. The clean value in each named profile refers to that individual case's median, despite the inherited formatter field saying `mean_per_case_median_us`.

Frozen source: `e3f517ebbb4005b4d63195a646c089b882f96f97`; pristine official base: `569d981db738682668ac97db457f5423382f9bd2`. Only new files were added; existing native operator, emitter and config bytes were checked against that base. The sorting helper at SGLang `a0fcebb3bb7043e62b6ca8885b9e723f2262e091` is byte-identical to the earlier verified helper. The frozen production module has **544 lines**. The immutable measured manifest's **538-line** value is an earlier estimate; this correction preserves all hash-bound measurement receipts. The failed `612` preparation attempt and successful four-case pilot are retained separately.

Precision: BF16 hidden/output, MXFP4 expert weights with native E8M0 scales, FP32 intermediate with source-derived `0x7fffff` roundup, and nine supplied FP32 route weights including shared expert 256 once. Captures came from a real native TP8 forward with FP8 KV and varied deterministic synthetic 8,192-token prompts. Capture source remains AITER `c8325e00` / SGLang `46beacf`; replay control is newly qualified `569d981`. This is four-token verification (`M=4C`).

## Reproduce

Use one owned MI355X allocation. Eight-GPU serving and a new Engine are not required to replay this bank. Clone the source feature at the frozen pin. Extract the frozen Python harness/package closure beside this README:

```sh
tar -xzf frozen-harness.tar.gz
```

The code-only archive preserves measured Python bytes without source formatters rewriting them. It contains the existing benchmark/package closure; its original runtime has PyHIP installed, but the new native AITER specialization does not import PyHIP or add a dependency. [sha256-manifest.json](sha256-manifest.json) hashes the archive, extracted sources, exact receipts and this README. Verify after extraction:

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
for name, expected in json.loads(Path('sha256-manifest.json').read_text()).items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual == expected, name
print('All evidence and frozen source hashes match')
PY
```

Use exact runtime/dependency identities in [runtime.json](runtime.json) and [measured-manifest.json](measured-manifest.json). Bulk captured tensors/weights are external and excluded here. Their locations and SHA hashes are in `full/identity.json` and `full/c1-capture.json`; copy or link a readable bank preserving its 30 layer files and 16 frames.

[recipe/](recipe/) contains the actual environment/cache paths and full commands. Record any task-path adjustments in a separate recipe. The measured binary patch hash used clone-local `core.abbrev=8`. Keep HOME unchanged and create writable task-specific caches before imports. Recorded runtime: FlyDSL 0.3.4.1, PyHIP 270225149, Torch 2.11 / HIP 7.2 and the immutable image identity.

The frozen `benchmarks/moe_replay/upstream_replay_launch.sh` consumes `MOE_PORT_*` variables and the measured manifest. Full settings are `--comparison-mode native-port --concurrencies 1 --samples 40 --warmups 10 --profile-cases 1`. It stops before timing if correctness, state, source or engagement gates fail. Every profile receives JSON and TXT output. Success requires the completion receipt and all gates, rather than only the shell exit status.

Job 86952 on `marlowe-mi355x-1` reused one GPU, 16 CPUs and 128 GiB. C1 finished; the same allocation continued for the separately scoped M8 lane. The campaign records cleanup after both lanes. This evidence branch contains no tensor or weight blobs and does not authorize an external PR.
