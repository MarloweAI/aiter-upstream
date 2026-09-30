# Internal review evidence: native GLM G2 padded-row guard

Selected source: AITER `296c72c1ed8b317cd00407f082ca6f743efe8978` on official base `c8325e00c88f1b97af75150a311f6704c2729f9c`. This branch archives small review evidence separately so the source PR stays kernel/selection/tests only. No external PR is authorized or opened.

The single selected True mode is used for all qualified actual M32/M64/M128 (C8/C16/C32 with M=4C); False/default remains native elsewhere. It preserves native BF16 hidden/output, MXFP4/E8M0 expert weights/activations, FP32-middle `0x7fffff` quantization and supplied FP32 route weights. Eight routed experts plus shared expert 256 exactly once are the input routes. This local boundary includes native sorting, G1, G2 and output reset/combine; it excludes router and TP collectives. FP8 KV is capture provenance. It is neither a full-serving latency nor an attention-stack measurement.

## Clean native operator results

Each cell is the mean of 480 per-case medians from 40 randomized paired unprofiled HIP graph samples: 30 real layer shards × 16 target-verification frames on one MI355X GPU. These are independent-reference qualified before timing.

| C | Native baseline, µs | Selected candidate, µs | Saved, µs | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 8 | 52.912018 | 52.552251 | 0.359767 | 0.680% |
| 16 | 74.003024 | 73.579734 | 0.423290 | 0.572% |
| 32 | 85.035551 | 84.701676 | 0.333875 | 0.393% |

Qualified signed saving sum: **1.116931 µs** over these three requested Cs. C1/C2/C4 have source-proven unchanged generated dispatch; C64 is outside this BM16 specialization and retains its historical native accuracy failure. C64 is missing, not zero measured saving. This sum is not a whole-model or additive multi-PR speedup.

The [immutable summary](summary.json), [identity/runtime receipt](identity.json) and [source/harness manifest](source-manifest.json) contain exact pins and scope. Raw clean primary samples: [C8](c8/clean.json), [C16](c16/clean.json), [C32](c32/clean.json). Qualification rollups preserve all gate counts, tolerance/maxima and SHA256 of their larger original checker receipts: [C8](c8/qualification-rollup.json), [C16](c16/qualification-rollup.json), [C32](c32/qualification-rollup.json). Both arms pass 480 eager + 1,440 graph comparisons and 90 alternating/zero/restored primary state checks per C, with unchanged 1% NRMSE and zero violations of the 2% + 0.02 elementwise policy.

## Readable profiles and physical mechanism

Profiles are separate runs, not clean timing sources. Each named-case dump lists actual kernel order, streams, start and duration, with a separate device span/kernel sum. The inherited `clean_mean_per_case_median_us` field in a named profile is actually that named case's clean median, not the campaign mean above.

| C | Baseline dump | Selected dump |
| --- | --- | --- |
| 8 | [TXT](c8/profile-baseline-l3-s0.txt) / [JSON](c8/profile-baseline-l3-s0.json) | [TXT](c8/profile-candidate-l3-s0.txt) / [JSON](c8/profile-candidate-l3-s0.json) |
| 16 | [TXT](c16/profile-baseline-l3-s0.txt) / [JSON](c16/profile-baseline-l3-s0.json) | [TXT](c16/profile-candidate-l3-s0.txt) / [JSON](c16/profile-candidate-l3-s0.json) |
| 32 | [TXT](c32/profile-baseline-l3-s0.txt) / [JSON](c32/profile-baseline-l3-s0.json) | [TXT](c32/profile-candidate-l3-s0.txt) / [JSON](c32/profile-candidate-l3-s0.json) |

Each C folder also contains actual engagement JSON/TXT/trace before clean timing and separately labeled G2+reset profiles/receipt. G2 controls use identical native G1 intermediates; their deltas are never added to the full-operator result. Actual `_sparseepi_mr1` presence, absent from baseline, is required.

[Pristine baseline ISA](isa/baseline-bm16.isa), [selected candidate ISA](isa/mr1-candidate.isa) and [compiled proof](isa/mr1-compiled-isa-proof.json) show the real program. Fresh default-off G2 `.text` is byte-identical to pristine official code (`9ebe9d83…f940`). Candidate `.text` is distinct (`40e555d3…4621`). MR1's wave validity branch skips its all-invalid LDS group. **The attempted preservation of MR0's compiler guard failed:** selected emitted MR0 loads at 0x1d58 are unconditional; MR1 loads at 0x1d80 are guarded by compare/branch 0x1d6c/0x1d70. Producer writes, barriers, BF16 operands and packed atomic updates are unchanged. The contribution is measured mask relocation/prefetch scheduling, not a claim that both groups have fewer instructions or that atomic arrival order is identical.

The initial second-variant attempt loaded a stale v1 nested-emitter cache and was [rejected by actual engagement](rejected-stale-cache/summary.json); it supplied no admitted timing. A fresh per-variant FlyDSL namespace fixed execution identity. The final API has one True mode versus False; it does not expose simultaneous v1/MR1 modes requiring a mode-dependent production cache key. [Archived v1 results](archived-v1-summary.json) remain separate. A hypothetical mixed per-C score was not head-to-head and is not deployed.

## Selection and reproduction

Bind once before graph capture:

```python
from aiter.ops.flydsl.kernels.mxmoe_sparse_epilogue import make_g2_sparse_epilogue_override
callback = make_g2_sparse_epilogue_override()
# Pass _stage2_override=callback to the ordinary native fused_moe call.
```

Default None preserves the old decorated custom-op/fake path. Opted-in eager/HIP graph calls invoke the same existing native `_fused_moe_impl` directly: Python callables cannot cross its Torch-library schema. TorchDynamo/torch.compile is untested for this opt-in. The real actual-M guard runs before top-k cache normalization. The special shared/FHMOE API and unsupported native configurations fall back; no general EP qualification is claimed.

Source tests (existing AITER runtime and pytest required):

```bash
python3 op_tests/test_mxmoe_sparse_epilogue_policy.py
python3 op_tests/flydsl_tests/test_mxmoe_sparse_epilogue.py
```

Reported checks: 44 source CPU policy/API tests, repository hooks, six native GPU fixtures, captured full-operator gates, independent reductions and clean/profile/source/compiled checks. Actual runtime is in identity.json: torch 2.11.0+rocm7.2, HIP 7.2.26015, Triton 3.7.0 and FlyDSL 0.3.4.1 on gfx950.

This folder also contains the **exact frozen replay runner/helpers/oracle** under `harness/benchmarks/moe_replay/`. To reproduce the captured campaign, obtain the authorized real capture bank and original packed weights; they remain external, are not downloadable from this branch, and are identified by paths/digests in receipts. Clone MarloweAI/marlowe-kernels at `cbbf415b9cfae2302b1175ade370be448470a33a` and overlay these replay files. Its library source digest independently matches the recorded `f3774124…707f`. Clone AITER at the selected source SHA and SGLang at sorting-adapter pin `46beacf13dbcd9d3aea456301f2d39ea3888540e`; that adapter's bytes match the plan's later SGLang pin. This is not a claim of latest-SGLang full-serving dispatch. The harness does not instantiate the coworker's PyHIP MoE arm in native-port mode; no new dependency/backend is added by the AITER source delta.

Run **inside an owned one-GPU Slurm allocation**, with the image/dependencies above and external bank paths provided. Set each variable to your writable/accessible path before importing AITER; keep the AITER source clean:

```bash
export MOE_PORT_ROOT=/your/writable/moe-replay-run
export MOE_PORT_AITER_SOURCE=/your/aiter-selected-checkout
export MOE_PORT_SGLANG_SOURCE=/your/sglang-pinned-checkout
export MOE_PORT_MARLOWE_SOURCE=/your/marlowe-kernels-replay-checkout
export MOE_PORT_CAPTURE=/your/real-capture-bank
export MOE_PORT_MANIFEST=/your/evidence/source-manifest.json
export MOE_PORT_SGLANG_SHA=46beacf13dbcd9d3aea456301f2d39ea3888540e
export MOE_PORT_RUNTIME_MANIFEST=/your/runtime-preflight.json
export MOE_PORT_FLYDSL_CACHE="$MOE_PORT_ROOT/cache/flydsl-296c72c1-fresh"
cd "$MOE_PORT_MARLOWE_SOURCE"
bash benchmarks/moe_replay/upstream_replay_launch.sh \
  --comparison-mode native-port --native-port-manifest "$MOE_PORT_MANIFEST" \
  --concurrencies 8 16 32 --samples 40 --warmups 10 --profile-cases 1 \
  --output "$MOE_PORT_ROOT/results/fresh-comparison"
```

The output directory must not exist. Source/harness hashes and actual marker must pass before timing. Do not reuse an earlier experimental True-mode cache. Cache/image repairs, the research runner and bulk weights are not part of the source PR. Frozen profile labels/group interpretation are clarified above rather than rewriting historical receipts. Inherited PyHIP/AITER license files are preserved with this harness archive; the coworker snapshot contains no root license, so it is not labeled wholesale as MIT.

## Resource closure

[Release](lifecycle/release.json), [cleanup health](lifecycle/cleanup-health.json) and [Slurm accounting](lifecycle/slurm-accounting.txt) record one reused GPU allocation (86754), 28m07s within the 60-minute experiment budget, all final measurement steps terminal and intentional release. One stale-cache probe failed and remains recorded. Queue absence and zero owned torch allocation/reservation were verified; unrelated jobs were preserved. No eight-GPU allocation, recapture or full serving run was performed for this native port.
