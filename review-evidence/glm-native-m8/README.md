# GLM native exact-M8 route merge: review evidence

Opt-in native FlyDSL supplied-route MoE for one MI355X TP8/EP1 rank, C2/M8 four-token verification. BF16 hidden/output, shuffled MXFP4 expert weights/E8M0 scales, native0x7fffff input and FP32 SiLU-middle quantization, actual supplied nine-route FP32 weights, weighted BF16 packed atomics. Shared expert256 is supplied once; native top8 are distinct within each token. No router, collective or serving/ITL claim.

| Full480-case mean of per-case clean medians | C2/M8 |
| --- | ---: |
| Official AITER569d981 native control |33.351204202821µs |
| Consumer route merge + G1 output reset, measured9b145 |30.166782304877µs |
| Saved |3.184421897943µs /9.548146683334% |

Forty randomized paired samples and10warmups per case;30 layers×16 verification frames. The complete operator boundary charges native sorting/reset or candidate integrated reset, both GEMMs and local atomic combination. All480 independent-reference eager checks/arm,1440 graph checks/arm,90 state checks/arm,480 native-capture parity checks, source/routes/dispatch/immutable gates passed. Policy unchanged: finite output, NRMSE≤1%, zero elementwise violations under atol0.02+rtol0.02×|reference|. `qualification-rollup.json` is a compact independently recomputed index; `full/` retains raw measurements/checks/dispatch/traceJSON+TXT. Profiles explain kernels, not table latency. A named profile's inherited clean label refers to its individual case median.

Measured source:9b145e0de47daacc939b8730d145138f182bdb24. Published source34795206af96156f6c61930ec8d2100e80bef3d7 adds only host input/scratch byte-range alias rejection and its CPU test; kernel/helper/GPUfixture bytes and eligible launch AST remain identical as recorded in `host-guard-provenance.json`. No later-head retiming is claimed. Actual production diff is406insertions/59deletions across4files, excluding491testlines;330 in the immutable measured manifest was an estimate, corrected here. Native defaultFalse G1/G2 .text equals pristine official569d981; `proofs/` contains exact hashes and small code objects. Candidate profile contains only G1_merge8_reset +G2_merge8, no separate zero launch.

Earlier unsupportedNT compile failure, finiteSiLU clamp7 correctness failure, and corrected separate-zero full result (3.24%slower) remain in distinct directories. The reset pilot13.58% is separate from the full result. No thresholds were relaxed. Inputs/routes must not alias a handle's private output/middle/scales; serial eager/capturedgraph reuse supported, concurrent handle reuse unsupported.

## Reproduce

Use one owned MI355X allocation and the exact runtime/dependency/image identities in `runtime-m8-fused-reset.json`. No Engine or eightGPU serving required for replay. Use measured source9b145 and frozen manifest for exact reproduction; clone official569d981 then checkout source9b145, set clone-local core.abbrev8 for measured binary patch identity, and extract `frozen-harness.tar.gz` and use its `frozen-harness/` as the benchmark/package tree. The manifest's harness hashes match every archived file. The inherited harness environment includes PyHIP; the proposed kernel uses only native FlyDSL and adds no PyHIP dependency.

Captured model tensors/weights stay external. Exact SHA/location/frame identities are in `full/identity.json` and `full/c2-capture.json`. Capture source is AITERc8325e00/SGLang46beacf nativeTP8+FP8KV with deterministic varied8192-token prompts; replay control is freshly compiled/qualified official569d981, not a relabeled recapture. Sorting helper is pinned SGLanga0fcebb3 and its source hash is in the manifest. Preserve all30layers×16frames.

`recipe/` preserves exact environment/cache/fixture/pilot/full commands; adjust only task paths in a separately recorded recipe. Keep HOME unchanged and allocate writable task-specific caches before imports. Full run uses `upstream_replay_launch.sh --comparison-mode native-port --concurrencies2 --samples40 --warmups10 --profile-cases1` through MOE_PORT variables and measured-manifest.json. Failed gates stop before timing. Every profile has readable TXT/JSON. `full/completed.json` plus all receipts determine success, not shell exit alone.

CPU: `python3 op_tests/test_mxmoe_tiny_m8_policy.py` (11passed publishedhead); GPU: `python3 op_tests/flydsl_tests/test_mxmoe_tiny_m8.py` (changingroutes/sharedweight/zero/hotactivations/twoworkspaces/graphs passed at measured source). Black/Ruff/hooks/py_compile/diffchecks pass. Job86952 reused oneGPU for both tiny lanes and was released19:11:48UTC; cleanup files verify empty job/step queues afterward. This branch contains no tensor/weight blobs. No external submission authorized.
