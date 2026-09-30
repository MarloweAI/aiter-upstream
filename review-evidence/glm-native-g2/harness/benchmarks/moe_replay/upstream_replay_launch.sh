#!/usr/bin/env bash
# Execute on an owned one-GPU allocation. Extra args select pilot/full coverage.
set -euo pipefail
NATIVE_PORT=0
if [[ " $* " == *" native-port "* ]]; then
  NATIVE_PORT=1
  CAMPAIGN=${MOE_PORT_ROOT:?Set the owned native-port root}
  AITER_SOURCE=${MOE_PORT_AITER_SOURCE:?Set the pinned feature checkout}
  SGLANG_SOURCE=${MOE_PORT_SGLANG_SOURCE:?Set the pinned sorting-adapter checkout}
  MARLOWE_SOURCE=${MOE_PORT_MARLOWE_SOURCE:-$CAMPAIGN/source/marlowe-kernels}
  CAPTURE=${MOE_PORT_CAPTURE:?Set the readable real-input capture bank}
  PORT_MANIFEST=${MOE_PORT_MANIFEST:?Set the frozen source/shape manifest}
  AITER_SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["aiter_feature_sha"])' "$PORT_MANIFEST")
  SGLANG_SHA=${MOE_PORT_SGLANG_SHA:?Set the actual sorting-adapter revision}
  RUNTIME_MANIFEST=${MOE_PORT_RUNTIME_MANIFEST:?Set the verified image/dependency receipt}
else
  CAMPAIGN=/workspace/hassane/moe-upstream-qualification-20260929
  AITER_SOURCE=$CAMPAIGN/source/aiter
  SGLANG_SOURCE=$CAMPAIGN/source/sglang
  MARLOWE_SOURCE=$CAMPAIGN/source/marlowe-kernels
  CAPTURE=$CAMPAIGN/results/fp8-native-target-verify-capture
  AITER_SHA=c8325e00c88f1b97af75150a311f6704c2729f9c
  SGLANG_SHA=46beacf13dbcd9d3aea456301f2d39ea3888540e
  RUNTIME_MANIFEST=$CAMPAIGN/results/runtime-preflight.json
fi
export TMPDIR="$CAMPAIGN/tmp"
export XDG_CACHE_HOME="$CAMPAIGN/cache/xdg"
export SGLANG_CACHE_DIR="$CAMPAIGN/cache/sglang-main-46beacf1"
export SGLANG_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/jit"
export TILELANG_CACHE_DIR="$SGLANG_CACHE_DIR/tilelang"
export TRITON_CACHE_DIR="$CAMPAIGN/cache/triton"
export TORCH_EXTENSIONS_DIR="$CAMPAIGN/cache/torch-extensions"
export AITER_JIT_DIR="$CAMPAIGN/cache/aiter-main-c8325e00"
export AITER_ROOT_DIR="$CAMPAIGN/cache/aiter-cpp-main-c8325e00"
if (( NATIVE_PORT )); then
  # Nested emitter source edits need an explicit variant-owned cache namespace.
  export FLYDSL_RUNTIME_CACHE_DIR=${MOE_PORT_FLYDSL_CACHE:-$CAMPAIGN/cache/flydsl-main-c8325e00}
else
  export FLYDSL_RUNTIME_CACHE_DIR="$CAMPAIGN/cache/flydsl-main-c8325e00"
fi
export PYHIP_CACHE_DIR="$CAMPAIGN/cache/pyhip-27022514"
export TORCHINDUCTOR_CACHE_DIR="$CAMPAIGN/cache/torch-inductor"
export PYTHONPATH="$AITER_SOURCE:$SGLANG_SOURCE/python:$MARLOWE_SOURCE:$CAMPAIGN/deps:${PYTHONPATH:-}"
export AITER_USE_SYSTEM_TRITON=1
export MAX_JOBS=4
export SGLANG_SET_CPU_AFFINITY=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$SGLANG_CACHE_DIR" "$TILELANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$AITER_JIT_DIR" "$AITER_ROOT_DIR" "$FLYDSL_RUNTIME_CACHE_DIR" "$PYHIP_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
if (( NATIVE_PORT )); then
  export SGLANG_USE_MARLOWE_ATTENTION=0
  export SGLANG_USE_MARLOWE_KERNELS=0
else
  python -c 'import pyhip; print("PYHIP_IMPORT_READY", flush=True)'
fi
LIBRARY_DIGEST=$(python -c 'from marlowe_kernels._runtime import source_digest; print(source_digest())')
VARIANT_DIGEST=$(python -c 'from benchmarks.moe_replay.upstream_contract import file_digest; print(file_digest("benchmarks/moe_replay/upstream_candidate_variants.py"))')
RECOVERY_ARGS=()
if [[ " $* " == *" ordered-recovery "* ]]; then
  ORDERED_DIGEST=$(python -c 'from benchmarks.moe_replay.upstream_contract import file_digest; print(file_digest("benchmarks/moe_replay/upstream_ordered_variant.py"))')
  RECOVERY_ARGS=(--ordered-digest "$ORDERED_DIGEST")
fi
python -m benchmarks.moe_replay.upstream_campaign \
  --capture "$CAPTURE" \
  --aiter-source "$AITER_SOURCE" --aiter-sha "$AITER_SHA" \
  --sglang-source "$SGLANG_SOURCE" --sglang-sha "$SGLANG_SHA" \
  --library-digest "$LIBRARY_DIGEST" --variant-digest "$VARIANT_DIGEST" \
  --runtime-manifest "$RUNTIME_MANIFEST" "${RECOVERY_ARGS[@]}" "$@"
printf 'REPLAY_EXIT=0\n'
