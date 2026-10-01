#!/usr/bin/env bash
set -euo pipefail
source /workspace/hassane/moe-native-tiny-20261001/launch/environment-m8-fused-reset.sh
trap 'rc=$?; printf "M8_PILOT_DONE rc=%s\n" "$rc"' EXIT
python - <<'HEALTH'
import hashlib,importlib.metadata,json,os,socket,subprocess,torch
from pathlib import Path
import aiter,flydsl,pyhip
from benchmarks.moe_replay.upstream_native_port import read_manifest,validate_source,validate_harness
assert torch.cuda.device_count()==1 and torch.version.hip
x=torch.ones(4,device='cuda');assert float(x.sum())==4
m=read_manifest(os.environ['MOE_PORT_MANIFEST']);assert validate_source(m,os.environ['MOE_PORT_AITER_SOURCE']);assert validate_harness(m,Path.cwd())
from sglang.kernels.ops.moe.moe_sorting_small import apply_aiter_small_moe_sort_patch
import sglang.kernels.ops.moe.moe_sorting_small as shim
assert hashlib.sha256(Path(shim.__file__).read_bytes()).hexdigest()==m['sorting_adapter_sha256']
apply_aiter_small_moe_sort_patch()
info=dict(job=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),host=socket.gethostname(),devices=torch.cuda.device_count(),device=torch.cuda.get_device_name(),torch=torch.__version__,hip=torch.version.hip,aiter=aiter.__file__,feature_sha=m['aiter_feature_sha'],base_sha=m['aiter_base_sha'],sglang_sha=os.environ['MOE_PORT_SGLANG_SHA'],sorting_adapter_sha256=m['sorting_adapter_sha256'],flydsl=importlib.metadata.version('flydsl'),pyhip=importlib.metadata.version('pyhip'),image_path='/workspace/hassane/attention-image-mi355x-8da12b8b-20260917-r3/raw/sglang-rocm-8da12b8b.sqsh',image_sha256='8375b26f6360f877628a363a3093173ac63b9e780f48e23389f10bdfd67cb5e4',image_bytes=60567834624,capture=os.environ['MOE_PORT_CAPTURE'],capture_native_source='AITER c8325e00 / SGLang46beacf; fresh standalone current569d981/a0fcebb3 control, no recapture',flydsl_cache=os.environ['FLYDSL_RUNTIME_CACHE_DIR'])
Path(os.environ['MOE_PORT_RUNTIME_MANIFEST']).write_text(json.dumps(info,indent=2)+'\n');print('CURRENT_NATIVE_HEALTH_READY',json.dumps(info),flush=True)
HEALTH
python - <<'FIXTURE'
import json,math,os,runpy
from pathlib import Path
from aiter.ops.flydsl import moe_kernels
original=moe_kernels.runtime_swiglu_limit
seen=[]
def observe(limit, act):
    value=original(limit,act)
    seen.append(dict(input_limit=limit,activation=act,normalized_limit='+inf' if math.isinf(value) and value>0 else value))
    return value
moe_kernels.runtime_swiglu_limit=observe
try:
    runpy.run_path(os.environ['MOE_PORT_AITER_SOURCE']+'/op_tests/flydsl_tests/test_mxmoe_tiny_m8.py',run_name='__main__')
finally:
    moe_kernels.runtime_swiglu_limit=original
assert seen and all(row['normalized_limit']=='+inf' for row in seen),seen
receipt=dict(passed=True,feature_sha='9b145e0de47daacc939b8730d145138f182bdb24',observed_native_host_normalization=seen,source='aiter.ops.flydsl.moe_kernels.runtime_swiglu_limit; actual public fused_moe native calls',high_activation_fixture='x=1/4; gate/up approximately12 > old clamp7')
(Path(os.environ['MOE_PORT_ROOT'])/'results/m8-fused-reset-activation-proof.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('M8_NATIVE_ACTIVATION_PROOF',json.dumps(receipt),flush=True)
FIXTURE
printf 'M8_NATIVE_FIXTURE_EXIT=0\n'
printf 'M8_FIXTURE_ONLY_VERDICT=passed\n'
