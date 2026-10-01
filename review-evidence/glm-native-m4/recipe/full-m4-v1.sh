#!/usr/bin/env bash
set -euo pipefail
source /workspace/hassane/moe-native-tiny-20261001/launch/environment-m4-abi-fixed.sh
trap 'rc=$?; printf "M4_FULL_DONE rc=%s\n" "$rc"' EXIT
bash benchmarks/moe_replay/upstream_replay_launch.sh --output "$MOE_PORT_ROOT/results/m4-full-v1" --comparison-mode native-port --native-port-manifest "$MOE_PORT_MANIFEST" --concurrencies 1 --samples 40 --warmups 10 --profile-cases 1
python - <<'VERDICT'
import json,os,pathlib
s=json.loads((pathlib.Path(os.environ['MOE_PORT_ROOT'])/'results/m4-full-v1/summary.json').read_text()); rows=[r for r in s['receipts'] if r.get('concurrency')==1];print('M4_FULL_RECEIPTS',json.dumps(rows),flush=True);assert len(rows)==1 and rows[0]['status']=='qualified' and rows[0]['cases']==480;print('M4_FULL_VERDICT=qualified',flush=True)
VERDICT
