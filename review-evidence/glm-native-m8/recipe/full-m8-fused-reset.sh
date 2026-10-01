#!/usr/bin/env bash
set -euo pipefail
source /workspace/hassane/moe-native-tiny-20261001/launch/environment-m8-fused-reset.sh
trap 'rc=$?; printf "M8_FULL_DONE rc=%s\n" "$rc"' EXIT
bash benchmarks/moe_replay/upstream_replay_launch.sh --output "$MOE_PORT_ROOT/results/m8-full-fused-reset" --comparison-mode native-port --native-port-manifest "$MOE_PORT_MANIFEST" --concurrencies 2 --samples 40 --warmups 10 --profile-cases 1
python - <<'VERDICT'
import json,os,pathlib
s=json.loads((pathlib.Path(os.environ['MOE_PORT_ROOT'])/'results/m8-full-fused-reset/summary.json').read_text()); rows=[r for r in s['receipts'] if r.get('concurrency')==2];print('M8_FULL_RECEIPTS',json.dumps(rows),flush=True);assert len(rows)==1 and rows[0]['status']=='qualified' and rows[0]['cases']==480;print('M8_FULL_VERDICT=qualified',flush=True)
VERDICT
