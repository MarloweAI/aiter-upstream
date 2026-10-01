#!/usr/bin/env bash
set -euo pipefail
source /workspace/hassane/moe-native-tiny-20261001/launch/environment-m8-fused-reset.sh
trap 'rc=$?; printf "M8_PILOT_AFTER_PROOF_DONE rc=%s\n" "$rc"' EXIT
python - <<'CHECK'
import json,os
from pathlib import Path
p=Path(os.environ['MOE_PORT_ROOT'])/'results/m8-fused-reset-pristine-compiled-proof.json';x=json.loads(p.read_text());assert x['passed'] and x['feature_sha']=='9b145e0de47daacc939b8730d145138f182bdb24';print('M8_PRISTINE_CONTROL_VERIFIED',flush=True)
CHECK
bash benchmarks/moe_replay/upstream_replay_launch.sh --output "$MOE_PORT_ROOT/results/m8-pilot-fused-reset" --comparison-mode native-port --native-port-manifest "$MOE_PORT_MANIFEST" --concurrencies 2 --layers 3 77 --steps 0 1 --samples 5 --warmups 2 --profile-cases 1
python - <<'VERDICT'
import json,os
from pathlib import Path
s=json.loads((Path(os.environ['MOE_PORT_ROOT'])/'results/m8-pilot-fused-reset/summary.json').read_text());rows=[r for r in s['receipts'] if r.get('concurrency')==2];print('M8_PILOT_RECEIPTS',json.dumps(rows),flush=True);assert len(rows)==1 and rows[0]['status']=='qualified';print('M8_PILOT_VERDICT=qualified',flush=True)
VERDICT
