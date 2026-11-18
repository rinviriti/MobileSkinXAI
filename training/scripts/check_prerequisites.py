"""Fail-closed local prerequisite inspection; never starts training."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
required=[ROOT/'dataset',ROOT/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8/data_approval.json']
missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
print(json.dumps({'ready':not missing,'missing_private_local_inputs':missing,'training_started':False,'note':'Public code/configs preserve original guards. Licensed source data and approved local manifests/locks are required; never remove guards or silently regenerate splits.'},indent=2))
raise SystemExit(bool(missing))
