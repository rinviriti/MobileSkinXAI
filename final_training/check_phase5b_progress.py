"""Compact read-only monitor for the authorized training sequence."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9';O=R/json.loads((R/'final_training/phase5b/LATEST.json').read_text())['execution_directory']
s=json.loads((O/'sequence_status.json').read_text());out={'sequence_status':s['status'],'models':{}}
for m in ['M1','M2','M3','M4','M5']:
 run=P/f'runs/{m}_seed42';h=run/'history.jsonl'
 if h.exists():
  records=[json.loads(line) for line in h.read_text().splitlines()];b=max(records,key=lambda x:x['validation']['macro_f1']);out['models'][m]={'status':s['models'].get(m,{}).get('status'),'epoch':records[-1]['epoch'],'best_epoch':b['epoch'],'best_validation_macro_F1':b['validation']['macro_f1'],'elapsed_seconds':records[-1]['elapsed_seconds']}
 elif m in s['models']:out['models'][m]=s['models'][m]
print(json.dumps(out,indent=2))
