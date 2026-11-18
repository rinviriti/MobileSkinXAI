"""Verify the frozen release and active data bindings without importing models."""
import csv,json,hashlib,stat
from pathlib import Path
from collections import defaultdict
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
a=json.loads((R/'final_training/data_approval.json').read_text());assert a['approved'] is True
D=(R/a['manifest_path']).parents[1];lock=json.loads((D/'RELEASE_LOCK.json').read_text())
assert sha(D/'checksums.sha256')==lock['checksums_sha256']
lines=(D/'checksums.sha256').read_text().splitlines()
for line in lines:
 h,name=line.split('  ',1);p=D/name;assert sha(p)==h,name
 assert not p.stat().st_mode & (stat.S_IWUSR|stat.S_IWGRP|stat.S_IWOTH),name
assert len(lines)==lock['files']
assert sha(R/a['manifest_path'])==a['manifest_sha256']
assert sha(R/a['release_manifest_path'])==a['release_manifest_sha256']
assert sha(R/a['protocol_path'])==a['protocol_sha256']
rows=read(D/'dataset_manifest.csv');active=[r for r in rows if r['release_eligible']=='True'];excluded=[r for r in rows if r['release_eligible']=='False']
assert len({(r['dataset'],r['image']) for r in rows})==len(rows)
assert all(r['optimization_allowed']=='False' and r['model_selection_allowed']=='False' and r['calibration_allowed']=='False' for r in excluded)
assert all(r['original_split']==r['split'] for r in rows)
assert all(r['optimization_allowed']==str(r['split']=='train') and r['model_selection_allowed']==str(r['split']=='validation') for r in active)
# Verify frozen configs change only the expected data bindings from the originals.
allowed={'split_manifest','dataset_version','dataset_release','external_evaluation_manifest','evaluation_scoring_units','dataset_protocol'}
for m in ['M1','M2','M3','M4','M5']:
 cfg=json.loads((R/f'final_training/configs/{m}.json').read_text());before=json.loads((D/f'previous_active_configs/{m}.json').read_text())
 assert sha(R/f'final_training/configs/{m}.json')==sha(D/f'experiment_configs/{m}.json')
 assert {k:v for k,v in cfg.items() if k not in allowed}=={k:v for k,v in before.items() if k not in allowed}
 assert cfg['split_manifest']==a['manifest_path']
 trainview=read(R/cfg['split_manifest']);assert len(trainview)==17170
 assert all(r['release_eligible']=='True' and r['split'] in ['train','validation','test'] for r in trainview)
 assert len({r['image'] for r in trainview})==len(trainview)
 unit_owner={}
 for r in trainview:assert unit_owner.setdefault(r['lesion_id'],r['split'])==r['split']
# Final original input checks after all active configuration changes.
snap=json.loads((D/'input_preservation_snapshot.json').read_text())
for p,h in snap.items():assert sha(R/p)==h,p
units=read(D/'evaluation_scoring_units.csv');members=defaultdict(list)
for r in units:members[r['lesion_scoring_unit']].append(r)
assert all(any(r['use_in_lesion_probability_mean']=='True' for r in rs) and len({r['label'] for r in rs})==1 for rs in members.values())
print(json.dumps({'release':str(D.relative_to(R)),'checksummed_files':len(lines),'original_inputs_preserved':len(snap),'eligible_images':len(active),'excluded_images':len(excluded),'five_config_data_bindings_verified':True,'approved':True},indent=2))
