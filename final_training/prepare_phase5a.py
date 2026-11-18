"""Create isolated five-config preparation; never executes full training."""
import csv,hashlib,json,shutil,uuid
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'final_training'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
D=R/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8'
A=json.loads((D/'data_approval.json').read_text());assert A['approved'] is True
lock=json.loads((D/'RELEASE_LOCK.json').read_text());assert sha(D/'checksums.sha256')==lock['checksums_sha256']
for line in (D/'checksums.sha256').read_text().splitlines():
 h,p=line.split('  ',1);assert sha(D/p)==h,p
checks=json.loads((D/'integrity_checks.json').read_text());weights=json.loads((D/'class_weights.json').read_text())
rows=list(csv.DictReader(open(D/'dataset_manifest.csv')))
for r in rows:
 if r['release_eligible']=='True':
  assert (R/r['processed_path']).is_file();assert r['label'] in weights['class_order'];assert int(r['label_index'])==weights['class_order'].index(r['label'])
  assert sha(R/r['processed_path'])==r['processed_sha256']
train=[r for r in rows if r['optimization_allowed']=='True'];assert len(train)==11632
for c,n in weights['counts'].items():assert sum(r['label']==c for r in train)==n
version='v'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
O=P/'phase5a'/version;O.mkdir(parents=True,exist_ok=False)
for name in ['configs','reports','code','export_probe','runs']:(O/name).mkdir()
project='MobileSkinXAI_final_'+version
for m in ['M1','M2','M3','M4','M5']:
 cfg=json.loads((P/f'configs/{m}.json').read_text());cfg['project_id']=project
 cfg['checkpoint_dir']=str((O/f'runs/{m}_seed42/checkpoints').relative_to(R));(R/cfg['checkpoint_dir']).mkdir(parents=True)
 cfg['max_finetune_epochs']=40 if m=='M5' else 25
 cfg['preparation_status']='awaiting_Phase5B_training_authorization'
 if m=='M5':
  cfg['initialization']['student_checkpoint']=str((O/'runs/M3_seed42/checkpoints/best.pt').relative_to(R))
  cfg['initialization']['teacher_checkpoint']=str((O/'runs/M4_seed42/checkpoints/best.pt').relative_to(R))
 (O/f'configs/{m}.json').write_text(json.dumps(cfg,indent=2)+'\n')
core=(P/'core.py').read_text().replace('PROJECT = Path(__file__).resolve().parent','PROJECT = Path(__file__).resolve().parents[1]').replace('ROOT = PROJECT.parent','ROOT = PROJECT.parents[2]').replace("PROJECT_ID = 'MobileSkinXAI_final_v1'",f'PROJECT_ID = {project!r}').replace("record_path = PROJECT / 'pretrained'", "record_path = ROOT / 'final_training' / 'pretrained'")
(O/'code/core.py').write_text(core)
trainer=(P/'train.py').read_text().replace("(PROJECT / 'pretrained'", "(ROOT / 'final_training' / 'pretrained'")
# Unique epoch checkpoint payloads; mutable best/last are symlinks to immutable payloads.
trainer=trainer.replace('import argparse','import argparse\nimport os')
trainer=trainer.replace("torch.save(checkpoint, checkpoint_dir / 'last.pt')\n            if improved:\n                torch.save(checkpoint, checkpoint_dir / 'best.pt')", "epoch_path = checkpoint_dir / f'epoch_{epoch+1:03d}.pt'\n            with epoch_path.open('xb') as handle:\n                torch.save(checkpoint, handle)\n            for alias in (['last.pt', 'best.pt'] if improved else ['last.pt']):\n                temporary = checkpoint_dir / (alias + '.next')\n                temporary.symlink_to(epoch_path.name)\n                os.replace(temporary, checkpoint_dir / alias)")
(O/'code/train.py').write_text(trainer)
for name in ['M5_LOSS.md','LITERATURE_REVIEW.md'] :shutil.copyfile(P/name,O/name)
shutil.copyfile(D/'data_approval.json',O/'data_approval.json')
(O/'reports/dataset_verification.json').write_text(json.dumps({'approved_release':str(D.relative_to(R)),'release_checksums_verified':lock['files'],'canonical_manifest_sha256':sha(D/'dataset_manifest.csv'),'integrity_checks_sha256':sha(D/'integrity_checks.json'),'eligible_image_paths_labels_and_prepared_hashes_checked':18473,'train_counts':weights['counts'],'train_weights':weights['weights'],'demonstration_checkpoint_access':False},indent=2)+'\n')
summary={'phase':'5A_preparation_only','version':version,'project_id':project,'models':{m:{'status':'prepared_not_trained','checkpoint_dir':f'runs/{m}_seed42/checkpoints'} for m in ['M1','M2','M3','M4','M5']},'main_configuration_count':5,'full_training_authorized':False}
(O/'experiment_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(P/'phase5a/LATEST.json').write_text(json.dumps({'directory':str(O.relative_to(R)),'dataset_version':D.name},indent=2)+'\n')
print(O)
