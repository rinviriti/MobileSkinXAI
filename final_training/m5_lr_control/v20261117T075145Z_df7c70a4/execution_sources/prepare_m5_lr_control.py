"""Prepare exactly one authorized LR-only M5 control, preserving original artifacts."""
import difflib,hashlib,json,shutil,uuid
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9';D=R/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8';B=R/'final_training/phase5b/v20261117T063340Z_d20cbfaa'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for folder,lockname in [(P,'PREPARATION_LOCK.json'),(D,'RELEASE_LOCK.json')]:
 lock=json.loads((folder/lockname).read_text());assert sha(folder/'checksums.sha256')==lock['checksums_sha256']
 for line in (folder/'checksums.sha256').read_text().splitlines():
  h,n=line.split('  ',1);assert sha(folder/n)==h,n
assert json.loads((D/'data_approval.json').read_text())['approved']
version='v'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8];O=R/'final_training/m5_lr_control'/version;O.mkdir(parents=True,exist_ok=False)
for name in ['code','configs','reports','run/checkpoints']:(O/name).mkdir(parents=True)
before=json.loads((P/'configs/M5.json').read_text());cfg=json.loads(json.dumps(before));cfg['head_lr']=1e-5;cfg['backbone_lr']=1e-6;cfg['checkpoint_dir']=str((O/'run/checkpoints').relative_to(R))
assert {k:v for k,v in cfg.items() if k not in ['head_lr','backbone_lr','checkpoint_dir']}=={k:v for k,v in before.items() if k not in ['head_lr','backbone_lr','checkpoint_dir']}
(O/'configs/M5.json').write_text(json.dumps(cfg,indent=2)+'\n')
# All original release, training and analysis outputs are snapshotted, never written.
snap={}
for folder in [D,P,B,R/'research/thesis_assets/M5_validation_diagnosis/v20261117T073553Z_99a1fb04']:
 for p in folder.rglob('*'):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:snap[str(p.relative_to(R))]=sha(p)
(O/'reports/original_artifact_snapshot.json').write_text(json.dumps(snap,indent=2)+'\n')
original=(P/'code/train.py').read_text();s=original.replace('import argparse','import argparse\nimport sys\nfrom pathlib import Path\nEXPERIMENT = Path(__file__).resolve().parents[1]\nsys.path.insert(0, str(EXPERIMENT.parents[2] / "final_training/phase5a/v20261117T061722Z_f5055bd9/code"))')
s=s.replace("path = PROJECT / 'experiment_summary.json'", "path = EXPERIMENT / 'experiment_summary.json'")
s=s.replace("(PROJECT / 'training_authorization.json')", "(EXPERIMENT / 'training_authorization.json')")
s=s.replace("digest(PROJECT / 'configs' / f'{cfg[\"id\"]}.json')", "digest(EXPERIMENT / 'configs' / f'{cfg[\"id\"]}.json')")
s=s.replace("'artifact_kind': 'final_training_checkpoint', 'classes': cfg['classes'],", "'artifact_kind': 'final_training_checkpoint', 'experiment_id': EXPERIMENT.name,\n                  'experiment_kind': 'single_M5_LR_only_control', 'classes': cfg['classes'],")
s=s.replace("cfg = config(args.model)", "cfg = json.loads((EXPERIMENT / 'configs' / f'{args.model}.json').read_text())")
s=s.replace("choices=['M1', 'M2', 'M3', 'M4', 'M5']", "choices=['M5']")
# Observability only: original gradients/optimizer/forward/ordering are unchanged.
s=s.replace("total_loss, n, gate_total = 0.0, 0, 0.0", "total_loss, n, gate_total = 0.0, 0, 0.0\n            components = {'ce': 0.0, 'kd': 0.0, 'consistency': 0.0, 'gate_mass': 0.0}")
s=s.replace("if stats:\n                    gate_total", "if stats:\n                    for key in components:\n                        components[key] += float(stats[key]) * size\n                    gate_total")
s=s.replace("'validation': metrics, 'global_updates': global_updates,", "'validation': metrics, 'global_updates': global_updates,\n                      'head_lr': optimizer.param_groups[0]['lr'], 'backbone_lr': optimizer.param_groups[1]['lr'],\n                      'loss_components': {key: value / n for key, value in components.items()},")
(O/'code/train.py').write_text(s);shutil.copyfile(P/'code/core.py',O/'code/original_core_reference.py');shutil.copyfile(P/'code/train.py',O/'code/original_train_reference.py')
(O/'reports/trainer_routing_and_logging_only.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='frozen_original_train.py',tofile='LR_control_train.py')))
(O/'reports/config_control_check.json').write_text(json.dumps({'training_differences_only':{'head_lr':[before['head_lr'],cfg['head_lr']],'backbone_lr':[before['backbone_lr'],cfg['backbone_lr']]},'nontraining_difference':'checkpoint_dir routes to new version; experiment provenance/logging extended only','all_other_training_fields_identical':True,'core_implementation_imported_unchanged':str((P/'code/core.py').relative_to(R)),'core_sha256':sha(P/'code/core.py'),'same_parent_paths':cfg['initialization'],'full_run_limit':1},indent=2)+'\n')
(O/'training_authorization.json').write_text(json.dumps({'authorized':True,'authorized_UTC':datetime.now(timezone.utc).isoformat(),'scope':'ONE M5 learning-rate-only control from original final M3/M4 parents; no auto replacement or additional runs','user_instruction':'Proceed with ONE controlled M5 learning-rate improvement experiment','experiment_directory':str(O.relative_to(R))},indent=2)+'\n')
(O/'experiment_summary.json').write_text(json.dumps({'experiment_id':version,'name':'LR-adjusted MobileSkinXAI-Enhanced','models':{'M5':{'status':'prepared_not_trained'}},'main_control_configurations':1,'dataset_release':D.name},indent=2)+'\n')
(O/'reports/experiment_binding.json').write_text(json.dumps({'original_project':str(P.relative_to(R)),'original_results':str((B/'final_results.json').relative_to(R)),'original_dataset':str(D.relative_to(R))},indent=2)+'\n')
(R/'final_training/m5_lr_control/LATEST.json').write_text(json.dumps({'experiment_directory':str(O.relative_to(R))},indent=2)+'\n')
print(O)
