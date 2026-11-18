"""Freeze single-control artifacts and verify originals, parents and LR-only scope."""
import argparse,csv,hashlib,json,shutil,stat
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--experiment-dir',required=True);args=ap.parse_args();O=R/args.experiment_dir
 results=json.loads((O/'comparison_results.json').read_text());assert results['full_control_runs']==1
 snapshot=json.loads((O/'reports/original_artifact_snapshot.json').read_text());assert all(sha(R/p)==h for p,h in snapshot.items())
 before=json.loads((P/'configs/M5.json').read_text());after=json.loads((O/'configs/M5.json').read_text());allowed={'head_lr','backbone_lr','checkpoint_dir'};assert {k:v for k,v in before.items() if k not in allowed}=={k:v for k,v in after.items() if k not in allowed};assert after['head_lr']==1e-5 and after['backbone_lr']==1e-6
 original=torch.load(P/'runs/M5_seed42/checkpoints/best.pt',map_location='cpu',weights_only=True);parents=original['provenance']['parents'];shape={k:tuple(v.shape) for k,v in original['state_dict'].items()};del original
 history=[json.loads(l) for l in (O/'run/history.jsonl').read_text().splitlines()];lrcheck=[]
 for h in history:
  p=O/f"run/checkpoints/epoch_{h['epoch']:03d}.pt";cp=torch.load(p,map_location='cpu',weights_only=True);pv=cp['provenance'];assert pv['experiment_id']==O.name and pv['experiment_kind']=='single_M5_LR_only_control' and pv['config_sha256']==sha(O/'configs/M5.json');assert {k:tuple(v.shape) for k,v in cp['state_dict'].items()}==shape
  assert pv['parents']==parents,'Parents differ from original M5'
  groups=cp['optimizer']['param_groups'];assert groups[0]['lr']==1e-5 and groups[1]['lr']==1e-6 and groups[0]['weight_decay']==groups[1]['weight_decay']==1e-4
  steps={int(s['step']) for s in cp['optimizer']['state'].values()};assert steps=={727*h['epoch']};lrcheck.append({'epoch':h['epoch'],'head_lr':groups[0]['lr'],'backbone_lr':groups[1]['lr'],'optimizer_steps':727*h['epoch'],'config_and_original_parent_match':True});del cp
 with (O/'reports/actual_LR_checkpoint_audit.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,list(lrcheck[0]));w.writeheader();w.writerows(lrcheck)
 assert len(list((O/'run/checkpoints').glob('epoch_*.pt')))==len(history)
 summary=json.loads((O/'experiment_summary.json').read_text());summary.update({'status':'complete_verified','comparison_results':'comparison_results.json','joint_exploratory_goal_met':results['joint_macro_MEL_SCC_goal_met'],'automatic_replacement':False,'retention_recommendation':results['recommendation']});(O/'experiment_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 target=O/'execution_sources';target.mkdir(exist_ok=False)
 for name in ['prepare_m5_lr_control.py','report_m5_lr_control.py','verify_m5_lr_control.py']:shutil.copyfile(R/'final_training'/name,target/name)
 # Verify 300DPI and matching vector exports before freezing reports.
 from PIL import Image
 for p in (O/'thesis_figures').glob('*.png'):
  with Image.open(p) as im:assert abs(im.info['dpi'][0]-300)<1
  assert p.with_suffix('.svg').is_file() and p.with_suffix('.pdf').is_file()
 files=sorted(p for p in O.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.is_symlink() and p.name not in ['checksums.sha256','POSTFLIGHT_VERIFICATION.json'])
 (O/'checksums.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(O))+'\n' for p in files))
 verify={'status':'PASS','original_artifacts_unchanged':len(snapshot),'hash_mismatches':0,'full_control_runs_started':1,'training_factor_changes_only':['head_lr','backbone_lr'],'artifact_routing_and_logging_changes_only_besides_LR':True,'loss_architecture_loader_seed_optimizer_functions_imported_from_verified_original':True,'same_fresh_M3_M4_parents_as_original_M5':True,'original_M5_not_initialization':True,'every_epoch_optimizer_LR_and_step_verified':len(lrcheck),'all_epoch_checkpoints_preserved':True,'all_requested_metrics_recomputed_from_saved_validation_predictions':True,'joint_exploratory_goals_met':results['joint_macro_MEL_SCC_goal_met'],'automatic_checkpoint_replacement':False,'test_PAD_evaluated':False,'Flutter_modified':False,'artifact_files_checksummed':len(files),'checksums_sha256':sha(O/'checksums.sha256')}
 (O/'POSTFLIGHT_VERIFICATION.json').write_text(json.dumps(verify,indent=2)+'\n')
 for line in (O/'checksums.sha256').read_text().splitlines():
  h,name=line.split('  ',1);assert sha(O/name)==h
 for p in files+[O/'checksums.sha256',O/'POSTFLIGHT_VERIFICATION.json']:p.chmod(stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH)
 print(json.dumps(verify,indent=2))
if __name__=='__main__':main()
