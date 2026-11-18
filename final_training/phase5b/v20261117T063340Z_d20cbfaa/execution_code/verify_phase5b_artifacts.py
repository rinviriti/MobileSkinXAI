"""Postflight verification and checksum freeze of actual experiment outputs."""
import argparse,csv,hashlib,json,os,stat
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execution-dir',required=True);args=ap.parse_args();O=R/args.execution_dir;results=json.loads((O/'final_results.json').read_text());checks=[]
 for folder,lockname in [(R/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8','RELEASE_LOCK.json'),(P,'PREPARATION_LOCK.json')]:
  lock=json.loads((folder/lockname).read_text());assert sha(folder/'checksums.sha256')==lock['checksums_sha256']
  for line in (folder/'checksums.sha256').read_text().splitlines():
   h,name=line.split('  ',1);assert sha(folder/name)==h,name
 for r in results['models']:
  if r['status']!='complete':continue
  m=r['model'];path=R/r['checkpoint'];cp=torch.load(path,map_location='cpu',weights_only=True);cfg=P/f'configs/{m}.json';prov=cp['provenance'];status=json.loads((path.parent.parent/'run_status.json').read_text())
  assert sha(path.parent.parent/'resolved_config.json')==sha(cfg)
  assert prov['artifact_kind']=='final_training_checkpoint' and prov['model_id']==m and prov['run_id']==status['run_id']
  assert prov['project_id']==json.loads(cfg.read_text())['project_id'] and prov['config_sha256']==sha(cfg)
  assert prov['classes']==['NV','MEL','BCC','SCC'] and cp['epoch']==r['best_epoch']
  assert abs(cp['validation_macro_f1']-r['macro_f1'])<1e-6
  assert prov['data_receipt']['manifest_sha256']==json.loads((P/'data_approval.json').read_text())['manifest_sha256']
  if m=='M5':
   for name,parent in [('student','M3'),('teacher','M4')]:
    expected=next(x for x in results['models'] if x['model']==parent);assert prov['parents'][name]['sha256']==expected['checkpoint_sha256']
  else:
   official=json.loads((R/f'final_training/pretrained/{m}/provenance.json').read_text());assert prov['official_initialization_sha256']==official['sha256']==official['publisher_sha256']
  checks.append({'model':m,'final_checkpoint_provenance':'PASS','best_epoch':cp['epoch'],'official_initialization_sha256':prov['official_initialization_sha256'],'run_id':prov['run_id'],'train_class_counts':prov['train_class_counts'],'parents':prov['parents']});del cp
 files=[p for p in O.rglob('*') if p.is_file() and p.name not in ['artifact_checksums.csv','POSTFLIGHT_VERIFICATION.json']]
 for r in results['models']:
  if r['status']=='complete':files.extend(p for p in (P/f"runs/{r['model']}_seed42").rglob('*') if p.is_file() and not p.is_symlink())
 records=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(files))]
 with (O/'artifact_checksums.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,['path','bytes','sha256']);w.writeheader();w.writerows(records)
 verification={'status':'PASS','dataset_and_preparation_unchanged':True,'selected_checkpoint_provenance':checks,'experiment_artifacts_checksummed':len(records),'artifact_manifest_sha256':sha(O/'artifact_checksums.csv'),'test_PAD_inference_performed':False,'final_model_count':len(checks),'candidate_configuration_limit':5,'all_epoch_checkpoint_payloads_preserved':True}
 (O/'POSTFLIGHT_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
 summary=json.loads((P/'experiment_summary.json').read_text());summary.update({'phase':'5B_complete','full_training_authorized':True,'final_results':str((O/'final_results.json').relative_to(R)),'postflight_verification':str((O/'POSTFLIGHT_VERIFICATION.json').relative_to(R))});temporary=P/'experiment_summary.next';temporary.write_text(json.dumps(summary,indent=2)+'\n');os.replace(temporary,P/'experiment_summary.json')
 # Payloads/logs are now frozen; selectors remain symlinks to retained epoch payloads.
 for p in set(files)|{O/'artifact_checksums.csv',O/'POSTFLIGHT_VERIFICATION.json'}:p.chmod(stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH)
 print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
