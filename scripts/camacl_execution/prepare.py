from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib,shutil,sys,ast
ROOT=Path(__file__).resolve().parents[2];APPROVED=ROOT/'research/thesis_assets/journal_development/v20261117T124247Z_CA_MACL'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
for line in (APPROVED/'PACKAGE_CHECKSUMS.sha256').read_text().splitlines():
 h,name=line.split('  ',1);assert sha(APPROVED/name)==h,name
version='v'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_five_run_seed42'
out=ROOT/'research/thesis_assets/journal_development/CA_MACL_training'/version;out.mkdir(parents=True)
folders=['00_Experiment_Protocol','01_Training_Histories','02_Validation_Results','03_Confusion_Matrices','04_ROC_PR','05_Classwise_Performance','06_Ablation_Comparisons','07_Synthetic_Shift_Robustness','08_Model_Complexity','09_Explainability','10_Publication_Figures','11_Source_Data','12_Reports','code','configs','source_data','runs']
for name in folders:(out/name).mkdir()
for p in (APPROVED/'code').glob('*.py'):
 if p.name=='train.py':shutil.copyfile(p,out/'code/approved_train.py')
 elif p.name in ['common.py','model.py','pipeline.py','losses.py']:shutil.copyfile(p,out/'code'/p.name)
for p in (APPROVED/'configs').glob('C?.json'):shutil.copyfile(p,out/'configs'/p.name)
for name in ['approved_train_validation.csv','train_class_statistics.json','official_ImageNet_provenance.json','smoke_training_tensors.pt']:
 shutil.copyfile(APPROVED/'source_data'/name,out/'source_data'/name)
for name in ['ARCHITECTURE_AND_LOSSES.md','TRAINING_PROTOCOL.md','FIVE_EXPERIMENTS.md','PREPROCESSING_AND_AUGMENTATION.md','NOVELTY_OVERLAP_MATRIX.md']:
 shutil.copyfile(APPROVED/name,out/'00_Experiment_Protocol'/('APPROVED_'+name))
for name in ['model_complexity.json','GPU_SMOKE.json','TRAINING_CONTRACT_CHECKS.json','CPU_IMPLEMENTATION_CHECKS.json']:
 shutil.copyfile(APPROVED/'reports'/name,out/'11_Source_Data'/('approved_'+name))
release=ROOT/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8'
assert sha(release/'checksums.sha256')==json.loads((release/'RELEASE_LOCK.json').read_text())['checksums_sha256']
for line in (release/'checksums.sha256').read_text().splitlines():
 h,name=line.split('  ',1);assert sha(release/name)==h
assert json.loads((release/'data_approval.json').read_text())['approved']
with (out/'source_data/approved_train_validation.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==14467
for r in rows:
 assert r['dataset']=='ISIC2019' and r['split'] in ['train','validation'] and r['release_eligible']=='True'
 assert ['NV','MEL','BCC','SCC'][int(r['label_index'])]==r['label']
 assert sha(Path(r['source']))==r['source_sha256'];assert sha(ROOT/r['processed_path'])==r['processed_sha256']
for field in ['lesion_id','identity_group','risk_component']:
 owners={}
 for r in rows:assert owners.setdefault(r[field],r['split'])==r['split']
configs=[json.loads((out/f'configs/C{i}.json').read_text()) for i in range(5)]
for i,cfg in enumerate(configs):
 assert sha(out/f'configs/C{i}.json')==sha(APPROVED/f'configs/C{i}.json')
 assert cfg['development_manifest_sha256']==sha(out/'source_data/approved_train_validation.csv')
 if i:
  prev=configs[i-1];diff={k for k in cfg if cfg[k]!=prev[k]}-{'id'};assert diff<={1:{'acquisition'},2:{'fusion'},3:{'consistency','loss'},4:{'contrastive','loss'}}[i]
# Approved update structure: two forwards, averaged CE, exactly one backward and optimizer.step per batch.
src=(out/'code/approved_train.py').read_text();tree=ast.parse(src);steps=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step'];assert len(steps)==1
free=shutil.disk_usage(out).free;assert free>20*2**30
# Snapshot protected indexes and checkpoint file stats; never read old test metrics/predictions.
protected={}
for p in list((ROOT/'final_training').rglob('*.pt')):
 if p.is_file():protected[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
for base in [ROOT/'research/thesis_assets/final_training_results',ROOT/'research/thesis_assets/final_test_results',ROOT/'research/thesis_assets/cross_dataset_ablation',APPROVED]:
 for p in base.rglob('*'):
  if p.is_file() and p.suffix!='.pyc':protected[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
dump(out/'11_Source_Data/protected_original_stat_snapshot.json',protected)
dump(out/'00_Experiment_Protocol/PREFLIGHT.json',{'status':'PASS','approved_implementation':str(APPROVED.relative_to(ROOT)),'approved_package_checksums_sha256':sha(APPROVED/'PACKAGE_CHECKSUMS.sha256'),'dataset_release':release.name,'raw_prepared_development_images_hash_checked':len(rows)*2,'counts':{s:sum(r['split']==s for r in rows) for s in ['train','validation']},'no_known_group_crossing':True,'five_configs_byte_identical':True,'single_optimizer_step_call_per_batch':True,'expected_updates_per_epoch':727,'expected_max_updates_per_config':29080,'free_storage_GiB':free/2**30,'minimum_free_required_GiB':20,'scope':'ISIC training/development validation only; no PAD or independent ISIC test loader/metrics access'})
dump(out/'TRAINING_AUTHORIZATION.json',{'authorized':True,'source':'User explicitly approved exactly C0–C4 full training in current turn','seed':42,'created_UTC':datetime.now(timezone.utc).isoformat(),'additional_variants_authorized':False})
print(out.relative_to(ROOT),flush=True)
