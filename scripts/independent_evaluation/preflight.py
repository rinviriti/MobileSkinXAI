from common import *
from collections import Counter,defaultdict
import shutil,torch,timm
from PIL import Image
out=ROOT/sys.argv[1];assert not out.exists();out.mkdir(parents=True)
for d in ['01_Confusion_Matrices','02_Classwise','03_ROC_PR','04_Comparisons','05_Calibration','06_Explainability','07_Confidence_Intervals','11_Source_Data','independent_prediction_records','code']: (out/d).mkdir()
for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,out/'code'/p.name)
checks={}
assert sha(RELEASE/'checksums.sha256')==load(RELEASE/'RELEASE_LOCK.json')['checksums_sha256']
checks['release_files']=check_list(RELEASE,'checksums.sha256')
assert load(RELEASE/'data_approval.json')['approved']
checks['training_package_files']=check_list(PACKAGE,'PACKAGE_CHECKSUMS.sha256')
assert len(read(PACKAGE/'FIGURE_INDEX.csv'))==104
checks['training_figures_preserved']=104
manifest=read(RELEASE/'dataset_manifest.csv');eligible=[r for r in manifest if r['release_eligible']=='True']
for r in manifest:
 assert r['label'] in CLASSES and int(r['label_index'])==CLASSES.index(r['label'])
 assert sha(ROOT/r['processed_path'])==r['processed_sha256'],r['image']
 assert sha(r['source'])==r['source_sha256'],r['source']
checks['raw_and_prepared_image_hashes']=len(manifest)*2
print('ALL DATA AND TRAINING PACKAGE HASHES PASS',flush=True)
for field in ['identity_group','risk_component','oriented_pixel_sha256','source_sha256','lesion_id']:
 groups=defaultdict(set)
 for r in eligible:
  if r[field]:groups[(r['dataset'] if field=='lesion_id' else '',r[field])].add(r['split'])
 assert not [k for k,v in groups.items() if len(v)>1],field
checks['identity_and_risk_intersections']=0
for r in eligible:
 assert (r['optimization_allowed']=='True')==(r['split']=='train')
 assert (r['model_selection_allowed']=='True')==(r['split']=='validation')
 assert r['calibration_allowed']=='False'
models=model_info()
for k,m in models.items():
 assert sha(ROOT/m['checkpoint'])==m['checkpoint_sha256']
 assert sha(PACKAGE/f'11_Source_Data/checkpoints/{k}_selected_checkpoint.pt')==m['checkpoint_sha256']
 payload=torch.load(ROOT/m['checkpoint'],map_location='cpu',weights_only=True)
 assert payload['provenance']['classes']==CLASSES
 assert payload['provenance']['artifact_kind']=='final_training_checkpoint'
 assert m['config']['classes']==CLASSES and m['config']['dataset_version']==RELEASE.name
 m['provenance']=payload['provenance'];m['checkpoint_epoch']=payload['epoch']
 definition=timm.create_model(m['config']['architecture'],pretrained=False,num_classes=4)
 actual=timm.data.resolve_model_data_config(definition)
 assert all(np.allclose(actual[x],m['config']['normalization'][x]) for x in ['mean','std']);del definition
 if k in ['M5','M5-LR']:
  assert models['M3']['checkpoint_sha256'] in json.dumps(m['provenance']['parents'])
  assert models['M4']['checkpoint_sha256'] in json.dumps(m['provenance']['parents'])
units=read(RELEASE/'evaluation_scoring_units.csv');unitmap={(r['dataset'],r['image']):r for r in units}
counts={};evaluation=[]
for split,expected in [('test',2703),('external',1303)]:
 rows=[r for r in eligible if r['split']==split];assert len(rows)==expected;assert len({(r['dataset'],r['image']) for r in rows})==expected
 counts[split]=dict(Counter(r['label'] for r in rows))
 for i,r in enumerate(rows):
  u=unitmap[(r['dataset'],r['image'])];assert u['split']==split and u['label']==r['label'];assert u['lesion_scoring_unit'] and u['bootstrap_cluster']
  with Image.open(ROOT/r['processed_path']) as im:assert im.mode=='RGB' and im.size==(224,224);im.verify()
  evaluation.append({**r,**u,'evaluation_row':i})
write(out/'11_Source_Data/locked_evaluation_manifest.csv',evaluation)
for name in ['DATASET_PROTOCOL.json','data_approval.json','integrity_checks.json','evaluation_denominators.csv','evaluation_scoring_units.csv','exclusions.csv','unresolved_pair_disposition.csv','DATASET_RELEASE_REPORT.md']:
 shutil.copyfile(RELEASE/name,out/'11_Source_Data'/name)
code=(PREP/'code/core.py').read_text();assert "if split not in ['train', 'validation']" in code
checks['recorded_data_use_review']={'trainer_guard':'only train and validation loaders; two-view training only','configuration':'test_access not loaded by trainer; no calibration permitted','release_flags':'optimization train only; selection validation only','prior_evaluation':'package verification records no heldout evaluation','limitation':'Recorded code and provenance support absence of heldout optimization/selection; cannot establish undocumented behavior. Pre-model image similarity audit and integrity hashing are not model fitting.'}
lock={'title':TITLE,'created_UTC':datetime.now(timezone.utc).isoformat(),'dataset_release':RELEASE.name,'dataset_manifest_sha256':sha(RELEASE/'dataset_manifest.csv'),'scoring_units_sha256':sha(RELEASE/'evaluation_scoring_units.csv'),'evaluation_manifest_sha256':sha(out/'11_Source_Data/locked_evaluation_manifest.csv'),'counts':counts,'models':models,'primary_unit':'declared lesion arithmetic mean softmax over frozen lexicographic exact-capture representatives','secondary_unit':'all eligible images','class_order':CLASSES,'decision':'argmax; fixed class ordering resolves ties; no thresholds','preprocessing':'Existing 224 RGB letterbox gray114; ToTensor then config normalization; no augmentation, resize, TTA or calibration','inference':'CUDA BF16 autocast; original batch sizes 16 except ConvNeXt8; frozen eval mode; deterministic manifest row order','metrics':['accuracy','balanced_accuracy','macro_precision','macro_recall','macro_f1','class_precision','class_recall','class_f1','class_specificity','OvR_AUROC','noninterpolated_average_precision','NLL','multiclass_Brier_sum','ECE','MCE'],'undefined':'zero_division=0 for precision/F1; ROC/AP require both classes, otherwise unavailable','calibration':'raw softmax only; NLL numerical clipping1e-12; Brier sum across4classes; top-label ECE/MCE10 equal-width bins; never fit calibration','bootstrap':{'replicates':2000,'seed':42,'interval':'percentile2.5,97.5','cluster':'ISIC declared lesion; PAD patient; unstratified resample clusters','paired_models':True,'CI_metrics':['accuracy','balanced_accuracy','macro_precision','macro_recall','macro_f1','MEL_recall','SCC_recall'],'other_metrics':'point estimates; ROC/AP and calibration no bootstrap CI','domain_gap':'ISIC minus PAD; independent dataset bootstrap draws; positive means performance drop','paired_comparisons':['M5 minus M3','M5-LR minus M3','M5-LR minus M5']},'GradCAM_selection':'For every model/dataset/true-class/correctness stratum choose image nearest median top1 confidence, tie by image ID; missing strata reported; target predicted logit at final spatial stage; original inference batch context; no optimization','analysis_rules':'Roles fixed. Primary original enhanced vs baseline; report secondary LR control and reference without selection replacement. Descriptive paired CIs; exploratory multiple comparisons, no clinical claim or established novelty. PAD50MEL images and patient-independence limits disclosed.','integrity_checks':checks}
dump(out/'MODEL_SELECTION_LOCK.json',lock);(out/'MODEL_SELECTION_LOCK.json').chmod(0o444)
dump(out/'LOCK_RECEIPT.json',{'sha256':sha(out/'MODEL_SELECTION_LOCK.json'),'created_before_test_predictions':True,'chmod':'0444 owner-reversible; content-hash seal'})
dump(out/'PREFLIGHT_VERIFICATION.json',{'status':'PASS','counts':counts,'checks':checks,'lock_sha256':sha(out/'MODEL_SELECTION_LOCK.json')})
print(str(out),counts,flush=True)
