from common import *
from collections import Counter,defaultdict
import shutil
out=ROOT/'research/thesis_assets/journal_development'/('v'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_CA_MACL')
out.mkdir(parents=True)
for d in ['configs','code','source_data','reports','figures','conversion','runs']:(out/d).mkdir()
assert sha(RELEASE/'checksums.sha256')==load(RELEASE/'RELEASE_LOCK.json')['checksums_sha256']
for line in (RELEASE/'checksums.sha256').read_text().splitlines():
 h,p=line.split('  ',1);assert sha(RELEASE/p)==h
assert load(RELEASE/'data_approval.json')['approved']
with (RELEASE/'dataset_manifest.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['dataset']=='ISIC2019' and r['split'] in ['train','validation'] and r['release_eligible']=='True']
assert Counter(r['split'] for r in rows)=={'train':11632,'validation':2835}
for field in ['lesion_id','identity_group','risk_component']:
 owner={}
 for r in rows:
  assert r[field];assert owner.setdefault(r[field],r['split'])==r['split']
for r in rows:assert CLASSES[int(r['label_index'])]==r['label'];assert sha(r['source'])==r['source_sha256'];assert sha(ROOT/r['processed_path'])==r['processed_sha256']
write(out/'source_data/approved_train_validation.csv',rows)
counts=[sum(r['split']=='train' and r['label']==c for r in rows) for c in CLASSES];weights=np.clip(np.sqrt(sum(counts)/(4*np.array(counts))),.5,3)
dump(out/'source_data/train_class_statistics.json',{'classes':CLASSES,'counts':counts,'weights':weights.tolist(),'formula':'clip(sqrt(N/(4*Nc)),0.5,3.0), train-only; weighted CE has batch weight-sum denominator'})
for i in range(5):
 cfg={'id':f'C{i}','method_name':'CA-MACL (working name; novelty unestablished)','architecture':ARCH,'classes':CLASSES,'seed':42,'pretrained':'fresh in-memory load of immutable official ImageNet artifact; NO previous lesion checkpoint','dataset_release':RELEASE.name,'development_manifest_sha256':sha(out/'source_data/approved_train_validation.csv'),'input':[3,224,224],'preprocessing':{'treatment':'RGB','alternatives':['CC','CLAHE'],'normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},'resize':'EXIF transpose, RGB, Lanczos letterbox gray114; version-locked PIL'},'acquisition':i>=1,'fusion':i>=2,'consistency':i>=3,'contrastive':i>=4,'loss':{'lambda_js':.05 if i>=3 else 0.,'lambda_supcon':.02 if i>=4 else 0.,'temperature':.1,'ramp_fine_epochs':5},'batch_size':16,'num_workers':4,'optimizer':'AdamW','weight_decay':1e-4,'frozen_epochs':5,'frozen_head_lr':1e-3,'fine_epochs':35,'head_lr':1e-4,'backbone_lr':1e-5,'scheduler':'fixed stage LR, no adaptive schedule','patience':7,'selection_metric':'clean validation image macro-F1','mixed_precision':'CUDA BF16 required for primary protocol','gradient_clip_norm':1.,'freeze_bn_running_stats':True,'paired_CE_for_all_configs':True,'stress_policy':'eight fixed source-only conditions; final selected checkpoint only; never checkpoint selection','scope':'development only; no held-out or PAD loader; full training requires separate authorization'}
 dump(out/f'configs/C{i}.json',cfg)
record=load(PRETRAIN/'provenance.json');assert sha(PRETRAIN/'model.safetensors')==record['sha256']==record['publisher_sha256']
dump(out/'reports/preflight.json',{'status':'PASS','release_files_hash_verified':True,'raw_and_prepared_ISIC_development_hashes_checked':len(rows)*2,'partition_counts':dict(Counter(r['split'] for r in rows)),'known_group_intersections':0,'pretrained':record,'heldout_images_or_predictions_read':False,'PAD_outcomes_read':False,'HIBA_accessed':False,'full_training_started':False,'patient_independence_limit':'Global patient identifiers absent for ISIC; known lesions/identity groups only'})
dump(out/'TRAINING_AUTHORIZATION.json',{'authorized':False,'reason':'Current task authorizes implementation and bounded smoke checks only'})
print(out.relative_to(ROOT),flush=True)
