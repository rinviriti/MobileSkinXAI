from pathlib import Path
import csv,json,hashlib,shutil,sys,collections
import numpy as np,torch,timm
ROOT=Path.cwd();OUT=ROOT/'research/thesis_assets/E1_E2_joint_domain/v20261117T221855Z_supervised_joint_seed42';REL=ROOT/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8';PH6=ROOT/'research/thesis_assets/final_test_results/v20261117T103000Z_phase6_final'
def read(p):return list(csv.DictReader(p.open()))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def write(p,rows):
 with p.open('x',newline='') as f:w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
for folder in ['00_Protocol','01_PAD_Splits','02_E1_Frozen_Evaluation','03_E2_Training','04_E2_Evaluation','05_Cross_Dataset_Comparison','06_Confusion_Matrices','07_ROC_PR','08_Classwise_Analysis','09_Explainability','10_Publication_Figures','11_Source_Data','12_Reports']: (OUT/folder).mkdir(exist_ok=True)
assert not (OUT/'01_PAD_Splits/PAD_SPLIT.csv').exists(),'Split already frozen; never regenerate'
preserved={}
for root in [PH6,ROOT/'research/thesis_assets/final_training_results/v20261117T091430Z_704ed3cc',ROOT/'research/thesis_assets/journal_development/CA_MACL_training/v20261117T132848Z_five_run_seed42',ROOT/'research/thesis_assets/journal_development/CA_MACL_multiseed/v20261117T171618Z_matched_seeds43_44',ROOT/'research/thesis_assets/journal_development/CA_MACL_cross_dataset_evaluation/v20261117T201030Z_locked_nine_checkpoints']:
 for line in (root/'PACKAGE_CHECKSUMS.sha256').read_text().splitlines():h,name=line.split('  ',1);assert sha(root/name)==h;preserved[str((root/name).relative_to(ROOT))]=h
for line in (REL/'checksums.sha256').read_text().splitlines():h,name=line.split('  ',1);assert sha(REL/name)==h;preserved[str((REL/name).relative_to(ROOT))]=h
assert json.loads((REL/'data_approval.json').read_text())['approved'];manifest=read(REL/'dataset_manifest.csv');active=[r for r in manifest if r['release_eligible']=='True'];classes=['NV','MEL','BCC','SCC'];pad=[r for r in active if r['dataset']=='PADUFES20'];assert len(pad)==1303;assert [sum(r['label']==c for r in pad) for c in classes]==[233,50,837,183]
for r in active:
 assert r['label']==classes[int(r['label_index'])] and sha(Path(r['source']))==r['source_sha256'] and sha(ROOT/r['processed_path'])==r['processed_sha256']
# Identity closure over patient groups; known duplicate/risk groups cannot cross new partitions.
patients=sorted({r['patient_id'] for r in pad});assert all(patients);parent={p:p for p in patients}
def find(p):
 while parent[p]!=p:parent[p]=parent[parent[p]];p=parent[p]
 return p
for field in ['identity_group','risk_component','source_sha256','oriented_pixel_sha256']:
 seen={}
 for r in pad:
  token=r[field]
  if token in seen:
   a,b=find(r['patient_id']),find(seen[token]);parent[max(a,b)]=min(a,b)
  else:seen[token]=r['patient_id']
groups=collections.defaultdict(list)
for r in pad:groups[find(r['patient_id'])].append(r)
keys=sorted(groups);vectors=[]
for g in keys:
 rs=groups[g];images=[sum(r['label']==c for r in rs) for c in classes];lesions=[len({(r['patient_id'],r['lesion_id']) for r in rs if r['label']==c}) for c in classes];vectors.append(images+lesions+[len({r['patient_id'] for r in rs}),len(rs)])
v=np.array(vectors,float);tot=v.sum(0);ratios=np.array([.6,.2,.2]);target=ratios[:,None]*tot;scale=np.maximum(target,1);rng=np.random.default_rng(42);assign=np.full(len(keys),-1,int);counts=np.zeros_like(target)
def loss(ct):return float((((ct-target)/scale)**2).sum())
# One deterministic metadata-only allocation plus a fixed number of local refinements; no model scores.
priority=np.max(v[:,:4]/tot[:4],axis=1);order=sorted(range(len(keys)),key=lambda i:(-priority[i],rng.random()))
for i in order:
 choices=[]
 for s in range(3):c=counts.copy();c[s]+=v[i];choices.append(loss(c))
 s=int(np.argmin(choices));assign[i]=s;counts[s]+=v[i]
for _ in range(60000):
 i,j=rng.integers(len(keys),size=2);a,b=assign[i],assign[j]
 if a==b:continue
 c=counts.copy();c[a]+=v[j]-v[i];c[b]+=v[i]-v[j]
 if (c[:,4:8]<1).any():continue
 if loss(c)<loss(counts)-1e-12:assign[i],assign[j]=b,a;counts=c
names=['train','validation','test'];mapping={k:names[int(assign[i])] for i,k in enumerate(keys)};units={(r['dataset'],r['image']):r for r in read(REL/'evaluation_scoring_units.csv')};records=[]
for r in pad:records.append({**r,**units[(r['dataset'],r['image'])],'study_split':mapping[find(r['patient_id'])],'identity_patient_component':find(r['patient_id'])})
assert all(len({r['study_split'] for r in records if r['patient_id']==p})==1 for p in patients)
for field in ['identity_group','risk_component','source_sha256','oriented_pixel_sha256']:
 sets=collections.defaultdict(set)
 for r in records:sets[r[field]].add(r['study_split'])
 assert all(len(s)==1 for s in sets.values()),field
for field in ['source_sha256','oriented_pixel_sha256','identity_group','risk_component']:
 isictest={r[field] for r in active if r['dataset']=='ISIC2019' and r['split'] in ['validation','test']};assert not isictest&{r[field] for r in records if r['study_split']=='train'}
write(OUT/'01_PAD_Splits/PAD_SPLIT.csv',records)
for s in names:write(OUT/f'01_PAD_Splits/PAD_{s}.csv',[r for r in records if r['study_split']==s])
summary=[]
for s in names:
 rs=[r for r in records if r['study_split']==s]
 for c in classes:
  cr=[r for r in rs if r['label']==c];assert cr
  summary.append({'split':s,'class':c,'images':len(cr),'declared_lesions':len({(r['patient_id'],r['lesion_id']) for r in cr}),'patients_with_class':len({r['patient_id'] for r in cr}),'all_partition_patients':len({r['patient_id'] for r in rs})})
write(OUT/'01_PAD_Splits/SPLIT_COUNTS.csv',summary)
joint=[{**r,'study_split':r['split'],'domain':'ISIC'} for r in active if r['dataset']=='ISIC2019' and r['split'] in ['train','validation']]+[{**r,'domain':'PAD'} for r in records if r['study_split'] in ['train','validation']];write(OUT/'11_Source_Data/DEVELOPMENT_MANIFEST.csv',joint)
write(OUT/'11_Source_Data/ISIC_TEST_MANIFEST.csv',[{**r,**units[(r['dataset'],r['image'])]} for r in active if r['dataset']=='ISIC2019' and r['split']=='test'])
m=json.loads((PH6/'MODEL_SELECTION_LOCK.json').read_text())['models']['M3'];assert sha(ROOT/m['checkpoint'])==m['checkpoint_sha256'];state=torch.load(ROOT/m['checkpoint'],map_location='cpu',weights_only=True);assert state['provenance']['artifact_kind']=='final_training_checkpoint' and state['provenance']['classes']==classes;net=timm.create_model(m['config']['architecture'],pretrained=False,num_classes=4);net.load_state_dict(state['state_dict'],strict=True);del net,state
cfg={'id':'E2','architecture':m['config']['architecture'],'classes':classes,'seed':42,'input':[3,224,224],'normalization':m['config']['normalization'],'batch_size':16,'steps_per_epoch':727,'ISIC_per_batch':8,'PAD_per_batch':8,'class_sampling':'none; uniform shuffled image cycles within each domain','weight_decay':1e-4,'frozen_epochs':5,'fine_epochs':30,'frozen_head_lr':1e-3,'head_lr':1e-4,'backbone_lr':1e-5,'patience':6,'checkpoint_metric':'PAD validation declared-lesion macro-F1','freeze_BN_running_stats':True,'gradient_clip':1.,'mixed_precision':'CUDA BF16','inference':'Same original M3 prepared224RGB JPEG ->ToTensor->ImageNet normalization; no raw reprocessing, augmentation or TTA','augmentations':'Training only: H/V0.5, rotationU(-10,10) expand+letterbox; brightness/contrastp0.8 U(.9,1.1); WBp0.5 gainsU(.97,1.03)geomeannormalized; JPEGp0.4quality75-95; blurp0.2radius.1-.6; framingp0.3scale.9-1withoutcrop','weighting':'clip(sqrt(1/(4*pi_c)),.5,2) where pi=.5*n_ISIC_c/N_ISIC+.5*n_PAD_c/N_PAD, training only','pretrained_artifact':'final_training/pretrained/M3/model.safetensors'}
isic=[r for r in joint if r['domain']=='ISIC' and r['study_split']=='train'];ptrain=[r for r in joint if r['domain']=='PAD' and r['study_split']=='train'];pi=np.array([.5*sum(r['label']==c for r in isic)/len(isic)+.5*sum(r['label']==c for r in ptrain)/len(ptrain) for c in classes]);cfg['class_weights']=np.clip(np.sqrt(1/(4*pi)),.5,2).tolist();dump(OUT/'00_Protocol/E2_CONFIG.json',cfg);dump(OUT/'00_Protocol/E1_PROVENANCE.json',m);dump(OUT/'11_Source_Data/PRESERVATION_HASHES.json',preserved)
for name in ['pipeline.py','common.py']:shutil.copyfile(ROOT/'research/thesis_assets/journal_development/CA_MACL_training/v20261117T132848Z_five_run_seed42/code'/name,OUT/'code'/name)
shutil.copyfile(PH6/'code/metrics.py',OUT/'code/phase6_scoring.py')
dump(OUT/'00_Protocol/SPLIT_LOCK.json',{'seed':42,'method':'patient+knownidentity closure; deterministic metadata-only greedy allocation +60000fixed pair swaps; normalized squared class-image/class-lesion/patient/total count deviation','manifests':{str(p.relative_to(OUT)):sha(p) for p in (OUT/'01_PAD_Splits').glob('*.csv')},'[private group]_partition':next(r['study_split'] for r in records if r['patient_id']==__import__('os').environ.get('MSX_PRIVATE_VALIDATION_GROUP', '__PRIVATE_GROUP_REQUIRED__')),'original_records_unchanged':True})
for p in (OUT/'01_PAD_Splits').glob('*.csv'):p.chmod(0o444)
dump(OUT/'00_Protocol/PREFLIGHT.json',{'status':'PASS','ISIC_training':len(isic),'PAD_training':len(ptrain),'PAD_counts':summary,'patient_components':len(groups),'patients':len(patients),'all_eligible_raw_prepared_hashes_checked':len(active)*2,'E1_sha256':m['checkpoint_sha256'],'free_GiB':shutil.disk_usage(OUT).free/2**30})
print('FROZEN SPLIT',summary,'PAT834',next(r['study_split'] for r in records if r['patient_id']==__import__('os').environ.get('MSX_PRIVATE_VALIDATION_GROUP', '__PRIVATE_GROUP_REQUIRED__')),flush=True)
