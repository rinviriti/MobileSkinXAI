from pathlib import Path
import csv,json,hashlib,shutil,ast
import torch,timm,numpy as np
R=Path.cwd();O=R/'research/thesis_assets/E1_E2_six_model_comparison/v20261117T231458Z_five_new_joint_runs';B=R/'research/thesis_assets/E1_E2_joint_domain/v20261117T221855Z_supervised_joint_seed42';P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9';PH=R/'research/thesis_assets/final_test_results/v20261117T103000Z_phase6_final'
def read(p):return list(csv.DictReader(p.open()))
def load(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
for d in ['00_Protocol','01_Frozen_Data_Manifests','02_E1_Evaluations','03_E2_Training','04_E2_Evaluations','05_Four_Setting_Comparisons','06_Confusion_Matrices','07_Classwise_Analysis','08_Training_Curves','09_Explainability','10_Model_Complexity','11_Publication_Figures','12_Source_Data','13_Reports']:(O/d).mkdir(exist_ok=True)
assert not (O/'00_Protocol/CONFIGURATIONS.json').exists();preserved={}
for root in [B,PH,R/'research/thesis_assets/final_training_results/v20261117T091430Z_704ed3cc']:
 for line in (root/'PACKAGE_CHECKSUMS.sha256').read_text().splitlines():h,name=line.split('  ',1);assert sha(root/name)==h,(root,name);preserved[str((root/name).relative_to(R))]=h
base=load(B/'00_Protocol/E2_CONFIG.json');m3=load(B/'03_E2_Training/TRAINING_COMPLETE.json');assert m3['epochs']==13 and m3['best_epoch']==7;assert sha(R/m3['best_checkpoint'])==m3['best_checkpoint_sha256'];assert abs(m3['training_seconds']-335.24892886999805)<1e-8
for metric,expected in [('accuracy',[.8522,.6049,.8410,.8000]),('macro_f1',[.6853,.4032,.6121,.7100])]:
 row=next(r for r in read(B/'E1_E2_SIDE_BY_SIDE_RESULTS.csv') if r['Metric']==metric);assert [round(float(v),4) for k,v in row.items() if k!='Metric']==expected
for name in ['PAD_SPLIT.csv','PAD_train.csv','PAD_validation.csv','PAD_test.csv','SPLIT_COUNTS.csv']:
 source=B/'01_PAD_Splits'/name;assert sha(source)==load(B/'00_Protocol/SPLIT_LOCK.json')['manifests']['01_PAD_Splits/'+name];shutil.copyfile(source,O/'01_Frozen_Data_Manifests'/name)
for name in ['DEVELOPMENT_MANIFEST.csv','ISIC_TEST_MANIFEST.csv']:shutil.copyfile(B/'11_Source_Data'/name,O/'01_Frozen_Data_Manifests'/name)
rows=read(O/'01_Frozen_Data_Manifests/DEVELOPMENT_MANIFEST.csv');pad=read(O/'01_Frozen_Data_Manifests/PAD_SPLIT.csv');assert len(pad)==1303 and [sum(r['study_split']==s for r in pad) for s in ['train','validation','test']]==[781,261,261];assert {r['study_split'] for r in pad if r['patient_id']==__import__('os').environ.get('MSX_PRIVATE_VALIDATION_GROUP', '__PRIVATE_GROUP_REQUIRED__')}=={'validation'}
for r in rows+read(O/'01_Frozen_Data_Manifests/PAD_test.csv')+read(O/'01_Frozen_Data_Manifests/ISIC_TEST_MANIFEST.csv'):assert sha(R/r['processed_path'])==r['processed_sha256'] and sha(Path(r['source']))==r['source_sha256']
models=load(PH/'MODEL_SELECTION_LOCK.json')['models'];E1={}
for k,m in models.items():
 assert sha(R/m['checkpoint'])==m['checkpoint_sha256'];s=torch.load(R/m['checkpoint'],map_location='cpu',weights_only=True);assert s['provenance']['artifact_kind']=='final_training_checkpoint' and s['provenance']['classes']==base['classes'];net=timm.create_model(m['config']['architecture'],pretrained=False,num_classes=4);net.load_state_dict(s['state_dict'],strict=True);E1[k]=m;del s,net
configs={}
for k in ['M1','M2','M4']:
 c={**base,'id':k,'name':models[k]['name'],'architecture':models[k]['config']['architecture'],'normalization':models[k]['config']['normalization'],'pretrained_artifact':f'final_training/pretrained/{k}/model.safetensors','enhanced':False};configs[k]=c
 record=load(R/f'final_training/pretrained/{k}/provenance.json');assert sha(R/c['pretrained_artifact'])==record['sha256']==record['publisher_sha256'];assert record['repo_id']=='timm/'+c['architecture']
original=load(P/'configs/M5.json');control=load(R/'final_training/m5_lr_control/v20261117T075145Z_df7c70a4/configs/M5.json');assert {k:v for k,v in original.items() if k not in ['head_lr','backbone_lr','checkpoint_dir']}=={k:v for k,v in control.items() if k not in ['head_lr','backbone_lr','checkpoint_dir']}
training=[r for r in rows if r['study_split']=='train'];weights=[len(training)/(4*sum(r['label']==cl for r in training)) for cl in base['classes']]
for k,orig in [('M5',original),('M5-LR',control)]:configs[k]={**base,'id':k,'name':models[k]['name'],'architecture':original['architecture'],'normalization':original['normalization'],'enhanced':True,'head_lr':orig['head_lr'],'backbone_lr':orig['backbone_lr'],'frozen_epochs':0,'fine_epochs':40,'loss':original['loss'],'augmentation':original['augmentation'],'class_weights':weights,'weighting':'original inverse frequency N_joint/(4*n_joint_class), training labels only; differs from standard E2 mixture-moderated weights','student_start':m3['best_checkpoint'],'student_sha256':m3['best_checkpoint_sha256'],'teacher_parent':'new M4 E2 best PAD-validation checkpoint; freeze/hash before M5','teacher_qualification':original['teacher_qualification'],'budget_kind':'up to40additional epochs from completed joint M3'}
assert {k:v for k,v in configs['M5'].items() if k not in ['id','name','head_lr','backbone_lr']}=={k:v for k,v in configs['M5-LR'].items() if k not in ['id','name','head_lr','backbone_lr']}
for name in ['common.py','pipeline.py']:shutil.copyfile(B/'code'/name,O/'code'/name)
shutil.copyfile(B/'code/engine.py',O/'code/base_engine.py');shutil.copyfile(B/'code/phase6_scoring.py',O/'code/phase6_scoring.py')
source=(P/'code/core.py').read_text();tree=ast.parse(source);segments=[ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['js_divergence','m5_loss']];assert len(segments)==2;(O/'code/original_m5_loss.py').write_text('import torch,math\nimport torch.nn.functional as F\n\n'+'\n\n'.join(segments)+'\n')
shutil.copyfile(P/'code/core.py',O/'12_Source_Data/ORIGINAL_FINAL_CORE.py');shutil.copyfile(P/'configs/M5.json',O/'12_Source_Data/ORIGINAL_M5_CONFIG.json');shutil.copyfile(R/'final_training/m5_lr_control/v20261117T075145Z_df7c70a4/configs/M5.json',O/'12_Source_Data/ORIGINAL_M5_LR_CONFIG.json')
dump(O/'00_Protocol/CONFIGURATIONS.json',configs);dump(O/'00_Protocol/E1_MODELS.json',E1);dump(O/'00_Protocol/REUSED_M3_E2.json',m3);dump(O/'12_Source_Data/PRESERVATION_HASHES.json',preserved);dump(O/'00_Protocol/PREFLIGHT.json',{'status':'PASS','E1_checkpoints_verified':6,'reused_M3_E2_verified':True,'M3_metrics_reproduced':True,'same_split_hashes':{p.name:sha(p) for p in (O/'01_Frozen_Data_Manifests').glob('PAD*.csv')},'free_GiB':shutil.disk_usage(O).free/2**30,'new_training_configurations':5,'teacher_condition':'must qualify on PAD development lesion macro-F1 before M5; fail stops enhancement rather than substitute'})
print('PREFLIGHT PASS; all six E1 and completed M3 E2 verified; exact frozen split reused',flush=True)
