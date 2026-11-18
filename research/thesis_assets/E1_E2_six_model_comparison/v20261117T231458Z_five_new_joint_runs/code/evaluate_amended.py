from common import *
import base_engine as b
from phase6_scoring import primary
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
import torch,timm,time,gc,collections
KEYS=['M1','M2','M3','M4','M5','M5-LR'];METRICS=['accuracy','macro_f1','weighted_f1','balanced_accuracy','macro_precision','macro_recall','MEL_recall','SCC_recall']
NAMES={'M1':'MobileNetV3-Small','M2':'EfficientNet-Lite0','M3':'MobileNetV4-Conv-Small','M4':'ConvNeXt-Tiny','M5':'MobileSkinXAI-Enhanced (MobileNetV4)','M5-LR':'MobileSkinXAI-Enhanced LR Control (MobileNetV4)'}
BASE=ROOT/'research/thesis_assets/E1_E2_joint_domain/v20261117T221855Z_supervised_joint_seed42';PH=ROOT/'research/thesis_assets/final_test_results/v20261117T103000Z_phase6_final'
def cmstats(cm):
 tp=np.diagonal(cm,axis1=-2,axis2=-1);sup=cm.sum(-1);pred=cm.sum(-2);pr=np.divide(tp,pred,out=np.zeros_like(tp,dtype=float),where=pred!=0);re=np.divide(tp,sup,out=np.zeros_like(tp,dtype=float),where=sup!=0);f=np.divide(2*pr*re,pr+re,out=np.zeros_like(tp,dtype=float),where=pr+re!=0);return np.stack([tp.sum(-1)/cm.sum((-2,-1)),f.mean(-1),(f*sup).sum(-1)/sup.sum(-1),re.mean(-1),pr.mean(-1),re.mean(-1),re[...,1],re[...,3]],axis=-1)
def state(out,key):return load(BASE/'03_E2_Training/TRAINING_COMPLETE.json') if key=='M3' else load(out/'03_E2_Training'/key/'TRAINING_COMPLETE.json')
def cfg_for(out,key):return load(BASE/'00_Protocol/E2_CONFIG.json') if key=='M3' else load(out/'03_E2_Training'/key/'resolved_configuration.json')
def frozen(out):
 out=Path(out);assert load(out/'ALL_FIVE_NEW_COMPLETE.json')['status']=='PASS';assert load(out/'00_Protocol/TEACHER_QUALIFICATION.json')['qualified'] is False;assert load(out/'00_Protocol/AMENDED_TEACHER_AUTHORIZATION.json')['permitted_exploratory_use'] is True;models=load(out/'00_Protocol/E1_MODELS.json');records=[];costs=[];features={}
 release_files=0
 assert load(RELEASE/'data_approval.json')['approved'] is True
 for line in (RELEASE/'checksums.sha256').read_text().splitlines():
  digest,name=line.split(maxsplit=1);assert sha(RELEASE/name.strip())==digest,name;release_files+=1
 inputs={}
 for ds,name,expected in [('ISIC','ISIC_TEST_MANIFEST.csv',2703),('PAD','PAD_test.csv',261)]:
  rs=read(out/'01_Frozen_Data_Manifests'/name);assert len(rs)==expected and len({r['image'] for r in rs})==expected
  for r in rs:
   assert r['dataset_version']=='v20261117T060846Z_608314c8' and CLASSES[int(r['label_index'])]==r['label'];assert sha(ROOT/r['processed_path'])==r['processed_sha256'];assert sha(ROOT/r['source'])==r['source_sha256']
  if ds=='PAD':assert len({r['patient_id'] for r in rs})==126 and all(r['patient_id'] for r in rs)
  inputs[ds]={'images':len(rs),'source_and_prepared_hashes_verified':True,'manifest_sha256':sha(out/'01_Frozen_Data_Manifests'/name)}
 dump(out/'00_Protocol/FINAL_PRE_EVALUATION_INTEGRITY.json',{'status':'PASS','release_files_checked':release_files,'evaluation_inputs':inputs,'checkpoint_checks':'strict loads and finite weights for all twelve performed below before lock','no_test_prediction_access':True})
 for key in KEYS:
  status=state(out,key);cfg=cfg_for(out,key);assert status['status'] in ['complete','recovered_prespecified_policy'] and sha(ROOT/status['best_checkpoint'])==status['best_checkpoint_sha256'];assert sha(ROOT/models[key]['checkpoint'])==models[key]['checkpoint_sha256'];net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=4).eval();mac=[0];handles=[]
  assert cfg['classes']==CLASSES==models[key]['config']['classes'];assert cfg['architecture']==models[key]['config']['architecture'];assert cfg['normalization']==models[key]['config']['normalization'];assert sha(ROOT/models[key]['configuration'])==models[key]['configuration_sha256']
  for condition,path,expected_epoch in [('E1',ROOT/models[key]['checkpoint'],models[key]['checkpoint_epoch']),('E2',ROOT/status['best_checkpoint'],status['best_epoch'])]:
   payload=torch.load(path,map_location='cpu',weights_only=True);assert payload['epoch']==expected_epoch and payload['provenance']['classes']==CLASSES;net.load_state_dict(payload['state_dict'],strict=True);assert all(torch.isfinite(v).all() for v in payload['state_dict'].values() if v.is_floating_point())
   if condition=='E2':assert payload['config']==cfg
   del payload
  def hook(module,inp,v):mac[0]+=v.numel()*((module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1] if isinstance(module,torch.nn.Conv2d) else module.in_features)
  for module in net.modules():
   if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):handles.append(module.register_forward_hook(hook))
  torch.set_num_threads(4)
  with torch.no_grad():assert net(torch.zeros(1,3,224,224)).shape==(1,4)
  for h in handles:h.remove()
  feature_info=net.feature_info;feature=(feature_info[-1] if isinstance(feature_info,list) else feature_info.get_dicts()[-1])['module'];shape=[];h=net.get_submodule(feature).register_forward_hook(lambda m,i,o:shape.extend(o.shape))
  with torch.no_grad():net(torch.zeros(1,3,224,224))
  h.remove();assert len(shape)==4 and min(shape[-2:])>1;features[key]={'layer':feature,'spatial_shape':shape};params=sum(p.numel() for p in net.parameters());del net
  records.append({'configuration':key,'name':NAMES[key],'E1_checkpoint':models[key]['checkpoint'],'E1_sha256':models[key]['checkpoint_sha256'],'E2_checkpoint':status['best_checkpoint'],'E2_sha256':status['best_checkpoint_sha256'],'E2_configuration':cfg,'best_epoch':status['best_epoch'],'PAD_validation_lesion_macro_F1':status['best_PAD_validation_lesion_F1']});costs.append({'configuration':key,'model_name':NAMES[key],'parameters':params,'ConvLinear_MACs':mac[0],'E2_training_seconds':status['training_seconds'],'E2_prescribed_training_seconds':status.get('prescribed_training_seconds',status['training_seconds']),'E2_actual_last_completed_epoch':status.get('actual_last_completed_epoch',status['epochs']),'E2_stop_reason':status.get('stop_reason','original completed run'),'E2_PAD_validation_lesion_macro_F1':status['best_PAD_validation_lesion_F1'],'E2_best_epoch':status['best_epoch'],'E2_total_epochs':status['epochs'],'E2_peak_allocated_GiB':status['peak_allocated_GiB'],'E2_checkpoint':status['best_checkpoint'],'E2_checkpoint_sha256':status['best_checkpoint_sha256'],'E2_checkpoint_bytes':(ROOT/status['best_checkpoint']).stat().st_size,'student_parent_seconds':state(out,'M3')['training_seconds'] if cfg.get('enhanced') else 0.,'teacher_parent_seconds':state(out,'M4')['training_seconds'] if cfg.get('enhanced') else 0.,'inference_architecture':cfg['architecture'],'control_status':'LR sensitivity, not independent architecture' if key=='M5-LR' else ('training-only enhancement; MobileNetV4 graph' if key=='M5' else 'standard backbone')})
 eligible=[r for r in costs if r['configuration']!='M4'];winner=sorted(eligible,key=lambda r:(-state(out,r['configuration'])['best_PAD_validation_lesion_F1'],r['parameters'],r['ConvLinear_MACs'],r['configuration']))[0]['configuration'];write(out/'MODEL_COMPLEXITY.csv',costs);write(out/'E2_TRAINING_SUMMARY.csv',costs);dump(out/'00_Protocol/SPATIAL_FEATURES.json',features)
 lock={'class_order':CLASSES,'input_shape':[3,224,224],'decision_rule':'argmax of uncalibrated raw softmax means over frozen capture representatives; no threshold fitting','metrics':['accuracy','macro_f1','weighted_f1','balanced_accuracy','macro_precision','macro_recall','classwise_precision_recall_F1_specificity','macro_and_class_AUROC','macro_and_class_average_precision_AP','NLL','Brier_sum','ECE10'],'bootstrap':{'valid_replicates':2000,'method':'paired percentile cluster bootstrap; shared draws across all frozen models','PAD_cluster':'patient','ISIC_cluster':'declared lesion; patient metadata unavailable','stream_seeds':{'ISIC':42,'PAD':43},'absent_class_draws':'reject and count, no undefined-class imputation','metrics_with_intervals':METRICS,'interpretation':'conditional on fixed checkpoints; not training-seed uncertainty or clinical confirmation'},'historical_full_PAD':'descriptive reference only; never enters matched E2-minus-E1 effect','teacher_protocol_amendment_id':load(out/'00_Protocol/AMENDED_TEACHER_AUTHORIZATION.json')['amendment_id'],'original_teacher_qualified':False,'teacher_original_qualification_sha256':sha(out/'00_Protocol/TEACHER_QUALIFICATION.json'),'amendment_sha256':sha(out/'13_Reports/TEACHER_PROTOCOL_AMENDMENT.md'),'created_UTC':datetime.now(timezone.utc).isoformat(),'models':records,'deployment_development_candidate':winner,'selection':'predeclared best PAD validation lesion F1 among lightweight models; no test result used','PAD_test_manifest_sha256':sha(out/'01_Frozen_Data_Manifests/PAD_test.csv'),'ISIC_test_manifest_sha256':sha(out/'01_Frozen_Data_Manifests/ISIC_TEST_MANIFEST.csv'),'normalization':'model-specific original config','scoring':'unmodified Phase6 frozen capture representatives, per-lesion arithmetic softmax means','PAD_status':'POST-HOC EXPLORATORY','inference_code_sha256':sha(Path(__file__))}
 path=out/'00_Protocol/FINAL_EVALUATION_LOCK.json';assert not path.exists();dump(path,lock);path.chmod(0o444);print('ALL E2 SELECTIONS FROZEN; DEVELOPMENT CANDIDATE',winner,flush=True)
def infer(out):
 out=Path(out);lock=load(out/'00_Protocol/FINAL_EVALUATION_LOCK.json');assert sha(Path(__file__))==lock['inference_code_sha256'];assert sha(out/'01_Frozen_Data_Manifests/PAD_test.csv')==lock['PAD_test_manifest_sha256'];assert sha(out/'01_Frozen_Data_Manifests/ISIC_TEST_MANIFEST.csv')==lock['ISIC_test_manifest_sha256'];sets={'ISIC':[dict(r,domain='ISIC',study_split='test') for r in read(out/'01_Frozen_Data_Manifests/ISIC_TEST_MANIFEST.csv')],'PAD':[dict(r,domain='PAD') for r in read(out/'01_Frozen_Data_Manifests/PAD_test.csv')]};assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();torch.set_num_threads(4);seed_all(42)
 for m in lock['models']:
  key=m['configuration'];assert sha(ROOT/m['E1_checkpoint'])==m['E1_sha256'] and sha(ROOT/m['E2_checkpoint'])==m['E2_sha256'];net=None
  for ds,rows in sets.items():
   for condition,folder in [('E1','02_E1_Evaluations'),('E2','04_E2_Evaluations')]:
    path=out/f'{folder}/{key}_{ds}_image.csv';side=path.with_suffix('.receipt.json')
    if side.exists():assert sha(path)==load(side)['sha256'];continue
    assert not path.exists() and not path.with_suffix('.pending.csv').exists(),'Partial inference preserved; investigate before replay'
    if condition=='E1' or key=='M3':
     source=(BASE/f'{"02_E1_Frozen_Evaluation" if condition=="E1" else "04_E2_Evaluation"}/{condition}_{ds}_image.csv') if key=='M3' else PH/f'independent_prediction_records/{key}_{"test" if ds=="ISIC" else "external"}_image_probabilities.csv';cached={r['image']:r for r in read(source)};records=[]
     for r in rows:
      q=cached[r['image']];assert q['checkpoint_sha256']==m[condition+'_sha256'] and q['processed_sha256']==r['processed_sha256'] and q['true_class']==r['label'];records.append({**r,**{k:q[k] for k in CLASSES+['true_class','true_index','predicted_class','predicted_index']},'model':key,'model_name':NAMES[key],'condition':condition,'split':'test' if ds=='ISIC' else 'external','checkpoint_sha256':m[condition+'_sha256'],'probability_type':'unchanged cached raw softmax'})
     write(path,records);dump(side,{'sha256':sha(path),'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'checkpoint_sha256':m[condition+'_sha256'],'reused':True});continue
    if net is None:
     cfg=m['E2_configuration'];net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=4);s=torch.load(ROOT/m['E2_checkpoint'],map_location='cpu',weights_only=True);net.load_state_dict(s['state_dict'],strict=True);del s;net=net.cuda().eval().requires_grad_(False)
    loader=DataLoader(b.Images(rows,m['E2_configuration']),batch_size=16,shuffle=False,num_workers=4,pin_memory=True);records=[];offset=0;start=time.perf_counter()
    with torch.inference_mode():
     for x,y,ids,meta in loader:
      with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x.cuda(non_blocking=True))
      p=logits.float().softmax(1).cpu().numpy();assert np.isfinite(p).all()
      for prob in p:
       r=rows[offset];pred=int(prob.argmax());records.append({**r,'model':key,'model_name':NAMES[key],'condition':condition,'split':'test' if ds=='ISIC' else 'external','true_class':r['label'],'true_index':int(r['label_index']),'predicted_class':CLASSES[pred],'predicted_index':pred,**dict(zip(CLASSES,map(float,prob))),'checkpoint_sha256':m['E2_sha256'],'probability_type':'raw uncalibrated softmax'});offset+=1
    assert offset==len(rows);pending=path.with_suffix('.pending.csv');write(pending,records);pending.replace(path);dump(side,{'sha256':sha(path),'checkpoint_sha256':m['E2_sha256'],'images':offset,'seconds':time.perf_counter()-start,'reused':False});print('INFERRED',key,ds,offset,flush=True)
  if net is not None:del net;gc.collect();torch.cuda.empty_cache()
 dump(out/'04_E2_Evaluations/INFERENCE_COMPLETE.json',{'status':'PASS','E1_models':6,'E2_models':6,'new_inference_sets':10,'reused_M3_E2_sets':2,'E1_retrained':False,'test_selection':False})
def analyze(out):
 out=Path(out);assert load(out/'04_E2_Evaluations/INFERENCE_COMPLETE.json')['status']=='PASS';summary=[];cw=[];cms=[];curves=[];reliability=[];boot={};rejects={};order=[(c,k) for c in ['E1','E2'] for k in KEYS]
 for ds in ['ISIC','PAD']:
  for level in ['lesion_primary','image_secondary']:
   groups=None;arrays=[];reference_ids=None
   for condition,key in order:
    folder='02_E1_Evaluations' if condition=='E1' else '04_E2_Evaluations';path=out/f'{folder}/{key}_{ds}_image.csv';assert sha(path)==load(path.with_suffix('.receipt.json'))['sha256'];rows=read(path)
    if level=='lesion_primary':rows=primary(rows);write(out/f'{folder}/{key}_{ds}_lesion.csv',rows)
    unitids=[r['lesion_scoring_unit'] if level=='lesion_primary' else r['image'] for r in rows];assert reference_ids is None or unitids==reference_ids;reference_ids=unitids;y=np.array([int(r['true_index']) for r in rows]);p=np.array([[float(r[c]) for c in CLASSES] for r in rows]);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6);pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(4));assert (cm.sum(1)>0).all();base={'model':key,'model_name':NAMES[key],'condition':condition,'dataset':ds,'analysis_level':level};au=[];ap=[]
    for j,cl in enumerate(CLASSES):
     tp=int(cm[j,j]);fn=int(cm[j].sum()-tp);fp=int(cm[:,j].sum()-tp);tn=int(cm.sum()-tp-fn-fp);pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn);f=2*pr*re/(pr+re) if pr+re else 0;a=float(roc_auc_score(y==j,p[:,j]));av=float(average_precision_score(y==j,p[:,j]));au.append(a);ap.append(av);cw.append({**base,'class':cl,'support':tp+fn,'correct':tp,'TP':tp,'FN':fn,'FP':fp,'TN':tn,'precision':pr,'recall':re,'f1':f,'specificity':tn/(tn+fp),'AUROC':a,'AUPRC_AP':av})
     x,z,_=roc_curve(y==j,p[:,j]);curves.extend({**base,'class':cl,'curve':'ROC','x':float(a),'y':float(b)} for a,b in zip(x,z));z,x,_=precision_recall_curve(y==j,p[:,j]);curves.extend({**base,'class':cl,'curve':'PR','x':float(a),'y':float(b)} for a,b in zip(x,z))
    top=p.max(1);correct=pred==y;bi=np.minimum((top*10).astype(int),9);ece=0
    for bidx in range(10):
     use=bi==bidx;n=int(use.sum());a=float(correct[use].mean()) if n else None;c=float(top[use].mean()) if n else None;err=abs(a-c) if n else 0;ece+=n/len(rows)*err;reliability.append({**base,'bin':bidx,'count':n,'accuracy':a,'confidence':c})
    summary.append({**base,'support':len(rows),**dict(zip(METRICS,map(float,cmstats(cm)))),'macro_AUROC':float(np.mean(au)),'macro_AUPRC_AP':float(np.mean(ap)),'NLL':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),'Brier_sum':float(((p-np.eye(4)[y])**2).sum(1).mean()),'ECE10':ece,'NV_to_MEL':int(cm[0,1]),'MEL_to_NV':int(cm[1,0]),'SCC_to_BCC':int(cm[3,2])});cms.extend({**base,'true_class':CLASSES[i],'predicted_class':CLASSES[j],'count':int(cm[i,j]),'row_percent':float(100*cm[i,j]/cm[i].sum())} for i in range(4) for j in range(4));ids=sorted({r['bootstrap_cluster'] for r in rows});assert groups is None or groups==ids;groups=ids;gm={v:i for i,v in enumerate(ids)};arr=np.zeros((len(ids),4,4),int)
    for r,a,b in zip(rows,y,pred):arr[gm[r['bootstrap_cluster']],a,b]+=1
    arrays.append(arr)
   counts=np.stack(arrays);rng=np.random.default_rng(42 if ds=='ISIC' else 43);rep=[];rejected=0
   while len(rep)<2000:
    weights=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups));cm=np.einsum('g,kgij->kij',weights,counts)
    if (cm.sum(-1)==0).any():rejected+=1;assert rejected<10000;continue
    rep.append(cmstats(cm))
   a=np.array(rep);boot[(ds,level)]=a;np.save(out/'12_Source_Data'/f'bootstrap_{ds}_{level}.npy',a);rejects[ds+'_'+level]=rejected
   for i,(condition,key) in enumerate(order):
    row=next(r for r in summary if r['model']==key and r['condition']==condition and r['dataset']==ds and r['analysis_level']==level)
    for j,m in enumerate(METRICS):row[m+'_CI_low'],row[m+'_CI_high']=map(float,np.percentile(a[:,i,j],[2.5,97.5]))
   print('METRICS',ds,level,flush=True)
 paired=[];gaps=[];side=[];improvements=[]
 def get(k,c,d,l='lesion_primary'):return next(r for r in summary if r['model']==k and r['condition']==c and r['dataset']==d and r['analysis_level']==l)
 for key in KEYS:
  for level in ['lesion_primary','image_secondary']:
   for ds in ['ISIC','PAD']:
    e1,e2=get(key,'E1',ds,level),get(key,'E2',ds,level);i1=order.index(('E1',key));i2=order.index(('E2',key))
    for j,m in enumerate(METRICS):lo,hi=np.percentile(boot[(ds,level)][:,i2,j]-boot[(ds,level)][:,i1,j],[2.5,97.5]);paired.append({'model':key,'dataset':ds,'analysis_level':level,'metric':m,'E2_minus_E1':e2[m]-e1[m],'CI_low':float(lo),'CI_high':float(hi)})
   e1i,e1p,e2i,e2p=[get(key,c,d,level) for c,d in [('E1','ISIC'),('E1','PAD'),('E2','ISIC'),('E2','PAD')]];gain=next(r for r in paired if r['model']==key and r['dataset']=='PAD' and r['analysis_level']==level and r['metric']=='macro_f1');gaps.append({'model':key,'model_name':NAMES[key],'analysis_level':level,'E1_ISIC_macro_f1':e1i['macro_f1'],'E1_PAD_macro_f1':e1p['macro_f1'],'E1_zero_shot_gap':e1i['macro_f1']-e1p['macro_f1'],'E2_ISIC_macro_f1':e2i['macro_f1'],'E2_PAD_macro_f1':e2p['macro_f1'],'E2_supervised_performance_difference':e2i['macro_f1']-e2p['macro_f1'],'PAD_macro_f1_gain':e2p['macro_f1']-e1p['macro_f1'],'PAD_gain_CI_low':gain['CI_low'],'PAD_gain_CI_high':gain['CI_high'],'ISIC_macro_f1_change':e2i['macro_f1']-e1i['macro_f1']})
   improvements.append({'model':key,'model_name':NAMES[key],'analysis_level':level,**{'PAD_'+m+'_change':e2p[m]-e1p[m] for m in ['accuracy','macro_f1','balanced_accuracy','MEL_recall','SCC_recall']},'PAD_macro_f1_CI_low':gain['CI_low'],'PAD_macro_f1_CI_high':gain['CI_high'],'ISIC_macro_f1_change':e2i['macro_f1']-e1i['macro_f1']})
  for metric in METRICS+['NV_recall','BCC_recall','macro_AUROC','macro_AUPRC_AP']:
   row={'Model':NAMES[key],'configuration':key,'Metric':metric}
   for condition,ds,label in [('E1','ISIC','E1 ISIC -> ISIC'),('E1','PAD','E1 ISIC -> PAD'),('E2','ISIC','E2 ISIC+PAD -> ISIC'),('E2','PAD','E2 ISIC+PAD -> PAD')]:row[label]=next(r for r in cw if r['model']==key and r['condition']==condition and r['dataset']==ds and r['analysis_level']=='lesion_primary' and r['class']==metric.split('_')[0])['recall'] if metric in ['NV_recall','BCC_recall'] else get(key,condition,ds)[metric]
   side.append(row)
 enhancement_pairs=[]
 for a,b in [('M5','M3'),('M5-LR','M5')]:
  for ds in ['ISIC','PAD']:
   for level in ['lesion_primary','image_secondary']:
    ia=order.index(('E2',a));ib=order.index(('E2',b))
    for j,metric in enumerate(METRICS):
     delta=get(a,'E2',ds,level)[metric]-get(b,'E2',ds,level)[metric];lo,hi=np.percentile(boot[(ds,level)][:,ia,j]-boot[(ds,level)][:,ib,j],[2.5,97.5]);enhancement_pairs.append({'comparison':a+' minus '+b,'dataset':ds,'analysis_level':level,'metric':metric,'difference':delta,'CI_low':float(lo),'CI_high':float(hi),'interpretation':'secondary exploratory fixed-model comparison; initialization/loss/budget confounding acknowledged'})
 write(out/'12_Source_Data/ENHANCEMENT_PAIRED_COMPARISONS.csv',enhancement_pairs)
 # Reproduce all original completed M3 values without rerunning that checkpoint.
 previous=read(BASE/'05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv')
 for r in previous:
  now=get('M3',r['model'],r['dataset'],r['analysis_level'])
  for m in METRICS+['macro_AUROC','macro_AUPRC_AP']:assert abs(now[m]-float(r[m]))<1e-12,('M3 reproduction',r,m)
 for name,data in [('12_Source_Data/ALL_SETTING_METRICS.csv',summary),('CLASSWISE_RESULTS.csv',cw),('06_Confusion_Matrices/CONFUSION_COUNTS.csv',cms),('12_Source_Data/ROC_PR_POINTS.csv',curves),('12_Source_Data/RELIABILITY_BINS.csv',reliability),('12_Source_Data/PAIRED_DIFFERENCES.csv',paired),('SOURCE_TARGET_GAP_COMPARISON.csv',gaps),('PAD_IMPROVEMENTS_BY_MODEL.csv',improvements),('SIX_MODEL_SIDE_BY_SIDE_RESULTS.csv',side)]:write(out/name,data)
 historical=[r for r in read(PH/'final_model_comparison.csv') if r['split']=='external'];write(out/'12_Source_Data/HISTORICAL_FULL_PAD_REFERENCE.csv',historical);dump(out/'13_Reports/METRICS_COMPLETE.json',{'status':'PASS','primary_PAD_lesions':205,'PAD_patients':126,'PAD_melanoma_lesions':7,'valid_bootstrap_replicates':2000,'absent_class_candidates_rejected':rejects,'M3_original_results_reproduced':True,'PAD_status':'POST-HOC EXPLORATORY'});print('SIX-MODEL METRICS COMPLETE',flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','infer','analyze']);p.add_argument('output');a=p.parse_args();{'freeze':frozen,'infer':infer,'analyze':analyze}[a.mode](ROOT/a.output)
