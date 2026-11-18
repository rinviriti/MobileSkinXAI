from common import *
from engine import Images
from phase6_scoring import primary
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
import torch,timm,time,gc
METRIC_NAMES=['accuracy','macro_f1','weighted_f1','balanced_accuracy','macro_precision','macro_recall','MEL_recall','SCC_recall']
def cmstats(cm):
 tp=np.diagonal(cm,axis1=-2,axis2=-1);sup=cm.sum(-1);pred=cm.sum(-2);pr=np.divide(tp,pred,out=np.zeros_like(tp,dtype=float),where=pred!=0);re=np.divide(tp,sup,out=np.zeros_like(tp,dtype=float),where=sup!=0);f=np.divide(2*pr*re,pr+re,out=np.zeros_like(tp,dtype=float),where=pr+re!=0);return np.stack([tp.sum(-1)/cm.sum((-2,-1)),f.mean(-1),(f*sup).sum(-1)/sup.sum(-1),re.mean(-1),pr.mean(-1),re.mean(-1),re[...,1],re[...,3]],axis=-1)
def cohorts(out):return {'ISIC':[dict(r,domain='ISIC',study_split='test') for r in read(out/'11_Source_Data/ISIC_TEST_MANIFEST.csv')],'PAD':[dict(r,domain='PAD') for r in read(out/'01_PAD_Splits/PAD_test.csv')]}
def infer(out):
 out=Path(out);cfg=load(out/'00_Protocol/E2_CONFIG.json');selection=load(out/'00_Protocol/E2_SELECTION_LOCK.json');assert selection['status']=='complete';assert sha(ROOT/selection['best_checkpoint'])==selection['best_checkpoint_sha256'];e1=load(out/'00_Protocol/E1_PROVENANCE.json');assert sha(ROOT/e1['checkpoint'])==e1['checkpoint_sha256'];sets=cohorts(out);ph6=ROOT/'research/thesis_assets/final_test_results/v20261117T103000Z_phase6_final'
 for ds,rows in sets.items():
  path=out/f'02_E1_Frozen_Evaluation/E1_{ds}_image.csv'
  if path.exists():assert sha(path)==load(path.with_suffix('.receipt.json'))['sha256'];continue
  source=ph6/f'independent_prediction_records/M3_{"test" if ds=="ISIC" else "external"}_image_probabilities.csv';old={r['image']:r for r in read(source)};preds=[]
  for r in rows:
   q=old[r['image']];assert q['checkpoint_sha256']==e1['checkpoint_sha256'] and q['processed_sha256']==r['processed_sha256'];assert q['true_class']==r['label'];preds.append({**r,**{k:q[k] for k in CLASSES+['true_index','true_class','predicted_index','predicted_class']},'model':'E1','model_name':'ISIC-only MobileNetV4-Conv-Small (original frozen M3)','split':'test' if ds=='ISIC' else 'external','checkpoint_sha256':e1['checkpoint_sha256'],'probability_type':'unchanged Phase6 raw uncalibrated softmax','source_prediction_file':str(source.relative_to(ROOT))})
  write(path,preds);dump(path.with_suffix('.receipt.json'),{'sha256':sha(path),'source_sha256':sha(source),'reused_predictions':True,'checkpoint_sha256':e1['checkpoint_sha256']})
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();torch.set_num_threads(4);seed_all(42);net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=4);cp=torch.load(ROOT/selection['best_checkpoint'],map_location='cpu',weights_only=True);net.load_state_dict(cp['state_dict'],strict=True);del cp;net=net.cuda().eval().requires_grad_(False)
 for ds,rows in sets.items():
  path=out/f'04_E2_Evaluation/E2_{ds}_image.csv'
  if path.exists():assert sha(path)==load(path.with_suffix('.receipt.json'))['sha256'];continue
  assert not path.with_suffix('.pending.csv').exists(),'Partial inference retained; investigate before resuming';loader=DataLoader(Images(rows,cfg),batch_size=16,shuffle=False,num_workers=4,pin_memory=True);preds=[];offset=0;start=time.perf_counter()
  with torch.inference_mode():
   for x,y,ids,_ in loader:
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x.cuda(non_blocking=True))
    probs=logits.float().softmax(1).cpu().numpy();assert np.isfinite(probs).all()
    for p in probs:
     r=rows[offset];guess=int(p.argmax());preds.append({**r,'model':'E2','model_name':'Joint ISIC+PAD MobileNetV4-Conv-Small','split':'test' if ds=='ISIC' else 'external','true_index':int(r['label_index']),'true_class':r['label'],'predicted_index':guess,'predicted_class':CLASSES[guess],**dict(zip(CLASSES,map(float,p))),'checkpoint_sha256':selection['best_checkpoint_sha256'],'probability_type':'raw uncalibrated softmax'});offset+=1
  assert offset==len(rows);pending=path.with_suffix('.pending.csv');write(pending,preds);pending.replace(path);dump(path.with_suffix('.receipt.json'),{'sha256':sha(path),'checkpoint_sha256':selection['best_checkpoint_sha256'],'inference_seconds':time.perf_counter()-start,'images':offset});print('FINAL INFERENCE',ds,offset,flush=True)
 dump(out/'04_E2_Evaluation/INFERENCE_COMPLETE.json',{'status':'PASS','E2_checkpoint_locked_before_test':True,'E1_retrained':False,'E2_test_images':{k:len(v) for k,v in sets.items()},'new_training':False})
def analyze(out):
 out=Path(out);assert load(out/'04_E2_Evaluation/INFERENCE_COMPLETE.json')['status']=='PASS';summary=[];classes=[];cms=[];curves=[];bins=[];bootstrap={};rejects={}
 for ds in ['ISIC','PAD']:
  for level in ['lesion_primary','image_secondary']:
   arrays=[];cluster_ids=None
   for key,folder in [('E1','02_E1_Frozen_Evaluation'),('E2','04_E2_Evaluation')]:
    rows=read(out/f'{folder}/{key}_{ds}_image.csv')
    if level=='lesion_primary':rows=primary(rows);write(out/f'{folder}/{key}_{ds}_lesion.csv',rows)
    y=np.array([int(r['true_index']) for r in rows]);p=np.array([[float(r[c]) for c in CLASSES] for r in rows]);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6);pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(4));assert (cm.sum(1)>0).all(),'Absent class requires explicit undefined-metric report';base={'model':key,'dataset':ds,'analysis_level':level};auc=[];aps=[]
    for j,c in enumerate(CLASSES):
     tp=int(cm[j,j]);fn=int(cm[j].sum()-tp);fp=int(cm[:,j].sum()-tp);tn=int(cm.sum()-tp-fn-fp);pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn);f=2*pr*re/(pr+re) if pr+re else 0;a=float(roc_auc_score(y==j,p[:,j]));ap=float(average_precision_score(y==j,p[:,j]));auc.append(a);aps.append(ap);classes.append({**base,'class':c,'support':tp+fn,'correct':tp,'TP':tp,'FN':fn,'FP':fp,'TN':tn,'precision':pr,'recall':re,'f1':f,'specificity':tn/(tn+fp),'AUROC':a,'AUPRC_AP':ap})
     from sklearn.metrics import roc_curve,precision_recall_curve
     x,z,_=roc_curve(y==j,p[:,j]);curves.extend({**base,'class':c,'curve':'ROC','x':float(a),'y':float(b)} for a,b in zip(x,z));z,x,_=precision_recall_curve(y==j,p[:,j]);curves.extend({**base,'class':c,'curve':'PR','x':float(a),'y':float(b)} for a,b in zip(x,z))
    top=p.max(1);correct=pred==y;bi=np.minimum((top*10).astype(int),9);ece=0
    for b in range(10):
     ix=bi==b;n=int(ix.sum());acc=float(correct[ix].mean()) if n else None;conf=float(top[ix].mean()) if n else None;err=abs(acc-conf) if n else 0;ece+=n/len(y)*err;bins.append({**base,'bin':b,'count':n,'accuracy':acc,'confidence':conf})
    summary.append({**base,'support':len(rows),**dict(zip(METRIC_NAMES,map(float,cmstats(cm)))),'macro_AUROC':float(np.mean(auc)),'macro_AUPRC_AP':float(np.mean(aps)),'NLL':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),'Brier_sum':float(((p-np.eye(4)[y])**2).sum(1).mean()),'ECE10':ece,'NV_to_MEL':int(cm[0,1]),'MEL_to_NV':int(cm[1,0]),'SCC_to_BCC':int(cm[3,2])})
    cms.extend({**base,'true_class':CLASSES[i],'predicted_class':CLASSES[j],'count':int(cm[i,j]),'row_percent':float(100*cm[i,j]/cm[i].sum())} for i in range(4) for j in range(4))
    gids=sorted({r['bootstrap_cluster'] for r in rows});assert cluster_ids is None or gids==cluster_ids;cluster_ids=gids;gm={g:i for i,g in enumerate(gids)};a=np.zeros((len(gids),4,4),int)
    for r,truth,guess in zip(rows,y,pred):a[gm[r['bootstrap_cluster']],truth,guess]+=1
    arrays.append(a)
   arrays=np.stack(arrays);rng=np.random.default_rng(42 if ds=='ISIC' else 43);reps=[];rejected=0
   while len(reps)<2000:
    weights=np.bincount(rng.integers(len(cluster_ids),size=len(cluster_ids)),minlength=len(cluster_ids));cm=np.einsum('g,kgij->kij',weights,arrays)
    if (cm.sum(-1)==0).any():rejected+=1;assert rejected<10000;continue
    reps.append(cmstats(cm))
   a=np.array(reps);bootstrap[(ds,level)]=a;np.save(out/'11_Source_Data'/f'bootstrap_{ds}_{level}.npy',a);rejects[ds+'_'+level]=rejected
   for ki,key in enumerate(['E1','E2']):
    r=next(r for r in summary if r['model']==key and r['dataset']==ds and r['analysis_level']==level)
    for j,m in enumerate(METRIC_NAMES):r[m+'_CI_low'],r[m+'_CI_high']=map(float,np.percentile(a[:,ki,j],[2.5,97.5]))
 comparisons=[];gaps=[]
 for level in ['lesion_primary','image_secondary']:
  for ds in ['ISIC','PAD']:
   a=next(r for r in summary if r['model']=='E2' and r['dataset']==ds and r['analysis_level']==level);b=next(r for r in summary if r['model']=='E1' and r['dataset']==ds and r['analysis_level']==level)
   for j,m in enumerate(METRIC_NAMES):lo,hi=np.percentile(bootstrap[(ds,level)][:,1,j]-bootstrap[(ds,level)][:,0,j],[2.5,97.5]);comparisons.append({'dataset':ds,'analysis_level':level,'metric':m,'E2_minus_E1':a[m]-b[m],'CI_low':float(lo),'CI_high':float(hi)})
  for ki,key in enumerate(['E1','E2']):
   a=next(r for r in summary if r['model']==key and r['dataset']=='ISIC' and r['analysis_level']==level);b=next(r for r in summary if r['model']==key and r['dataset']=='PAD' and r['analysis_level']==level);lo,hi=np.percentile(bootstrap[('ISIC',level)][:,ki,1]-bootstrap[('PAD',level)][:,ki,1],[2.5,97.5]);gaps.append({'model':key,'analysis_level':level,'ISIC_minus_PAD_macro_F1':a['macro_f1']-b['macro_f1'],'CI_low':float(lo),'CI_high':float(hi),'interpretation':'zero-shot transfer gap' if key=='E1' else 'supervised joint-domain performance difference; NOT zero-shot gap'})
 side=[]
 for m in METRIC_NAMES+['macro_AUROC','macro_AUPRC_AP']+['NV_recall','BCC_recall']:
  row={'Metric':m}
  for key,ds,label in [('E1','ISIC','E1 ISIC -> ISIC'),('E1','PAD','E1 ISIC -> PAD'),('E2','ISIC','E2 ISIC+PAD -> ISIC'),('E2','PAD','E2 ISIC+PAD -> PAD')]:
   if m in ['NV_recall','BCC_recall']:value=next(r for r in classes if r['model']==key and r['dataset']==ds and r['analysis_level']=='lesion_primary' and r['class']==m[:3].replace('_',''))['recall']
   else:value=next(r for r in summary if r['model']==key and r['dataset']==ds and r['analysis_level']=='lesion_primary')[m]
   row[label]=value
  side.append(row)
 for gaprow in gaps:
  gain=next(z for z in comparisons if z['dataset']=='PAD' and z['analysis_level']==gaprow['analysis_level'] and z['metric']=='macro_f1');gaprow['PAD_macro_F1_gain_vs_E1']=gain['E2_minus_E1'] if gaprow['model']=='E2' else 0.;gaprow['PAD_gain_CI_low']=gain['CI_low'] if gaprow['model']=='E2' else 0.;gaprow['PAD_gain_CI_high']=gain['CI_high'] if gaprow['model']=='E2' else 0.
 for name,data in [('05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv',summary),('CLASSWISE_RESULTS.csv',classes),('06_Confusion_Matrices/CONFUSION_COUNTS.csv',cms),('07_ROC_PR/CURVE_POINTS.csv',curves),('11_Source_Data/RELIABILITY_BINS.csv',bins),('05_Cross_Dataset_Comparison/PAIRED_DIFFERENCES.csv',comparisons),('GENERALIZATION_COMPARISON.csv',gaps),('E1_E2_SIDE_BY_SIDE_RESULTS.csv',side)]:write(out/name,data)
 dump(out/'05_Cross_Dataset_Comparison/METRICS_COMPLETE.json',{'status':'PASS','bootstrap_valid_replicates':2000,'rejected_absent_class_candidates':rejects,'interval_limitation':'conditional allclassespresent, very small PAD melanoma test7patients; single trainingseed','calibration_fitted':False});print('FOUR-SETTING METRICS COMPLETE',flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['infer','analyze']);p.add_argument('output');a=p.parse_args();(infer if a.mode=='infer' else analyze)(ROOT/a.output)
