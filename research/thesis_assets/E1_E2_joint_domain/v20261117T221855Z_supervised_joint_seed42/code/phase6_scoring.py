from common import *
from collections import defaultdict
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
METRIC_NAMES=['accuracy','balanced_accuracy','macro_precision','macro_recall','macro_f1','MEL_recall','SCC_recall']
def cmstats(cm):
 tp=np.diagonal(cm,axis1=-2,axis2=-1);support=cm.sum(-1);pred=cm.sum(-2);total=cm.sum((-2,-1))
 rec=np.divide(tp,support,out=np.zeros_like(tp,dtype=float),where=support!=0);prec=np.divide(tp,pred,out=np.zeros_like(tp,dtype=float),where=pred!=0)
 f=np.divide(2*prec*rec,prec+rec,out=np.zeros_like(tp,dtype=float),where=(prec+rec)!=0)
 return np.stack([tp.sum(-1)/total,rec.mean(-1),prec.mean(-1),rec.mean(-1),f.mean(-1),rec[...,1],rec[...,3]],axis=-1)
def primary(rows):
 groups=defaultdict(list)
 for r in rows:
  if r['use_in_lesion_probability_mean']=='True':groups[r['lesion_scoring_unit']].append(r)
 out=[]
 for lesion,rs in sorted(groups.items()):
  assert len({r['true_class'] for r in rs})==1 and len({r['bootstrap_cluster'] for r in rs})==1
  p=np.mean([[float(r[c]) for c in CLASSES] for r in rs],axis=0);pred=int(p.argmax());r=rs[0]
  out.append({'model':r['model'],'model_name':r['model_name'],'dataset':r['dataset'],'split':r['split'],'lesion_scoring_unit':lesion,'bootstrap_cluster':r['bootstrap_cluster'],'true_class':r['true_class'],'true_index':int(r['true_index']),'predicted_class':CLASSES[pred],'predicted_index':pred,'exact_capture_representatives':len(rs),'representative_images':';'.join(x['image'] for x in rs),**{c:float(p[i]) for i,c in enumerate(CLASSES)},'probability_type':'mean uncalibrated raw softmax over exact-capture representatives'})
 return out

def main(out):
 out=Path(out);lock=lockcheck(out);assert load(out/'INFERENCE_COMPLETE.json')['status']=='PASS'
 summary=[];classwise=[];bins=[];curves=[];cms=[];bootrows=[];boot={}
 for split in ['test','external']:
  for level in ['lesion_primary','image_secondary']:
   allrows=[];arrays=[];clusters=None
   for k in KEYS:
    rows=read(out/f'independent_prediction_records/{k}_{split}_image_probabilities.csv')
    if level=='lesion_primary':
     rows=primary(rows);write(out/f'independent_prediction_records/{k}_{split}_lesion_probabilities.csv',rows)
    y=np.array([int(r['true_index']) for r in rows]);p=np.array([[float(r[c]) for c in CLASSES] for r in rows]);pred=p.argmax(1);assert all(int(r['predicted_index'])==v for r,v in zip(rows,pred))
    cm=confusion_matrix(y,pred,labels=range(4));st=cmstats(cm);base={'model':k,'model_name':NAMES[k],'split':split,'analysis_level':level}
    top=p.max(1);correct=(pred==y);bi=np.minimum((top*10).astype(int),9);ece=0;mce=0
    for b in range(10):
     ix=bi==b;n=int(ix.sum());acc=float(correct[ix].mean()) if n else None;conf=float(top[ix].mean()) if n else None;err=abs(acc-conf) if n else 0;ece+=n/len(rows)*err;mce=max(mce,err)
     bins.append({**base,'bin':b,'lower':b/10,'upper':(b+1)/10,'count':n,'accuracy':acc,'mean_confidence':conf,'absolute_gap':err})
    aurocs=[];aps=[]
    for j,c in enumerate(CLASSES):
     tp=int(cm[j,j]);fn=int(cm[j].sum()-tp);fp=int(cm[:,j].sum()-tp);tn=int(cm.sum()-tp-fn-fp);precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn);f1=2*precision*recall/(precision+recall) if precision+recall else 0
     auc=float(roc_auc_score(y==j,p[:,j]));ap=float(average_precision_score(y==j,p[:,j]));aurocs.append(auc);aps.append(ap)
     classwise.append({**base,'class':c,'support':int((y==j).sum()),'TP':tp,'FN':fn,'FP':fp,'TN':tn,'precision':precision,'recall':recall,'f1':f1,'specificity':tn/(tn+fp),'AUROC':auc,'AUPRC_average_precision':ap})
     x,z,thresholds=roc_curve(y==j,p[:,j]);curves.extend({**base,'class':c,'curve':'ROC','x':float(a),'y':float(b),'threshold':float(t) if np.isfinite(t) else 'inf'} for a,b,t in zip(x,z,thresholds))
     z,x,thresholds=precision_recall_curve(y==j,p[:,j]);curves.extend({**base,'class':c,'curve':'PR','x':float(a),'y':float(b),'threshold':float(thresholds[i]) if i<len(thresholds) else ''} for i,(a,b) in enumerate(zip(x,z)))
    summary.append({**base,'support':len(rows),'bootstrap_clusters':len({r['bootstrap_cluster'] for r in rows}),**dict(zip(METRIC_NAMES,map(float,st))),'macro_AUROC':float(np.mean(aurocs)),'macro_AUPRC_average_precision':float(np.mean(aps)),'NLL':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),'Brier_sum':float(((p-np.eye(4)[y])**2).sum(1).mean()),'ECE10':ece,'MCE10':mce,'NV_to_MEL':int(cm[0,1]),'SCC_to_BCC':int(cm[3,2])})
    cms.extend({**base,'true_class':CLASSES[i],'predicted_class':CLASSES[j],'count':int(cm[i,j]),'row_percent':float(100*cm[i,j]/cm[i].sum())} for i in range(4) for j in range(4))
    groupids=sorted({r['bootstrap_cluster'] for r in rows});assert clusters is None or clusters==groupids;clusters=groupids;gmap={v:i for i,v in enumerate(groupids)};arr=np.zeros((len(groupids),4,4),dtype=int)
    for r,a,b in zip(rows,y,pred):arr[gmap[r['bootstrap_cluster']],a,b]+=1
    arrays.append(arr)
   arrays=np.stack(arrays);rng=np.random.default_rng(42 if split=='test' else 43);rep=[]
   for b in range(2000):
    weights=np.bincount(rng.integers(len(clusters),size=len(clusters)),minlength=len(clusters));cm=np.einsum('g,kgij->kij',weights,arrays);assert np.all(cm.sum(-1)>0),'Bootstrap class absent; stop and disclose';rep.append(cmstats(cm))
   values=np.stack(rep);boot[(split,level)]=values
   for ki,k in enumerate(KEYS):
    s=next(r for r in summary if r['model']==k and r['split']==split and r['analysis_level']==level)
    for mi,metric in enumerate(METRIC_NAMES):
     lo,hi=np.percentile(values[:,ki,mi],[2.5,97.5]);s[metric+'_CI_low']=float(lo);s[metric+'_CI_high']=float(hi)
    bootrows.extend({'split':split,'analysis_level':level,'replicate':b,'model':k,**dict(zip(METRIC_NAMES,map(float,values[b,ki])))} for b in range(2000))
   print('METRICS',split,level,'clusters',len(clusters),flush=True)
 write(out/'final_model_comparison.csv',summary);write(out/'classwise_test_metrics.csv',classwise);write(out/'11_Source_Data/reliability_bins.csv',bins);write(out/'11_Source_Data/ROC_PR_curve_points.csv',curves);write(out/'11_Source_Data/confusion_matrices.csv',cms);write(out/'11_Source_Data/grouped_bootstrap_replicates.csv',bootrows)
 comparisons=[]
 for (split,level),values in boot.items():
  for a,b in [('M5','M3'),('M5-LR','M3'),('M5-LR','M5')]:
   sa=next(r for r in summary if r['model']==a and r['split']==split and r['analysis_level']==level);sb=next(r for r in summary if r['model']==b and r['split']==split and r['analysis_level']==level)
   for mi,metric in enumerate(METRIC_NAMES):
    lo,hi=np.percentile(values[:,KEYS.index(a),mi]-values[:,KEYS.index(b),mi],[2.5,97.5]);comparisons.append({'split':split,'analysis_level':level,'comparison':f'{NAMES[a]} minus {NAMES[b]}','metric':metric,'difference':sa[metric]-sb[metric],'CI_low':float(lo),'CI_high':float(hi)})
 gaps=[]
 for level in ['lesion_primary','image_secondary']:
  for ki,k in enumerate(KEYS):
   a=next(r for r in summary if r['model']==k and r['split']=='test' and r['analysis_level']==level);b=next(r for r in summary if r['model']==k and r['split']=='external' and r['analysis_level']==level)
   for mi,metric in enumerate(METRIC_NAMES):
    lo,hi=np.percentile(boot[('test',level)][:,ki,mi]-boot[('external',level)][:,ki,mi],[2.5,97.5]);gaps.append({'model':k,'model_name':NAMES[k],'analysis_level':level,'metric':metric,'ISIC_minus_PAD':a[metric]-b[metric],'CI_low':float(lo),'CI_high':float(hi)})
 write(out/'11_Source_Data/paired_model_differences.csv',comparisons);write(out/'11_Source_Data/domain_gaps.csv',gaps)
 dump(out/'METRICS_COMPLETE.json',{'status':'PASS','bootstrap_replicates':2000,'ISIC_seed':42,'PAD_seed':43,'seed_policy':'Independent dataset stream derived from base seed42 (offset1 PAD)','primary':'lesion scoring units','secondary':'images','invalid_replicates':0,'calibration_fitted':False})
if __name__=='__main__':main(ROOT/sys.argv[1])
