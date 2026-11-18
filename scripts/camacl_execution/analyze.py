from common import *
from approved_train import metric
from pipeline import STRESS
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
import sys,shutil
KEYS=[f'C{i}' for i in range(5)]
NAMES={'C0':'MobileNetV4-Conv-Small (standard reference)','C1':'MobileNetV4-Conv-Small + acquisition augmentation','C2':'MobileNetV4-Conv-Small + acquisition + multi-scale fusion','C3':'MobileNetV4-Conv-Small + acquisition + fusion + JS consistency','C4':'CA-MACL MobileNetV4-Conv-Small (full method)'}
METRICS=['accuracy','macro_f1','balanced_accuracy','MEL_recall','SCC_recall']
def cmstats(cm):
 tp=np.diagonal(cm,axis1=-2,axis2=-1);support=cm.sum(-1);pred=cm.sum(-2);rec=np.divide(tp,support,out=np.zeros_like(tp,dtype=float),where=support>0);pre=np.divide(tp,pred,out=np.zeros_like(tp,dtype=float),where=pred>0);f=np.divide(2*pre*rec,pre+rec,out=np.zeros_like(pre),where=pre+rec>0)
 return np.stack([tp.sum(-1)/cm.sum((-2,-1)),f.mean(-1),rec.mean(-1),rec[...,1],rec[...,3]],-1)
def assess(rows):
 y=np.array([int(r['label_index']) for r in rows]);p=np.array([[float(r[c]) for c in CLASSES] for r in rows]);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6);pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(4));pr,re,f,s=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0);values=cmstats(cm)
 return y,p,pred,cm,pr,re,f,s,dict(zip(METRICS,map(float,values)))
def main(out):
 out=Path(out);assert load(out/'ALL_FIVE_COMPLETE.json')['status']=='PASS';meta={r['image']:r for r in read(out/'source_data/approved_train_validation.csv') if r['split']=='validation'};assert len(meta)==2835
 summary=[];cw=[];preds=[];cms=[];curves=[];hist=[];reliability=[];stress=[];groups=sorted({r['lesion_id'] for r in meta.values()});gmap={v:i for i,v in enumerate(groups)};allcounts=[];conditions=['clean']+list(STRESS);complexity=load(out/'11_Source_Data/approved_model_complexity.json');base,fusion=complexity['models']
 for k in KEYS:
  run=out/'runs'/k;status=load(run/'status.json');assert status['status']=='complete';assert sha(run/'checkpoints/best.pt')==status['best_checkpoint_sha256'];best=status['best_epoch'];h=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()];assert len(h)==status['epochs'] and all(r['optimizer_updates_this_epoch']==727 for r in h);assert status['global_updates']==727*status['epochs']
  for r in h:hist.append({'model':k,'model_name':NAMES[k],'epoch':r['epoch'],'stage':r['stage'],'fine_epoch':r['fine_epoch'],'train_total_loss':r['loss_components']['total'],'validation_unweighted_CE':r['validation']['unweighted_CE'],'train_accuracy':r['train']['accuracy'],'validation_accuracy':r['validation']['accuracy'],'train_macro_f1':r['train']['macro_f1'],'validation_macro_f1':r['validation']['macro_f1'],'train_balanced_accuracy':r['train']['balanced_accuracy'],'validation_balanced_accuracy':r['validation']['balanced_accuracy'],'train_CE':r['loss_components']['ce'],'train_JS':r['loss_components']['js'],'train_SupCon':r['loss_components']['supcon'],'auxiliary_ramp':r['auxiliary_ramp'],'head_lr':r['learning_rates']['head'],'backbone_lr':r['learning_rates'].get('backbone',0),'elapsed_seconds':r['elapsed_seconds'],'peak_allocated_GiB':r['peak_allocated_GiB'],'optimizer_updates_this_epoch':r['optimizer_updates_this_epoch'],'global_updates':r['global_updates']})
  modelcounts=[]
  for condition in conditions:
   path=run/f'validation_epoch_{best:03d}.csv' if condition=='clean' else run/f'stress_{condition}.csv';rows=read(path);assert len(rows)==len(meta) and {r['image'] for r in rows}==set(meta);assert all(int(r['label_index'])==int(meta[r['image']]['label_index']) for r in rows);y,p,pre,cm,precision,recall,f1,support,values=assess(rows)
   arr=np.zeros((len(groups),4,4),int)
   for r,a,b in zip(rows,y,pre):arr[gmap[meta[r['image']]['lesion_id']],a,b]+=1
   modelcounts.append(arr)
   for j,c in enumerate(CLASSES):
    tp=cm[j,j];fp=cm[:,j].sum()-tp;tn=cm.sum()-cm[j].sum()-fp;cw.append({'model':k,'model_name':NAMES[k],'condition':condition,'class':c,'support':int(support[j]),'precision':float(precision[j]),'recall':float(recall[j]),'f1':float(f1[j]),'specificity':float(tn/(tn+fp)),'false_positives':int(fp),'AUROC':float(roc_auc_score(y==j,p[:,j])),'AUPRC_AP':float(average_precision_score(y==j,p[:,j]))})
   stress.append({'model':k,'model_name':NAMES[k],'condition':condition,'images':len(rows),**values})
   if condition!='clean':continue
   model=base if k in ['C0','C1'] else fusion
   s={'model':k,'model_name':NAMES[k],**values,'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),'macro_AUROC':float(np.mean([roc_auc_score(y==j,p[:,j]) for j in range(4)])),'macro_AUPRC_AP':float(np.mean([average_precision_score(y==j,p[:,j]) for j in range(4)])),'NV_to_MEL':int(cm[0,1]),'SCC_to_BCC':int(cm[3,2]),'best_epoch':best,'epochs_completed':status['epochs'],'training_seconds':status['training_seconds'],'stress_seconds':status['stress_seconds'],'peak_allocated_GiB':status['peak_allocated_GiB'],'inference_parameters':status['inference_parameters'],'training_projector_parameters':status['training_only_projector_parameters'],'ConvLinear_MACs':model['ConvLinear_MACs'],'checkpoint':status['best_checkpoint'],'checkpoint_sha256':status['best_checkpoint_sha256'],'NLL':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),'Brier_sum':float(((p-np.eye(4)[y])**2).sum(1).mean())}
   top=p.max(1);correct=pre==y;binids=np.minimum((top*10).astype(int),9);ece=0
   for b in range(10):
    use=binids==b;n=int(use.sum());a=float(correct[use].mean()) if n else None;c=float(top[use].mean()) if n else None;gap=abs(a-c) if n else 0;ece+=n/len(y)*gap;reliability.append({'model':k,'bin':b,'count':n,'mean_confidence':c,'accuracy':a,'lower':b/10,'upper':(b+1)/10})
   s['ECE10']=ece;summary.append(s)
   preds.extend({'model':k,'model_name':NAMES[k],**r,'true_class':CLASSES[int(r['label_index'])],'predicted_class':CLASSES[int(v)],'lesion_id':meta[r['image']]['lesion_id'],'confidence':float(prob.max()),'checkpoint_sha256':s['checkpoint_sha256']} for r,v,prob in zip(rows,pre,p))
   cms.extend({'model':k,'model_name':NAMES[k],'true_class':CLASSES[a],'predicted_class':CLASSES[b],'count':int(cm[a,b]),'row_percentage':float(100*cm[a,b]/cm[a].sum())} for a in range(4) for b in range(4))
   for j,c in enumerate(CLASSES):
    xx,yy,_=roc_curve(y==j,p[:,j]);curves.extend({'model':k,'class':c,'curve':'ROC','x':float(a),'y':float(b)} for a,b in zip(xx,yy));yy,xx,_=precision_recall_curve(y==j,p[:,j]);curves.extend({'model':k,'class':c,'curve':'PR','x':float(a),'y':float(b)} for a,b in zip(xx,yy))
  allcounts.append(modelcounts)
  shutil.copyfile(run/'history.jsonl',out/f'01_Training_Histories/{k}_history.jsonl')
 # Fixed shared known-lesion group draws, preserving within-image dependence across all synthetic conditions.
 counts=np.array(allcounts);rng=np.random.default_rng(42);boot=[]
 for rep in range(2000):
  weights=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups));cm=np.einsum('g,kcgij->kcij',weights,counts);assert np.all(cm.sum(-1)>0);boot.append(cmstats(cm))
 boot=np.array(boot);np.save(out/'11_Source_Data/lesion_group_bootstrap.npy',boot);dump(out/'11_Source_Data/bootstrap_schema.json',{'axes':['replicate','model','condition','metric'],'models':KEYS,'conditions':conditions,'metrics':METRICS,'replicates':2000,'seed':42,'cluster':'known ISIC lesion ID','lesions':len(groups),'interval':'percentile95','selection_bias':'selected reused-validation descriptive uncertainty only; no training seed variance'})
 for ki,k in enumerate(KEYS):
  s=summary[ki];ss=[r for r in stress if r['model']==k and r['condition']!='clean'];s['mean_stress_macro_f1']=float(np.mean([r['macro_f1'] for r in ss]));s['robustness_degradation']=s['macro_f1']-s['mean_stress_macro_f1'];s['mean_stress_MEL_recall']=float(np.mean([r['MEL_recall'] for r in ss]));s['mean_stress_SCC_recall']=float(np.mean([r['SCC_recall'] for r in ss]))
  for mi,m in enumerate(METRICS):
   lo,hi=np.percentile(boot[:,ki,0,mi],[2.5,97.5]);s[m+'_CI_low']=float(lo);s[m+'_CI_high']=float(hi)
  lo,hi=np.percentile(boot[:,ki,1:,1].mean(1),[2.5,97.5]);s['mean_stress_macro_f1_CI_low']=float(lo);s['mean_stress_macro_f1_CI_high']=float(hi)
  for r in ss:r['robustness_degradation']=s['macro_f1']-r['macro_f1']
 contrasts=[]
 for i in range(1,5):
  for metric,column in [('macro_f1',1),('MEL_recall',3),('SCC_recall',4)]:
   lo,hi=np.percentile(boot[:,i,0,column]-boot[:,i-1,0,column],[2.5,97.5]);contrasts.append({'contrast':f'{KEYS[i]} minus {KEYS[i-1]}','added_component':['acquisition','fusion','JS','class-aware SupCon'][i-1],'metric':metric,'difference':summary[i][metric]-summary[i-1][metric],'CI_low':float(lo),'CI_high':float(hi),'interpretation':'conditional selected-validation single-seed difference; not significance/causal-generalization proof'})
  lo,hi=np.percentile((boot[:,i,1:,1]-boot[:,i-1,1:,1]).mean(1),[2.5,97.5]);contrasts.append({'contrast':f'{KEYS[i]} minus {KEYS[i-1]}','metric':'mean_stress_macro_f1','difference':summary[i]['mean_stress_macro_f1']-summary[i-1]['mean_stress_macro_f1'],'CI_low':float(lo),'CI_high':float(hi),'interpretation':'source-only, conditional grouped interval; no external superiority'})
 write(out/'02_Validation_Results/CA_MACL_COMPARISON.csv',summary);write(out/'02_Validation_Results/selected_validation_probabilities.csv',preds);write(out/'05_Classwise_Performance/classwise_metrics.csv',cw);write(out/'03_Confusion_Matrices/confusion_matrix_source.csv',cms);write(out/'04_ROC_PR/curve_points.csv',curves);write(out/'01_Training_Histories/all_histories.csv',hist);write(out/'07_Synthetic_Shift_Robustness/condition_metrics.csv',stress);write(out/'06_Ablation_Comparisons/paired_component_differences.csv',contrasts);write(out/'11_Source_Data/reliability_bins.csv',reliability)
 criteria={'C4_vs_C0_mean_stress_F1_improves':summary[4]['mean_stress_macro_f1']>summary[0]['mean_stress_macro_f1'],'clean_F1_drop_at_most_.01':summary[4]['macro_f1']>=summary[0]['macro_f1']-.01,'MEL_drop_at_most_.03':summary[4]['MEL_recall']>=summary[0]['MEL_recall']-.03,'SCC_drop_at_most_.03':summary[4]['SCC_recall']>=summary[0]['SCC_recall']-.03,'clinical_or_external_criterion':False};dump(out/'07_Synthetic_Shift_Robustness/PREDECLARED_CRITERIA.json',criteria)
 dump(out/'12_Reports/ANALYSIS_COMPLETE.json',{'status':'PASS','models':5,'epochs':len(hist),'known_validation_lesions':len(groups),'images_per_condition':2835,'stress_conditions':8,'bootstrap_replicates':2000,'heldout_access':False})
 print('ANALYSIS COMPLETE',[(r['model'],r['macro_f1'],r['mean_stress_macro_f1']) for r in summary],flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
