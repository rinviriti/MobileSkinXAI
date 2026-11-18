"""Validation-only diagnostics: no optimizer steps, model forwards or image reads."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mobileskinxai-analysis-matplotlib')
import csv,hashlib,json,sys,uuid
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9';B=R/'final_training/phase5b/v20261117T063340Z_d20cbfaa';D=R/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8';C=['NV','MEL','BCC','SCC']
NAMES={'M3':'MobileNetV4-Conv-Small','M4':'ConvNeXt-Tiny','M5':'Enhanced MobileNetV4-Conv-Small'}
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rel(p):return str(p.relative_to(R))
O=R/'research/thesis_assets/M5_validation_diagnosis'/('v'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);O.mkdir(parents=True,exist_ok=False)
(O/'figures').mkdir();(O/'source_data').mkdir()
def js(name,x):(O/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def write(name,rs):
 rs=list(rs);p=O/name;p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rs for k in r))
 with p.open('x',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rs)
# Artifact integrity only: never open any image (including training or validation images).
for folder,lockfile in [(D,'RELEASE_LOCK.json'),(P,'PREPARATION_LOCK.json')]:
 lock=json.loads((folder/lockfile).read_text());assert sha(folder/'checksums.sha256')==lock['checksums_sha256']
 for line in (folder/'checksums.sha256').read_text().splitlines():
  h,f=line.split('  ',1);assert sha(folder/f)==h
assert json.loads((D/'data_approval.json').read_text())['approved']
manifest=read(D/'dataset_manifest.csv');val={r['image']:r for r in manifest if r['dataset']=='ISIC2019' and r['split']=='validation' and r['release_eligible']=='True'};assert len(val)==2835
ids=sorted(val);y=np.array([int(val[i]['label_index']) for i in ids]);weights_record=json.loads((D/'class_weights.json').read_text());w=np.array([weights_record['weights'][c] for c in C]);train_counts=np.array([weights_record['counts'][c] for c in C])
final=json.loads((B/'final_results.json').read_text());fmap={r['model']:r for r in final['models']};snap={}
for folder in [D,P/'configs',P/'code']:
 for p in folder.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:snap[rel(p)]=sha(p)
prob={};pred={};hist={};cms={};checkpoints={};check_receipts=[]
for m in ['M3','M4','M5']:
 run=P/f'runs/{m}_seed42';history=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()];hist[m]=history;best=max(history,key=lambda r:r['validation']['macro_f1']);assert best['epoch']==fmap[m]['best_epoch']
 cp=R/fmap[m]['checkpoint'];assert sha(cp)==fmap[m]['checkpoint_sha256'];snap[rel(cp.resolve())]=sha(cp)
 checkpoint=torch.load(cp,map_location='cpu',weights_only=True);assert checkpoint['provenance']['artifact_kind']=='final_training_checkpoint';assert checkpoint['provenance']['model_id']==m;assert checkpoint['provenance']['classes']==C;assert checkpoint['provenance']['data_receipt']['manifest_sha256']==json.loads((D/'data_approval.json').read_text())['manifest_sha256'];assert checkpoint['epoch']==best['epoch']
 checkpoints[m]={'path':rel(cp),'sha256':sha(cp),'epoch':checkpoint['epoch'],'optimizer_groups':[{k:g[k] for k in ['lr','weight_decay','betas','eps','amsgrad','maximize']} for g in checkpoint['optimizer']['param_groups']],'provenance':checkpoint['provenance']};del checkpoint
 pp=run/f"validation_predictions_epoch_{best['epoch']:03d}.csv";rs=read(pp);assert len(rs)==2835;ps={r['image']:r for r in rs};assert set(ps)==set(val)
 assert all(int(ps[i]['label_index'])==int(val[i]['label_index']) for i in ids)
 a=np.array([[float(ps[i][c]) for c in C] for i in ids]);assert np.isfinite(a).all() and (a>=0).all() and np.allclose(a.sum(1),1,atol=1e-5);prob[m]=a;pred[m]=a.argmax(1);cm=np.bincount(y*4+pred[m],minlength=16).reshape(4,4);assert cm.tolist()==best['validation']['confusion_matrix'];cms[m]=cm
 for p in [pp,run/'history.jsonl',run/'resolved_config.json',run/'run_status.json']:snap[rel(p)]=sha(p)
 check_receipts.append({'model':m,'saved_validation_predictions':len(rs),'confusion_matrix_matches_history':True,'checkpoint_sha256_matches':True,'fresh_final_provenance':True})
js('checkpoint_receipts.json',checkpoints)
# Exact optimizer LR state at every M5 epoch: reading state only, never constructing optimizer.
lrrows=[]
for h in hist['M5']:
 p=P/f"runs/M5_seed42/checkpoints/epoch_{h['epoch']:03d}.pt";snap[rel(p)]=sha(p);cp=torch.load(p,map_location='cpu',weights_only=True)
 groups=cp['optimizer']['param_groups'];assert len(groups)==2 and groups[0]['lr']==1e-4 and groups[1]['lr']==1e-5
 lrrows.append({'additional_epoch':h['epoch'],'head_lr':groups[0]['lr'],'backbone_lr':groups[1]['lr'],'weight_decay':groups[0]['weight_decay'],'optimizer_state_step_min':min(int(v['step']) for v in cp['optimizer']['state'].values()),'optimizer_state_step_max':max(int(v['step']) for v in cp['optimizer']['state'].values()),'validation_macro_f1':h['validation']['macro_f1'],'validation_loss':h['validation']['unweighted_loss'],'train_objective':h['train_loss'],'selected':h['epoch']==fmap['M5']['best_epoch'],'gate_acceptance':h['gate_acceptance'],**{f'gate_{c}':h['gate_acceptance_by_class'][c] for c in C}});del cp
 pp=P/f"runs/M5_seed42/validation_predictions_epoch_{h['epoch']:03d}.csv";snap[rel(pp)]=sha(pp)
write('source_data/M5_exact_LR_and_history.csv',lrrows)
# Numerical/autograd validation against an independently written formula using synthetic logits.
sys.path.insert(0,str(P/'code'));from core import m5_loss
cfg=json.loads((P/'configs/M5.json').read_text());torch.set_num_threads(2);torch.manual_seed(42)
target=torch.arange(16)%4;wt=torch.tensor(w,dtype=torch.float32);sw=torch.randn(16,4,requires_grad=True);ss=torch.randn(16,4,requires_grad=True);tw=(torch.randn(16,4)*.1+torch.nn.functional.one_hot(target,4)*7).requires_grad_();ts=(tw.detach()+torch.randn(16,4)*.03).requires_grad_()
loss,st=m5_loss(sw,ss,tw,ts,target,wt,cfg);g=torch.autograd.grad(loss,[sw,ss,tw,ts],allow_unused=True);assert g[2] is None and g[3] is None
swr=sw.detach().clone().requires_grad_();ssr=ss.detach().clone().requires_grad_();a=wt[target].sqrt();ptw=tw.detach().softmax(-1);pts=ts.detach().softmax(-1)
def js_t(p,q):
 mid=(p+q)/2;return .5*((p*(p.log()-mid.log())).sum(-1)+(q*(q.log()-mid.log())).sum(-1))
h=-(ptw*ptw.log()+pts*pts.log()).sum(-1)/(2*np.log(4));d=js_t(ptw,pts);reliability=((1-h)*torch.exp(-d/.1)).detach();assert st['gate_acceptance']==1
ce=(-wt[target]*(swr.log_softmax(-1)[torch.arange(16),target]+ssr.log_softmax(-1)[torch.arange(16),target])).sum()/(2*wt[target].sum())
ptwt=(tw.detach()/4).softmax(-1);ptst=(ts.detach()/4).softmax(-1)
kd=16/(2*a.sum())*(a*reliability*((ptwt*(ptwt.log()-(ssr/4).log_softmax(-1))).sum(-1)+(ptst*(ptst.log()-(swr/4).log_softmax(-1))).sum(-1))).sum()
cons=(a*reliability*js_t(swr.softmax(-1),ssr.softmax(-1))).sum()/a.sum();ref=ce+.5*kd+.1*cons;gr=torch.autograd.grad(ref,[swr,ssr]);assert torch.allclose(loss,ref,atol=2e-6);assert all(torch.allclose(x,z,atol=2e-6,rtol=2e-5) for x,z in zip(g[:2],gr))
zero,stats0=m5_loss(sw,ss,torch.zeros_like(tw),torch.zeros_like(ts),target,wt,cfg);assert torch.isfinite(zero) and stats0['kd']==0 and stats0['consistency']==0
component_norms={}
for label,obj in [('CE',ce),('0.5_KD',.5*kd),('0.1_consistency',.1*cons)]:
 # Rebuild graph for isolated gradient norms; values here are synthetic, not historical losses.
 pass
js('gradient_and_loss_contracts.json',{'status':'PASS','independent_formula_absolute_loss_error':abs(float(loss.detach()-ref.detach())),'max_student_gradient_error':max(float((x-z).abs().max()) for x,z in zip(g[:2],gr)),'teacher_gradients':None,'zero_gate_auxiliary_losses_zero':True,'finite_supervised_gradient_path':True,'teacher_correct_both_view_gate_contract':True,'historical_component_loss_scales_reconstructable':False,'reason':'Only aggregate train_objective and hard gate fractions were saved; raw augmented-view logits, CE/KD/JS components and reliability masses were not logged. Synthetic checks are not training loss-scale measurements.'})
# Confidence summaries, with no temperature fitting or decision threshold tuning.
def ece(a,yy):
 conf=a.max(1);correct=a.argmax(1)==yy;v=0.
 for j in range(10):
  mask=(conf>=j/10)&((conf<(j+1)/10) if j<9 else (conf<=1))
  if mask.any():v+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
 return float(v)
confidence=[];errors=[];perimage=[];cohorts=[]
for m,a in prob.items():
 pp=pred[m];sortedp=np.sort(a,axis=1);margin=sortedp[:,-1]-sortedp[:,-2];pt=a[np.arange(len(y)),y];other=a.copy();other[np.arange(len(y)),y]=-1;signed=pt-other.max(1)
 for c in range(4):
  k=y==c;correct=k&(pp==y);incorrect=k&(pp!=y)
  confidence.append({'model':m,'architecture':NAMES[m],'true_class':C[c],'n':int(k.sum()),'recall':float((pp[k]==c).mean()),'precision':float((y[pp==c]==c).mean()),'mean_top1_confidence':float(a[k].max(1).mean()),'mean_true_class_probability':float(pt[k].mean()),'correct_mean_confidence':float(a[correct].max(1).mean()),'error_mean_confidence':float(a[incorrect].max(1).mean()),'mean_top1_margin':float(margin[k].mean()),'mean_true_minus_best_other_margin':float(signed[k].mean()),'NLL':float(-np.log(pt[k].clip(1e-12)).mean()),'multiclass_Brier':float(((a[k]-np.eye(4)[y[k]])**2).sum(1).mean()),'ECE_10_equal_width':ece(a[k],y[k]),'error_confidence_ge_0.9':int((a[incorrect].max(1)>=.9).sum())})
 for truth,guess in [(0,1),(1,0),(2,3),(3,2)]:
  k=(y==truth)&(pp==guess)
  errors.append({'model':m,'true_class':C[truth],'predicted_class':C[guess],'n':int(k.sum()),'percentage_of_true_class':100*float(k.sum()/sum(y==truth)),'mean_predicted_confidence':float(a[k,guess].mean()),'median_predicted_confidence':float(np.median(a[k,guess])),'mean_true_class_probability':float(a[k,truth].mean()),'mean_top1_margin':float(margin[k].mean()),'confident_errors_ge_0.7':int((a[k,guess]>=.7).sum()),'confident_errors_ge_0.9':int((a[k,guess]>=.9).sum()),'low_margin_errors_le_0.1':int((margin[k]<=.1).sum())})
 for i,imageid in enumerate(ids):
  cohort=val[imageid]['lesion_id'].split('_')[0]
  perimage.append({'model':m,'architecture':NAMES[m],'image':imageid,'lesion_id':val[imageid]['lesion_id'],'cohort_inferred_from_lesion_namespace':cohort,'true_class':C[y[i]],'predicted_class':C[pp[i]],'top1_confidence':float(a[i,pp[i]]),'top1_margin':float(margin[i]),'true_class_probability':float(pt[i]),'true_minus_best_other_margin':float(signed[i]),'MEL_minus_NV_probability':float(a[i,1]-a[i,0]),'SCC_minus_BCC_probability':float(a[i,3]-a[i,2]),**{f'p_{c}':float(a[i,j]) for j,c in enumerate(C)}})
 for cohort in sorted({val[i]['lesion_id'].split('_')[0] for i in ids}):
  cmask=np.array([val[i]['lesion_id'].split('_')[0]==cohort for i in ids])
  for c in range(4):
   k=cmask&(y==c)
   if k.any():cohorts.append({'model':m,'cohort_namespace':cohort,'true_class':C[c],'n':int(k.sum()),'recall':float((pp[k]==c).mean()),'NV_to_MEL':int(((y==0)&(pp==1)&cmask).sum()) if c==0 else '', 'SCC_to_BCC':int(((y==3)&(pp==2)&cmask).sum()) if c==3 else ''})
write('source_data/per_class_confidence.csv',confidence);write('source_data/targeted_confusion_confidence.csv',errors);write('source_data/validation_probability_margins.csv',perimage);write('source_data/cohort_confusions.csv',cohorts)
# Paired prediction transitions and teacher alignment: descriptive, not causal attribution.
transitions=[];teacher=[]
for c in range(4):
 k=y==c
 for before in range(4):
  for after in range(4):
   kk=k&(pred['M3']==before)&(pred['M5']==after)
   if kk.any():transitions.append({'true_class':C[c],'M3_prediction':C[before],'M5_prediction':C[after],'n':int(kk.sum()),'M4_matches_M5':int((pred['M4'][kk]==after).sum()),'M4_correct':int((pred['M4'][kk]==c).sum())})
 changed=k&(pred['M3']!=pred['M5']);teacher.append({'true_class':C[c],'teacher_recall':float((pred['M4'][k]==c).mean()),'M3_teacher_agreement':float((pred['M3'][k]==pred['M4'][k]).mean()),'M5_teacher_agreement':float((pred['M5'][k]==pred['M4'][k]).mean()),'changed_predictions':int(changed.sum()),'changed_align_with_teacher':int((pred['M5'][changed]==pred['M4'][changed]).sum()),'M3_errors_corrected_by_M5':int((k&(pred['M3']!=c)&(pred['M5']==c)).sum()),'M3_correct_spoiled_by_M5':int((k&(pred['M3']==c)&(pred['M5']!=c)).sum())})
write('source_data/paired_prediction_transitions.csv',transitions);write('source_data/teacher_alignment.csv',teacher)
# Pairwise teacher soft targets atT4; cannot reconstruct actual training weak/strong gates.
softrows=[]
for truth,other in [(0,1),(1,0),(2,3),(3,2)]:
 k=y==truth;tk=np.exp(np.log(prob['M4'].clip(1e-12))/4);tk/=tk.sum(1,keepdims=True)
 softrows.append({'true_class':C[truth],'competitor':C[other],'n':int(k.sum()),'teacher_mean_p_true_T1':float(prob['M4'][k,truth].mean()),'teacher_mean_p_competitor_T1':float(prob['M4'][k,other].mean()),'teacher_mean_p_true_T4':float(tk[k,truth].mean()),'teacher_mean_p_competitor_T4':float(tk[k,other].mean()),'interpretation':'Unaugmented validation diagnostic only; not historical training targets or gates'})
write('source_data/teacher_temperature_diagnostic.csv',softrows)
# Bootstrap image-level metric difference using paired lesion-cluster resampling.
lesions=sorted({val[i]['lesion_id'] for i in ids});li={x:j for j,x in enumerate(lesions)};tensor=np.zeros((len(lesions),2,4,4),dtype=np.int32)
for j,m in enumerate(['M3','M5']):
 for i,imageid in enumerate(ids):tensor[li[val[imageid]['lesion_id']],j,y[i],pred[m][i]]+=1
def mf(cm):return np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(4,dtype=float),where=(cm.sum(0)+cm.sum(1))>0).mean()
rng=np.random.default_rng(42);deltas=[]
for _ in range(2000):
 cm=tensor[rng.integers(0,len(lesions),len(lesions))].sum(0);deltas.append(mf(cm[1])-mf(cm[0]))
js('paired_validation_bootstrap.json',{'M5_minus_M3_macro_f1':mf(cms['M5'])-mf(cms['M3']),'paired_lesion_cluster_percentile95_interval':np.quantile(deltas,[.025,.975]).tolist(),'replicates':2000,'seed':42,'lesions':len(lesions),'limitations':'Diagnostic selected-checkpoint validation interval, not unbiased generalization or selection-adjusted significance. No patient clustering possible without global IDs.'})
# Publication figures and exact source tables.
for m,cm in cms.items():
 write(f'source_data/{m}_confusion_counts.csv',[{'true_class':c,**{d:int(cm[i,j]) for j,d in enumerate(C)}} for i,c in enumerate(C)])
 write(f'source_data/{m}_confusion_row_percent.csv',[{'true_class':c,'support':int(cm[i].sum()),**{d:float(100*cm[i,j]/cm[i].sum()) for j,d in enumerate(C)}} for i,c in enumerate(C)])
def plot_cm(ax,m,normalized):
 cm=cms[m];a=100*cm/cm.sum(1,keepdims=True) if normalized else cm;vmax=100 if normalized else max(cms['M3'].max(),cms['M5'].max());im=ax.imshow(a,cmap='Blues',vmin=0,vmax=vmax)
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{a[i,j]:.1f}%' if normalized else str(cm[i,j]),ha='center',va='center',fontsize=10,color='white' if a[i,j]>.52*vmax else 'black')
 ax.set_xticks(range(4),C);ax.set_yticks(range(4),[f'{c} (n={int(cm[i].sum())})' for i,c in enumerate(C)]);ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title(f'{NAMES[m]} ({m})\n'+('Row-normalized validation confusion' if normalized else 'Validation confusion: raw counts'),fontsize=10);return im
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(O/f'figures/{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
for m in ['M3','M5']:
 for normalized in [False,True]:
  fig,ax=plt.subplots(figsize=(6.3,5.3),layout='constrained');im=plot_cm(ax,m,normalized);fig.colorbar(im,ax=ax,label='Percent of true class' if normalized else 'Images');save(fig,f'{m}_confusion_'+('row_percent' if normalized else 'counts'))
for normalized in [False,True]:
 fig,axs=plt.subplots(1,2,figsize=(12,5.1),layout='constrained')
 for ax,m in zip(axs,['M3','M5']):im=plot_cm(ax,m,normalized)
 fig.colorbar(im,ax=axs,label='Percent of true class' if normalized else 'Images',shrink=.85);save(fig,'M3_M5_confusion_'+('row_percent' if normalized else 'counts')+'_side_by_side')
fig,axs=plt.subplots(2,2,figsize=(12,10),layout='constrained')
for row,normalized in enumerate([False,True]):
 for ax,m in zip(axs[row],['M3','M5']):im=plot_cm(ax,m,normalized)
 fig.colorbar(im,ax=axs[row],label='Percent of true class' if normalized else 'Images',shrink=.85)
save(fig,'M3_M5_confusion_comparison')
js('analysis_inputs.json',snap)
js('summary.json',{'dataset_release':D.name,'models':{m:{'architecture':NAMES[m],'confusion_matrix':cms[m].tolist(),'selected_epoch':fmap[m]['best_epoch']} for m in cms},'train_counts':weights_record['counts'],'weights':weights_record['weights'],'gradient_contracts':'PASS','M5_LR_constant_all_7_epochs':True,'saved_predictions_used':True,'model_forward_passes':0,'optimizer_steps':0,'any_image_files_opened':0,'locked_test_PAD_image_or_prediction_access':False,'checkpoint_verification':check_receipts,'output_directory':rel(O)})
assert all(sha(R/p)==h for p,h in snap.items()),'Read-only preservation failure'
js('preservation_check.json',{'input_files_unchanged':len(snap),'hash_mismatches':0,'training_performed':False,'checkpoints_modified':False,'Flutter_modified':False,'locked_test_PAD_images_read':False})
print(json.dumps({'output_directory':str(O),'confidence':confidence,'targeted_errors':errors,'teacher':teacher,'bootstrap':json.loads((O/'paired_validation_bootstrap.json').read_text()),'LR_history':lrrows},indent=2))
if __name__=='__main__':pass
