from common import *
from collections import Counter
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap,shutil
from PIL import Image
CLASS_COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
ALL=['M3','M5']+KEYS
class Figures:
 def __init__(self,out):self.out=out;self.rows=read(out/'FIGURE_INDEX.csv') if (out/'FIGURE_INDEX.csv').exists() else []
 def save(self,folder,name,fig,title,caption,sources,split='validation'):
  paths={e:self.out/folder/f'{name}.{e}' for e in ['png','svg','pdf']}
  for ext,p in paths.items():assert not p.exists();fig.savefig(p,dpi=300,bbox_inches='tight');shutil.copyfile(p,self.out/'10_Publication_Ready'/p.name)
  plt.close(fig);self.rows.append({'id':name,'title':title,'caption':caption,'split':split,'source_data':';'.join(sources),'source_sha256':json.dumps({s:sha(self.out/s) for s in sources}),'generation_script':'code/posthoc.py' if folder.startswith('07_') else 'code/analyze.py',**{e:str(p.relative_to(self.out)) for e,p in paths.items()}});write(self.out/'FIGURE_INDEX.csv',self.rows);print('FIGURE',name,flush=True)
def wrap(s):return '\n'.join(textwrap.wrap(s,32))
def assess(rows):
 y=np.array([int(r['label_index']) for r in rows]);p=np.array([[float(r[c]) for c in CLASSES] for r in rows]);pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(4));pr,re,f,s=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0)
 return {'accuracy':float((y==pred).mean()),'macro_f1':float(f.mean()),'balanced_accuracy':float(re.mean()),'MEL_recall':float(re[1]),'SCC_recall':float(re[3]),'NV_to_MEL':int(cm[0,1]),'SCC_to_BCC':int(cm[3,2])},cm,pr,re,f,s,y,p

def main(out):
 out=Path(out);sealcheck(out);assert load(out/'00_Protocol/THREE_RUNS_COMPLETE.json')['status']=='PASS';figs=Figures(out);summary=[];cws=[];probs=[];hist=[];matrices={};modelrows={}
 for k in ALL:
  run=out/f'runs/{k}' if k in KEYS else PREP/f'runs/{k}_seed42';status=load(run/('status.json' if k in KEYS else 'run_status.json'));epoch=status['best_epoch'] if k in KEYS else (18 if k=='M3' else 1);rows=read(run/f'validation_predictions_epoch_{epoch:03d}.csv');assert len(rows)==2835
  metrics,cm,pr,re,f,s,y,p=assess(rows);matrices[k]=cm;modelrows[k]=(y,p)
  components={'M3':'single-view weighted CE; original backbone training','M5':'two-view CE + gated/reliability-weighted cross-view KD + gated student JS','B1':'two-view CE + ungated class-aware KD + ungated student JS','B2':'two-view CE + ungated class-aware student JS; no teacher','B3':'two-view CE + a priori acquisition perturbations; no teacher/JS'}[k]
  cp=run/'checkpoints/best.pt';seconds=status.get('seconds',status.get('elapsed_seconds'));params=status.get('parameters',None)
  if params is None:
   import torch
   payload=torch.load(cp,map_location='cpu',weights_only=True);net=core.model(core.config('M3'));net.load_state_dict(payload['state_dict']);params=sum(a.numel() for a in net.parameters());del net,payload
  summary.append({'model':k,'model_name':NAMES[k],'components':components,**metrics,'parameters':params,'training_seconds':seconds,'epochs':status.get('epochs',status.get('epochs_completed')),'best_epoch':epoch,'epoch_definition':'additional from final M3' if k!='M3' else 'original total epochs','checkpoint':str(cp.relative_to(ROOT)),'checkpoint_sha256':sha(cp),'peak_allocated_GiB':status.get('peak_allocated_GiB','unavailable')})
  for j,c in enumerate(CLASSES):cws.append({'model':k,'model_name':NAMES[k],'class':c,'precision':float(pr[j]),'recall':float(re[j]),'f1':float(f[j]),'support':int(s[j]),'AUROC':float(roc_auc_score(y==j,p[:,j])),'AUPRC_average_precision':float(average_precision_score(y==j,p[:,j]))})
  probs.extend({'model':k,'model_name':NAMES[k],**r,'predicted_class':CLASSES[int(v)]} for r,v in zip(rows,p.argmax(1)))
  hs=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()]
  for h in hs:hist.append({'model':k,'epoch':h['epoch'],'train_loss':h['train_loss'],'validation_loss':h['validation']['unweighted_loss'],'train_accuracy':h['train']['accuracy'],'validation_accuracy':h['validation']['accuracy'],'train_macro_f1':h['train']['macro_f1'],'validation_macro_f1':h['validation']['macro_f1'],'elapsed_seconds':h['elapsed_seconds'],'head_lr':h.get('head_lr',1e-4 if k!='M3' else ''),'backbone_lr':h.get('backbone_lr',1e-5 if k!='M3' else ''),**{f'loss_{a}':b for a,b in h.get('loss_components',{}).items()},**{f'gate_{a}':b for a,b in (h.get('gate_acceptance_by_class') or {}).items()}})
 write(out/'ABLATION_COMPARISON.csv',summary);write(out/'02_Validation_Results/classwise_validation_metrics.csv',cws);write(out/'02_Validation_Results/selected_validation_probabilities.csv',probs);write(out/'01_Training_Histories/all_histories.csv',hist)
 cmrows=[]
 for k,cm in matrices.items():
  for i in range(4):
   for j in range(4):cmrows.append({'model':k,'true_class':CLASSES[i],'predicted_class':CLASSES[j],'count':int(cm[i,j]),'row_percentage':100*int(cm[i,j])/int(cm[i].sum())})
 write(out/'03_Confusion_Matrices/confusion_matrix_source.csv',cmrows)
 def cmplot(ax,k,norm):
  cm=matrices[k];v=100*cm/cm.sum(1,keepdims=True) if norm else cm;im=ax.imshow(v,cmap='Blues',vmin=0,vmax=100 if norm else 1700)
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{v[i,j]:.1f}%' if norm else str(int(v[i,j])),ha='center',va='center',fontsize=9,color='white' if v[i,j]>(55 if norm else 850) else 'black')
  ax.set(xticks=range(4),xticklabels=CLASSES,yticks=range(4),yticklabels=[f'{c} (n={n})' for c,n in zip(CLASSES,cm.sum(1))],xlabel='Predicted label',ylabel='True label',title=wrap(NAMES[k]));return im
 for k in ALL:
  for norm in [False,True]:
   fig,ax=plt.subplots(figsize=(6.5,6),layout='constrained');im=cmplot(ax,k,norm);fig.colorbar(im,ax=ax,label='Row percentage' if norm else 'Count');figs.save('03_Confusion_Matrices',k+('_percent' if norm else '_counts'),fig,NAMES[k]+' validation confusion matrix','Image-level eligible validation; fixed NV/MEL/BCC/SCC order and shared scales. Support counts shown.',['03_Confusion_Matrices/confusion_matrix_source.csv'])
 for norm in [False,True]:
  fig,axs=plt.subplots(2,3,figsize=(17,11),layout='constrained')
  for ax,k in zip(axs.flat,ALL):im=cmplot(ax,k,norm)
  axs.flat[-1].axis('off');fig.colorbar(im,ax=list(axs.flat),shrink=.7,label='Row percentage' if norm else 'Count');figs.save('03_Confusion_Matrices','all_models_'+('percent' if norm else 'counts'),fig,'Validation baseline, original enhanced and three new interventions','Shared counts0–1700 or row percentages0–100; no held-out data.',['03_Confusion_Matrices/confusion_matrix_source.csv'])
 for k in KEYS:
  hs=[r for r in hist if r['model']==k];best=next(r['best_epoch'] for r in summary if r['model']==k)
  for metric in ['loss','accuracy','macro_f1']:
   fig,ax=plt.subplots(figsize=(7.5,4.8),layout='constrained');x=[r['epoch'] for r in hs];ax.plot(x,[r['train_'+metric] for r in hs],label='Training weighted composite objective' if metric=='loss' else 'Training (weak view)',color=COLORS[k]);ax.plot(x,[r['validation_'+metric] for r in hs],label='Validation unweighted CE' if metric=='loss' else 'Validation',color='#333333',ls='--');ax.axvline(best,color='gray',ls=':',label=f'Best checkpoint: {best}');ax.set(xlabel='Additional training epoch',ylabel=metric.replace('_',' '),title=wrap(NAMES[k]));ax.legend();figs.save('01_Training_Histories',f'{k}_{metric}',fig,NAMES[k]+' '+metric,'Losses have different weighting/objective definitions and are explicitly labeled; best checkpoint chosen by validation macro-F1, patience6.',['01_Training_Histories/all_histories.csv'])
 curves=[]
 for k in ALL:
  y,p=modelrows[k]
  for kind in ['ROC','PR']:
   fig,ax=plt.subplots(figsize=(6.8,5.3),layout='constrained')
   for j,c in enumerate(CLASSES):
    if kind=='ROC':x,z,th=roc_curve(y==j,p[:,j]);score=roc_auc_score(y==j,p[:,j])
    else:z,x,th=precision_recall_curve(y==j,p[:,j]);score=average_precision_score(y==j,p[:,j]);ax.axhline((y==j).mean(),color=CLASS_COLORS[j],ls=':',alpha=.3)
    ax.plot(x,z,color=CLASS_COLORS[j],label=f'{c}: {"AUROC" if kind=="ROC" else "AP"}={score:.3f}');curves.extend({'model':k,'class':c,'curve':kind,'x':float(a),'y':float(b)} for a,b in zip(x,z))
   ax.set(xlabel='False-positive rate' if kind=='ROC' else 'Recall',ylabel='True-positive rate' if kind=='ROC' else 'Precision',xlim=(0,1),ylim=(0,1.02),title=wrap(NAMES[k])+'\nValidation '+kind);ax.legend();write(out/'04_ROC_PR/curve_points.csv',curves);figs.save('04_ROC_PR',f'{k}_{kind}',fig,NAMES[k]+' validation '+kind,'Raw softmax one-vs-rest curves; AP is noninterpolated average precision; no calibration or hard-label probability fabrication.',['02_Validation_Results/selected_validation_probabilities.csv'])
 for metric in ['accuracy','macro_f1','balanced_accuracy','MEL_recall','SCC_recall','NV_to_MEL','SCC_to_BCC','training_seconds','parameters']:
  fig,ax=plt.subplots(figsize=(11,5.8),layout='constrained');v=[r[metric] for r in summary];bars=ax.barh(range(5),v,color=[COLORS[k] for k in ALL]);ax.set(yticks=range(5),yticklabels=[wrap(NAMES[k]) for k in ALL],xlabel=metric.replace('_',' '),title='Validation ablation comparison' if metric not in ['training_seconds','parameters'] else 'Actual model complexity / training resource');ax.invert_yaxis();ax.bar_label(bars,fmt='%.3f' if max(v)<2 else '%.0f',padding=3);ax.set_xlim(0,max(v)*1.18);folder='08_Complexity' if metric in ['training_seconds','parameters'] else '06_Ablation_Comparisons';figs.save(folder,'comparison_'+metric,fig,'Comparison: '+metric,'Single seed and unequal realized budgets; M3 includes original head/fine-tuning while enhanced arms are additional. Parameters count inference student only, excluding frozen teacher. Runtime is actual training time, not phone inference latency. No causal/superiority claim.',['ABLATION_COMPARISON.csv'])
 for metric in ['precision','recall','f1']:
  fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
  for j,k in enumerate(ALL):ax.bar(np.arange(4)+(j-2)*.16,[next(r[metric] for r in cws if r['model']==k and r['class']==c) for c in CLASSES],width=.15,label=NAMES[k],color=COLORS[k])
  ax.set(xticks=range(4),xticklabels=CLASSES,ylim=(0,1.05),ylabel=metric,title='Validation class-wise '+metric);ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.13));figs.save('05_Classwise_Performance','classwise_'+metric,fig,'Class-wise '+metric,'SupportsNV1635/MEL586/BCC519/SCC95; validation used for selection, not independent inference.',['02_Validation_Results/classwise_validation_metrics.csv'])
 hs=[r for r in hist if r['model']=='B1'];fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
 for name in ['ce','kd','student_JS','original_gated_kd','original_gated_JS']:axs[0].plot([r['epoch'] for r in hs],[r['loss_'+name] for r in hs],label=name)
 for c in CLASSES:axs[1].plot([r['epoch'] for r in hs],[r['gate_'+c] for r in hs],label=c)
 axs[0].set(xlabel='Additional epoch',ylabel='Unscaled loss component',title='Ungated auxiliary scale and gated counterfactual');axs[1].set(xlabel='Additional epoch',ylabel='Original teacher hard-gate acceptance',ylim=(0,1),title='Diagnostic gate only; NOT used by this model')
 for ax in axs:ax.legend()
 figs.save('06_Ablation_Comparisons','B1_loss_gate_diagnostics',fig,NAMES['B1']+' loss and counterfactual gates','B1 uses reliability1; diagnostic original gates do not affect optimization. Same full-batch class-weight denominator; loss magnitudes are not gradient-direction equivalence.',['01_Training_Histories/all_histories.csv'])
 # Descriptive improvements relative to fixed validation references.
 deltas=[]
 for r in summary:
  for reference in ['M3','M5']:
   base=next(z for z in summary if z['model']==reference);deltas.append({'model':r['model'],'reference':reference,'macro_f1_difference':r['macro_f1']-base['macro_f1']})
 write(out/'06_Ablation_Comparisons/validation_macro_f1_differences.csv',deltas)
 fig,ax=plt.subplots(figsize=(11,5.8),layout='constrained')
 for j,reference in enumerate(['M3','M5']):
  vals=[next(r['macro_f1_difference'] for r in deltas if r['model']==k and r['reference']==reference) for k in KEYS];bars=ax.barh(np.arange(3)+(j-.5)*.32,vals,height=.3,label='Relative to '+NAMES[reference],color=[COLORS[k] for k in KEYS],hatch='' if j==0 else '///',alpha=1 if j==0 else .65);ax.bar_label(bars,fmt='%+.4f',padding=4)
 ax.axvline(0,color='black',lw=1);ax.set(yticks=range(3),yticklabels=[wrap(NAMES[k]) for k in KEYS],xlabel='Validation macro-F1 difference',title='Descriptive improvements/deterioration versus fixed references');ax.invert_yaxis();ax.legend(loc='lower right');figs.save('06_Ablation_Comparisons','validation_macro_f1_differences',fig,'Validation macro-F1 differences','Single-seed differences selected on validation; no statistical-significance or mechanism-only attribution.',['06_Ablation_Comparisons/validation_macro_f1_differences.csv'])
 # Freeze every selected checkpoint BEFORE access to any held-out images, predictions or previous metrics.
 lock={'created_UTC':datetime.now(timezone.utc).isoformat(),'study':'POST-HOC EXPLORATORY; Phase6 previously exposed','dataset_release':RELEASE.name,'models':{k:next(r for r in summary if r['model']==k) for k in KEYS},'reference_roles':{'M3':'fixed baseline','M5':'fixed original primary enhanced'},'selection':'validation macro-F1 only; all runs complete; no winner replacement or further training','protocol_sha256':sha(out/'00_Protocol/PROTOCOL_LOCK.json'),'analysis':'Released lesion-mean primary; image secondary; seed42/43 grouped bootstrap2000, percentile95%; no calibration; no significance claims','previous_exposure':'Original Phase6 PAD findings were viewed before study direction was chosen; reuse cannot confirm external superiority independently.'}
 dump(out/'00_Protocol/FINAL_MODEL_SELECTION_LOCK.json',lock);(out/'00_Protocol/FINAL_MODEL_SELECTION_LOCK.json').chmod(0o444);dump(out/'00_Protocol/FINAL_LOCK_RECEIPT.json',{'sha256':sha(out/'00_Protocol/FINAL_MODEL_SELECTION_LOCK.json')});print('VALIDATION ANALYSIS AND FINAL LOCK COMPLETE',flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
