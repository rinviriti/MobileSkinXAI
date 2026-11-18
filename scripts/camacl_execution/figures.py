from common import *
from analyze import KEYS,NAMES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap,shutil
COLORS=dict(zip(KEYS,['#4E79A7','#76B7B2','#F28E2B','#B07AA1','#E15759']))
CLASS_COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def wrap(x,n=39):return '\n'.join(textwrap.wrap(x,n))
def pick(rows,**kw):return [r for r in rows if all(r[k]==v for k,v in kw.items())]
class Writer:
 def __init__(self,out):self.out=Path(out);self.rows=read(out/'FIGURE_INDEX.csv') if (out/'FIGURE_INDEX.csv').exists() else []
 def save(self,folder,name,fig,title,caption,sources,split='ISIC development validation',script='figures.py'):
  paths={ext:self.out/folder/f'{name}.{ext}' for ext in ['png','svg','pdf']}
  for ext,p in paths.items():assert not p.exists(),p;fig.savefig(p,dpi=300,bbox_inches='tight');shutil.copyfile(p,self.out/'10_Publication_Figures'/p.name)
  plt.close(fig);self.rows.append({'figure_id':name,'title':title,'caption':caption,'split':split,'source_data':';'.join(sources),'source_sha256':json.dumps({s:sha(self.out/s) for s in sources}),'generation_script':'code/'+script,'PNG_DPI':300,**{ext:str(p.relative_to(self.out)) for ext,p in paths.items()}});write(self.out/'FIGURE_INDEX.csv',self.rows);print('FIGURE',name,flush=True)
def cmplot(ax,cm,title,norm):
 v=cm/cm.sum(1,keepdims=True)*100 if norm else cm;im=ax.imshow(v,cmap='Blues',vmin=0,vmax=100 if norm else 1700)
 for a in range(4):
  for b in range(4):ax.text(b,a,f'{v[a,b]:.1f}%' if norm else str(int(v[a,b])),ha='center',va='center',fontsize=9,color='white' if v[a,b]>(55 if norm else 850) else 'black')
 ax.set(xticks=range(4),xticklabels=CLASSES,yticks=range(4),yticklabels=[f'{c} (n={n})' for c,n in zip(CLASSES,cm.sum(1))],xlabel='Predicted label',ylabel='True label',title=wrap(title));return im

def main(out):
 out=Path(out);assert load(out/'12_Reports/ANALYSIS_COMPLETE.json')['status']=='PASS';w=Writer(out);summary=read(out/'02_Validation_Results/CA_MACL_COMPARISON.csv');hist=read(out/'01_Training_Histories/all_histories.csv');cw=read(out/'05_Classwise_Performance/classwise_metrics.csv');cms=read(out/'03_Confusion_Matrices/confusion_matrix_source.csv');curves=read(out/'04_ROC_PR/curve_points.csv');stress=read(out/'07_Synthetic_Shift_Robustness/condition_metrics.csv');matrices={}
 for k in KEYS:
  s=pick(summary,model=k)[0];h=pick(hist,model=k);epochs=[int(r['epoch']) for r in h];best=int(s['best_epoch']);matrices[k]=np.array([int(r['count']) for r in pick(cms,model=k)]).reshape(4,4)
  for m in ['loss','accuracy','macro_f1']:
   fig,ax=plt.subplots(figsize=(8.5,5.7),layout='constrained')
   if m=='loss':a='train_total_loss';b='validation_unweighted_CE';la='Training: weighted composite objective';lb='Validation: unweighted cross-entropy'
   else:a='train_'+m;b='validation_'+m;la='Training: online augmented weak view';lb='Validation: deterministic clean images'
   ax.plot(epochs,[float(r[a]) for r in h],color=COLORS[k],marker='.',label=la);ax.plot(epochs,[float(r[b]) for r in h],color='#333333',ls='--',marker='.',label=lb);ax.axvline(best,ls=':',color='#555555',label=f'Best checkpoint epoch {best}');ax.axvline(5.5,color='#999999',ls='-.',label='Head → fine-tuning');ax.set(xlabel='Total training epoch',ylabel=m.replace('_',' '),title=wrap(NAMES[k])+f'\nTraining and clean validation {m.replace("_"," ")}');ax.legend(loc='best')
   w.save('01_Training_Histories',f'{k}_{m}_curves',fig,NAMES[k]+' '+m+' curves','Actual epoch logs. Training weighted/composite loss and validation unweighted CE have different definitions; curves are labeled accordingly. Checkpoint selected by clean validation macro-F1; head/fine boundary shown.',['01_Training_Histories/all_histories.csv'])
  for norm in [False,True]:
   fig,ax=plt.subplots(figsize=(7,6.5),layout='constrained');im=cmplot(ax,matrices[k],NAMES[k],norm);fig.colorbar(im,ax=ax,label='Row percentage' if norm else 'Count');w.save('03_Confusion_Matrices',f'{k}_confusion_'+('percent' if norm else 'counts'),fig,NAMES[k]+' clean validation confusion','Supports shown. Shared count scale0–1700, percentage scale0–100, fixedNV/MEL/BCC/SCC order.',['03_Confusion_Matrices/confusion_matrix_source.csv'])
  for kind in ['ROC','PR']:
   fig,ax=plt.subplots(figsize=(7.4,6),layout='constrained')
   for j,c in enumerate(CLASSES):
    r=pick(curves,model=k,**{'class':c,'curve':kind});m=pick(cw,model=k,condition='clean',**{'class':c})[0];v=float(m['AUROC' if kind=='ROC' else 'AUPRC_AP']);ax.plot([float(z['x']) for z in r],[float(z['y']) for z in r],color=CLASS_COLORS[j],label=f'{c}: {"AUROC" if kind=="ROC" else "AP"}={v:.3f}')
    if kind=='PR':ax.axhline(int(m['support'])/2835,color=CLASS_COLORS[j],ls=':',alpha=.25,lw=1)
   if kind=='ROC':ax.plot([0,1],[0,1],ls='--',color='gray',lw=1)
   ax.set(xlabel='False-positive rate' if kind=='ROC' else 'Recall',ylabel='True-positive rate' if kind=='ROC' else 'Precision',xlim=(0,1),ylim=(0,1.02),title=wrap(NAMES[k])+'\nClean development validation '+kind);ax.legend(loc='lower right' if kind=='ROC' else 'lower left');w.save('04_ROC_PR',f'{k}_{kind}',fig,NAMES[k]+' '+kind,'One-vs-rest raw uncalibrated softmax probabilities. AP is noninterpolated average precision; dotted PR baselines indicate prevalence. Validation was used for selection.',['04_ROC_PR/curve_points.csv','05_Classwise_Performance/classwise_metrics.csv'])
 for norm in [False,True]:
  fig,axs=plt.subplots(2,3,figsize=(18,12),layout='constrained')
  for ax,k in zip(axs.flat,KEYS):im=cmplot(ax,matrices[k],NAMES[k],norm)
  axs.flat[-1].axis('off');fig.colorbar(im,ax=list(axs.flat),shrink=.7,label='Row percentage' if norm else 'Count');w.save('03_Confusion_Matrices','all_five_confusions_'+('percent' if norm else 'counts'),fig,'All five clean validation confusion matrices','Actual selected checkpoints, fixed class order and common color scales.',['03_Confusion_Matrices/confusion_matrix_source.csv'])
 for metric in ['accuracy','macro_f1','balanced_accuracy','MEL_recall','SCC_recall','NV_to_MEL','SCC_to_BCC']:
  fig,ax=plt.subplots(figsize=(13,7.5),layout='constrained');v=[float(pick(summary,model=k)[0][metric]) for k in KEYS];bars=ax.barh(range(5),v,color=[COLORS[k] for k in KEYS]);ax.set(yticks=range(5),yticklabels=[wrap(NAMES[k]) for k in KEYS],xlabel=metric.replace('_',' '),title='Clean development validation · '+metric.replace('_',' '));ax.invert_yaxis();ax.bar_label(bars,fmt='%.4f' if max(v)<2 else '%.0f',padding=4);ax.set_xlim(0,1.14 if max(v)<2 else max(v)*1.18);folder='05_Classwise_Performance' if metric in ['MEL_recall','SCC_recall'] else '06_Ablation_Comparisons';w.save(folder,'comparison_'+metric,fig,'Five-configuration '+metric,'Single-seed selected-validation comparison; no independent test, external or clinical claim.',['02_Validation_Results/CA_MACL_COMPARISON.csv'])
 # Grouped overall three-metric comparison.
 fig,ax=plt.subplots(figsize=(13,8),layout='constrained');x=np.arange(5)
 for j,m in enumerate(['accuracy','macro_f1','balanced_accuracy']):ax.barh(x+(j-1)*.23,[float(pick(summary,model=k)[0][m]) for k in KEYS],height=.21,label=m.replace('_',' '))
 ax.set(yticks=x,yticklabels=[wrap(NAMES[k]) for k in KEYS],xlabel='Metric',xlim=(0,1.05),title='Clean source validation · three-metric comparison');ax.invert_yaxis();ax.legend(loc='lower right');w.save('06_Ablation_Comparisons','overall_three_metrics',fig,'Accuracy, macro-F1 and balanced accuracy','All metrics from the same clean-validation-selected checkpoint per configuration.',['02_Validation_Results/CA_MACL_COMPARISON.csv'])
 for m in ['precision','recall','f1']:
  fig,ax=plt.subplots(figsize=(13,6),layout='constrained');x=np.arange(4)
  for j,k in enumerate(KEYS):ax.bar(x+(j-2)*.16,[float(pick(cw,model=k,condition='clean',**{'class':c})[0][m]) for c in CLASSES],width=.15,color=COLORS[k],label=wrap(NAMES[k],42))
  ax.set(xticks=x,xticklabels=CLASSES,ylim=(0,1.08),ylabel=m,title='Clean validation class-wise '+m);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8);w.save('05_Classwise_Performance','classwise_'+m,fig,'Class-wise '+m,'SupportsNV1635/MEL586/BCC519/SCC95. No threshold tuning.',['05_Classwise_Performance/classwise_metrics.csv'])
 # Synthetic shift comparison and degradation, fixed conditions rather than external domains.
 conditions=[r['condition'] for r in pick(stress,model='C0') if r['condition']!='clean']
 for m in ['macro_f1','MEL_recall','SCC_recall']:
  fig,ax=plt.subplots(figsize=(13,6.3),layout='constrained');x=np.arange(len(conditions))
  for k in KEYS:ax.plot(x,[float(pick(stress,model=k,condition=c)[0][m]) for c in conditions],marker='o',color=COLORS[k],label=wrap(NAMES[k],42))
  ax.set(xticks=x,xticklabels=[c.replace('_','\n') for c in conditions],ylim=(0,1.03),ylabel=m.replace('_',' '),title='Source-only synthetic acquisition stress · '+m.replace('_',' '));ax.legend(loc='upper left',bbox_to_anchor=(1.01,1));w.save('07_Synthetic_Shift_Robustness','synthetic_'+m,fig,'Synthetic-shift '+m,'Fixed approved conditions applied to validation after clean checkpoint selection; no PAD information or external generalization evidence.',['07_Synthetic_Shift_Robustness/condition_metrics.csv'])
 fig,ax=plt.subplots(figsize=(13,7.5),layout='constrained');matrix=np.array([[float(pick(stress,model=k,condition=c)[0]['robustness_degradation']) for c in conditions] for k in KEYS]);lim=max(abs(matrix).max(),.01);im=ax.imshow(matrix,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
 for i in range(5):
  for j in range(8):ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=8)
 ax.set(yticks=range(5),yticklabels=[wrap(NAMES[k]) for k in KEYS],xticks=range(8),xticklabels=[c.replace('_','\n') for c in conditions],title='Synthetic robustness degradation · clean minus shifted macro-F1');fig.colorbar(im,ax=ax,label='Positive = degradation; negative = improvement');w.save('07_Synthetic_Shift_Robustness','condition_degradation_heatmap',fig,'Per-condition robustness degradation','Same selected model, same source validation identities; fixed conditions. Heatmap scale symmetric across configurations.',['07_Synthetic_Shift_Robustness/condition_metrics.csv'])
 for m in ['mean_stress_macro_f1','robustness_degradation']:
  fig,ax=plt.subplots(figsize=(13,7.5),layout='constrained');v=[float(pick(summary,model=k)[0][m]) for k in KEYS];bars=ax.barh(range(5),v,color=[COLORS[k] for k in KEYS]);ax.set(yticks=range(5),yticklabels=[wrap(NAMES[k]) for k in KEYS],xlabel=m.replace('_',' '),title='Source-only mean stress robustness');ax.invert_yaxis();ax.axvline(0,color='black',lw=1);ax.bar_label(bars,fmt='%.4f',padding=4);w.save('07_Synthetic_Shift_Robustness',m,fig,m.replace('_',' '),'Equal mean over eight predeclared conditions; this is synthetic development robustness, not a measured smartphone domain gap.',['02_Validation_Results/CA_MACL_COMPARISON.csv'])
 # Params, MACs, resources; scatter denotes inference footprint, never phone latency.
 for m in ['inference_parameters','ConvLinear_MACs','training_seconds','peak_allocated_GiB']:
  fig,ax=plt.subplots(figsize=(13,7.5),layout='constrained');v=[float(pick(summary,model=k)[0][m]) for k in KEYS];bars=ax.barh(range(5),v,color=[COLORS[k] for k in KEYS]);ax.set(yticks=range(5),yticklabels=[wrap(NAMES[k]) for k in KEYS],xlabel=m.replace('_',' '),title='Actual student complexity and training resource');ax.invert_yaxis();ax.bar_label(bars,fmt='%.3f' if max(v)<2 else '%.0f',padding=4);ax.set_xlim(0,max(v)*1.2);w.save('08_Model_Complexity',m,fig,m,'Parameters/MACs refer to inference student. C4projector training-only. MAC convention counts Conv/Linear dots only; VRAM allocated, not full-device usage; runtime actual realized training budget.',['02_Validation_Results/CA_MACL_COMPARISON.csv','11_Source_Data/approved_model_complexity.json'])
 for xmetric in ['inference_parameters','ConvLinear_MACs']:
  fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
  for k in KEYS:
   r=pick(summary,model=k)[0];ax.scatter(float(r[xmetric])/1e6,float(r['macro_f1']),s=100,color=COLORS[k],label=wrap(NAMES[k],42))
  ax.set(xlabel=xmetric.replace('_',' ')+' (millions)',ylabel='Clean validation macro-F1',title='Inference complexity versus selected validation macro-F1');ax.legend(loc='upper left',bbox_to_anchor=(1.01,1));w.save('08_Model_Complexity',xmetric+'_vs_F1',fig,xmetric+' vs macro-F1','Equal footprint for C2–C4; plot does not imply measured phone latency or clinical performance.',['02_Validation_Results/CA_MACL_COMPARISON.csv'])
 # Auxiliary component actual histories.
 for k in ['C3','C4']:
  h=pick(hist,model=k);fig,axs=plt.subplots(1,2,figsize=(12,5.5),layout='constrained');x=[int(r['epoch']) for r in h]
  for name in ['train_CE','train_JS']+(['train_SupCon'] if k=='C4' else []):axs[0].plot(x,[float(r[name]) for r in h],label=name)
  axs[1].plot(x,[float(r['auxiliary_ramp']) for r in h],color=COLORS[k],label='Fine-stage auxiliary ramp');axs[0].set(xlabel='Epoch',ylabel='Unscaled loss component');axs[1].set(xlabel='Epoch',ylabel='r(e)',ylim=(-.05,1.05))
  for ax in axs:ax.axvline(5.5,color='gray',ls=':');ax.legend()
  fig.suptitle(wrap(NAMES[k]));w.save('01_Training_Histories',k+'_loss_components',fig,NAMES[k]+' auxiliary losses','CE, JS and contrastive are unscaled actual logged components; zero coefficient/head-stage values do not imply a separately optimized objective.',['01_Training_Histories/all_histories.csv'])
 # C4 versus C3 prespecified descriptive deltas.
 a=pick(summary,model='C4')[0];b=pick(summary,model='C3')[0];metrics=['accuracy','macro_f1','balanced_accuracy','MEL_recall','SCC_recall','mean_stress_macro_f1'];data=[{'metric':m,'C4_minus_C3':float(a[m])-float(b[m])} for m in metrics];write(out/'06_Ablation_Comparisons/C4_C3_metric_differences.csv',data);fig,ax=plt.subplots(figsize=(10,5.4),layout='constrained');v=[r['C4_minus_C3'] for r in data];bars=ax.barh(range(len(v)),v,color=['#009E73' if x>=0 else '#D55E00' for x in v]);ax.axvline(0,color='black');ax.set(yticks=range(len(v)),yticklabels=[m.replace('_',' ') for m in metrics],xlabel='Full CA-MACL minus JS-consistent multi-scale MobileNetV4',title='Class-aware contrastive addition: improvement / deterioration');ax.invert_yaxis();ax.bar_label(bars,fmt='%+.4f',padding=4);w.save('06_Ablation_Comparisons','C4_vs_C3_deltas',fig,'CA-MACL vs JS-consistent MobileNetV4','Conditional single-seed differences; checkpoint selection and realized epochs differ, with no independent clinical inference.',['06_Ablation_Comparisons/C4_C3_metric_differences.csv'])
 dump(out/'12_Reports/FIGURES_COMPLETE.json',{'status':'PASS','non_XAI_figure_families':len(w.rows),'formats':['PNG300DPI','SVG','PDF']})
if __name__=='__main__':main(ROOT/sys.argv[1])
