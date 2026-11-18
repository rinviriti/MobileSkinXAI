from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
from PIL import Image
from sklearn.metrics import roc_curve,precision_recall_curve
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
DS={'test':'ISIC 2019 internal test','external':'PAD-UFES-20 external'}
CP=['#0072B2','#D55E00','#009E73','#CC79A7']
def wrap(s):return '\n'.join(textwrap.wrap(s,34))
class Writer:
 def __init__(self,out):self.out=out;self.records=[]
 def save(self,category,name,fig,title,caption,sources,split,level):
  folder=self.out/category;folder.mkdir(exist_ok=True);paths={e:folder/(name+'.'+e) for e in ['png','svg','pdf']}
  for e,p in paths.items():assert not p.exists(),p;fig.savefig(p,dpi=300,bbox_inches='tight')
  plt.close(fig);self.records.append({'figure_id':name,'title':title,'caption':caption,'dataset_split':split,'analysis_level':level,'source_data':';'.join(sources),'source_sha256':json.dumps({s:sha(self.out/s) for s in sources}),'generation_script':'code/figures.py',**{e+'_path':str(p.relative_to(self.out)) for e,p in paths.items()},'png_dpi':300})
  write(self.out/'FIGURE_INDEX.csv',self.records);print('FIGURE',name,flush=True)
def cmplot(ax,cm,norm,title):
 values=cm/cm.sum(1,keepdims=True)*100 if norm else cm
 im=ax.imshow(values,cmap='Blues',vmin=0,vmax=100 if norm else 1700)
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{values[i,j]:.1f}%' if norm else str(int(values[i,j])),ha='center',va='center',fontsize=10,color='white' if values[i,j]>(55 if norm else 850) else 'black')
 ax.set_xticks(range(4),CLASSES);ax.set_yticks(range(4),[f'{c} (n={int(n)})' for c,n in zip(CLASSES,cm.sum(1))]);ax.set(xlabel='Predicted label',ylabel='True label',title=title);return im

def main(out):
 out=Path(out);lockcheck(out);w=Writer(out);summary=read(out/'final_model_comparison.csv');cw=read(out/'classwise_test_metrics.csv');cms=read(out/'11_Source_Data/confusion_matrices.csv');bins=read(out/'11_Source_Data/reliability_bins.csv')
 def select(rows,**kw):return [r for r in rows if all(r[k]==v for k,v in kw.items())]
 for split in DS:
  for level in ['lesion_primary','image_secondary']:
   label='Lesion-level (primary)' if level=='lesion_primary' else 'Image-level (secondary)'
   matrices={}
   for k in KEYS:
    rs=select(cms,model=k,split=split,analysis_level=level);cm=np.array([int(r['count']) for r in rs]).reshape(4,4);matrices[k]=cm
    for norm in [False,True]:
     title=f'{NAMES[k]}\n{DS[split]} · {label}';fig,ax=plt.subplots(figsize=(6.3,5.8),layout='constrained');im=cmplot(ax,cm,norm,title);fig.colorbar(im,ax=ax,label='Row percentage' if norm else 'Count')
     w.save('01_Confusion_Matrices',f'{k}_{split}_{level}_{"percent" if norm else "counts"}',fig,title,'Frozen argmax predictions; NV/MEL/BCC/SCC order. Supports shown in row labels. Count scale fixed 0–1700 across all models and datasets; percentages fixed0–100. '+label+'.',['11_Source_Data/confusion_matrices.csv'],split,level)
   for norm in [False,True]:
    fig,axs=plt.subplots(1,3,figsize=(18,5.8),layout='constrained')
    for ax,k in zip(axs,['M3','M5','M5-LR']):im=cmplot(ax,matrices[k],norm,wrap(NAMES[k]))
    title=f'{DS[split]} · {label}';fig.suptitle(title);fig.colorbar(im,ax=axs,label='Row percentage' if norm else 'Count',shrink=.75)
    w.save('01_Confusion_Matrices',f'focused_{split}_{level}_{"percent" if norm else "counts"}',fig,title,'Predeclared baseline, primary enhanced and secondary LR control; consistent color scales; no post-test selection.',['11_Source_Data/confusion_matrices.csv'],split,level)
  level='lesion_primary'
  for k in KEYS:
   rows=read(out/f'independent_prediction_records/{k}_{split}_lesion_probabilities.csv');y=np.array([int(r['true_index']) for r in rows]);p=np.array([[float(r[c]) for c in CLASSES] for r in rows])
   for kind in ['ROC','PR']:
    fig,ax=plt.subplots(figsize=(6.6,5.5),layout='constrained')
    for j,c in enumerate(CLASSES):
     cr=select(cw,model=k,split=split,analysis_level=level,**{'class':c})[0]
     if kind=='ROC':x,z,_=roc_curve(y==j,p[:,j]);metric=float(cr['AUROC'])
     else:z,x,_=precision_recall_curve(y==j,p[:,j]);metric=float(cr['AUPRC_average_precision']);ax.axhline((y==j).mean(),color=CP[j],alpha=.25,ls=':',lw=1)
     ax.plot(x,z,color=CP[j],label=f'{c}: {"AUROC" if kind=="ROC" else "AP"}={metric:.3f}')
    if kind=='ROC':ax.plot([0,1],[0,1],ls='--',color='gray',lw=1)
    title=f'{NAMES[k]}\n{DS[split]} · lesion-level {kind}';ax.set(xlabel='False-positive rate' if kind=='ROC' else 'Recall',ylabel='True-positive rate' if kind=='ROC' else 'Precision',xlim=(0,1),ylim=(0,1.02),title=title);ax.legend(loc='lower right' if kind=='ROC' else 'lower left')
    w.save('03_ROC_PR',f'{k}_{split}_{kind}',fig,title,'One-vs-rest curves from mean raw softmax probabilities, without calibration. PR summary is noninterpolated average precision; dotted PR baselines show prevalence.',['11_Source_Data/ROC_PR_curve_points.csv','classwise_test_metrics.csv'],split,level)
   br=select(bins,model=k,split=split,analysis_level=level);fig,axs=plt.subplots(1,2,figsize=(10,4.8),layout='constrained');valid=[r for r in br if int(r['count'])]
   axs[0].plot([0,1],[0,1],color='gray',ls='--');axs[0].plot([float(r['mean_confidence']) for r in valid],[float(r['accuracy']) for r in valid],marker='o',color=COLORS[k]);axs[0].set(xlabel='Mean top-label confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1),title='Reliability (10 fixed-width bins)')
   axs[1].bar([float(r['lower'])+.05 for r in br],[int(r['count']) for r in br],width=.09,color=COLORS[k]);axs[1].set(xlabel='Top-label confidence',ylabel='Lesion count',xlim=(0,1),title='Confidence distribution');s=select(summary,model=k,split=split,analysis_level=level)[0];title=f'{NAMES[k]}\n{DS[split]} · ECE={float(s["ECE10"]):.3f}, Brier={float(s["Brier_sum"]):.3f}';fig.suptitle(title)
   w.save('05_Calibration',f'{k}_{split}_reliability',fig,title,'Uncalibrated predictions; no fitting. ECE uses10equal-width bins; multiclass Brier sums four squared errors; small bins have sampling uncertainty.',['11_Source_Data/reliability_bins.csv','final_model_comparison.csv'],split,level)
  for metric in ['precision','recall','f1']:
   fig,ax=plt.subplots(figsize=(12,5.5),layout='constrained');x=np.arange(4)
   for i,k in enumerate(KEYS):
    vals=[float(select(cw,model=k,split=split,analysis_level=level,**{'class':c})[0][metric]) for c in CLASSES];ax.bar(x+(i-2.5)*.13,vals,width=.125,color=COLORS[k],label=NAMES[k])
   ax.set(xticks=x,xticklabels=CLASSES,ylim=(0,1.07),ylabel=metric.capitalize(),title=DS[split]+' · lesion-level class-wise '+metric);ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.13))
   w.save('02_Classwise',f'{split}_{metric}',fig,DS[split]+' class-wise '+metric,'Primary lesion-level class metrics; equal class importance; no threshold tuning.',['classwise_test_metrics.csv'],split,level)
  for metric in ['macro_f1','accuracy','balanced_accuracy']:
   fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained');rs=[select(summary,model=k,split=split,analysis_level=level)[0] for k in KEYS];v=np.array([float(r[metric]) for r in rs]);lo=np.array([float(r[metric+'_CI_low']) for r in rs]);hi=np.array([float(r[metric+'_CI_high']) for r in rs]);y=np.arange(6);ax.barh(y,v,color=[COLORS[k] for k in KEYS]);ax.errorbar(v,y,xerr=[np.maximum(v-lo,0),np.maximum(hi-v,0)],fmt='none',color='black',capsize=3);ax.set(yticks=y,yticklabels=[wrap(NAMES[k]) for k in KEYS],xlim=(0,1.08),xlabel=metric.replace('_',' '),title=DS[split]+' · primary lesion-level metric and95%CI');ax.invert_yaxis()
   for i,a in enumerate(v):ax.text(min(1.01,hi[i]+.015),i,f'{a:.3f}',va='center')
   w.save('07_Confidence_Intervals',f'{split}_{metric}_CI',fig,DS[split]+' '+metric,'Percentile95% intervals from2000 unstratified cluster bootstrap draws; ISIC lesion/PAD patient grouping. Comparisons are descriptive; roles remain fixed.',['final_model_comparison.csv','11_Source_Data/grouped_bootstrap_replicates.csv'],split,level)
  fig,axs=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
  for ax,cl in zip(axs,['MEL','SCC']):
   rs=[select(summary,model=k,split=split,analysis_level=level)[0] for k in KEYS];vals=[float(r[cl+'_recall']) for r in rs];ax.barh(range(6),vals,color=[COLORS[k] for k in KEYS]);ax.set(yticks=range(6),yticklabels=[wrap(NAMES[k]) for k in KEYS],xlim=(0,1.05),xlabel='Recall / sensitivity',title=cl);ax.invert_yaxis()
   for i,v in enumerate(vals):ax.text(v+.01,i,f'{v:.3f}',va='center')
  title=DS[split]+' · melanoma and SCC lesion-level sensitivity';fig.suptitle(title)
  w.save('02_Classwise',f'{split}_MEL_SCC_recall',fig,title,'Sensitivity describes this research dataset, not clinical safety. PAD melanoma support50 images/34 declared lesions.',['classwise_test_metrics.csv','final_model_comparison.csv'],split,level)
 # Cross-domain model comparison and gap.
 for metric in ['accuracy','macro_f1','balanced_accuracy']:
  fig,ax=plt.subplots(figsize=(12,6),layout='constrained');x=np.arange(6)
  for j,split in enumerate(DS):
   vals=[float(select(summary,model=k,split=split,analysis_level='lesion_primary')[0][metric]) for k in KEYS];bars=ax.barh(x+(j-.5)*.32,vals,height=.3,color=[COLORS[k] for k in KEYS],hatch='' if j==0 else '///',alpha=1 if j==0 else .65,label=DS[split]);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
  ax.set(yticks=x,yticklabels=[wrap(NAMES[k]) for k in KEYS],xlim=(0,1.12),xlabel=metric.replace('_',' '),title='Independent internal versus external performance · lesion-level');ax.invert_yaxis();ax.legend(loc='lower right')
  w.save('04_Comparisons',f'domain_comparison_{metric}',fig,'Internal versus external '+metric,'Predeclared model roles preserved; PAD smartphone clinical images differ from ISIC dermoscopy. Primary lesion denominators differ.',['final_model_comparison.csv'],'test;external','lesion_primary')
 gaps=read(out/'11_Source_Data/domain_gaps.csv');fig,ax=plt.subplots(figsize=(12,6),layout='constrained');x=np.arange(6)
 for j,metric in enumerate(['accuracy','macro_f1','balanced_accuracy']):
  rs=[select(gaps,model=k,analysis_level='lesion_primary',metric=metric)[0] for k in KEYS];v=np.array([float(r['ISIC_minus_PAD']) for r in rs]);ax.barh(x+(j-1)*.24,v,height=.22,label=metric.replace('_',' '));
 ax.axvline(0,color='black',lw=1);ax.set(yticks=x,yticklabels=[wrap(NAMES[k]) for k in KEYS],xlabel='ISIC minus PAD metric (positive = external drop)',title='Generalization gap · primary lesion-level scoring');ax.invert_yaxis();ax.legend()
 w.save('04_Comparisons','domain_gaps',fig,'Internal–external generalization gaps','Domain gaps are descriptive, with CIs in source CSV. Cohort, modality, prevalence and label differences prevent attributing the gap to one cause.',['11_Source_Data/domain_gaps.csv'],'test;external','lesion_primary')
 dump(out/'FIGURES_COMPLETE.json',{'status':'PASS','figures_without_XAI':len(w.records),'formats':['PNG300DPI','SVG','PDF']})
if __name__=='__main__':main(ROOT/sys.argv[1])
