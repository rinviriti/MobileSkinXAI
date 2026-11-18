from common import *
from evaluate_amended import KEYS,NAMES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
COLORS=dict(zip(KEYS,['#4E79A7','#76B7B2','#F28E2B','#9C755F','#B07AA1','#E15759']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def wrap(s,n=36):return '\n'.join(textwrap.wrap(s,n))
def sel(rows,**kw):return [r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())]
def main(out):
 out=Path(out);assert load(out/'13_Reports/METRICS_COMPLETE.json')['status']=='PASS';summary=read(out/'12_Source_Data/ALL_SETTING_METRICS.csv');idx=read(out/'FIGURE_INDEX.csv')
 for r in idx:
  r['thesis_shortlist']=str(r['thesis_shortlist']).lower()=='true'
  assert sha(out/r['source_data'])==r['source_sha256']
  for e in ['png','svg','pdf']:assert (out/r[e]).exists()
 flat=read(out/'12_Source_Data/ALL_E2_HISTORIES.csv')
 for r in flat:
  for k in r:
   if k not in ['model','stage']:r[k]=int(r[k]) if k=='epoch' else float(r[k])
 y=np.arange(6);names=[wrap(NAMES[k]) for k in KEYS]
 def point(k,c,d,m):return float(sel(summary,model=k,condition=c,dataset=d,analysis_level='lesion_primary')[0][m])
 def save(folder,name,fig,title,caption,source,shortlist=False):
  if any(r['figure_id']==name for r in idx):plt.close(fig);return
  paths={e:f'{folder}/{name}.{e}' for e in ['png','svg','pdf']}
  for e,path in paths.items():
   p=out/path;p.parent.mkdir(parents=True,exist_ok=True)
   if not p.exists():tmp=p.with_name(p.stem+'.pending.'+e);fig.savefig(tmp,dpi=300,bbox_inches='tight');tmp.replace(p)
  plt.close(fig);idx.append({'figure_id':name,'title':title,'caption':caption+' PAD findings: POST-HOC EXPLORATORY. E2 uses labeled PAD training exposure.','source_data':source,'source_sha256':sha(out/source),'script':'code/figures_resume_v1.py','PNG_DPI':300,'thesis_shortlist':shortlist,**paths});write(out/'FIGURE_INDEX.csv',idx)
 # Paired source probability-based ROC/PR, without fitting calibration or thresholds.
 curves=read(out/'12_Source_Data/ROC_PR_POINTS.csv')
 for key in KEYS:
  for curve in ['ROC','PR']:
   fig,axs=plt.subplots(2,2,figsize=(12,10),layout='constrained')
   for ax,cl in zip(axs.flat,CLASSES):
    for condition in ['E1','E2']:
     rows=sel(curves,model=key,condition=condition,dataset='PAD',analysis_level='lesion_primary',**{'class':cl,'curve':curve})
     ax.plot([float(r['x']) for r in rows],[float(r['y']) for r in rows],label=condition+' '+('ISIC-only' if condition=='E1' else 'joint ISIC+PAD'),color='#4E79A7' if condition=='E1' else '#E15759')
    if curve=='ROC':ax.plot([0,1],[0,1],ls=':',color='gray')
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False-positive rate' if curve=='ROC' else 'Recall',ylabel='True-positive rate' if curve=='ROC' else 'Precision',title=cl);ax.legend(fontsize=8)
   fig.suptitle(wrap(NAMES[key],60)+' · PAD lesion '+curve)
   save('07_ROC_PR',key+'_PAD_'+curve,fig,NAMES[key]+' PAD '+curve,'Raw uncalibrated softmax probabilities averaged by frozen declared-lesion capture rule; no hard-label reconstruction or calibration fitting.','12_Source_Data/ROC_PR_POINTS.csv')
 # Historical versus current scores are descriptive, never paired across populations.
 history=read(out/'HISTORICAL_VS_CURRENT_PAD_RESULTS.csv');fig,ax=plt.subplots(figsize=(13,8),layout='constrained')
 for j,(field,label) in enumerate([('historical_full_PAD_macro_f1','Historical full PAD · different population'),('current_fixed_PAD_E1_macro_f1','Current fixed PAD test · 205 lesions')]):
  ax.barh(y+(j-.5)*.3,[float(sel(history,model=k)[0][field]) for k in KEYS],height=.28,color=[COLORS[k] for k in KEYS],alpha=.5 if j==0 else 1,hatch='//' if j==0 else '',label=label)
 ax.set(yticks=y,yticklabels=names,xlabel='Lesion macro-F1',xlim=(0,1),title='Descriptive historical/current source-only scores\nDifferent PAD populations; no causal or paired interpretation');ax.invert_yaxis();ax.legend();save('11_Publication_Figures','historical_current_PAD_E1',fig,'Historical versus current source-only PAD','Same frozen E1 checkpoints but different evaluation populations; no adaptation gain is calculated between these bars.','HISTORICAL_VS_CURRENT_PAD_RESULTS.csv')
 fig,axs=plt.subplots(1,2,figsize=(15,6),layout='constrained');xs=np.arange(4)
 for j,k in enumerate(['M5','M5-LR']):axs[0].bar(xs+(j-.5)*.3,[point(k,'E2','PAD',m) for m in ['macro_f1','balanced_accuracy','MEL_recall','SCC_recall']],width=.28,color=COLORS[k],label=wrap(NAMES[k]))
 axs[0].set(xticks=xs,xticklabels=['Macro-F1','Balanced\naccuracy','MEL recall','SCC recall'],ylim=(0,1.05),title='Frozen joint-domain PAD test · descriptive');axs[0].legend(fontsize=8)
 for k in ['M5','M5-LR']:
  hs=sel(flat,model=k);axs[1].plot([h['epoch'] for h in hs],[h['PAD_val_lesion_F1'] for h in hs],color=COLORS[k],label=wrap(NAMES[k]))
 axs[1].set(xlabel='Additional student epoch',ylabel='PAD validation lesion macro-F1',title='Same parents/loss/views; prescribed LR control');axs[1].legend(fontsize=8);save('11_Publication_Figures','enhanced_LR_control',fig,'Enhanced learning-rate sensitivity comparison','Only prespecified head/backbone LRs differ; realized early-stopping budgets may differ. Both are amended weaker-teacher exploratory experiments.','12_Source_Data/ENHANCED_CONTROL_PLOT_DATA.csv',True)
 gates=read(out/'12_Source_Data/TEACHER_GATE_AND_LOSS.csv')
 fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
 for ax,k in zip(axs,['M5','M5-LR']):
  for cl,col in zip(CLASSES,['#0072B2','#D55E00','#009E73','#CC79A7']):
   rs=sel(gates,model=k,**{'class':cl});ax.plot([int(r['epoch']) for r in rs],[float(r['acceptance']) for r in rs],label=cl,color=col)
  ax.set(xlabel='Additional epoch',ylabel='Accepted training presentations / class presentations',ylim=(0,1),title=wrap(NAMES[k]));ax.legend()
 save('08_Training_Curves','teacher_gate_acceptance',fig,'Weaker-teacher gate acceptance','TRAIN labels only; original confidence/entropy/agreement/correctness gates unchanged by amendment. Acceptance is not proof of useful guidance.','12_Source_Data/TEACHER_GATE_AND_LOSS.csv')
 fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
 for ax,field,label in zip(axs,['ce','kd_weighted','JS_weighted'],['Weighted classification CE','0.5 × cross-view KD','0.1 × gated JS']):
  for k in ['M5','M5-LR']:
   rs=sel(gates,model=k,**{'class':'NV'});ax.plot([int(r['epoch']) for r in rs],[float(r[field]) for r in rs],color=COLORS[k],label=wrap(NAMES[k]))
  ax.set(xlabel='Additional epoch',ylabel='Mean training component',title=label);ax.legend(fontsize=7)
 save('08_Training_Curves','enhanced_loss_components',fig,'Recorded enhanced loss components','Different panels retain actual component magnitudes; no auxiliary loss is fabricated when gates reject examples.','12_Source_Data/TEACHER_GATE_AND_LOSS.csv')
 dump(out/'13_Reports/FIGURES_COMPLETE.json',{'status':'PASS','figure_families':len(idx),'shortlist':sum(r['thesis_shortlist'] for r in idx),'formats':['PNG300DPI','SVG','PDF']});print('FIGURES COMPLETE',len(idx),flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
