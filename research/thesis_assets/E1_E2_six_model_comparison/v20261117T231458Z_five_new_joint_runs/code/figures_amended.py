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
 out=Path(out);assert load(out/'13_Reports/METRICS_COMPLETE.json')['status']=='PASS';summary=read(out/'12_Source_Data/ALL_SETTING_METRICS.csv');cw=read(out/'CLASSWISE_RESULTS.csv');cmrows=read(out/'06_Confusion_Matrices/CONFUSION_COUNTS.csv');gaps=read(out/'SOURCE_TARGET_GAP_COMPARISON.csv');gain=read(out/'PAD_IMPROVEMENTS_BY_MODEL.csv');complexity=read(out/'MODEL_COMPLEXITY.csv');idx=[]
 def save(folder,name,fig,title,caption,source,shortlist=False):
  paths={e:f'{folder}/{name}.{e}' for e in ['png','svg','pdf']}
  for e,path in paths.items():
   p=out/path
   if not p.exists():tmp=p.with_name(p.stem+'.pending.'+e);fig.savefig(tmp,dpi=300,bbox_inches='tight');tmp.replace(p)
  plt.close(fig);idx.append({'figure_id':name,'title':title,'caption':caption+' PAD findings: POST-HOC EXPLORATORY. E2 uses labeled PAD training exposure.','source_data':source,'source_sha256':sha(out/source),'script':'code/figures_amended.py','PNG_DPI':300,'thesis_shortlist':shortlist,**paths});write(out/'FIGURE_INDEX.csv',idx)
 def point(k,c,d,m):return float(sel(summary,model=k,condition=c,dataset=d,analysis_level='lesion_primary')[0][m])
 counts=read(out/'01_Frozen_Data_Manifests/SPLIT_COUNTS.csv');fig,ax=plt.subplots(figsize=(9,5),layout='constrained');x=np.arange(4)
 for j,s in enumerate(['train','validation','test']):ax.bar(x+(j-1)*.25,[int(sel(counts,split=s,**{'class':c})[0]['images']) for c in CLASSES],width=.23,label=s)
 ax.set(xticks=x,xticklabels=CLASSES,ylabel='Images',title='Unchanged patient-disjoint PAD split');ax.legend();save('11_Publication_Figures','fixed_PAD_split',fig,'Fixed PAD class distribution','781/261/261 images, 550/127/126 patients; no new split generated.','01_Frozen_Data_Manifests/SPLIT_COUNTS.csv')
 dev=read(out/'01_Frozen_Data_Manifests/DEVELOPMENT_MANIFEST.csv');distribution=[]
 for d in ['ISIC','PAD']:
  for cl in CLASSES:distribution.append({'domain':d,'class':cl,'images':sum(r['domain']==d and r['study_split']=='train' and r['label']==cl for r in dev)})
 write(out/'12_Source_Data/TRAINING_DISTRIBUTION.csv',distribution);fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
 for j,d in enumerate(['ISIC','PAD']):ax.bar(x+(j-.5)*.3,[sel(distribution,domain=d,**{'class':c})[0]['images'] for c in CLASSES],width=.28,label=d)
 ax.set(xticks=x,xticklabels=CLASSES,ylabel='Unique training images',title='Unique source/target training composition\nOptimization batches use 50:50 domains');ax.legend();save('11_Publication_Figures','training_composition',fig,'Training domain composition','Unique counts differ from logged per-image presentation frequency.','12_Source_Data/TRAINING_DISTRIBUTION.csv')
 y=np.arange(6);names=[wrap(NAMES[k]) for k in KEYS]
 for condition in ['E1','E2']:
  fig,ax=plt.subplots(figsize=(12,8),layout='constrained');bars=ax.barh(y,[point(k,condition,'PAD','macro_f1') for k in KEYS],color=[COLORS[k] for k in KEYS]);ax.bar_label(bars,fmt='%.4f',padding=4);ax.set(yticks=y,yticklabels=names,xlim=(0,1.1),xlabel='PAD lesion macro-F1',title=condition+' · fixed PAD test patients');ax.invert_yaxis();save('11_Publication_Figures',condition+'_PAD_macro_F1',fig,condition+' PAD macro-F1','All six configurations, including a separately labeled LR sensitivity control.','12_Source_Data/ALL_SETTING_METRICS.csv')
 for metric in ['macro_f1','accuracy','balanced_accuracy','MEL_recall','SCC_recall']:
  fig,ax=plt.subplots(figsize=(13,8),layout='constrained')
  for j,c in enumerate(['E1','E2']):ax.barh(y+(j-.5)*.3,[point(k,c,'PAD',metric) for k in KEYS],height=.28,color=[COLORS[k] for k in KEYS],alpha=.5 if c=='E1' else 1,hatch='//' if c=='E1' else '',label='E1 ISIC-only' if c=='E1' else 'E2 joint ISIC+PAD')
  ax.set(yticks=y,yticklabels=names,xlim=(0,1.05),xlabel=metric.replace('_',' '),title='Same PAD lesion units · '+metric.replace('_',' '));ax.invert_yaxis();ax.legend(loc='lower right');save('11_Publication_Figures','paired_PAD_'+metric,fig,'E1/E2 PAD '+metric,'Single seed 42; minority supports MEL 7 lesions / SCC 28 lesions; patient intervals in source CSV.','12_Source_Data/ALL_SETTING_METRICS.csv',metric in ['macro_f1','MEL_recall','SCC_recall'])
 fig,axs=plt.subplots(1,2,figsize=(16,9),layout='constrained')
 for ax,condition in zip(axs,['E1','E2']):
  for j,ds in enumerate(['ISIC','PAD']):ax.barh(y+(j-.5)*.3,[point(k,condition,ds,'macro_f1') for k in KEYS],height=.28,color=[COLORS[k] for k in KEYS],alpha=1 if ds=='ISIC' else .5,hatch='' if ds=='ISIC' else '//',label=ds)
  ax.set(yticks=y,yticklabels=names if condition=='E1' else [],xlim=(0,1),xlabel='Lesion macro-F1',title=condition+(' zero-shot source-only' if condition=='E1' else ' supervised joint-domain'));ax.invert_yaxis();ax.legend()
 save('11_Publication_Figures','ISIC_PAD_all_models',fig,'ISIC and PAD across all configurations','E2 target exposure acknowledged; performance differences are not zero-shot gaps.','12_Source_Data/ALL_SETTING_METRICS.csv',True)
 for field,title in [('E1_zero_shot_gap','E1 source-to-target gap'),('E2_supervised_performance_difference','E2 supervised source-target difference'),('ISIC_macro_f1_change','Joint-training ISIC macro-F1 trade-off')]:
  fig,ax=plt.subplots(figsize=(13,8),layout='constrained');values=[float(sel(gaps,model=k,analysis_level='lesion_primary')[0][field]) for k in KEYS];bars=ax.barh(y,values,color=[COLORS[k] for k in KEYS]);ax.bar_label(bars,fmt='%+.4f',padding=4);ax.axvline(0,color='black',lw=1);ax.set(yticks=y,yticklabels=names,xlabel='Macro-F1 difference',title=title);ax.invert_yaxis();save('11_Publication_Figures',field,fig,title,'Inspect absolute source/target scores; a smaller difference alone does not establish generalization.','SOURCE_TARGET_GAP_COMPARISON.csv',field=='ISIC_macro_f1_change')
 fig,ax=plt.subplots(figsize=(13,8),layout='constrained');rs=[sel(gain,model=k,analysis_level='lesion_primary')[0] for k in KEYS];v=np.array([100*float(r['PAD_macro_f1_change']) for r in rs]);lo=np.array([100*float(r['PAD_macro_f1_CI_low']) for r in rs]);hi=np.array([100*float(r['PAD_macro_f1_CI_high']) for r in rs]);ax.barh(y,v,color=[COLORS[k] for k in KEYS]);ax.hlines(y,lo,hi,color='black');ax.plot(lo,y,linestyle='none',marker='|',color='black');ax.plot(hi,y,linestyle='none',marker='|',color='black');ax.axvline(0,color='gray');ax.set(yticks=y,yticklabels=names,xlabel='E2 minus E1 macro-F1 (percentage points)',title='Paired PAD adaptation effects · conditional patient 95% intervals');ax.invert_yaxis();save('11_Publication_Figures','PAD_macro_F1_gain_CI',fig,'PAD paired macro-F1 gains','2,000 valid patient-group bootstrap replicates, fixed checkpoints, no seed uncertainty or clinical significance claim.','PAD_IMPROVEMENTS_BY_MODEL.csv',True)
 fig,axs=plt.subplots(1,2,figsize=(16,8),layout='constrained')
 for ax,cl in zip(axs,['NV','BCC']):
  for j,c in enumerate(['E1','E2']):ax.barh(y+(j-.5)*.3,[float(sel(cw,model=k,condition=c,dataset='PAD',analysis_level='lesion_primary',**{'class':cl})[0]['recall']) for k in KEYS],height=.28,color=[COLORS[k] for k in KEYS],alpha=.5 if c=='E1' else 1,hatch='//' if c=='E1' else '',label=c)
  ax.set(yticks=y,yticklabels=names if cl=='NV' else [],xlim=(0,1.05),xlabel='Recall',title=cl+' PAD recall');ax.invert_yaxis();ax.legend()
 save('11_Publication_Figures','NV_BCC_recall',fig,'NV and BCC recall comparison','Same declared lesion supports; no outcome-based exclusions.','CLASSWISE_RESULTS.csv')
 fig,axs=plt.subplots(1,3,figsize=(17,10),layout='constrained');labels=[c+' '+NAMES[k] for c in ['E1','E2'] for k in KEYS]
 for ax,metric in zip(axs,['precision','recall','f1']):
  mat=np.array([[float(sel(cw,model=k,condition=c,dataset='PAD',analysis_level='lesion_primary',**{'class':cl})[0][metric]) for cl in CLASSES] for c in ['E1','E2'] for k in KEYS]);im=ax.imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto')
  for i in range(12):
   for j in range(4):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if mat[i,j]>.55 else 'black')
  ax.set(xticks=range(4),xticklabels=CLASSES,yticks=range(12),yticklabels=[wrap(s,38) for s in labels] if metric=='precision' else [],title='PAD '+metric)
 fig.colorbar(im,ax=list(axs),shrink=.6);save('07_Classwise_Analysis','classwise_heatmap',fig,'PAD class-wise precision/recall/F1','All twelve fixed condition/configuration combinations; supports small for melanoma.','CLASSWISE_RESULTS.csv')
 matrices={(c,k):np.array([int(r['count']) for r in sel(cmrows,model=k,condition=c,dataset='PAD',analysis_level='lesion_primary')]).reshape(4,4) for c in ['E1','E2'] for k in KEYS};limit=max(cm.max() for cm in matrices.values())
 def draw(ax,c,k,norm):
  cm=matrices[(c,k)];v=100*cm/cm.sum(1,keepdims=True) if norm else cm;im=ax.imshow(v,vmin=0,vmax=100 if norm else limit,cmap='Blues')
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{v[i,j]:.1f}%' if norm else str(int(v[i,j])),ha='center',va='center',color='white' if v[i,j]>(55 if norm else limit*.55) else 'black')
  ax.set(xticks=range(4),xticklabels=CLASSES,yticks=range(4),yticklabels=[f'{cl} (n={n})' for cl,n in zip(CLASSES,cm.sum(1))],xlabel='Predicted class',ylabel='True class',title=c+' · '+wrap(NAMES[k]));return im
 for key in KEYS:
  for norm in [False,True]:
   fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
   for ax,c in zip(axs,['E1','E2']):im=draw(ax,c,key,norm)
   fig.colorbar(im,ax=list(axs),shrink=.65,label='Row percentage' if norm else 'Lesion count');save('06_Confusion_Matrices',key+'_paired_'+('percent' if norm else 'counts'),fig,NAMES[key]+' matched PAD confusions','Raw and normalized E1/E2 matrices share scales and fixed class order; no cases omitted.','06_Confusion_Matrices/CONFUSION_COUNTS.csv',norm and key in ['M3','M5'])
  # Individual matrices retained for appendix; paired panels form the thesis shortlist.
  for c in ['E1','E2']:
   for norm in [False,True]:
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained');im=draw(ax,c,key,norm);fig.colorbar(im,ax=ax,label='Row percentage' if norm else 'Lesion count');save('06_Confusion_Matrices',key+'_'+c+'_'+('percent' if norm else 'counts'),fig,c+' '+NAMES[key]+' confusion','Appendix export; use paired panels to avoid redundant thesis charts.','06_Confusion_Matrices/CONFUSION_COUNTS.csv')
 # All new runs and reused M3, accurately labeled histories and best epochs.
 from evaluate_amended import state,cfg_for,BASE
 flat=[]
 for key in KEYS:
  run=BASE/'03_E2_Training' if key=='M3' else out/'03_E2_Training'/key;hs=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()]
  for h in hs[:state(out,key)['epochs']]:flat.append({'model':key,'epoch':h['epoch'],'stage':h['stage'],'train_loss':h.get('training_total_loss',h['training_weighted_CE']),'train_accuracy':h['train']['accuracy'],'train_macro_f1':h['train']['macro_f1'],'PAD_val_CE':h['PAD_validation']['unweighted_CE'],'PAD_val_lesion_F1':h['PAD_validation']['lesion']['macro_f1'],'PAD_val_image_accuracy':h['PAD_validation']['image']['accuracy'],'ISIC_val_image_F1':h['ISIC_validation']['image']['macro_f1']})
 write(out/'12_Source_Data/ALL_E2_HISTORIES.csv',flat)
 for key in KEYS:
  hs=sel(flat,model=key);st=state(out,key)
  for metric in ['loss','accuracy','macro_f1']:
   fig,ax=plt.subplots(figsize=(10,6),layout='constrained');ep=[h['epoch'] for h in hs]
   if metric=='loss':ax.plot(ep,[h['train_loss'] for h in hs],label='Training weighted '+('composite loss' if key in ['M5','M5-LR'] else 'CE'));ax.plot(ep,[h['PAD_val_CE'] for h in hs],label='PAD validation unweighted image CE')
   elif metric=='accuracy':ax.plot(ep,[h['train_accuracy'] for h in hs],label='Training augmented weak/image accuracy');ax.plot(ep,[h['PAD_val_image_accuracy'] for h in hs],label='PAD validation IMAGE accuracy')
   else:ax.plot(ep,[h['train_macro_f1'] for h in hs],label='Training augmented weak/image macro-F1');ax.plot(ep,[h['PAD_val_lesion_F1'] for h in hs],label='PAD validation LESION macro-F1(selection)');ax.plot(ep,[h['ISIC_val_image_F1'] for h in hs],label='ISIC validation IMAGE macro-F1(diagnostic)')
   ax.axvline(st['best_epoch'],ls=':',color='black',label='Best epoch '+str(st['best_epoch']));ax.set(xlabel='Additional student epoch' if key in ['M5','M5-LR'] else 'Total epoch',ylabel=metric,title='E2 '+wrap(NAMES[key]));ax.legend(fontsize=8);save('08_Training_Curves',key+'_'+metric,fig,'E2 '+NAMES[key]+' '+metric,('M1 displays the prescribed epochs1–16 only; extra epochs17–23 remain preserved as out-of-protocol artifacts. ' if key=='M1' else '')+'Definitions differ: weighted training objective versus unweighted validation CE; image versus lesion units explicitly labeled. Enhanced epochs are additional to shared M3 parent.','12_Source_Data/ALL_E2_HISTORIES.csv')
 for field,label,div in [('parameters','Inference parameters (millions)',1e6),('ConvLinear_MACs','Conv/Linear MACs (millions)',1e6),('E2_training_seconds','Own E2 training time (minutes)',60)]:
  fig,ax=plt.subplots(figsize=(13,8),layout='constrained');ax.barh(y,[float(sel(complexity,configuration=k)[0][field])/div for k in KEYS],color=[COLORS[k] for k in KEYS]);ax.set(yticks=y,yticklabels=names,xlabel=label,title='Model configuration '+label);ax.invert_yaxis();save('10_Model_Complexity',field,fig,label,'Inference excludes teacher. Enhanced own time excludes shared M3/M4 parent training. M1 actual committed time includes excluded epochs17-23; prescribed time is separately recorded; partial epoch24 time is unknown. MAC convention excludes pooling/norm/activation/adds.','MODEL_COMPLEXITY.csv')
 for r in complexity:r['PAD_E2_macro_F1']=point(r['configuration'],'E2','PAD','macro_f1')
 write(out/'12_Source_Data/COMPLEXITY_WITH_PAD_RESULTS.csv',complexity)
 for field,label,div in [('parameters','Inference parameters (millions)',1e6),('ConvLinear_MACs','Conv/Linear MACs (millions)',1e6)]:
  fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
  for k in KEYS:ax.scatter(float(sel(complexity,configuration=k)[0][field])/div,point(k,'E2','PAD','macro_f1'),color=COLORS[k],label=wrap(NAMES[k]),s=60)
  ax.set(xlabel=label,ylabel='E2 PAD lesion macro-F1',title='Inference complexity versus exploratory target performance');ax.legend(loc='upper left',bbox_to_anchor=(1.01,1));save('10_Model_Complexity',field+'_versus_PAD_F1',fig,label+' versus PAD F1','No phone latency inferred; deployment selection uses predeclared validation policy, not test points.','12_Source_Data/COMPLEXITY_WITH_PAD_RESULTS.csv')
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
