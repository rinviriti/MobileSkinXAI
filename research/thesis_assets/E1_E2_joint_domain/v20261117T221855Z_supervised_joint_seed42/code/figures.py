from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
NAMES={'E1':'ISIC-only MobileNetV4-Conv-Small (frozen original M3)','E2':'Joint ISIC+PAD MobileNetV4-Conv-Small'};COLORS={'E1':'#4E79A7','E2':'#E15759'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def wrap(s,n=44):return '\n'.join(textwrap.wrap(s,n))
def main(out):
 out=Path(out);assert load(out/'05_Cross_Dataset_Comparison/METRICS_COMPLETE.json')['status']=='PASS';summary=read(out/'05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv');cw=read(out/'CLASSWISE_RESULTS.csv');cmrows=read(out/'06_Confusion_Matrices/CONFUSION_COUNTS.csv');curves=read(out/'07_ROC_PR/CURVE_POINTS.csv');counts=read(out/'01_PAD_Splits/SPLIT_COUNTS.csv');hist=[json.loads(l) for l in (out/'03_E2_Training/history.jsonl').read_text().splitlines()];status=load(out/'03_E2_Training/TRAINING_COMPLETE.json');index=[]
 def sel(rows,**kw):return [r for r in rows if all(r[k]==v for k,v in kw.items())]
 def point(k,ds,m):return float(sel(summary,model=k,dataset=ds,analysis_level='lesion_primary')[0][m])
 def save(folder,name,fig,title,caption,source):
  paths={ext:f'{folder}/{name}.{ext}' for ext in ['png','svg','pdf']}
  for ext,path in paths.items():
   p=out/path
   if not p.exists():tmp=p.with_name(p.stem+'.pending.'+ext);fig.savefig(tmp,dpi=300,bbox_inches='tight');tmp.replace(p)
   mirror=out/'10_Publication_Figures'/p.name
   if mirror!=p:
    import shutil
    if mirror.exists():assert sha(mirror)==sha(p)
    else:shutil.copyfile(p,mirror)
  plt.close(fig);index.append({'figure_id':name,'title':title,'caption':caption+' PAD findings are POST-HOC EXPLORATORY; E2 is supervised joint-domain training, not zero-shot transfer.','source_data':source,'source_sha256':sha(out/source),'script':'code/figures.py','PNG_DPI':300,**paths});write(out/'FIGURE_INDEX.csv',index)
 fig,ax=plt.subplots(figsize=(9,5),layout='constrained');x=np.arange(4)
 for j,s in enumerate(['train','validation','test']):ax.bar(x+(j-1)*.25,[int(sel(counts,split=s,**{'class':c})[0]['images']) for c in CLASSES],width=.23,label=s)
 ax.set(xticks=x,xticklabels=CLASSES,ylabel='PAD images',title='Fixed patient-disjoint PAD60/20/20 split');ax.legend();save('01_PAD_Splits','PAD_split_distribution',fig,'PAD patient-disjoint split distribution','Image proportions prioritized; patient counts550/127/126, melanoma lesions20/7/7.','01_PAD_Splits/SPLIT_COUNTS.csv')
 dev=read(out/'11_Source_Data/DEVELOPMENT_MANIFEST.csv');distribution=[]
 for d in ['ISIC','PAD']:
  rs=[r for r in dev if r['domain']==d and r['study_split']=='train']
  for c in CLASSES:distribution.append({'domain':d,'class':c,'training_images':sum(r['label']==c for r in rs)})
 write(out/'11_Source_Data/TRAINING_DOMAIN_DISTRIBUTION.csv',distribution);fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
 for j,d in enumerate(['ISIC','PAD']):ax.bar(x+(j-.5)*.3,[int(sel(distribution,domain=d,**{'class':c})[0]['training_images']) for c in CLASSES],width=.28,label=d)
 ax.set(xticks=x,xticklabels=CLASSES,ylabel='Unique training images',title='Joint-training source composition\nBatches use50:50domains despite unequal unique-image counts');ax.legend();save('05_Cross_Dataset_Comparison','training_domain_distribution',fig,'ISIC and PAD training distributions','Unique images differ from actual sampled presentation counts; uniform domain queues, no class oversampling.','11_Source_Data/TRAINING_DOMAIN_DISTRIBUTION.csv')
 for metric in ['accuracy','macro_f1']:
  fig,ax=plt.subplots(figsize=(10,6),layout='constrained');bars=ax.bar(range(2),[point(k,'PAD',metric) for k in ['E1','E2']],color=[COLORS[k] for k in ['E1','E2']]);ax.set(xticks=range(2),xticklabels=[wrap(NAMES[k],34) for k in ['E1','E2']],ylim=(0,1.08),ylabel=metric,title='Same PAD held-out patients · lesion-level '+metric);ax.bar_label(bars,fmt='%.4f',padding=4);save('05_Cross_Dataset_Comparison','PAD_'+metric,fig,'E1 versus E2 PAD '+metric,'No test-driven selection; paired patient intervals in source data.','05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv')
 fig,ax=plt.subplots(figsize=(12,6),layout='constrained');settings=[('E1','ISIC'),('E1','PAD'),('E2','ISIC'),('E2','PAD')];x=np.arange(4)
 for j,m in enumerate(['accuracy','macro_f1','balanced_accuracy']):ax.bar(x+(j-1)*.25,[point(k,d,m) for k,d in settings],width=.23,label=m.replace('_',' '))
 ax.set(xticks=x,xticklabels=[wrap(NAMES[k],29)+'\n→ '+d for k,d in settings],ylim=(0,1.05),ylabel='Lesion-level metric',title='Four fixed source-only/joint-domain evaluation settings');ax.legend(loc='upper left',bbox_to_anchor=(1.01,1));save('05_Cross_Dataset_Comparison','four_setting_metrics',fig,'Four-setting ISIC/PAD comparison','Same ISIC/PAD test IDs for E1/E2. Different training recipes and selection sets prevent data-only causal attribution.','05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv')
 gaps=read(out/'GENERALIZATION_COMPARISON.csv');fig,ax=plt.subplots(figsize=(10,6),layout='constrained');bars=ax.bar(range(2),[float(sel(gaps,model=k,analysis_level='lesion_primary')[0]['ISIC_minus_PAD_macro_F1']) for k in ['E1','E2']],color=[COLORS[k] for k in ['E1','E2']]);ax.set(xticks=range(2),xticklabels=[wrap(NAMES[k],34) for k in ['E1','E2']],ylabel='ISIC minus PAD macro-F1',title='Cross-dataset performance differences\nE2 includes labeled PAD training exposure');ax.axhline(0,color='black',lw=1);ax.bar_label(bars,fmt='%.4f',padding=4);save('05_Cross_Dataset_Comparison','performance_difference',fig,'ISIC/PAD performance differences','E1 is zero-shot transfer; E2 difference is NOT a zero-shot generalization measure.','GENERALIZATION_COMPARISON.csv')
 matrices={k:np.array([int(r['count']) for r in sel(cmrows,model=k,dataset='PAD',analysis_level='lesion_primary')]).reshape(4,4) for k in ['E1','E2']};limit=max(m.max() for m in matrices.values())
 def cmplot(ax,k,norm):
  cm=matrices[k];v=100*cm/cm.sum(1,keepdims=True) if norm else cm;im=ax.imshow(v,cmap='Blues',vmin=0,vmax=100 if norm else limit)
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{v[i,j]:.1f}%' if norm else str(int(v[i,j])),ha='center',va='center',color='white' if v[i,j]>(55 if norm else limit*.55) else 'black')
  ax.set(xticks=range(4),xticklabels=CLASSES,yticks=range(4),yticklabels=[f'{c} (n={n})' for c,n in zip(CLASSES,cm.sum(1))],xlabel='Predicted class',ylabel='True class',title=wrap(NAMES[k])+'\nPAD declared-lesion scoring');return im
 for k in ['E1','E2']:
  fig,ax=plt.subplots(figsize=(8,7),layout='constrained');im=cmplot(ax,k,False);fig.colorbar(im,ax=ax,label='Lesion count');save('06_Confusion_Matrices',k+'_PAD_counts',fig,NAMES[k]+' PAD confusion counts','Fixed class order; identical lesion supports and color scales.','06_Confusion_Matrices/CONFUSION_COUNTS.csv')
 fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
 for ax,k in zip(axs,['E1','E2']):im=cmplot(ax,k,True)
 fig.colorbar(im,ax=list(axs),label='Row percentage',shrink=.7);save('06_Confusion_Matrices','PAD_normalized_comparison',fig,'PAD row-normalized confusion comparison','Same denominator per row; small melanoma support7lesions, no omitted errors.','06_Confusion_Matrices/CONFUSION_COUNTS.csv')
 for metric in ['precision','recall','f1']:
  fig,ax=plt.subplots(figsize=(11,6),layout='constrained');x=np.arange(4)
  for j,k in enumerate(['E1','E2']):ax.bar(x+(j-.5)*.3,[float(sel(cw,model=k,dataset='PAD',analysis_level='lesion_primary',**{'class':c})[0][metric]) for c in CLASSES],width=.28,color=COLORS[k],label=wrap(NAMES[k]))
  ax.set(xticks=x,xticklabels=CLASSES,ylim=(0,1.08),ylabel=metric,title='PAD held-out lesion class-wise '+metric);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1));save('08_Classwise_Analysis','PAD_class_'+metric,fig,'PAD class-wise '+metric,'Macro metrics and minority classes shown rather than majority accuracy alone.','CLASSWISE_RESULTS.csv')
 for kind in ['loss','accuracy','macro_f1']:
  fig,ax=plt.subplots(figsize=(11,6),layout='constrained');epochs=[h['epoch'] for h in hist]
  if kind=='loss':ax.plot(epochs,[h['training_weighted_CE'] for h in hist],label='Training weighted CE');ax.plot(epochs,[h['PAD_validation']['unweighted_CE'] for h in hist],label='PAD validation unweighted CE');ax.plot(epochs,[h['ISIC_validation']['unweighted_CE'] for h in hist],label='ISIC validation unweighted CE')
  else:
   ax.plot(epochs,[h['train'][kind] for h in hist],label='Training augmented image '+kind);ax.plot(epochs,[h['PAD_validation']['image'][kind] for h in hist],label='PAD validation image '+kind);ax.plot(epochs,[h['PAD_validation']['lesion'][kind] for h in hist],label='PAD validation LESION '+kind);ax.plot(epochs,[h['ISIC_validation']['image'][kind] for h in hist],label='ISIC validation image '+kind)
  ax.axvline(status['best_epoch'],ls=':',color='black',label='Selected epoch '+str(status['best_epoch']));ax.axvline(5.5,color='gray',ls='--',label='Head → fine-tuning');ax.set(xlabel='Total training epoch',ylabel=kind,title='Joint ISIC+PAD MobileNetV4-Conv-Small\nActual training/development histories');ax.legend(fontsize=8);save('03_E2_Training','E2_'+kind+'_curves',fig,'E2 '+kind+' history','Training weighted loss differs from unweighted validation CE; image and lesion metrics explicitly separated.','03_E2_Training/history.jsonl')
 for kind in ['ROC','PR']:
  fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
  for ax,c in zip(axs.flat,CLASSES):
   for k in ['E1','E2']:
    rs=sel(curves,model=k,dataset='PAD',analysis_level='lesion_primary',**{'class':c,'curve':kind});metric='AUROC' if kind=='ROC' else 'AUPRC_AP';v=float(sel(cw,model=k,dataset='PAD',analysis_level='lesion_primary',**{'class':c})[0][metric]);ax.plot([float(r['x']) for r in rs],[float(r['y']) for r in rs],color=COLORS[k],label=wrap(NAMES[k],34)+f'\n{metric}={v:.3f}')
   if kind=='ROC':ax.plot([0,1],[0,1],ls='--',color='gray')
   ax.set(xlabel='False-positive rate' if kind=='ROC' else 'Recall',ylabel='Recall' if kind=='ROC' else 'Precision',xlim=(0,1),ylim=(0,1.03),title=c);ax.legend(fontsize=7)
  fig.suptitle('PAD lesion-level '+kind+' comparison · raw uncalibrated probabilities');save('07_ROC_PR','PAD_'+kind+'_comparison',fig,'PAD '+kind+' comparison','One-vs-rest curves. AP is noninterpolated average precision; minority support small.','07_ROC_PR/CURVE_POINTS.csv')
 efficiency=read(out/'MODEL_EFFICIENCY.csv');fig,axs=plt.subplots(1,3,figsize=(15,7),layout='constrained')
 for ax,m,div,label in zip(axs,['parameters','ConvLinear_MACs','training_seconds'],[1e6,1e6,60],['Inference parameters (millions)','Conv/Linear MACs (millions)','Measured training duration (minutes)']):
  ax.barh(range(2),[float(sel(efficiency,model=k)[0][m])/div for k in ['E1','E2']],color=[COLORS[k] for k in ['E1','E2']]);ax.set(yticks=range(2),yticklabels=[wrap(NAMES[k],32) for k in ['E1','E2']] if m=='parameters' else [],xlabel=label);ax.invert_yaxis()
 fig.suptitle('Same inference architecture · different supervised training budgets');save('05_Cross_Dataset_Comparison','model_efficiency',fig,'E1/E2 model complexity and training cost','Training wall times have different data/loading/augmentation and stopping budgets; no phone latency estimate.','MODEL_EFFICIENCY.csv')
 dump(out/'12_Reports/FIGURES_COMPLETE.json',{'status':'PASS','figure_families':len(index),'formats':['PNG300DPI','SVG','PDF']});print('E1/E2 FIGURES COMPLETE',len(index),flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
