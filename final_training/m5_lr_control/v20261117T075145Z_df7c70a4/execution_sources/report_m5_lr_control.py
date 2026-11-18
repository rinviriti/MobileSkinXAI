"""Report one completed LR control using validation only, with thesis figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mobileskinxai-lr-figures')
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
R=Path(__file__).resolve().parents[1];P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9';B=R/'final_training/phase5b/v20261117T063340Z_d20cbfaa';C=['NV','MEL','BCC','SCC']
LABEL={'M3':'Original MobileNetV4-Conv-Small','M5':'Original MobileSkinXAI-Enhanced\n(MobileNetV4-Conv-Small)','LR':'LR-adjusted MobileSkinXAI-Enhanced\n(MobileNetV4-Conv-Small)'}
COLOR={'M3':'#0072B2','M5':'#D55E00','LR':'#009E73'}
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def savecsv(p,rows):
 rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('x',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def measure(cm):
 cm=np.array(cm,dtype=float);tp=cm.diagonal();s=cm.sum(1);pr=cm.sum(0);precision=np.divide(tp,pr,out=np.zeros(4),where=pr>0);recall=tp/s;f1=np.divide(2*tp,s+pr,out=np.zeros(4),where=(s+pr)>0)
 return {'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean()),'melanoma_precision':float(precision[1]),'melanoma_recall':float(recall[1]),'SCC_precision':float(precision[3]),'SCC_recall':float(recall[3]),'MEL_correct':int(tp[1]),'SCC_correct':int(tp[3]),'NV_to_MEL':int(cm[0,1]),'SCC_to_BCC':int(cm[3,2])},precision,recall,f1

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--experiment-dir',required=True);args=ap.parse_args();O=R/args.experiment_dir;status=json.loads((O/'execution_status.json').read_text());assert status['status']=='complete' and status['full_control_runs_started']==1
 original=json.loads((B/'final_results.json').read_text());orig={x['model']:x for x in original['models']};cfg=json.loads((O/'configs/M5.json').read_text());base=json.loads((P/'configs/M5.json').read_text());allowed={'head_lr','backbone_lr','checkpoint_dir'};assert {k:v for k,v in cfg.items() if k not in allowed}=={k:v for k,v in base.items() if k not in allowed}
 # Validation metadata only; no image is opened by this reporter.
 val={r['image']:r for r in read(R/cfg['split_manifest']) if r['split']=='validation'};assert len(val)==2835
 F=O/'thesis_figures';F.mkdir(exist_ok=False);S=F/'source_data';S.mkdir();plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
 runs={'M3':P/'runs/M3_seed42','M5':P/'runs/M5_seed42','LR':O/'run'};results=[];cms={};hist={};classrows=[];epochrows=[];cpdata={};pred_checks=[]
 for key,run in runs.items():
  h=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()];hist[key]=h;st=json.loads((run/'run_status.json').read_text());assert st['status']=='complete' and len(h)==st['epochs_completed'];best=max(h,key=lambda r:r['validation']['macro_f1']);ep=best['epoch'];path=run/'checkpoints/best.pt';assert path.resolve().name==f'epoch_{ep:03d}.pt' and sha(path)==st['best_checkpoint_sha256']
  rs=read(run/f'validation_predictions_epoch_{ep:03d}.csv');assert len(rs)==2835 and {r['image'] for r in rs}==set(val);cm=np.zeros((4,4),dtype=int)
  for r in rs:
   y=int(r['label_index']);assert y==int(val[r['image']]['label_index']);prob=np.array([float(r[c]) for c in C]);assert np.isfinite(prob).all() and np.isclose(prob.sum(),1,atol=1e-5);cm[y,int(prob.argmax())]+=1
  assert cm.tolist()==best['validation']['confusion_matrix'];cms[key]=cm;met,precision,recall,f1=measure(cm);assert abs(met['macro_f1']-st['best_validation_macro_f1'])<1e-6
  cp=torch.load(path,map_location='cpu',weights_only=True);prov=cp['provenance'];assert cp['epoch']==ep and prov['artifact_kind']=='final_training_checkpoint';cpdata[key]={'path':str(path.relative_to(R)),'payload':str(path.resolve().relative_to(R)),'sha256':sha(path),'provenance':prov}
  if key=='LR':
   assert prov['experiment_id']==O.name and prov['config_sha256']==sha(O/'configs/M5.json')
   assert cp['optimizer']['param_groups'][0]['lr']==1e-5 and cp['optimizer']['param_groups'][1]['lr']==1e-6
   for name,parent in [('student','M3'),('teacher','M4')]:assert prov['parents'][name]['sha256']==orig[parent]['checkpoint_sha256']
  del cp
  result={'model_key':key,'architecture':LABEL[key].replace('\n',' '),**met,'best_additional_or_total_epoch':ep,'epochs_completed':len(h),'training_wall_seconds':st['elapsed_seconds'],'peak_allocated_GiB':max(x['peak_allocated_GiB'] for x in h),'checkpoint':str(path.relative_to(R)),'checkpoint_payload':str(path.resolve().relative_to(R)),'checkpoint_sha256':sha(path)};results.append(result)
  pred_checks.append({'model':key,'validation_prediction_count':len(rs),'prediction_metrics_match_history':True,'best_checkpoint_hash_matches':True,'best_epoch':ep})
  for i,c in enumerate(C):classrows.append({'model_key':key,'architecture':LABEL[key].replace('\n',' '),'class':c,'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(cm[i].sum())})
  for r in h:
   row={'model_key':key,'architecture':LABEL[key].replace('\n',' '),'epoch':r['epoch'],'epoch_definition':'total epoch' if key=='M3' else 'additional epoch from same M3 parent','stage':r['stage'],'train_objective':r['train_loss'],'validation_CE':r['validation']['unweighted_loss'],'train_accuracy':r['train']['accuracy'],'validation_accuracy':r['validation']['accuracy'],'train_macro_f1':r['train']['macro_f1'],'validation_macro_f1':r['validation']['macro_f1'],'elapsed_seconds':r['elapsed_seconds'],'gate_acceptance':r['gate_acceptance']}
   if key=='LR':
    assert r['head_lr']==1e-5 and r['backbone_lr']==1e-6
    comp=r['loss_components'];assert np.isclose(r['train_loss'],comp['ce']+.5*comp['kd']+.1*comp['consistency'],atol=1e-5);row.update({'head_lr':r['head_lr'],'backbone_lr':r['backbone_lr'],'CE':comp['ce'],'weighted_KD':.5*comp['kd'],'weighted_JS':.1*comp['consistency'],'mean_gate_mass':comp['gate_mass']})
   epochrows.append(row)
  savecsv(S/f'{key}_confusion_counts.csv',[{'true_class':c,**{p:int(cm[i,j]) for j,p in enumerate(C)}} for i,c in enumerate(C)])
  savecsv(S/f'{key}_confusion_row_percent.csv',[{'true_class':c,'support':int(cm[i].sum()),**{p:float(100*cm[i,j]/cm[i].sum()) for j,p in enumerate(C)}} for i,c in enumerate(C)])
 savecsv(S/'model_comparison.csv',results);savecsv(S/'classwise_metrics.csv',classrows);savecsv(S/'epoch_histories.csv',epochrows)
 by={r['model_key']:r for r in results};new=by['LR'];old=by['M5'];guards={'macro_F1_exceeds_0.691175':new['macro_f1']>0.691175,'MEL_at_least_423_of_586':new['MEL_correct']>=423,'SCC_at_least_36_of_95':new['SCC_correct']>=36,'NV_to_MEL_reduced_from_221':new['NV_to_MEL']<221,'SCC_to_BCC_reduced_from_38':new['SCC_to_BCC']<38};joint=all(guards[k] for k in ['macro_F1_exceeds_0.691175','MEL_at_least_423_of_586','SCC_at_least_36_of_95'])
 retain='LR-adjusted checkpoint is the recommended development candidate; original M5 remains preserved and no automatic replacement is performed.' if joint else 'Retain original M5 as the enhanced-model development checkpoint: the LR control did not meet all three predeclared joint goals. Preserve the LR checkpoint and report its tradeoffs without automatic replacement.'
 summary={'experiment_directory':str(O.relative_to(R)),'dataset_version':cfg['dataset_version'],'models':results,'classwise_metrics':classrows,'exploratory_development_goals':guards,'joint_macro_MEL_SCC_goal_met':joint,'M5_LR_minus_original_M5':{k:new[k]-old[k] for k in ['accuracy','macro_f1','balanced_accuracy','melanoma_recall','SCC_recall','NV_to_MEL','SCC_to_BCC']},'recommendation':retain,'clinical_thresholds':False,'novelty_established':False,'test_PAD_evaluated':False,'full_control_runs':1,'prediction_checks':pred_checks,'process_wall_seconds':status['process_wall_seconds']}
 captions=[]
 def save(fig,name,caption):
  for ext in ['png','svg','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=300,bbox_inches='tight')
  plt.close(fig);captions.append({'figure':name,'caption':caption,'PNG':f'{name}.png','SVG':f'{name}.svg','PDF':f'{name}.pdf'})
 keys=['M3','M5','LR'];x=np.arange(3);width=.25
 fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
 # Metric colors are consistent across this figure; model colors are used in model-wise comparisons.
 for offset,metric,label,color in [(-width,'accuracy','Accuracy','#0072B2'),(0,'macro_f1','Macro-F1','#D55E00'),(width,'balanced_accuracy','Balanced accuracy','#009E73')]:
  bars=ax.bar(x+offset,[by[k][metric] for k in keys],width,label=label,color=color);ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
 ax.set_xticks(x,[LABEL[k] for k in keys]);ax.set_ylim(0,1.05);ax.set_ylabel('Validation metric');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);save(fig,'three_model_validation_comparison','Validation accuracy, macro-F1 and balanced accuracy at the checkpoint independently selected by maximum validation macro-F1 for each model. All models use the same2,835-image validation set. These are development-selection results, not locked-test estimates.')
 vmax=max(cm.max() for cm in cms.values())
 def cmplot(ax,key,norm):
  cm=cms[key];a=cm/cm.sum(1,keepdims=True)*100 if norm else cm;top=100 if norm else vmax;im=ax.imshow(a,vmin=0,vmax=top,cmap='Blues')
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{a[i,j]:.1f}%' if norm else str(cm[i,j]),ha='center',va='center',fontsize=10,color='white' if a[i,j]>.52*top else 'black')
  ax.set_xticks(range(4),C);ax.set_yticks(range(4),[f'{c} (n={cm[i].sum()})' for i,c in enumerate(C)]);ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title(LABEL[key]+'\n'+('Row-normalized validation confusion' if norm else 'Validation confusion: raw counts'),fontsize=10);return im
 for key in keys:
  for norm in [False,True]:
   fig,ax=plt.subplots(figsize=(6.7,5.8),layout='constrained');im=cmplot(ax,key,norm);fig.colorbar(im,ax=ax,label='Percent of true class' if norm else 'Images');suffix='row_percent' if norm else 'counts';save(fig,f'{key}_confusion_{suffix}',f"{LABEL[key].replace(chr(10),' ')} validation confusion matrix at selected epoch{by[key]['best_additional_or_total_epoch']}. Rows are true labels and columns predicted labels; class orderNV,MEL,BCC,SCC. "+('Each row sums to100% before rounding.' if norm else 'Cell entries are actual image counts.')+' Supports are1635,586,519,95. Common scales are used across all three models.')
 for norm in [False,True]:
  fig,axs=plt.subplots(1,3,figsize=(18,5.6),layout='constrained')
  for ax,key in zip(axs,keys):im=cmplot(ax,key,norm)
  fig.colorbar(im,ax=axs,label='Percent of true class' if norm else 'Images',shrink=.8);save(fig,'three_confusion_'+('row_percent' if norm else 'counts')+'_side_by_side','Side-by-side selected-checkpoint validation confusion matrices for original MobileNetV4-Conv-Small, original MobileSkinXAI-Enhanced (MobileNetV4-Conv-Small), and LR-adjusted MobileSkinXAI-Enhanced (MobileNetV4-Conv-Small). The identical validation examples, class order and common color scale permit direct descriptive comparison; row-normalization does not imply class-balanced sample sizes.')
 fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained');cx=np.arange(4);width=.24
 for j,key in enumerate(keys):
  rs=[next(r for r in classrows if r['model_key']==key and r['class']==c) for c in C]
  for ax,metric,title in zip(axs,['precision','recall','f1'],['Precision','Recall / sensitivity','F1']):ax.bar(cx+(j-1)*width,[r[metric] for r in rs],width,color=COLOR[key],label=LABEL[key]);ax.set_xticks(cx,C);ax.set_ylim(0,1.06);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8);save(fig,'classwise_precision_recall_f1','Per-class validation precision, recall and F1 for each selected checkpoint. NV,MEL,BCC,SCC supports are1635,586,519,95 respectively. The small SCC denominator and adaptive validation reuse limit inferential interpretation.')
 fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 for ax,key in zip(axs,['M5','LR']):
  hs=hist[key];epochs=[h['epoch'] for h in hs];ax.plot(epochs,[h['train_loss'] for h in hs],color=COLOR[key],label='Training objective');ax.plot(epochs,[h['validation']['unweighted_loss'] for h in hs],color=COLOR[key],ls='--',label='Validation CE');ax.axvline(by[key]['best_additional_or_total_epoch'],color='#555555',ls=':',lw=1);ax.set(xlabel='Additional epoch from same M3 parent',ylabel='Loss',title=LABEL[key]);ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
 save(fig,'original_and_LR_training_validation_loss','Training and validation loss trajectories for the original and LR-adjusted enhanced MobileNetV4-Conv-Small. Solid lines are the weighted two-view CE+0.5KD+0.1JS objective; dashed lines are unweighted validation CE. They are different loss functions. Dotted vertical lines mark the macro-F1-selected epoch; no loss-based checkpoint reselection was performed.')
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for ax,metric,title in zip(axs,['accuracy','macro_f1'],['Validation accuracy','Validation macro-F1']):
  for key in ['M5','LR']:ax.plot([h['epoch'] for h in hist[key]],[h['validation'][metric] for h in hist[key]],label=LABEL[key],color=COLOR[key])
  ax.axhline(by['M3'][metric],label=LABEL['M3']+' selected baseline',color=COLOR['M3'],ls=':');ax.set(xlabel='Additional enhanced-model epoch',ylabel=title);ax.set_ylim(0,1);ax.spines[['top','right']].set_visible(False)
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8);save(fig,'validation_accuracy_macro_f1_curves','Validation accuracy and macro-F1 by additional enhanced-model epoch. Original MobileNetV4-Conv-Small is shown as a fixed selected-checkpoint reference, not an additional-epoch trajectory. Learning rates are the sole changed training factor; actual stopping epochs can differ under the shared patience6 rule.')
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for ax,metric,title,den in zip(axs,['NV_to_MEL','SCC_to_BCC'],['NV incorrectly predicted as MEL','SCC incorrectly predicted as BCC'],[1635,95]):
  vals=[by[k][metric] for k in keys];bars=ax.bar(range(3),vals,color=[COLOR[k] for k in keys]);ax.bar_label(bars,padding=3);ax.set_xticks(range(3),[LABEL[k] for k in keys],fontsize=8);ax.set_ylim(0,max(vals)*1.22);ax.set_ylabel(f'Errors among{den} true-class images');ax.set_title(title);ax.spines[['top','right']].set_visible(False)
 save(fig,'NV_MEL_and_SCC_BCC_error_comparison','Counts of NV-to-MEL and SCC-to-BCC mistakes at the macro-F1-selected checkpoints. Denominators are1635NV and95SCC validation images, identical for all models. Lower bars are fewer errors; no decision thresholds were fitted.')
 lrhist=hist['LR'];fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
 for key,factor,label,col in [('ce',1,'Weighted CE','#0072B2'),('kd',.5,'0.5 × cross-view KD','#D55E00'),('consistency',.1,'0.1 × gated JS','#009E73')]:ax.plot([h['epoch'] for h in lrhist],[factor*h['loss_components'][key] for h in lrhist],label=label,color=col)
 ax.set(xlabel='Additional epoch',ylabel='Mean objective contribution',title=LABEL['LR']);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);save(fig,'LR_objective_component_curves','Measured LR-control training objective contributions, averaged over actual training batches: weighted CE,0.5times cross-view KD and0.1times gated JS. Original M5 component trajectories were not logged and are not reconstructed or fabricated.')
 savecsv(S/'targeted_error_counts.csv',[{'model_key':k,'architecture':LABEL[k].replace('\n',' '),'NV_to_MEL':by[k]['NV_to_MEL'],'NV_support':1635,'SCC_to_BCC':by[k]['SCC_to_BCC'],'SCC_support':95} for k in keys])
 savecsv(F/'FIGURE_INDEX.csv',captions)
 (F/'FIGURE_CAPTIONS.md').write_text('# Thesis figure captions\n\n'+'\n\n'.join(f"**{i+1}. {c['figure']}**\n\n{c['caption']}\n\nExports: `{c['PNG']}`, `{c['SVG']}`, `{c['PDF']}`. PNG300DPI; SVG/PDF vector; colorblind-friendly Okabe–Ito colors and common sequential confusion scales." for i,c in enumerate(captions)))
 (O/'comparison_results.json').write_text(json.dumps(summary,indent=2)+'\n');(O/'reports/checkpoint_provenance.json').write_text(json.dumps(cpdata,indent=2)+'\n')
 tab='\n'.join(f"| {r['architecture']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['melanoma_precision']:.4f}/{r['melanoma_recall']:.4f} | {r['SCC_precision']:.4f}/{r['SCC_recall']:.4f} | {r['NV_to_MEL']} | {r['SCC_to_BCC']} | {r['best_additional_or_total_epoch']}/{r['epochs_completed']} | {r['training_wall_seconds']/60:.2f} |" for r in results)
 checkpointlist='\n'.join('- '+LABEL[k].replace('\n',' ')+': `'+cpdata[k]['path']+'`; payload `'+cpdata[k]['payload']+'`.' for k in keys)
 goals='\n'.join('- '+k+': **'+str(v)+'**.' for k,v in guards.items())
 report=f'''# Single controlled M5 LR experiment

**Completed exactly one authorized control.** Dataset `{cfg['dataset_version']}` only. The same final M3 initialization SHA256`{orig['M3']['checkpoint_sha256']}` and same final M4 teacher SHA256`{orig['M4']['checkpoint_sha256']}` were used. Original M5 weights were not initialization. Original artifacts are preserved and no checkpoint is automatically replaced.

Only training-factor changes: headLR1e-4→1e-5 and backboneLR1e-5→1e-6. Architecture, class orderNV/MEL/BCC/SCC, AdamW decay1e-4, batch16, seed42, BF16, frozen BN statistics, data-order policy, two-view augmentations, class weights, T4, CE+0.5KD+0.1JS, reliability gates,40epoch cap, patience6 and validation macro-F1 selection are unchanged. Run routing/provenance and non-RNG logging were added in a new copy; the original model/loss/loader/seed/optimizer implementation is imported byte-for-byte. Config and source diffs are saved under reports/.

| Model | Accuracy | Macro-F1 | Balanced accuracy | MEL precision/recall | SCC precision/recall | NV→MEL | SCC→BCC | Selected/actual epochs | Training min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{tab}

All values are from each model's **same highest-macro-F1 checkpoint**, not separately optimized metrics. M3 epoch numbers are total run epochs; original/control M5 numbers are additional epochs from the same M3 epoch18 parent. Control process wall time{status['process_wall_seconds']/60:.2f}minutes; training wall time{new['training_wall_seconds']/60:.2f}minutes includes validation/checkpoint writing. Peak allocated CUDA memory{new['peak_allocated_GiB']:.3f}GiB, excluding driver/desktop memory.

## Exploratory development goals

{goals}

Joint macro-F1/MEL/SCC goal met: **{joint}**. These are exploratory acceptance goals on reused development validation, not clinical thresholds, noninferiority proof or independent generalization evidence. The strict macro-F1 goal is>0.691175 and sensitivity goals are>=423/586MEL and>=36/95SCC. Error-reduction goals are reported separately. No other epoch is substituted to satisfy a different endpoint.

**Recommendation:** {retain}

## Checkpoints and artifacts

{checkpointlist}

New run: `run/history.jsonl`, `run/resolved_config.json`, `run/run_status.json`, `run/validation_predictions_epoch_###.csv`, all exclusive-create epoch checkpoint payloads, and `training_stdout.log`. `best.pt`/`last.pt` are selectors, not overwritten payloads. `comparison_results.json` stores aggregate and all per-class precision/recall/F1; actual CSVs are in thesis_figures/source_data/. Parent provenance, optimizer LR and validation-probability metrics were verified. Smoke used a disposable process with zero optimizer steps, nonzero finite head/backbone gradients and absent teacher gradients; full training reset seed/model/optimizer/loader states independently.

## Thesis figures and limits

`thesis_figures/FIGURE_INDEX.csv` and `FIGURE_CAPTIONS.md` list all full-name figures and thesis-ready captions. PNG300DPI, SVG and PDF are exported. Counts/row-normalized matrices (three each), common-scale side-by-side comparisons, three-model validation comparison, class-wise precision/recall/F1, loss trajectories, validation curves and both targeted error comparisons use actual experiment data. Source CSVs accompany every family; original/control loss components are never invented. Full weighted training objective differs from unweighted validation CE.

One seed and one controlled run cannot establish reproducible superiority or causal benefit of the distillation mechanism. Actual early-stopping budgets can differ despite the same cap and rule. These remain adaptively reused validation-selection results. SCC validation has95images and missing global patient identifiers limit independence guarantees. No locked ISIC test/PAD image was inferred or used for optimization/model selection; no Flutter file was modified. No novelty claim is established, and no additional run was launched automatically.
'''
 for a,z in [('SHA256`','SHA256 `'),('headLR','head LR '),('backboneLR','backbone LR '),('orderNV','order NV'),('decay1e','decay 1e'),('batch16','batch 16'),('seed42','seed 42'),('gates,40','gates, 40'),('patience6','patience 6'),('epoch18','epoch 18'),('time'+f'{status["process_wall_seconds"]/60:.2f}', 'time '+f'{status["process_wall_seconds"]/60:.2f}'),('PNG300DPI','PNG 300 DPI'),('has95images','has 95 images')]:report=report.replace(a,z)
 (O/'CONTROL_RESULTS.md').write_text(report)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
