"""Validation-only result verification and thesis figures from real histories."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mobileskinxai-phase5b-matplotlib')
import argparse,csv,hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9';CLASSES=['NV','MEL','BCC','SCC'];MODELS=['M1','M2','M3','M4','M5']
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def savecsv(p,rows):
 assert not p.exists(),p
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def metrics(cm):
 cm=np.asarray(cm,dtype=float);t=cm.sum(1);p=cm.sum(0);tp=cm.diagonal();rec=np.divide(tp,t,out=np.zeros(4),where=t>0);prec=np.divide(tp,p,out=np.zeros(4),where=p>0);f1=np.divide(2*tp,t+p,out=np.zeros(4),where=t+p>0)
 return {'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(f1.mean()),'balanced_accuracy':float(rec.mean()),'melanoma_recall':float(rec[1]),'SCC_recall':float(rec[3])},prec,rec,f1
def savefig(fig,folder,name):
 fig.tight_layout()
 for ext in ['png','svg','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execution-dir',required=True);args=ap.parse_args();O=R/args.execution_dir;status=json.loads((O/'sequence_status.json').read_text());assert status['status'] in ['complete_all_five','completed_M1_M4_M5_blocked']
 F=O/'thesis_figures';F.mkdir(exist_ok=False);Data=F/'source_data';Data.mkdir();results=[];historyrows=[];classrows=[];receipt=[];bestrecords={};histories={}
 validation=read(R/json.loads((P/'configs/M1.json').read_text())['split_manifest']);validation={r['image']:r for r in validation if r['split']=='validation'}
 assert len(validation)==2835
 for m in MODELS:
  st=status['models'].get(m,{});run=P/f'runs/{m}_seed42'
  if st.get('status')!='complete':
   results.append({'model':m,'status':st.get('status','not_run'),'epochs_completed':0,'reason':st.get('reason','')});continue
  lines=(run/'history.jsonl').read_text().splitlines();hist=[json.loads(x) for x in lines];assert len(hist)==st['epochs_completed']
  epochs=[r['epoch'] for r in hist];assert epochs==list(range(1,len(hist)+1))
  best=max(hist,key=lambda x:x['validation']['macro_f1']);ep=best['epoch'];bestrecords[m]=best;histories[m]=hist
  cp=run/'checkpoints/best.pt';payload=cp.resolve();assert payload.name==f'epoch_{ep:03d}.pt';assert sha(cp)==st['best_checkpoint_sha256'];assert len(list((run/'checkpoints').glob('epoch_*.pt')))==len(hist)
  predictions=read(run/f'validation_predictions_epoch_{ep:03d}.csv');assert len(predictions)==2835;assert {r['image'] for r in predictions}==set(validation)
  cm=np.zeros((4,4),dtype=np.int64)
  for p in predictions:
   target=int(p['label_index']);assert target==int(validation[p['image']]['label_index']);probs=np.array([float(p[c]) for c in CLASSES]);assert np.isfinite(probs).all() and np.isclose(probs.sum(),1,atol=1e-5)
   cm[target,int(probs.argmax())]+=1
  assert cm.tolist()==best['validation']['confusion_matrix']
  measured,prec,rec,f1=metrics(cm)
  for k,v in measured.items():assert np.isclose(v,best['validation'][k],atol=1e-6),(m,k)
  assert np.isclose(measured['macro_f1'],st['best_validation_macro_f1'],atol=1e-6)
  result={'model':m,'status':'complete','best_epoch':ep,'epochs_completed':len(hist),**measured,'training_wall_seconds':st['elapsed_seconds'],'process_wall_seconds':st['process_wall_seconds'],'peak_allocated_GiB':max(r['peak_allocated_GiB'] for r in hist),'checkpoint':str(cp.relative_to(R)),'checkpoint_payload':str(payload.relative_to(R)),'checkpoint_sha256':sha(cp)};results.append(result)
  receipt.append({'model':m,'prediction_records':len(predictions),'prediction_confusion_matrix_matches_history':True,'best_checkpoint_hash_matches':True,'immutable_epoch_payloads':len(hist),'validation_only':True})
  for i,c in enumerate(CLASSES):classrows.append({'model':m,'class':c,'precision':float(prec[i]),'recall':float(rec[i]),'f1':float(f1[i]),'support':int(cm[i].sum())})
  for h in hist:
   historyrows.append({'model':m,'epoch':h['epoch'],'stage':h['stage'],'train_loss':h['train_loss'],'validation_loss':h['validation']['unweighted_loss'],'train_accuracy':h['train']['accuracy'],'validation_accuracy':h['validation']['accuracy'],'train_macro_f1':h['train']['macro_f1'],'validation_macro_f1':h['validation']['macro_f1'],'train_balanced_accuracy':h['train']['balanced_accuracy'],'validation_balanced_accuracy':h['validation']['balanced_accuracy'],'train_melanoma_recall':h['train']['melanoma_recall'],'validation_melanoma_recall':h['validation']['melanoma_recall'],'train_SCC_recall':h['train']['SCC_recall'],'validation_SCC_recall':h['validation']['SCC_recall'],'elapsed_seconds':h['elapsed_seconds'],'peak_allocated_GiB':h['peak_allocated_GiB'],'gate_acceptance':h['gate_acceptance']})
  savecsv(Data/f'{m}_best_validation_confusion_matrix.csv',[{'true_class':c,**{p:int(cm[i,j]) for j,p in enumerate(CLASSES)}} for i,c in enumerate(CLASSES)])
  fig,axes=plt.subplots(1,3,figsize=(13,3.7))
  for ax,key,title in zip(axes,['loss','accuracy','macro_f1'],['Loss','Accuracy','Macro-F1']):
   train=[h['train_loss'] if key=='loss' else h['train'][key] for h in hist];val=[h['validation']['unweighted_loss'] if key=='loss' else h['validation'][key] for h in hist]
   ax.plot(epochs,train,label='Training',color='#4477AA');ax.plot(epochs,val,label='Validation',color='#EE6677');ax.axvline(ep,color='#555555',ls=':',lw=1,label='Selected epoch');ax.set(xlabel='Epoch',ylabel=title,title=f'{m} {title}');ax.spines[['top','right']].set_visible(False)
   if key!='loss':ax.set_ylim(0,1)
  axes[0].legend(frameon=False,fontsize=8);fig.suptitle('Weighted training loss; unweighted validation loss',fontsize=9);savefig(fig,F,f'{m}_learning_curves')
 savecsv(Data/'epoch_histories.csv',historyrows);savecsv(Data/'model_validation_summary.csv',results);savecsv(Data/'classwise_validation_metrics.csv',classrows)
 # Five fixed slots; unavailable conditional model is explicitly NA, never a fabricated bar.
 completed={r['model']:r for r in results if r['status']=='complete'};x=np.arange(5);fig,ax=plt.subplots(figsize=(8,4.5));width=.26
 for offset,key,label,color in [(-width,'accuracy','Accuracy','#4477AA'),(0,'macro_f1','Macro-F1','#EE6677'),(width,'balanced_accuracy','Balanced accuracy','#228833')]:
  vals=[completed[m][key] if m in completed else np.nan for m in MODELS];bars=ax.bar(x+offset,vals,width,label=label,color=color);ax.bar_label(bars,fmt='%.3f',fontsize=7,padding=2)
 for i,m in enumerate(MODELS):
  if m not in completed:ax.text(i,.08,'Not trained\nteacher gate failed',ha='center',fontsize=8)
 ax.set_xticks(x,MODELS);ax.set_ylim(0,1.09);ax.set_ylabel('Best-checkpoint validation metric');ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False);savefig(fig,F,'five_model_validation_comparison')
 fig,axes=plt.subplots(1,2,figsize=(11,4));present=list(completed);width=.8/max(1,len(present));class_x=np.arange(4)
 for j,m in enumerate(present):
  cs=[next(r for r in classrows if r['model']==m and r['class']==c) for c in CLASSES]
  for ax,key,title in zip(axes,['recall','f1'],['Class sensitivity / recall','Class F1']):ax.bar(class_x+(j-(len(present)-1)/2)*width,[r[key] for r in cs],width,label=m);ax.set_xticks(class_x,CLASSES);ax.set_ylim(0,1.08);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
 axes[0].legend(frameon=False,ncol=3,fontsize=8);savefig(fig,F,'classwise_validation_comparison')
 # Combined validation trajectories from measured histories only.
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for m,hist in histories.items():
  for ax,key in zip(axes,['accuracy','macro_f1']):ax.plot([h['epoch'] for h in hist],[h['validation'][key] for h in hist],label=m);ax.set(xlabel='Epoch',ylabel='Validation '+key.replace('_',' '));ax.set_ylim(0,1);ax.spines[['top','right']].set_visible(False)
 axes[0].legend(frameon=False,fontsize=8);savefig(fig,F,'validation_metric_curves')
 gate=json.loads((O/'teacher_qualification.json').read_text());delta=completed['M5']['macro_f1']-completed['M3']['macro_f1'] if 'M5' in completed else None
 if 'M5' in completed:
  hist=histories['M5'];gaterows=[]
  for h in hist:
   for c in CLASSES:gaterows.append({'epoch':h['epoch'],'class':c,'accepted_fraction':h['gate_acceptance_by_class'][c]})
  savecsv(Data/'M5_gate_coverage.csv',gaterows)
  fig,ax=plt.subplots(figsize=(7,4))
  for c in CLASSES:ax.plot([h['epoch'] for h in hist],[h['gate_acceptance_by_class'][c] for h in hist],label=c)
  ax.set(xlabel='M5 additional epoch',ylabel='Training teacher-gate acceptance');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);savefig(fig,F,'M5_training_gate_coverage')
 final={'execution_directory':str(O.relative_to(R)),'training_directory':str(P.relative_to(R)),'dataset_version':status['dataset_version'],'models':results,'teacher_qualification':gate,'M5_minus_M3_validation_macro_F1':delta,'M5_improved_on_M3':delta>0 if delta is not None else None,'sum_training_wall_seconds':sum(r.get('training_wall_seconds',0) for r in results),'sequence_wall_seconds':status['sequence_wall_seconds'],'test_PAD_evaluated':False,'prediction_checkpoint_consistency_checks':receipt,'extra_budget_limitation':'M5 inherits M3 and receives additional epochs; any gain cannot be attributed exclusively to the candidate objective without separately authorized matched-budget controls','novelty':'Unestablished; direct overlap with confidence/cross-view and class-balanced distillation documented in Phase5A literature review'}
 (O/'final_results.json').write_text(json.dumps(final,indent=2)+'\n')
 table='\n'.join(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['melanoma_recall']:.4f} | {r['SCC_recall']:.4f} | {r['epochs_completed']} | {r['training_wall_seconds']/60:.2f} |" if r['status']=='complete' else f"| {r['model']} | NA | NA | NA | NA | 0 | Not trained |" for r in results)
 checkpoint_list='\n'.join('- '+r['model']+': `'+r['checkpoint']+'` (payload `'+r['checkpoint_payload']+'`)' for r in results if r['status']=='complete')
 report=f'''# Phase5B final training results — validation only

Dataset `{status['dataset_version']}`; five fixed Phase5A configurations. These are fresh final research runs, not demonstration checkpoints. No locked ISIC test or PAD image was evaluated or optimized. Selection is highest validation macro-F1 (first maximum on ties); validation results are selection evidence, not unbiased heldout generalization estimates.

| Model | Validation accuracy | Validation macro-F1 | MEL sensitivity | SCC sensitivity | Actual epochs | Training minutes |
|---|---:|---:|---:|---:|---:|---:|
{table}

All values are from the **same selected checkpoint per model**, not independently maximized accuracy/recall epochs. Training time includes epochs, validation and checkpoint writing; process time is separately recorded. Total training time {final['sum_training_wall_seconds']/60:.2f}minutes; sequence wall {status['sequence_wall_seconds']/60:.2f}minutes. Peak PyTorch allocated VRAM is in source CSV and final_results.json; it excludes WSL/desktop/driver allocations.

Teacher qualified: **{gate['qualified']}**; fresh M4 validation macro-F1 {gate['M4_validation_macro_F1']:.6f} versus M3 {gate['M3_validation_macro_F1']:.6f}, required margin {gate['required_margin']}. M5 minus M3 validation macro-F1: **{delta if delta is not None else 'NA — teacher gate failed; M5 not trained'}**. If M5 runs, its M3 starting point and M4 teacher hashes/run IDs are in checkpoint provenance. No substitute or extra full configuration was launched.

M5 receives up to40additional student epochs after M3's training plus M4 teacher compute. Any improvement cannot be attributed exclusively to the candidate mechanism. Confidence/class-aware/cross-view distillation has prior art; no novelty claim is established by these results. Phase5A/LITERATURE_REVIEW.md records overlaps and full-text access limitations. Biological patient independence remains limited by missing identifiers despite audited release quarantine.

## Checkpoints

{checkpoint_list}

Each run contains resolved_config.json, history.jsonl, validation_predictions_epoch_###.csv, run_status.json and exclusive-create epoch checkpoint payloads. best.pt/last.pt are selectors to retained immutable payloads. Training logs are {str(O.relative_to(R))}/M*_stdout.log. Persistent experiment summary: {str((P/'experiment_summary.json').relative_to(R))}.

## Thesis assets

`thesis_figures/` contains per-model loss/accuracy/macro-F1 curves, five-slot validation comparison, combined validation curves and class-wise validation recall/F1 comparisons. M5 gate coverage is included only if M5 actually ran. Files are PNG300DPI plus SVG/PDF. Exact source CSVs are in `thesis_figures/source_data/`. Weighted train/unweighted validation losses have different definitions, as indicated in figures. For M5, training loss is the full CE+KD+consistency objective while validation loss remains unweighted CE; they are not the same loss function. Training scores are measured online with train augmentation and changing parameters; validation is deterministic. No methodology figures were generated.

Verification recomputed selected-checkpoint confusion matrices and all requested metrics from2,835validation prediction records per completed model, verified selected payload hashes and retained epoch counts. `final_results.json` holds machine-readable results and remaining limitations. Locked test/PAD evaluation remains a separate later stage.
'''
 (O/'PHASE5B_RESULTS.md').write_text(report)
 print(json.dumps(final,indent=2))
if __name__=='__main__':main()
