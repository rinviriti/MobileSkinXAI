"""Verify provenance and create portable actual-result sources; never reads test/PAD images."""
import gc,importlib.metadata as md,platform,shutil
from common import *
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,auc
import torch

def prepare(c):
 out=c['out'];data=c['data'];assert not (data/'package_inputs.json').exists(),'Prepare must run once per new package'
 for f,lockname in [(RELEASE,'RELEASE_LOCK.json'),(PREP,'PREPARATION_LOCK.json')]:
  lock=json.loads((f/lockname).read_text());assert sha(f/'checksums.sha256')==lock['checksums_sha256']
  for line in (f/'checksums.sha256').read_text().splitlines():h,p=line.split('  ',1);assert sha(f/p)==h,p
 assert json.loads((RELEASE/'data_approval.json').read_text())['approved'] is True
 snapshot=json.loads((CONTROL/'reports/original_artifact_snapshot.json').read_text())
 for p in CONTROL.rglob('*'):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:snapshot[rel(p)]=sha(p)
 for p,h in snapshot.items():assert sha(ROOT/p)==h,p
 write_json(data/'package_inputs.json',snapshot)
 for n in ['dataset_manifest.csv','class_weights.json','class_split_counts.csv','data_approval.json','integrity_checks.json','DATASET_PROTOCOL.json','DATASET_RELEASE_REPORT.md','evaluation_denominators.csv']:shutil.copyfile(RELEASE/n,data/n)
 val=c['validation'];ids=sorted(val);y=np.array([int(val[i]['label_index']) for i in ids]);assert len(ids)==2835
 allmetrics=[];classwise=[];aucs=[];curves=[];epochs=[];lrs=[];inventory=[];gate=[];provenance=[];probrows=[]
 for k,r in c['models'].items():
  assert sha(r['checkpoint'])==r['checkpoint_sha256'],k
  destination=data/'checkpoints'/f'{k}_selected_checkpoint.pt';destination.parent.mkdir(exist_ok=True);assert not destination.exists();shutil.copyfile(r['checkpoint'],destination);assert sha(destination)==r['checkpoint_sha256']
  run_copy=data/'recorded_runs'/k;run_copy.mkdir(parents=True)
  for p in r['run'].glob('*'):
   if p.is_file() and p.suffix in ['.json','.jsonl','.csv']:shutil.copyfile(p,run_copy/p.name)
  shutil.copyfile(r['config_path'],run_copy/'frozen_configuration.json')
  logfile=CONTROL/'training_stdout.log' if k=='M5-LR' else BASE/f'{k}_stdout.log';shutil.copyfile(logfile,run_copy/'training_stdout.log')
  best=max(r['history'],key=lambda h:h['validation']['macro_f1']);assert best['epoch']==r['best_epoch']
  saved=read_csv(run_copy/f"validation_predictions_epoch_{r['best_epoch']:03d}.csv");pr={x['image']:x for x in saved};assert len(saved)==len(pr)==len(val) and set(pr)==set(ids)
  assert all(int(pr[i]['label_index'])==int(val[i]['label_index']) for i in ids)
  p=np.array([[float(pr[i][s]) for s in CLASSES] for i in ids]);assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5);pred=p.argmax(1);cm=np.bincount(y*4+pred,minlength=16).reshape(4,4);assert cm.tolist()==best['validation']['confusion_matrix']
  tp=np.diag(cm);support=cm.sum(1);pp=cm.sum(0);precision=tp/np.maximum(pp,1);recall=tp/support;f1=2*tp/np.maximum(support+pp,1);accuracy=tp.sum()/cm.sum();macro=f1.mean();balanced=recall.mean();assert abs(macro-r['macro_f1'])<1e-6
  localauc=[];localap=[]
  for j,cl in enumerate(CLASSES):
   truth=(y==j).astype(int);roc=roc_auc_score(truth,p[:,j]);ap=average_precision_score(truth,p[:,j]);fpr,tpr,threshold=roc_curve(truth,p[:,j],drop_intermediate=False);pre,rec,prthr=precision_recall_curve(truth,p[:,j]);trapezoid=auc(rec,pre)
   localauc.append(roc);localap.append(ap);aucs.append({'model':k,'model_name':NAMES[k],'class':cl,'support':int(support[j]),'AUROC_OvR':float(roc),'AUPRC_average_precision':float(ap),'PR_trapezoidal_area_secondary':float(trapezoid),'class_prevalence':float(truth.mean()),'probability_type':'saved raw softmax; uncalibrated; BF16 CUDA validation logits originally converted tofloat32 before softmax'})
   classwise.append({'model':k,'model_name':NAMES[k],'class':cl,'precision':float(precision[j]),'recall':float(recall[j]),'f1':float(f1[j]),'support':int(support[j]),'true_positives':int(tp[j]),'false_positives':int(pp[j]-tp[j]),'false_negatives':int(support[j]-tp[j]),'true_negatives':int(cm.sum()-support[j]-pp[j]+tp[j])})
   for i in range(len(fpr)):curves.append({'model':k,'model_name':NAMES[k],'class':cl,'curve':'ROC','x':float(fpr[i]),'y':float(tpr[i]),'threshold':float(threshold[i]) if np.isfinite(threshold[i]) else 'inf'})
   for i in range(len(pre)):curves.append({'model':k,'model_name':NAMES[k],'class':cl,'curve':'PR','x':float(rec[i]),'y':float(pre[i]),'threshold':float(prthr[i]) if i<len(prthr) else ''})
  for i,imageid in enumerate(ids):probrows.append({'model':k,'model_name':NAMES[k],'image':imageid,'true_class':CLASSES[y[i]],'label_index':int(y[i]),'predicted_class':CLASSES[pred[i]],'probability_type':'raw uncalibrated softmax',**{cl:float(p[i,j]) for j,cl in enumerate(CLASSES)}})
  allmetrics.append({'model':k,'model_name':NAMES[k],'accuracy':float(accuracy),'macro_f1':float(macro),'balanced_accuracy':float(balanced),'macro_AUROC_OvR':float(np.mean(localauc)),'macro_AUPRC_average_precision':float(np.mean(localap)),'MEL_precision':float(precision[1]),'MEL_recall':float(recall[1]),'SCC_precision':float(precision[3]),'SCC_recall':float(recall[3]),'NV_to_MEL':int(cm[0,1]),'SCC_to_BCC':int(cm[3,2]),'best_epoch':r['best_epoch'],'total_epochs':r['epochs_completed'],'epoch_definition':'additional from same final MobileNetV4-Conv-Small parent' if k in ['M5','M5-LR'] else 'total training epoch','training_wall_seconds':r['training_wall_seconds'],'peak_allocated_GiB':r['peak_allocated_GiB'],'full_checkpoint_bytes':r['checkpoint'].stat().st_size,'original_checkpoint':rel(r['checkpoint']),'packaged_checkpoint':str(destination.relative_to(out)),'checkpoint_sha256':sha(destination)})
  write_csv(data/f'{k}_confusion_counts.csv',[{'true_class':cl,**{d:int(cm[i,j]) for j,d in enumerate(CLASSES)}} for i,cl in enumerate(CLASSES)])
  write_csv(data/f'{k}_confusion_row_percent.csv',[{'true_class':cl,'support':int(support[i]),**{d:float(100*cm[i,j]/support[i]) for j,d in enumerate(CLASSES)}} for i,cl in enumerate(CLASSES)])
  for h in r['history']:
   epochs.append({'model':k,'model_name':NAMES[k],'epoch':h['epoch'],'stage':h['stage'],'train_loss':h['train_loss'],'train_loss_definition':'weighted two-viewCE+0.5KD+0.1JS' if k in ['M5','M5-LR'] else 'weighted single-viewCE','validation_loss':h['validation']['unweighted_loss'],'validation_loss_definition':'unweighted cross entropy','train_accuracy':h['train']['accuracy'],'validation_accuracy':h['validation']['accuracy'],'train_macro_f1':h['train']['macro_f1'],'validation_macro_f1':h['validation']['macro_f1'],'elapsed_seconds':h['elapsed_seconds'],'selected_epoch':h['epoch']==r['best_epoch'],'head_lr':h.get('head_lr',''),'backbone_lr':h.get('backbone_lr',''),'CE':h.get('loss_components',{}).get('ce',''),'weighted_KD':.5*h['loss_components']['kd'] if 'loss_components' in h else '', 'weighted_JS':.1*h['loss_components']['consistency'] if 'loss_components' in h else '', 'gate_acceptance':h.get('gate_acceptance')})
   if k in ['M5','M5-LR']:
    for cl in CLASSES:gate.append({'model':k,'model_name':NAMES[k],'additional_epoch':h['epoch'],'class':cl,'gate_accepted_fraction':h['gate_acceptance_by_class'][cl],'overall_hard_gate_fraction':h['gate_acceptance'],'mean_reliability_mass':h.get('loss_components',{}).get('gate_mass','')})
   cp_path=r['run']/f"checkpoints/epoch_{h['epoch']:03d}.pt";cp=torch.load(cp_path,map_location='cpu',weights_only=True);prov=cp['provenance'];assert prov['artifact_kind']=='final_training_checkpoint' and prov['classes']==CLASSES and prov['data_receipt']['manifest_sha256']==json.loads((RELEASE/'data_approval.json').read_text())['manifest_sha256']
   groups=cp['optimizer']['param_groups'];lrs.append({'model':k,'model_name':NAMES[k],'epoch':h['epoch'],'head_lr':groups[0]['lr'],'backbone_lr':groups[1]['lr'] if len(groups)>1 else 0.,'backbone_optimizer_group_present':len(groups)>1,'backbone_zero_definition':'0 denotes frozen/no optimizer group, not a logged optimizer LR' if len(groups)==1 else 'actual optimizer LR','stage':h['stage'],'source_checkpoint':rel(cp_path)})
   inventory.append({'model':k,'epoch':h['epoch'],'selected':h['epoch']==r['best_epoch'],'original_checkpoint':rel(cp_path),'bytes':cp_path.stat().st_size,'sha256':sha(cp_path),'copy_in_package':str(destination.relative_to(out)) if h['epoch']==r['best_epoch'] else 'not duplicated; original immutable payload retained'})
   if h['epoch']==r['best_epoch']:provenance.append({'model':k,'model_name':NAMES[k],'provenance':prov})
   del cp
  gc.collect();print('VERIFIED '+NAMES[k],flush=True)
 write_csv(data/'model_metrics.csv',allmetrics);write_csv(data/'classwise_metrics.csv',classwise);write_csv(data/'roc_pr_metrics.csv',aucs);write_csv(data/'roc_pr_curve_points.csv',curves);write_csv(data/'epoch_histories.csv',epochs);write_csv(data/'exact_learning_rates.csv',lrs);write_csv(data/'teacher_gates.csv',gate);write_csv(data/'checkpoint_inventory.csv',inventory);write_csv(data/'selected_validation_probabilities.csv',probrows);write_json(data/'checkpoint_provenance.json',provenance)
 write_json(data/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'versions':{n:md.version(n) for n in ['torch','torchvision','timm','numpy','scikit-learn','matplotlib','Pillow']},'analysis_inference':'CPUfloat32 for model shapes/MACs/Grad-CAM only; saved BF16 validation probabilities used for ROC/PR','no_training':True})
 write_json(data/'PREPARE_COMPLETE.json',{'models':6,'validation_images_per_model':2835,'checkpoint_payloads_audited':len(inventory),'selected_checkpoints_copied':6,'probability_source':'actual saved raw uncalibrated softmax','all_four_classes_present':True,'test_PAD_images_opened':False})
