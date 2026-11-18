"""Full trainer is implemented but disabled unless separately authorized. Never loads held-out sets."""
from common import *
from model import Student
from pipeline import SourceDataset,STRESS
from losses import Projector,objective
import argparse,time,os,torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support

def optimizer_for(net,projector,cfg,frozen):
 head=[];backbone=[]
 for name,p in net.named_parameters():
  is_head=(name.startswith('backbone.classifier.') or not name.startswith('backbone.'));p.requires_grad_(is_head or not frozen)
  (head if is_head else backbone).append(p)
 groups=[{'params':head,'lr':cfg['frozen_head_lr'] if frozen else cfg['head_lr'],'role':'head'}]
 if not frozen:groups.append({'params':backbone,'lr':cfg['backbone_lr'],'role':'backbone'})
 if cfg['contrastive']:groups.append({'params':list(projector.parameters()),'lr':cfg['frozen_head_lr'] if frozen else cfg['head_lr'],'role':'training_only_projector'})
 return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
def mode(net):
 net.train()
 for m in net.modules():
  if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
def metric(y,p):
 import numpy as np
 y=np.array(y);p=np.array(p);pred=p.argmax(1);pr,re,f,s=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0);cm=confusion_matrix(y,pred,labels=range(4))
 return {'accuracy':float((pred==y).mean()),'macro_f1':float(f.mean()),'balanced_accuracy':float(re.mean()),'classwise':{c:{'precision':float(pr[j]),'recall':float(re[j]),'f1':float(f[j]),'support':int(s[j])} for j,c in enumerate(CLASSES)},'NV_to_MEL':int(cm[0,1]),'SCC_to_BCC':int(cm[3,2]),'confusion_matrix':cm.tolist()}
@torch.no_grad()
def validate(out,cfg,net,path,stress=None):
 net.eval();loader=DataLoader(SourceDataset(out,cfg,'validation',stress=stress),batch_size=cfg['batch_size'],shuffle=False,num_workers=cfg['num_workers'],pin_memory=True);records=[];ys=[];ps=[];loss=0;n=0
 for x,y,ids in loader:
  x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x)
  ce=torch.nn.functional.cross_entropy(logits.float(),y);p=logits.float().softmax(1).cpu().tolist();labels=y.cpu().tolist();loss+=float(ce)*len(y);n+=len(y);ys.extend(labels);ps.extend(p)
  records.extend({'image':name,'label_index':label,**dict(zip(CLASSES,prob))} for name,label,prob in zip(ids,labels,p))
 assert not path.exists();write(path,records);return {**metric(ys,ps),'unweighted_CE':loss/n}
def guard(out,cfg):
 assert cfg['id'] in ['C0','C1','C2','C3','C4'];assert cfg['classes']==CLASSES;assert cfg['dataset_release']==RELEASE.name
 assert sha(out/'source_data/approved_train_validation.csv')==cfg['development_manifest_sha256']
 # Freeze implementation/config before training; never silently run edited code.
 seal=load(out/'IMPLEMENTATION_LOCK.json')
 for name,h in seal['sha256'].items():assert sha(out/name)==h,name
 assert load(out/'TRAINING_AUTHORIZATION.json')['authorized'],'FULL TRAINING DISABLED: separate explicit authorization required'
 assert load(out/'reports/IMPLEMENTATION_AUDIT.json')['technical_ready_for_RGB_training']
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
def execute(out,cfg):
 guard(out,cfg);run=out/'runs'/cfg['id'];assert not run.exists(),'Refuse overwrite/resume';run.mkdir();(run/'checkpoints').mkdir();seed_all(cfg['seed']);torch.set_num_threads(4);net=Student(cfg['fusion'],True,cfg['seed']).cuda();projector=Projector(cfg['seed']).cuda();weights=torch.tensor(load(out/'source_data/train_class_statistics.json')['weights'],device='cuda');best=-1;bestepoch=None;stale=0;frozenprev=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats();summary={}
 dump(run/'status.json',{'status':'running','config_sha256':sha(out/f'configs/{cfg["id"]}.json')})
 for epoch in range(1,cfg['frozen_epochs']+cfg['fine_epochs']+1):
  frozen=epoch<=cfg['frozen_epochs'];fine_epoch=max(0,epoch-cfg['frozen_epochs'])
  if frozenprev is None or frozen!=frozenprev:opt=optimizer_for(net,projector,cfg,frozen);stale=0;frozenprev=frozen
  mode(net);projector.train();dataset=SourceDataset(out,cfg,'train',epoch=epoch);loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(cfg['seed']+epoch),num_workers=cfg['num_workers'],pin_memory=True);sums={};n=0;ys=[];ps=[];gns=[];order=hashlib.sha256();class_images=[0]*4;distinct_lesions=[set() for _ in range(4)]
  with (run/f'augmentation_epoch_{epoch:03d}.jsonl').open('x') as augfile, (run/f'batch_diagnostics_epoch_{epoch:03d}.jsonl').open('x') as diagfile:
   for weak,strong,y,ids,lesions,metas in loader:
    for name,meta in zip(ids,metas):order.update((name+'\n').encode());augfile.write(meta+'\n')
    batch_labels=y.tolist()
    batch_counts=[batch_labels.count(j) for j in range(4)]
    batch_lesion_counts=[len({lesion for label,lesion in zip(batch_labels,lesions) if label==j}) for j in range(4)]
    for label,lesion in zip(batch_labels,lesions):class_images[label]+=1;distinct_lesions[label].add(lesion)
    weak,strong,y=weak.cuda(non_blocking=True),strong.cuda(non_blocking=True),y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):sw,fw=net.training_outputs(weak);ss,fs=net.training_outputs(strong);loss,stats=objective(sw,ss,fw,fs,y,weights,projector,cfg,fine_epoch)
    assert torch.isfinite(loss);loss.backward();params=[p for p in list(net.parameters())+list(projector.parameters()) if p.grad is not None];assert torch.stack([torch.isfinite(p.grad).all() for p in params]).all();norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip_norm']);assert torch.isfinite(norm);gns.append(float(norm));opt.step();size=len(y);n+=size;sums['total']=sums.get('total',0)+float(loss.detach())*size
    for name in ['ce','js','supcon']:sums[name]=sums.get(name,0)+float(stats[name])*size
    ys.extend(y.cpu().tolist());ps.extend(sw.detach().float().softmax(1).cpu().tolist())
    diagfile.write(json.dumps({'class_images':batch_counts,'class_distinct_lesions':batch_lesion_counts,'SCC_single_lesion_in_batch':batch_lesion_counts[3]==1,'contrastive_active':cfg['contrastive'] and fine_epoch>0,'minimum_positives':stats.get('minimum_positives'),'skipped_single_class':stats.get('skipped_single_class',False),'preclip_gradient_norm':float(norm)})+'\n')
  val=validate(out,cfg,net,run/f'validation_epoch_{epoch:03d}.csv');improved=val['macro_f1']>best
  if improved:best=val['macro_f1'];bestepoch=epoch;stale=0
  elif not frozen:stale+=1
  rec={'epoch':epoch,'stage':'head' if frozen else 'fine','fine_epoch':fine_epoch,'loss_components':{k:v/n for k,v in sums.items()},'train':metric(ys,ps),'validation':val,'learning_rates':{g['role']:g['lr'] for g in opt.param_groups},'mean_preclip_gradient_norm':float(np.mean(gns)),'maximum_preclip_gradient_norm':max(gns),'class_images':class_images,'class_distinct_lesions':[len(x) for x in distinct_lesions],'training_order_sha256':order.hexdigest(),'elapsed_seconds':time.perf_counter()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
  with (run/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  path=run/f'checkpoints/epoch_{epoch:03d}.pt';payload={'student':{k:v.detach().cpu() for k,v in net.state_dict().items()},'training_projector':projector.state_dict() if cfg['contrastive'] else None,'optimizer':opt.state_dict(),'epoch':epoch,'validation':val,'config':cfg,'config_sha256':sha(out/f'configs/{cfg["id"]}.json'),'ImageNet_sha256':sha(PRETRAIN/'model.safetensors'),'dataset_manifest_sha256':cfg['development_manifest_sha256'],'artifact_kind':'CA_MACL_new_research_run'}
  with path.open('xb') as f:torch.save(payload,f)
  for alias in (['best.pt','last.pt'] if improved else ['last.pt']):
   temp=path.parent/(alias+'.next');temp.symlink_to(path.name);os.replace(temp,path.parent/alias)
  print(cfg['id'],epoch,val['macro_f1'],flush=True)
  if not frozen and stale>=cfg['patience']:break
 # Freeze selected hash before source-only synthetic stress assessment; no further optimization.
 state=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=True);net.load_state_dict(state['student'],strict=True);dump(run/'SELECTED_CHECKPOINT_LOCK.json',{'checkpoint_sha256':sha(run/'checkpoints/best.pt'),'epoch':bestepoch,'selection':'clean validation macro-F1 only'})
 stress={name:validate(out,cfg,net,run/f'stress_{name}.csv',stress=name) for name in STRESS};dump(run/'source_stress_results.json',stress);dump(run/'status.json',{'status':'complete','best_epoch':bestepoch,'epochs':epoch,'seconds':time.perf_counter()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'best_checkpoint_sha256':sha(run/'checkpoints/best.pt')})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('configuration',choices=['C0','C1','C2','C3','C4']);p.add_argument('--execute',action='store_true');args=p.parse_args();out=ROOT/args.output;cfg=load(out/f'configs/{args.configuration}.json')
 if not args.execute:print(json.dumps(cfg,indent=2));print('INSPECTION ONLY; training authorization remains false')
 else:execute(out,cfg)
