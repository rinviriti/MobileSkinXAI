"""Exactly five runs. Approved model/loss/data/optimizer functions unchanged; durable epoch resume."""
from common import *
from model import Student
from pipeline import SourceDataset,STRESS
from losses import Projector,objective
from approved_train import optimizer_for,mode,metric,validate
import argparse,time,traceback,os,shutil,gc
import torch
from torch.utils.data import DataLoader

def sealcheck(out):
 lock=load(out/'00_Experiment_Protocol/PROTOCOL_LOCK.json')
 for name,h in lock['sha256'].items():assert sha(out/name)==h,name
 for p in Path(__file__).parent.glob('*.py'):
  key='code/'+p.name
  if key in lock['sha256']:assert sha(p)==lock['sha256'][key],'Executing unsealed source: '+str(p)
 assert load(out/'TRAINING_AUTHORIZATION.json')['authorized']
 assert load(out/'00_Experiment_Protocol/PREFLIGHT.json')['status']=='PASS'
 assert load(out/'00_Experiment_Protocol/CUDA_PREREQUISITES.json')['status']=='PASS'

def rng_state():
 n=np.random.get_state()
 return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy':{'name':n[0],'keys':n[1].tolist(),'pos':int(n[2]),'has_gauss':int(n[3]),'cached_gaussian':float(n[4])}}
def restore_rng(r):
 torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda']);random.setstate(r['python']);n=r['numpy'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
def atomic_json(path,value):
 tmp=path.with_name(path.name+'.next');dump(tmp,value);os.replace(tmp,path)
def aliases(cpdir,epoch,improved):
 for name in (['best.pt','last.pt'] if improved else ['last.pt']):
  temp=cpdir/(name+'.next')
  if temp.is_symlink():temp.unlink()
  temp.symlink_to(f'epoch_{epoch:03d}.pt');os.replace(temp,cpdir/name)
def history(run):return [json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()] if (run/'history.jsonl').exists() else []
def archive_uncommitted(run,last):
 patterns=['augmentation_epoch_*.jsonl','batch_diagnostics_epoch_*.jsonl','validation_epoch_*.csv']
 candidates=[]
 for pattern in patterns:
  for p in run.glob(pattern):
   if int(p.stem.split('_')[-1])>last:candidates.append(p)
 for p in (run/'checkpoints').glob('epoch_*.pt'):
  if int(p.stem.split('_')[-1])>last:candidates.append(p)
 hs=history(run);extra=[r for r in hs if r['epoch']>last]
 if candidates or extra:
  archive=run/'recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');archive.mkdir(parents=True)
  for p in candidates:shutil.move(str(p),archive/p.name)
  if extra:
   shutil.copyfile(run/'history.jsonl',archive/'uncommitted_history_full.jsonl');(run/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in hs if r['epoch']<=last))
  dump(archive/'recovery_receipt.json',{'last_committed_epoch':last,'archived_files':[p.name for p in candidates],'uncommitted_history_rows':len(extra),'reason':'replay incomplete epoch from verified boundary; no completed model restarted'})

def stress_evaluate(out,cfg,net,run,cp_sha):
 results={}
 for condition in STRESS:
  pred=run/f'stress_{condition}.csv';side=run/f'stress_{condition}_metrics.json'
  if side.exists():
   record=load(side);assert record['checkpoint_sha256']==cp_sha and sha(pred)==record['predictions_sha256'];results[condition]=record['metrics'];continue
  if pred.exists():
   archive=run/'recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');archive.mkdir(parents=True);shutil.move(str(pred),archive/pred.name)
  values=validate(out,cfg,net,pred,stress=condition);dump(side,{'checkpoint_sha256':cp_sha,'predictions_sha256':sha(pred),'metrics':values});results[condition]=values
 dump(run/'source_stress_results.json',results)

def execute(out,cfg,resume=False):
 sealcheck(out);assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();assert shutil.disk_usage(out).free>5*2**30
 key=cfg['id'];run=out/'runs'/key;fresh=not run.exists()
 if not fresh:
  status=load(run/'status.json')
  if status['status']=='complete':
   assert sha(run/'checkpoints/best.pt')==status['best_checkpoint_sha256'];print('SKIP VERIFIED COMPLETE',key,flush=True);return
  assert resume,'Partial run exists; explicit --resume required'
 else:run.mkdir();(run/'checkpoints').mkdir()
 seed_all(cfg['seed']);torch.set_num_threads(4);net=Student(cfg['fusion'],True,cfg['seed']).cuda();projector=Projector(cfg['seed']).cuda();weights=torch.tensor(load(out/'source_data/train_class_statistics.json')['weights'],device='cuda');best=-1.;bestepoch=None;stale=0;previous=None;last=0;updates=0;accumulated=0.;old_peak=0.;opt=None
 if not fresh:
  commits=sorted(run.glob('commit_epoch_*.json'));last=int(commits[-1].stem.split('_')[-1]) if commits else 0
  archive_uncommitted(run,last)
  if commits:
   receipt=load(commits[-1]);cp=run/'checkpoints'/receipt['checkpoint'];assert sha(cp)==receipt['sha256'];state=torch.load(cp,map_location='cpu',weights_only=True);assert state['config_sha256']==sha(out/f'configs/{key}.json');net.load_state_dict(state['student'],strict=True)
   if cfg['contrastive']:projector.load_state_dict(state['training_projector'],strict=True)
   control=state['training_control'];best=control['best'];bestepoch=control['best_epoch'];stale=control['stale'];previous=control['frozen'];updates=control['global_updates'];accumulated=control['elapsed_seconds'];old_peak=control['peak_allocated_GiB'];opt=optimizer_for(net,projector,cfg,previous);opt.load_state_dict(state['optimizer']);restore_rng(state['rng_state']);del state
   aliases(run/'checkpoints',last,False)
   temp=run/'checkpoints/best.pt.next'
   if temp.is_symlink():temp.unlink()
   temp.symlink_to(f'epoch_{bestepoch:03d}.pt');os.replace(temp,run/'checkpoints/best.pt')
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter();status_path=run/'status.json';atomic_json(status_path,{'status':'running','resuming_after_epoch':last,'config_sha256':sha(out/f'configs/{key}.json'),'started_UTC':datetime.now(timezone.utc).isoformat()})
 print('START' if fresh else 'RESUME',key,'epoch',last+1,flush=True)
 try:
  stop_already=last>=cfg['frozen_epochs']+cfg['fine_epochs'] or (last>cfg['frozen_epochs'] and stale>=cfg['patience'])
  epochs=[] if stop_already else range(last+1,cfg['frozen_epochs']+cfg['fine_epochs']+1)
  for epoch in epochs:
   frozen=epoch<=cfg['frozen_epochs'];fine=max(0,epoch-cfg['frozen_epochs'])
   if previous is None or frozen!=previous:opt=optimizer_for(net,projector,cfg,frozen);stale=0;previous=frozen
   mode(net);projector.train();ds=SourceDataset(out,cfg,'train',epoch=epoch);loader=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(cfg['seed']+epoch),num_workers=cfg['num_workers'],pin_memory=True);sums={};n=0;ys=[];ps=[];norms=[];order=hashlib.sha256();class_images=[0]*4;lesion_sets=[set() for _ in range(4)];epoch_updates=0
   with (run/f'augmentation_epoch_{epoch:03d}.jsonl').open('x') as augfile,(run/f'batch_diagnostics_epoch_{epoch:03d}.jsonl').open('x') as diagfile:
    for weak,strong,y,ids,lesions,metas in loader:
     for name,meta in zip(ids,metas):order.update((name+'\n').encode());augfile.write(meta+'\n')
     labels=y.tolist();counts=[labels.count(j) for j in range(4)];lesion_counts=[len({lid for lab,lid in zip(labels,lesions) if lab==j}) for j in range(4)]
     for label,lid in zip(labels,lesions):class_images[label]+=1;lesion_sets[label].add(lid)
     weak,strong,y=weak.cuda(non_blocking=True),strong.cuda(non_blocking=True),y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.bfloat16):sw,fw=net.training_outputs(weak);ss,fs=net.training_outputs(strong);loss,stats=objective(sw,ss,fw,fs,y,weights,projector,cfg,fine)
     assert torch.isfinite(loss);loss.backward();params=[p for p in list(net.parameters())+list(projector.parameters()) if p.grad is not None];assert torch.stack([torch.isfinite(p.grad).all() for p in params]).all();norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip_norm']);assert torch.isfinite(norm);norms.append(float(norm));opt.step();updates+=1;epoch_updates+=1;size=len(y);n+=size;sums['total']=sums.get('total',0)+float(loss.detach())*size
     for name in ['ce','js','supcon']:sums[name]=sums.get(name,0)+float(stats[name])*size
     ys.extend(y.cpu().tolist());ps.extend(sw.detach().float().softmax(1).cpu().tolist());diagfile.write(json.dumps({'class_images':counts,'class_distinct_lesions':lesion_counts,'SCC_single_lesion_in_batch':lesion_counts[3]==1,'contrastive_active':cfg['contrastive'] and fine>0,'minimum_positives':stats.get('minimum_positives'),'skipped_single_class':stats.get('skipped_single_class',False),'preclip_gradient_norm':float(norm)})+'\n')
   assert n==11632 and epoch_updates==727,'Two views must NOT double optimizer updates'
   val=validate(out,cfg,net,run/f'validation_epoch_{epoch:03d}.csv');improved=val['macro_f1']>best
   if improved:best=val['macro_f1'];bestepoch=epoch;stale=0
   elif not frozen:stale+=1
   elapsed=accumulated+time.perf_counter()-started;peak=max(old_peak,torch.cuda.max_memory_allocated()/2**30)
   rec={'epoch':epoch,'stage':'head' if frozen else 'fine','fine_epoch':fine,'loss_components':{k:v/n for k,v in sums.items()},'train':metric(ys,ps),'validation':val,'learning_rates':{g['role']:g['lr'] for g in opt.param_groups},'auxiliary_ramp':min(fine/5,1),'mean_preclip_gradient_norm':float(np.mean(norms)),'maximum_preclip_gradient_norm':max(norms),'class_images':class_images,'class_distinct_lesions':[len(x) for x in lesion_sets],'training_order_sha256':order.hexdigest(),'elapsed_seconds':elapsed,'peak_allocated_GiB':peak,'optimizer_updates_this_epoch':epoch_updates,'global_updates':updates,'stale':stale,'best_epoch_so_far':bestepoch}
   with (run/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n');f.flush();os.fsync(f.fileno())
   cp=run/f'checkpoints/epoch_{epoch:03d}.pt';payload={'student':{k:v.detach().cpu() for k,v in net.state_dict().items()},'training_projector':projector.state_dict() if cfg['contrastive'] else None,'optimizer':opt.state_dict(),'epoch':epoch,'validation':val,'config':cfg,'config_sha256':sha(out/f'configs/{key}.json'),'ImageNet_sha256':sha(PRETRAIN/'model.safetensors'),'dataset_manifest_sha256':cfg['development_manifest_sha256'],'artifact_kind':'CA_MACL_new_research_run','training_control':{'best':best,'best_epoch':bestepoch,'stale':stale,'frozen':frozen,'global_updates':updates,'elapsed_seconds':elapsed,'peak_allocated_GiB':peak},'rng_state':rng_state()}
   with cp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
   dump(run/f'commit_epoch_{epoch:03d}.json',{'epoch':epoch,'checkpoint':cp.name,'sha256':sha(cp),'history_epoch':epoch});aliases(cp.parent,epoch,improved);last=epoch;del payload
   atomic_json(status_path,{'status':'running','last_committed_epoch':last,'best_epoch':bestepoch,'best_macro_f1':best,'elapsed_training_seconds':elapsed,'global_updates':updates})
   print(key,'epoch',epoch,'stage',rec['stage'],'val_F1',round(val['macro_f1'],6),'stale',stale,'seconds',round(elapsed,1),flush=True)
   if not frozen and stale>=cfg['patience']:break
  training_seconds=accumulated+time.perf_counter()-started;cp=run/'checkpoints/best.pt';cp_sha=sha(cp);state=torch.load(cp,map_location='cpu',weights_only=True);net.load_state_dict(state['student'],strict=True);del state
  lock={'checkpoint_sha256':cp_sha,'epoch':bestepoch,'selection':'clean validation macro-F1 only','created_UTC':datetime.now(timezone.utc).isoformat()}
  if (run/'SELECTED_CHECKPOINT_LOCK.json').exists():assert load(run/'SELECTED_CHECKPOINT_LOCK.json')['checkpoint_sha256']==cp_sha
  else:dump(run/'SELECTED_CHECKPOINT_LOCK.json',lock);(run/'SELECTED_CHECKPOINT_LOCK.json').chmod(0o444)
  stress_start=time.perf_counter();stress_evaluate(out,cfg,net,run,cp_sha);stress_seconds=time.perf_counter()-stress_start
  status={'status':'complete','best_epoch':bestepoch,'epochs':last,'training_seconds':training_seconds,'stress_seconds':stress_seconds,'process_elapsed_seconds_this_attempt':time.perf_counter()-started,'peak_allocated_GiB':max(old_peak,torch.cuda.max_memory_allocated()/2**30),'inference_parameters':sum(p.numel() for p in net.parameters()),'training_only_projector_parameters':sum(p.numel() for p in projector.parameters()) if cfg['contrastive'] else 0,'best_checkpoint':str(cp.relative_to(ROOT)),'best_checkpoint_sha256':cp_sha,'last_checkpoint_sha256':sha(run/'checkpoints/last.pt'),'global_updates':updates,'config_sha256':sha(out/f'configs/{key}.json')};atomic_json(status_path,status)
  atomic_json(out/'01_Training_Histories'/f'{key}_completion.json',status)
  all_status=load(out/'EXPERIMENT_SUMMARY.json');all_status['models'][key]=status;atomic_json(out/'EXPERIMENT_SUMMARY.json',all_status)
  print('COMPLETE',key,'best_epoch',bestepoch,'training_seconds',round(training_seconds,1),flush=True)
 except BaseException as exc:
  with (run/'attempt_errors.log').open('a') as f:f.write(datetime.now(timezone.utc).isoformat()+'\n'+traceback.format_exc()+'\n')
  atomic_json(status_path,{'status':'failed','last_committed_epoch':last,'error':repr(exc),'resumable_from_committed_boundary':True});raise
 finally:
  del net,projector,opt;gc.collect();torch.cuda.empty_cache()

def main(out,resume=False):
 out=Path(out);sealcheck(out)
 for i in range(5):execute(out,load(out/f'configs/C{i}.json'),resume=resume)
 atomic_json(out/'ALL_FIVE_COMPLETE.json',{'status':'PASS','exactly_five':['C0','C1','C2','C3','C4'],'heldout_access':False,'completed_UTC':datetime.now(timezone.utc).isoformat()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--resume',action='store_true');a=p.parse_args();main(ROOT/a.output,a.resume)
