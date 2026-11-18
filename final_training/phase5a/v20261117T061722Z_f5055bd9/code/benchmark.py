"""Bounded disposable GPU updates on approved training data; no weight saves."""
import gc,importlib.metadata as md,json,math,platform,statistics,subprocess,time
from pathlib import Path
import torch,torchvision,timm
from core import PROJECT,ROOT,config,model,seed_all,configure_stage,training_mode,loader,train_weights,precision,autocast,optimization_loss,digest
STEPS=20;WARMUP=3;BOUNDED=96

def bench(cfg,frozen):
 seed_all();torch.cuda.empty_cache();torch.cuda.synchronize()
 init=time.perf_counter()
 # Reload verified ImageNet artifacts for every disposable stage.
 net=model(config('M3') if cfg['id']=='M5' else cfg,pretrained=True).cuda()
 teacher=model(config('M4'),pretrained=True).cuda().eval().requires_grad_(False) if cfg['id']=='M5' else None
 initialization_seconds=time.perf_counter()-init
 opt=configure_stage(net,cfg,frozen);training_mode(net,cfg,frozen)
 counts,w=train_weights(cfg);w=w.cuda();scaler=torch.amp.GradScaler('cuda',enabled=precision()==torch.float16)
 batches=loader(cfg,'train',cfg['id']=='M5');iterator=iter(batches)
 torch.cuda.reset_peak_memory_stats();timings=[];gates=[];bounded_images=0
 for i in range(WARMUP+STEPS+BOUNDED):
  torch.cuda.synchronize();start=time.perf_counter();batch=tuple(t.cuda(non_blocking=True) for t in next(iterator));opt.zero_grad(set_to_none=True)
  with autocast():loss,stats=optimization_loss(net,batch,w,cfg,teacher)
  assert torch.isfinite(loss),'Nonfinite loss'
  scaler.scale(loss).backward();scaler.step(opt);scaler.update();torch.cuda.synchronize();elapsed=time.perf_counter()-start
  if i>=WARMUP:timings.append(elapsed)
  if i>=WARMUP+STEPS:bounded_images+=len(batch[-1])
  if stats:gates.append(float(stats['gate_acceptance']))
 net.eval();evaltimes=[]
 for _ in range(20):
  torch.cuda.synchronize();start=time.perf_counter();batch=next(iterator);images=batch[0].cuda(non_blocking=True)
  with torch.no_grad(),autocast():out=net(images)
  torch.cuda.synchronize();evaltimes.append(time.perf_counter()-start)
 steady=timings[:STEPS];bound=timings[STEPS:]
 result={'model':cfg['id'],'stage':'head' if frozen else 'finetune','batch_size':cfg['batch_size'],'initialization':'verified official ImageNet weights; M5 uses architecture proxies, NOT future final parents','initialization_seconds':initialization_seconds,'warmup_updates':WARMUP,'representative_updates':STEPS,'bounded_partial_epoch_updates':BOUNDED,'bounded_partial_epoch_images':bounded_images,'full_training_epoch_images':sum(counts),'steady_step_mean_seconds':statistics.mean(steady),'steady_step_p90_seconds':sorted(steady)[math.ceil(.9*len(steady))-1],'bounded_partial_epoch_seconds':sum(bound),'bounded_step_mean_seconds':statistics.mean(bound),'images_per_second':bounded_images/sum(bound),'inference_batch_mean_seconds_training_images_only':statistics.mean(evaltimes),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'precision':str(precision()),'learned_weights_saved':False,'M5_proxy_gate_acceptance':statistics.mean(gates) if gates else None,'teacher_and_student_views_computed':4 if teacher is not None else None}
 del iterator,batches,batch,opt,net,teacher,loss,out,w,scaler;gc.collect();torch.cuda.empty_cache()
 return result

def main():
 assert torch.cuda.is_available(),'CUDA not accessible';torch.set_num_threads(4)
 free,total=torch.cuda.mem_get_info();probe=(torch.ones(32,32,device='cuda')@torch.ones(32,32,device='cuda')).sum().item();assert probe==32768
 props=torch.cuda.get_device_properties(0)
 env={'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'timm':timm.__version__,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'cuda_available':True,'GPU':torch.cuda.get_device_name(0),'compute_capability':list(torch.cuda.get_device_capability(0)),'total_GiB':total/2**30,'free_GiB_before_benchmark':free/2**30,'bf16_supported':torch.cuda.is_bf16_supported(),'gpu_tensor_test_sum':probe,'pip_check':subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pip','check'],capture_output=True,text=True).stdout.strip()}
 (PROJECT/'reports/environment_snapshot.json').write_text(json.dumps(env,indent=2)+'\n')
 (PROJECT/'reports/requirements_training.lock.txt').write_text(subprocess.check_output([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],text=True))
 report={'purpose':'Disposable preflight only; no final performance evidence','environment':env,'bounded_epoch_definition':'96 actual training updates, not a full epoch; includes image loading, augmentation, transfers, forward/backward, AdamW and AMP. No validation/test/PAD data loaded.','cases':[]}
 start=time.perf_counter()
 for m in ['M1','M2','M3','M4','M5']:
  cfg=config(m)
  for frozen in ([True,False] if m!='M5' else [False]):
   attempts=[]
   for b in [cfg['batch_size'],*cfg['batch_size_fallbacks']]:
    cfg['batch_size']=b
    try:r=bench(cfg,frozen);break
    except torch.cuda.OutOfMemoryError:attempts.append(b);gc.collect();torch.cuda.empty_cache()
   else:raise RuntimeError('All memory fallbacks exhausted')
   r['OOM_batch_attempts']=attempts;report['cases'].append(r);print(json.dumps(r),flush=True)
   (PROJECT/'reports/benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
 estimates={}
 for m in ['M1','M2','M3','M4','M5']:
  cs=[c for c in report['cases'] if c['model']==m];cfg=config(m);stages={}
  for c in cs:
   stages[c['stage']]=math.ceil(11632/c['batch_size'])*c['bounded_step_mean_seconds']+math.ceil(2835/c['batch_size'])*c['inference_batch_mean_seconds_training_images_only']
  nominal=stages.get('head',0)*cfg['frozen_epochs']+stages['finetune']*(cfg['max_total_epochs']-cfg['frozen_epochs'])
  estimates[m]={'epoch_seconds_including_validation_proxy':stages,'maximum_epochs':cfg['max_total_epochs'],'maximum_budget_nominal_minutes':nominal/60,'planning_interval_minutes':[nominal*1.25/60,nominal*2/60],'batch_size':cs[-1]['batch_size'],'budget_kind':cfg['budget_kind']}
 report['estimates']=estimates;report['total_nominal_hours']=sum(v['maximum_budget_nominal_minutes'] for v in estimates.values())/60;report['total_planning_interval_hours']=[report['total_nominal_hours']*1.25,report['total_nominal_hours']*2];report['benchmark_wall_seconds']=time.perf_counter()-start
 report['limitations']=['Short bounded partial epoch extrapolated to11632train+2835validation; no complete training epoch or run.','Validation inference timing uses training images; no heldout loader used.','Actual early stopping may shorten runs; thermal/desktop load/checkpoint serialization can lengthen them.','M5 measured with ImageNet teacher/student architecture proxies; final accuracy and teacher gate usefulness unknown.','Maximum budget counts M3 30epochs plus M5 40additional and M4 teacher30; no budget-matched mechanism attribution possible.']
 (PROJECT/'reports/benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'estimates':estimates,'total_planning_interval_hours':report['total_planning_interval_hours']},indent=2),flush=True)
if __name__=='__main__':main()
