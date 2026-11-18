"""Exactly one disposable epoch per model; official/proxy initialization, no saves."""
import gc,json,math,time
import torch
from core import PROJECT,config,model,seed_all,configure_stage,training_mode,loader,train_weights,precision,autocast,optimization_loss

def run(cfg):
 seed_all();torch.set_num_threads(4)
 student=model(config('M3') if cfg['id']=='M5' else cfg,pretrained=True).cuda()
 teacher=model(config('M4'),pretrained=True).cuda().eval().requires_grad_(False) if cfg['id']=='M5' else None
 opt=configure_stage(student,cfg,False);training_mode(student,cfg,False);counts,weights=train_weights(cfg);weights=weights.cuda();scaler=torch.amp.GradScaler('cuda',enabled=precision()==torch.float16)
 batches=loader(cfg,'train',cfg['id']=='M5');torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();images=0;steps=0
 limit=math.ceil(sum(counts)/cfg['batch_size'])
 for batch in batches:
  assert steps<limit,'Epoch update cap exceeded'
  batch=tuple(t.cuda(non_blocking=True) for t in batch);opt.zero_grad(set_to_none=True)
  with autocast():loss,_=optimization_loss(student,batch,weights,cfg,teacher)
  assert torch.isfinite(loss)
  scaler.scale(loss).backward();scaler.step(opt);scaler.update();images+=len(batch[-1]);steps+=1
 torch.cuda.synchronize();seconds=time.perf_counter()-start;assert images==11632 and steps==limit
 result={'model':cfg['id'],'stage':'finetune','batch_size':cfg['batch_size'],'epochs':1,'updates':steps,'training_images':images,'epoch_wall_seconds_including_loader_startup':seconds,'images_per_second':images/seconds,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'precision':str(precision()),'learned_checkpoint_saved':False,'heldout_images_loaded':False,'initialization':'Official ImageNet weights; M5 uses ImageNet architecture proxies only; future final M3/M4 not available'}
 del batches,student,teacher,opt,batch,weights,loss,scaler;gc.collect();torch.cuda.empty_cache();return result

def main():
 assert torch.cuda.is_available()
 report={'purpose':'Single-epoch bounded timing per architecture, not five final training runs; all updated weights discarded','cases':[]}
 for m in ['M1','M2','M3','M4','M5']:
  r=run(config(m));report['cases'].append(r);print(json.dumps(r),flush=True);(PROJECT/'reports/bounded_epoch.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
