from common import *
from model import Student
from losses import Projector,objective
from pipeline import SourceDataset
import torch,time,gc
from torch.utils.data import DataLoader

def main(out):
 out=Path(out);assert torch.cuda.is_available(),'CUDA sandbox inaccessible; no smoke or training executed';assert torch.cuda.is_bf16_supported();torch.set_num_threads(4);seed_all(42);sample=torch.load(out/'source_data/smoke_training_tensors.pt',map_location='cpu',weights_only=True);w=torch.tensor(load(out/'source_data/train_class_statistics.json')['weights'],device='cuda');x=sample['weak'].cuda();strong=sample['strong'].cuda();y=sample['target'].cuda();reports=[]
 for i in range(5):
  cfg=load(out/f'configs/C{i}.json');seed_all(42);net=Student(cfg['fusion'],True).cuda();projector=Projector().cuda();net.train();projector.train()
  for module in net.modules():
   if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
  torch.cuda.reset_peak_memory_stats();times=[];first_stats=None;pnorm=None
  for step in range(8):
   net.zero_grad(set_to_none=True);projector.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
   with torch.autocast('cuda',dtype=torch.bfloat16):
    sw,fw=net.training_outputs(x);ss,fs=net.training_outputs(strong if cfg['acquisition'] else x);loss,st=objective(sw,ss,fw,fs,y,w,projector,cfg,5)
   assert sw.shape==ss.shape==(16,4) and torch.isfinite(loss);loss.backward();params=[p for p in list(net.parameters())+list(projector.parameters()) if p.grad is not None];assert torch.stack([torch.isfinite(p.grad).all() for p in params]).all();norm=float(torch.nn.utils.clip_grad_norm_(params,1.));assert norm>0
   if cfg['fusion']:assert net.proj8[0].weight.grad.norm()>0 and net.proj16[0].weight.grad.norm()>0 and net.rho.grad.abs()>0
   if cfg['contrastive']:assert sum(float(p.grad.norm()) for p in projector.parameters() if p.grad is not None)>0
   else:assert all(p.grad is None for p in projector.parameters())
   torch.cuda.synchronize();times.append(time.perf_counter()-start)
   if first_stats is None:first_stats={k:float(v) if isinstance(v,torch.Tensor) else v for k,v in st.items()};pnorm=norm
  reports.append({'configuration':cfg['id'],'status':'PASS','batch':16,'views':2,'output_shape':[16,4],'zero_optimizer_updates':True,'warmup_backward_batches':2,'timed_backward_batches':6,'median_compute_seconds_per_batch':float(np.median(times[2:])),'mean_compute_seconds_per_batch':float(np.mean(times[2:])),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'first_loss_components':first_stats,'pre_clip_gradient_norm':pnorm,'finite_gradients':True,'fusion_gradients':'nonzero' if cfg['fusion'] else 'not applicable','training_projector_gradients':'nonzero' if cfg['contrastive'] else 'absent'})
  print(cfg['id'],reports[-1],flush=True);del net,projector;gc.collect();torch.cuda.empty_cache()
 # Bounded end-to-end pipeline: fresh stateless raw-decode views, no optimizer.
 cfg=load(out/'configs/C4.json');ds=SourceDataset(out,cfg,'train',epoch=1,limit=128);batches=DataLoader(ds,batch_size=16,shuffle=False,num_workers=4,pin_memory=True);net=Student(True,True).cuda().eval();start=time.perf_counter();samples=0
 with torch.inference_mode():
  for a,b,*rest in batches:
   with torch.autocast('cuda',dtype=torch.bfloat16):net(a.cuda(non_blocking=True));net(b.cuda(non_blocking=True))
   samples+=len(a)
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 dump(out/'reports/GPU_SMOKE.json',{'status':'PASS','GPU':torch.cuda.get_device_name(),'total_VRAM_bytes':torch.cuda.get_device_properties(0).total_memory,'free_total_VRAM_bytes':torch.cuda.mem_get_info(),'torch':torch.__version__,'CUDA':torch.version.cuda,'BF16':True,'runs':reports,'end_to_end_forward_pipeline':{'images':samples,'seconds':elapsed,'images_per_second':samples/elapsed,'includes_worker_startup':True,'includes_backward':False},'training_updates':0,'skin_lesion_checkpoint_loaded':False,'benchmark_limit':'Compute/backward uses16stratifiedtrainimages; data-pipeline128sourceimages. Not a full epoch and OS caches/hardware conditions affect projection.'})
if __name__=='__main__':
 if '--pipeline-only' in sys.argv:
  from benchmark_pipeline import main as benchmark
  benchmark(ROOT/sys.argv[1])
 else:main(ROOT/sys.argv[1])
