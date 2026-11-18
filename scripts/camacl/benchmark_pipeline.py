"""Bounded steady-state C4 forward/backward timing; zero optimizer steps, state discarded."""
from common import *
from model import Student
from losses import Projector,objective
from pipeline import SourceDataset
import time,torch,cv2
from torch.utils.data import DataLoader

def main(out):
 out=Path(out);path=out/'reports/STEADY_PIPELINE_BENCHMARK.json';assert not path.exists();assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();torch.set_num_threads(4);cv2.setNumThreads(1);seed_all(42);cfg=load(out/'configs/C4.json');ds=SourceDataset(out,cfg,'train',epoch=1);indices=np.random.default_rng(42).choice(len(ds),256,replace=False);ds.rows=[ds.rows[int(i)] for i in indices];write(out/'source_data/steady_benchmark_training_ids.csv',[{'image':r['image'],'split':'train'} for r in ds.rows]);loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=4,pin_memory=True);net=Student(True,True).cuda().train();projector=Projector().cuda();weights=torch.tensor(load(out/'source_data/train_class_statistics.json')['weights'],device='cuda')
 for layer in net.modules():
  if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
 torch.cuda.reset_peak_memory_stats();iterator=iter(loader);times=[];wall=time.perf_counter()
 for i in range(16):
  torch.cuda.synchronize();start=time.perf_counter();a,b,y,*_=next(iterator);a,b,y=a.cuda(non_blocking=True),b.cuda(non_blocking=True),y.cuda(non_blocking=True);net.zero_grad(set_to_none=True);projector.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):sw,fw=net.training_outputs(a);ss,fs=net.training_outputs(b);loss,_=objective(sw,ss,fw,fs,y,weights,projector,cfg,5)
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(projector.parameters()),1.);torch.cuda.synchronize();times.append(time.perf_counter()-start)
 dump(path,{'configuration':'C4','status':'PASS','source_images':256,'warmup_batches':4,'steady_batches':12,'batch_size':16,'mean_steady_seconds_per_batch':float(np.mean(times[4:])),'median_steady_seconds_per_batch':float(np.median(times[4:])),'p95_steady_seconds_per_batch':float(np.percentile(times[4:],95)),'steady_images_per_second':16/float(np.mean(times[4:])),'wall_seconds_with_worker_startup':time.perf_counter()-wall,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'optimizer_steps':0,'initialization':'official ImageNet only; discarded smoke state','scope':'random source-only subset; raw decoding, acquisition, two forward passes, losses/backward, gradient clipping; excludes AdamW updates, validation/checkpoint writes; cold/warm cache and worker startup documented','per_batch_seconds':times})
 print(load(path),flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
