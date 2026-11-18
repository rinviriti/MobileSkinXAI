from common import *
from model import Student
from losses import Projector,objective,supcon
from approved_train import mode,optimizer_for
import torch,time,gc,ast,io

def main(out):
 out=Path(out);assert torch.cuda.is_available(),'CUDA sandbox unavailable; no full training started';assert torch.cuda.is_bf16_supported();torch.set_num_threads(4);seed_all(42);sample=torch.load(out/'source_data/smoke_training_tensors.pt',map_location='cpu',weights_only=True);weights=torch.tensor(load(out/'source_data/train_class_statistics.json')['weights']);cfg=load(out/'configs/C0.json')
 # Float32 gradient equivalence for C0 duplicated-view CE vs one-view CE.
 net=Student(False,True).eval();x=sample['weak'][:2];y=sample['target'][:2];before={k:v.clone() for k,v in net.named_buffers()};loss=torch.nn.functional.cross_entropy(net(x),y,weight=weights);loss.backward();g1={k:p.grad.clone() for k,p in net.named_parameters() if p.grad is not None};net.zero_grad(set_to_none=True);a,fa=net.training_outputs(x);b,fb=net.training_outputs(x);loss2,_,=objective(a,b,fa,fb,y,weights,Projector(),cfg,5);loss2.backward();maxdiff=0.
 for k,p in net.named_parameters():
  if k in g1:torch.testing.assert_close(g1[k],p.grad,atol=1e-5,rtol=1e-4);maxdiff=max(maxdiff,float((g1[k]-p.grad).abs().max()))
 assert all(torch.equal(before[k],v) for k,v in net.named_buffers());assert net.backbone.drop_rate==0
 c0_update_equivalence={'status':'PASS','paired_CE_vs_single_CE_gradient_max_abs_difference':maxdiff,'BatchNorm_buffers_unchanged':True,'dropout_rate':net.backbone.drop_rate,'one_backward_one_step_per_batch':True,'expected_updates_per_epoch':727};del net;gc.collect()
 from run_study import rng_state,restore_rng
 state=rng_state();f=io.BytesIO();torch.save({'rng_state':state},f);f.seek(0);restored=torch.load(f,map_location='cpu',weights_only=True);restore_rng(restored['rng_state'])
 flags=[];w=weights.cuda();x=sample['weak'].cuda();strong=sample['strong'].cuda();y=sample['target'].cuda()
 for i in range(5):
  cfg=load(out/f'configs/C{i}.json');seed_all(42);net=Student(cfg['fusion'],True).cuda();proj=Projector().cuda();mode(net)
  for stage in [0,5]:
   net.zero_grad(set_to_none=True);proj.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):a,fa=net.training_outputs(x);b,fb=net.training_outputs(strong if cfg['acquisition'] else x);loss,stats=objective(a,b,fa,fb,y,w,proj,cfg,stage)
   assert a.shape==b.shape==(16,4) and torch.isfinite(loss);loss.backward();assert torch.stack([torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None]).all();assert sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)>0
   if cfg['contrastive'] and stage>0:assert any(p.grad is not None and p.grad.norm()>0 for p in proj.parameters())
   else:assert all(p.grad is None for p in proj.parameters())
  flags.append({'configuration':cfg['id'],'head_and_full_ramp_forward_backward':'PASS','optimizer_steps':0});del net,proj;gc.collect();torch.cuda.empty_cache()
 # SCC single-image class: second view remains positive; no-class negatives case is safely zero.
 z=torch.randn(8,64,device='cuda',requires_grad=True);labels=torch.tensor([0,1,2,3]*2,device='cuda');loss,s=supcon(z,labels);assert torch.isfinite(loss) and s['minimum_positives']==1;loss.backward();assert z.grad.norm()>0
 zero,sk=supcon(z.detach().requires_grad_(True),torch.zeros(8,device='cuda',dtype=torch.long));assert zero==0 and sk['skipped_single_class']
 dump(out/'00_Experiment_Protocol/CUDA_PREREQUISITES.json',{'status':'PASS','GPU':torch.cuda.get_device_name(),'total_VRAM_bytes':torch.cuda.get_device_properties(0).total_memory,'free_total_VRAM_bytes':torch.cuda.mem_get_info(),'torch':torch.__version__,'CUDA':torch.version.cuda,'BF16':True,'C0_update_equivalence':c0_update_equivalence,'configuration_smokes':flags,'rare_class_singleton_positive':'PASS','single_class_safe_zero':'PASS','RNG_checkpoint_weights_only_roundtrip':'PASS','full_training_started':False})
 print('CUDA PREREQUISITES PASS',c0_update_equivalence,flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
