from common import *
import torch,torch.nn.functional as F,timm,gc,io,shutil
from PIL import Image,ImageEnhance,ImageFilter
from torch.utils.data import DataLoader
from train import validate,classification_metrics

def uniform(a,b):return float(torch.empty(()).uniform_(a,b))
class RobustImages(core.Images):
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(self.root/r['processed_relative_path']) as im:im.load();assert im.mode=='RGB' and im.size==(224,224);im=im.copy()
  weak=self.geometry(im);strong=ImageEnhance.Brightness(weak.copy()).enhance(uniform(.9,1.1));strong=ImageEnhance.Contrast(strong).enhance(uniform(.9,1.1));gain=np.array([uniform(.95,1.05) for _ in range(3)]);gain/=np.exp(np.log(gain).mean());strong=Image.fromarray(np.clip(np.asarray(strong,dtype=np.float32)*gain,0,255).astype('uint8'))
  if float(torch.rand(()))<.5:
   buf=io.BytesIO();strong.save(buf,format='JPEG',quality=int(torch.randint(75,96,()).item()));buf.seek(0)
   with Image.open(buf) as im:strong=im.convert('RGB').copy()
  if float(torch.rand(()))<.2:strong=strong.filter(ImageFilter.GaussianBlur(uniform(.1,.6)))
  if float(torch.rand(()))<.3:
   n=round(224*uniform(.9,1));resized=strong.resize((n,n),Image.Resampling.BILINEAR);pad=224-n;x=int(torch.randint(pad+1,()).item());y=int(torch.randint(pad+1,()).item());strong=Image.new('RGB',(224,224),(114,)*3);strong.paste(resized,(x,y))
  return self.tensor(weak),self.tensor(strong),CLASSES.index(r['label']),r['image']

def batches(cfg,split):
 if split=='validation':return core.loader(cfg,split,return_ids=True)
 assert split=='train'
 if cfg['study_key']!='B3':return core.loader(cfg,'train',two_views=True,return_ids=True)
 return DataLoader(RobustImages(cfg,'train',True,True),batch_size=16,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,worker_init_fn=core.seed_worker,generator=torch.Generator().manual_seed(42),drop_last=False)

def initialize(out,cfg):
 core.seed_all(42);torch.set_num_threads(4);v=load(out/'00_Protocol/verification.json');parents=v['parents'];net=core.model(cfg)
 cp=torch.load(ROOT/parents['M3']['path'],map_location='cpu',weights_only=True);net.load_state_dict(cp['state_dict'],strict=True);del cp
 # Match original construction RNG consumption even when teacher guidance is absent.
 teacher=core.model(core.config('M4'))
 if cfg['study_key']=='B1':
  cp=torch.load(ROOT/parents['M4']['path'],map_location='cpu',weights_only=True);teacher.load_state_dict(cp['state_dict'],strict=True);del cp;teacher=teacher.cuda().eval().requires_grad_(False)
 else:del teacher;teacher=None
 return net.cuda(),teacher

def objective(sw,ss,tw,ts,y,weights,cfg):
 sw,ss=sw.float(),ss.float();a=weights[y].sqrt();den=a.sum().clamp_min(1e-8);ce=.5*(F.cross_entropy(sw,y,weight=weights)+F.cross_entropy(ss,y,weight=weights));js=(a*core.js_divergence(sw.softmax(-1),ss.softmax(-1))).sum()/den;zero=sw.new_zeros(())
 stats={'ce':ce.detach(),'student_JS':js.detach(),'kd':zero,'original_gated_kd':zero,'original_gated_JS':zero,'gate_acceptance':zero,'reliability_mass':zero,'class_weighted_reliability_mass':zero}
 mask=None
 if cfg['study_key']=='B1':
  tw,ts=tw.detach().float(),ts.detach().float();p,q=tw.softmax(-1),ts.softmax(-1);eps=1e-8;h=-.5*((p*p.clamp_min(eps).log()).sum(-1)+(q*q.clamp_min(eps).log()).sum(-1))/np.log(4);d=core.js_divergence(p,q);mask=(torch.minimum(p.max(-1).values,q.max(-1).values)>=.7)&(h<=.6)&(d<=.1)&(p.argmax(-1)==y)&(q.argmax(-1)==y);r=(mask.float()*(1-h).clamp(0,1)*torch.exp(-d/.1)).detach();kdper=.5*16*(F.kl_div(F.log_softmax(ss/4,-1),F.softmax(tw/4,-1),reduction='none').sum(-1)+F.kl_div(F.log_softmax(sw/4,-1),F.softmax(ts/4,-1),reduction='none').sum(-1));kd=(a*kdper).sum()/den;gkd=(a*r*kdper).sum()/den;gjs=(a*r*core.js_divergence(sw.softmax(-1),ss.softmax(-1))).sum()/den
  loss=ce+.5*kd+.1*js;stats.update({'kd':kd.detach(),'original_gated_kd':gkd.detach(),'original_gated_JS':gjs.detach(),'gate_acceptance':mask.float().mean(),'reliability_mass':r.mean(),'class_weighted_reliability_mass':(a*r).sum()/den})
 elif cfg['study_key']=='B2':loss=ce+.1*js
 else:assert cfg['study_key']=='B3';loss=ce
 return loss,stats,mask

def forward(net,teacher,batch,weights,cfg):
 weak,strong,y=batch
 tw=ts=None
 if teacher is not None:
  with torch.no_grad():tw,ts=teacher(weak),teacher(strong)
 sw,ss=net(weak),net(strong);loss,stats,mask=objective(sw,ss,tw,ts,y,weights,cfg)
 return loss,stats,mask,sw,ss

def check(out):
 assert torch.cuda.is_available(),'CUDA required; no smoke/full training started';assert torch.cuda.is_bf16_supported();sealcheck(out);result={}
 # Exact math parity + detached teacher + gradients, independent of any images.
 torch.manual_seed(42);cfg=load(out/'00_Protocol/B1.json');w=torch.tensor(load(out/'00_Protocol/verification.json')['class_weights'],device='cuda');y=torch.arange(4,device='cuda');sw=torch.randn(4,4,device='cuda',requires_grad=True);ss=torch.randn(4,4,device='cuda',requires_grad=True);tw=(torch.eye(4,device='cuda')*10).requires_grad_();ts=tw.clone();orig,st=core.m5_loss(sw,ss,tw,ts,y,w,cfg);loss,new,mask=objective(sw,ss,tw,ts,y,w,cfg);assert torch.allclose(orig,new['ce']+.5*new['original_gated_kd']+.1*new['original_gated_JS'],atol=1e-6)
 loss.backward();assert sw.grad.norm()>0 and ss.grad.norm()>0 and tw.grad is None
 # Both-view consistency gradient and all-rejected original gate behavior.
 zz=torch.zeros(4,4,device='cuda');_,reject=core.m5_loss(sw.detach(),ss.detach(),zz,zz,y,w,cfg);assert reject['kd']==0 and reject['consistency']==0
 augchecks={}
 for k in KEYS:
  cfg=load(out/f'00_Protocol/{k}.json');net,teacher=initialize(out,cfg);core.training_mode(net,cfg);train=batches(cfg,'train');times=[];first=None;gn=[];stats0=None;shape=None
  for i,batch in enumerate(train):
   ids=batch[-1];batch=tuple(x.cuda(non_blocking=True) for x in batch[:-1]);net.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
   with core.autocast():loss,stats,mask,sw,ss=forward(net,teacher,batch,w,cfg)
   assert sw.shape==ss.shape==(len(batch[-1]),4);assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize();times.append(time.perf_counter()-start);norm=float(sum(p.grad.detach().float().square().sum() for p in net.parameters() if p.grad is not None).sqrt());assert norm>0 and np.isfinite(norm);gn.append(norm)
   if teacher is not None:assert all(p.grad is None for p in teacher.parameters())
   if first is None:first=list(ids);stats0={a:float(b) for a,b in stats.items()};shape=list(sw.shape);aughash=hashlib.sha256(batch[0].detach().cpu().numpy().tobytes()+batch[1].detach().cpu().numpy().tobytes()).hexdigest()
   if i==7:break
  augchecks[k]={'ids':first,'augmentation_sha256':aughash}
  seconds=float(np.median(times[2:]));result[k]={'status':'PASS','shape':shape,'finite_losses_nonzero_gradients':True,'teacher_gradients':False,'first_batch_ids':first,'first_batch_augmentation_sha256':aughash,'loss_components_first_batch':stats0,'gradient_norm_first_batch':gn[0],'bounded_backward_batches':8,'median_forward_backward_seconds':seconds,'estimated_epoch_compute_seconds':seconds*727,'estimated_25_epoch_compute_minutes':seconds*727*25/60,'estimate_limit':'Compute-only lower bound; add validation, loaders and checkpoint IO; smoke does not optimizer.step; discarded state','peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
  del net,teacher,train;gc.collect();torch.cuda.empty_cache()
 assert augchecks['B1']==augchecks['B2'],'B1/B2 initial data/augmentation not matched'
 result['checks']={'original_loss_math_parity':True,'all_rejected_gate_zero_original_auxiliaries':True,'student_weak_strong_gradient':True,'teacher_detached':True,'B1_B2_initial_data_and_augmentation_parity':True,'full_training_started':False}
 dump(out/'00_Protocol/SMOKE_AND_BENCHMARK.json',result);print(json.dumps(result),flush=True)

def execute(out):
 sealcheck(out);assert load(out/'00_Protocol/SMOKE_AND_BENCHMARK.json')['checks']['original_loss_math_parity'];assert torch.cuda.is_available();assert torch.cuda.is_bf16_supported()
 for k in KEYS:
  cfg=load(out/f'00_Protocol/{k}.json');run=out/f'runs/{k}';assert not run.exists(),'Refuse overwrite/resume';cpdir=run/'checkpoints';cpdir.mkdir(parents=True);net,teacher=initialize(out,cfg);optimizer=core.configure_stage(net,cfg,False);train=batches(cfg,'train');valid=batches(cfg,'validation');counts,weights=core.train_weights(cfg);weights=weights.cuda();best=-1;stale=0;updates=0;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();history=[];provenance={'artifact_kind':'three_run_ablation_checkpoint','study_id':cfg['study_id'],'study_key':k,'classes':CLASSES,'config_sha256':sha(out/f'00_Protocol/{k}.json'),'protocol_lock_sha256':sha(out/'00_Protocol/PROTOCOL_LOCK.json'),'parent_student':load(out/'00_Protocol/verification.json')['parents']['M3'],'teacher':load(out/'00_Protocol/verification.json')['parents']['M4'] if k=='B1' else None,'train_class_counts':counts,'train_class_weights':weights.cpu().tolist(),'development_splits':['train','validation']}
  dump(run/'status.json',{'status':'running','started_UTC':datetime.now(timezone.utc).isoformat()});print('START',k,flush=True)
  for epoch in range(1,26):
   core.training_mode(net,cfg);matrix=torch.zeros(4,4,device='cuda',dtype=torch.long);sums={};n=0;gateclasses=torch.zeros(4,device='cuda');idhasher=hashlib.sha256()
   for batch in train:
    for x in batch[-1]:idhasher.update((x+'\n').encode())
    batch=tuple(x.cuda(non_blocking=True) for x in batch[:-1]);optimizer.zero_grad(set_to_none=True)
    with core.autocast():loss,stats,mask,sw,ss=forward(net,teacher,batch,weights,cfg)
    assert torch.isfinite(loss);loss.backward();assert torch.stack([torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None]).all();optimizer.step();size=len(batch[-1]);n+=size;updates+=1;sums['objective']=sums.get('objective',0)+float(loss.detach())*size
    for name,value in stats.items():sums[name]=sums.get(name,0)+float(value)*size
    if mask is not None:gateclasses+=torch.bincount(batch[-1][mask],minlength=4)
    matrix+=torch.bincount(batch[-1]*4+sw.detach().argmax(-1),minlength=16).reshape(4,4)
   metrics=validate(net,valid,run/f'validation_predictions_epoch_{epoch:03d}.csv');score=metrics['macro_f1'];improved=score>best
   if improved:best=score;stale=0;bestepoch=epoch
   else:stale+=1
   rec={'epoch':epoch,'epoch_definition':'additional from fresh final M3','train_loss':sums.pop('objective')/n,'loss_components':{name:value/n for name,value in sums.items()},'train':classification_metrics(matrix),'validation':metrics,'head_lr':optimizer.param_groups[0]['lr'],'backbone_lr':optimizer.param_groups[1]['lr'],'gate_acceptance_by_class':{c:float(gateclasses[j])/counts[j] for j,c in enumerate(CLASSES)} if k=='B1' else None,'train_order_sha256':idhasher.hexdigest(),'global_updates':updates,'elapsed_seconds':time.perf_counter()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
   history.append(rec)
   with (run/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   payload={'state_dict':{a:b.detach().cpu() for a,b in net.state_dict().items()},'optimizer':optimizer.state_dict(),'epoch':epoch,'validation_macro_f1':score,'provenance':provenance,'global_updates':updates};path=cpdir/f'epoch_{epoch:03d}.pt'
   with path.open('xb') as f:torch.save(payload,f)
   for alias in (['best.pt','last.pt'] if improved else ['last.pt']):
    temp=cpdir/(alias+'.next');temp.symlink_to(path.name);os.replace(temp,cpdir/alias)
   del payload;print(k,'epoch',epoch,'F1',round(score,6),'stale',stale,flush=True)
   if stale>=6:break
  torch.cuda.synchronize();status={'status':'complete','epochs':epoch,'best_epoch':bestepoch,'best_validation_macro_f1':best,'seconds':time.perf_counter()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'parameters':sum(p.numel() for p in net.parameters()),'best_checkpoint':str((cpdir/'best.pt').relative_to(ROOT)),'best_checkpoint_sha256':sha(cpdir/'best.pt'),'last_checkpoint_sha256':sha(cpdir/'last.pt'),'global_updates':updates};dump(run/'status.json',status);dump(out/f'01_Training_Histories/{k}_completion.json',status)
  del net,teacher,optimizer,train,valid;gc.collect();torch.cuda.empty_cache();sealcheck(out)
 dump(out/'00_Protocol/THREE_RUNS_COMPLETE.json',{'status':'PASS','keys':KEYS,'no_extra_full_configurations':True})
if __name__=='__main__':
 out=ROOT/sys.argv[2]
 if sys.argv[1]=='check':check(out)
 elif sys.argv[1]=='train':execute(out)
