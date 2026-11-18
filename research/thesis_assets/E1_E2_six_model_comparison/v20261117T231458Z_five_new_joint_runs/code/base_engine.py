"""One authorized joint-domain run. No held-out loader is used during training."""
from common import *
from pipeline import geometry,acquisition,stable_rng
from phase6_scoring import primary
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset,DataLoader,Sampler
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from safetensors.torch import load_file
import torch,timm,time,os,gc,collections,argparse

def metrics(rows):
 y=np.array([int(r['true_index']) for r in rows]);p=np.array([[float(r[c]) for c in CLASSES] for r in rows]);pred=p.argmax(1);pr,re,f,s=precision_recall_fscore_support(y,pred,labels=range(4),zero_division=0)
 return {'accuracy':float((pred==y).mean()),'macro_f1':float(f.mean()),'balanced_accuracy':float(re.mean()),'classwise':{c:{'precision':float(pr[j]),'recall':float(re[j]),'f1':float(f[j]),'support':int(s[j])} for j,c in enumerate(CLASSES)},'confusion_matrix':confusion_matrix(y,pred,labels=range(4)).tolist()}
class Images(Dataset):
 def __init__(self,rows,cfg,epoch=0,augment=False):self.rows=rows;self.cfg=cfg;self.epoch=epoch;self.augment=augment;self.transform=T.Compose([T.ToTensor(),T.Normalize(**cfg['normalization'])])
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  if isinstance(index,tuple):i,repeat=index
  else:i=index;repeat=0
  r=self.rows[i]
  with Image.open(ROOT/r['processed_path']) as f:assert f.mode=='RGB' and f.size==(224,224);im=f.copy()
  meta={'image':r['image'],'domain':r['domain'],'epoch':self.epoch,'occurrence':repeat}
  if self.augment:
   assert r['study_split']=='train'
   key=r['image']+f':occurrence{repeat}';im,g=geometry(im,stable_rng(42,self.epoch,key,'geometry'));im,a=acquisition(im,stable_rng(42,self.epoch,key,'acquisition'));meta.update(geometry=g,acquisition=a)
  return self.transform(im),int(r['label_index']),r['image'],json.dumps(meta,sort_keys=True)
class Balanced(Sampler):
 def __init__(self,rows,epoch,steps=727):
  self.steps=steps;self.batches=[];indices={d:np.array([i for i,r in enumerate(rows) if r['domain']==d]) for d in ['ISIC','PAD']};streams={};occ=collections.Counter()
  for di,d in enumerate(['ISIC','PAD']):
   rng=np.random.default_rng(42+100000*epoch+10000*di);stream=[]
   while len(stream)<steps*8:stream.extend(rng.permutation(indices[d]).tolist())
   streams[d]=stream[:steps*8]
  rng=np.random.default_rng(42+200000*epoch)
  for step in range(steps):
   batch=[]
   for d in ['ISIC','PAD']:
    for i in streams[d][step*8:(step+1)*8]:batch.append((int(i),occ[int(i)]));occ[int(i)]+=1
   self.batches.append([batch[int(j)] for j in rng.permutation(16)])
 def __iter__(self):return iter(self.batches)
 def __len__(self):return self.steps

def make(cfg,pretrained=True):
 seed_all(42);net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=1000)
 if pretrained:
  p=ROOT/cfg['pretrained_artifact'];record=load(p.parent/'provenance.json');assert sha(p)==record['sha256']==record['publisher_sha256'];net.load_state_dict(load_file(str(p)),strict=True)
 net.reset_classifier(4);return net

def optimizer(net,cfg,frozen):
 head=[];body=[]
 for n,p in net.named_parameters():h=n.startswith('classifier.');p.requires_grad_(h or not frozen);(head if h else body).append(p)
 groups=[{'params':head,'lr':cfg['frozen_head_lr'] if frozen else cfg['head_lr'],'role':'head'}]
 if not frozen:groups.append({'params':body,'lr':cfg['backbone_lr'],'role':'backbone'})
 return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
def trainmode(net):
 net.train()
 for module in net.modules():
  if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()

def rngstate():
 n=np.random.get_state();return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy':{'name':n[0],'keys':n[1].tolist(),'pos':int(n[2]),'has_gauss':int(n[3]),'cached_gaussian':float(n[4])}}
def restore(r):
 torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda']);random.setstate(r['python']);n=r['numpy'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
def atomic(p,v):q=p.with_name(p.name+'.next');dump(q,v);q.replace(p)
def alias(folder,name,target):
 q=folder/(name+'.next')
 if q.is_symlink():q.unlink()
 q.symlink_to(target);q.replace(folder/name)
def guard(out):
 lock=load(out/'00_Protocol/TRAINING_LOCK.json')
 for name,h in lock['sha256'].items():assert sha(out/name)==h,name
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();return load(out/'00_Protocol/E2_CONFIG.json')
@torch.no_grad()
def validate(net,rows,cfg,path):
 assert all(r['study_split']=='validation' for r in rows);net.eval();preds=[];ys=[];loss=0;offset=0;loader=DataLoader(Images(rows,cfg),batch_size=16,shuffle=False,num_workers=4,pin_memory=True)
 for x,y,ids,_ in loader:
  x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x)
  loss+=float(torch.nn.functional.cross_entropy(logits.float(),y))*len(y);p=logits.float().softmax(1).cpu().numpy()
  for prob in p:
   r=rows[offset];guess=int(prob.argmax());preds.append({**r,'model':'E2','model_name':'Joint ISIC+PAD MobileNetV4-Conv-Small','split':'validation','true_class':r['label'],'true_index':int(r['label_index']),'predicted_class':CLASSES[guess],'predicted_index':guess,'lesion_scoring_unit':r.get('lesion_scoring_unit','ISIC2019:'+r['lesion_id']),'bootstrap_cluster':r.get('bootstrap_cluster',r['lesion_id']),**dict(zip(CLASSES,map(float,prob)))});offset+=1
 assert offset==len(rows);assert not path.exists();write(path,preds);result={'image':metrics(preds),'unweighted_CE':loss/offset}
 if rows[0]['domain']=='PAD':result['lesion']=metrics(primary(preds))
 return result

def engineering(out):
 cfg=guard(out);assert not (out/'00_Protocol/SMOKE_BENCHMARK.json').exists();torch.set_num_threads(4);rows=[r for r in read(out/'11_Source_Data/DEVELOPMENT_MANIFEST.csv') if r['study_split']=='train'];net=make(cfg).cuda();weights=torch.tensor(cfg['class_weights'],device='cuda');loader=DataLoader(Images(rows,cfg,1,True),batch_sampler=Balanced(rows,1,20),num_workers=4,pin_memory=True);batches=list(loader);records=[]
 for frozen in [True,False]:
  opt=optimizer(net,cfg,frozen);trainmode(net);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();losses=[];norms=[]
  for x,y,_,_ in batches:
   x,y=x.cuda(),y.cuda();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x)
   loss=torch.nn.functional.cross_entropy(logits.float(),y,weight=weights);assert logits.shape==(16,4) and torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(norm) and norm>0;opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
  torch.cuda.synchronize();elapsed=time.perf_counter()-start;records.append({'frozen':frozen,'batches':20,'seconds':elapsed,'batches_per_second':20/elapsed,'estimated_727_step_epoch_seconds':elapsed/20*727,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'loss_min':min(losses),'loss_max':max(losses),'gradient_norm_min':min(norms)})
 dump(out/'00_Protocol/SMOKE_BENCHMARK.json',{'status':'PASS','GPU':torch.cuda.get_device_name(0),'VRAM_total':torch.cuda.get_device_properties(0).total_memory,'VRAM_free':torch.cuda.mem_get_info()[0],'records':records,'benchmark_model_discarded':True,'main_run_initialization':'fresh official ImageNet again, seed42 reset','training_only_samples':True});print('SMOKE BENCHMARK PASS',records,flush=True)

def train(out):
 cfg=guard(out);assert load(out/'00_Protocol/SMOKE_BENCHMARK.json')['status']=='PASS';run=out/'03_E2_Training';cpdir=run/'checkpoints';cpdir.mkdir(exist_ok=True);assert not (run/'TRAINING_COMPLETE.json').exists();rows=read(out/'11_Source_Data/DEVELOPMENT_MANIFEST.csv');training=[r for r in rows if r['study_split']=='train'];padval=[r for r in rows if r['study_split']=='validation' and r['domain']=='PAD'];isicval=[r for r in rows if r['study_split']=='validation' and r['domain']=='ISIC'];assert len([r for r in training if r['domain']=='ISIC'])==11632;torch.set_num_threads(4);net=make(cfg).cuda();weights=torch.tensor(cfg['class_weights'],device='cuda');best=-1.;bestepoch=0;stale=0;prev=None;last=0;accum=0.;hist=[];opt=None
 commits=sorted(run.glob('commit_*.json'))
 if commits:
  receipt=load(commits[-1]);path=cpdir/receipt['checkpoint'];assert sha(path)==receipt['sha256'];s=torch.load(path,map_location='cpu',weights_only=True);assert s['config']==cfg;net.load_state_dict(s['state_dict'],strict=True);ctl=s['control'];last=s['epoch'];best=ctl['best'];bestepoch=ctl['best_epoch'];stale=ctl['stale'];prev=ctl['frozen'];accum=ctl['seconds'];opt=optimizer(net,cfg,prev);opt.load_state_dict(s['optimizer']);restore(s['rng']);hist=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()];assert len(hist)==last,'Uncommitted history: preserve and investigate before resume';del s
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 for epoch in range(last+1,36):
  frozen=epoch<=5
  if prev is None or prev!=frozen:opt=optimizer(net,cfg,frozen);stale=0;prev=frozen
  trainmode(net);loader=DataLoader(Images(training,cfg,epoch,True),batch_sampler=Balanced(training,epoch),num_workers=4,pin_memory=True);total=0;n=0;ys=[];probs=[];exposure=collections.Counter();order=hashlib.sha256()
  with (run/f'augmentation_epoch_{epoch:03d}.jsonl').open('x') as f:
   for x,y,ids,metas in loader:
    for name,meta in zip(ids,metas):exposure[name]+=1;order.update((name+'\n').encode());f.write(meta+'\n')
    x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=net(x)
    loss=torch.nn.functional.cross_entropy(logits.float(),y,weight=weights);assert torch.isfinite(loss);loss.backward();params=[p for p in net.parameters() if p.grad is not None];assert all(torch.isfinite(p.grad).all() for p in params);norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);opt.step();total+=float(loss.detach())*len(y);n+=len(y);ys.extend(y.cpu().tolist());probs.extend(logits.detach().float().softmax(1).cpu().tolist())
  assert n==11632;trainrecords=[{'true_index':y,**dict(zip(CLASSES,p))} for y,p in zip(ys,probs)];pv=validate(net,padval,cfg,run/f'PAD_validation_epoch_{epoch:03d}.csv');iv=validate(net,isicval,cfg,run/f'ISIC_validation_epoch_{epoch:03d}.csv');score=pv['lesion']['macro_f1'];improved=score>best
  if improved:best=score;bestepoch=epoch;stale=0
  elif not frozen:stale+=1
  elapsed=accum+time.perf_counter()-start;expo=[]
  for r in training:expo.append({'image':r['image'],'domain':r['domain'],'class':r['label'],'patient_id':r['patient_id'],'presentations':exposure[r['image']]})
  write(run/f'exposure_epoch_{epoch:03d}.csv',expo)
  rec={'epoch':epoch,'stage':'head' if frozen else 'fine','training_weighted_CE':total/n,'train':metrics(trainrecords),'PAD_validation':pv,'ISIC_validation':iv,'learning_rates':{g['role']:g['lr'] for g in opt.param_groups},'best_epoch_so_far':bestepoch,'stale':stale,'elapsed_seconds':elapsed,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'optimizer_updates':727,'training_order_sha256':order.hexdigest()};hist.append(rec)
  with (run/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n');f.flush();os.fsync(f.fileno())
  path=cpdir/f'epoch_{epoch:03d}.pt';payload={'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'optimizer':opt.state_dict(),'rng':rngstate(),'epoch':epoch,'config':cfg,'control':{'best':best,'best_epoch':bestepoch,'stale':stale,'frozen':frozen,'seconds':elapsed},'provenance':{'artifact_kind':'E2_supervised_joint_training','classes':CLASSES,'initialization':'official ImageNet only','ImageNet_sha256':sha(ROOT/cfg['pretrained_artifact']),'PAD_split_lock_sha256':sha(out/'00_Protocol/SPLIT_LOCK.json'),'selection':'PAD validation declared-lesion macro-F1 only'}}
  with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
  dump(run/f'commit_{epoch:03d}.json',{'checkpoint':path.name,'sha256':sha(path)});alias(cpdir,'last.pt',path.name)
  if improved:alias(cpdir,'best.pt',path.name)
  atomic(run/'STATUS.json',{'status':'running','last_completed_epoch':epoch,'best_epoch':bestepoch,'best_PAD_validation_lesion_F1':best,'training_seconds':elapsed});print('E2 epoch',epoch,'PAD_lesion_F1',round(score,6),'ISIC_image_F1',round(iv['image']['macro_f1'],6),'stale',stale,'seconds',round(elapsed,1),flush=True)
  if not frozen and stale>=6:break
 status={'status':'complete','epochs':hist[-1]['epoch'],'best_epoch':bestepoch,'best_PAD_validation_lesion_F1':best,'training_seconds':accum+time.perf_counter()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'best_checkpoint':str((cpdir/'best.pt').relative_to(ROOT)),'best_checkpoint_sha256':sha(cpdir/'best.pt'),'last_checkpoint_sha256':sha(cpdir/'last.pt')};dump(run/'TRAINING_COMPLETE.json',status);dump(out/'00_Protocol/E2_SELECTION_LOCK.json',status);(out/'00_Protocol/E2_SELECTION_LOCK.json').chmod(0o444);print('E2 COMPLETE',status,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['engineering','train']);p.add_argument('output');a=p.parse_args();(engineering if a.mode=='engineering' else train)(ROOT/a.output)
