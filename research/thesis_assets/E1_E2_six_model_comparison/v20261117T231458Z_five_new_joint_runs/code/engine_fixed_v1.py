from common import *
from stopping_v1 import update, should_stop
import base_engine as b
from original_m5_loss import m5_loss
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from safetensors.torch import load_file
import torch,timm,time,os,gc,collections,shutil,argparse
ORDER=['M1','M2','M4','M5','M5-LR']
class Views(Dataset):
 def __init__(self,rows,cfg,epoch):
  self.rows=rows;self.cfg=cfg;self.epoch=epoch;self.single=b.Images(rows,cfg,epoch,True);self.tensor=T.Compose([T.ToTensor(),T.Normalize(**cfg['normalization'])])
  if cfg['enhanced']:
   a=cfg['augmentation'];g=a['shared_geometry'];self.geom=T.Compose([T.RandomHorizontalFlip(g['horizontal_flip']),T.RandomVerticalFlip(g['vertical_flip']),T.RandomRotation(g['rotation_degrees'],interpolation=InterpolationMode.BILINEAR,fill=(g['fill'],)*3)]);self.strong=T.Compose([T.ColorJitter(*a['strong_color_jitter']),T.RandomGrayscale(a['strong_grayscale_probability'])])
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  if not self.cfg['enhanced']:
   x,y,name,meta=self.single[index];return x,x,y,name,meta
  i,occ=index;r=self.rows[i]
  with Image.open(ROOT/r['processed_path']) as f:assert f.mode=='RGB' and f.size==(224,224);im=f.copy()
  token=hashlib.sha256(f'42|{self.epoch}|{r["image"]}|{occ}|original_M5_views'.encode()).digest();seed=int.from_bytes(token[:8],'little')%(2**63-1)
  with torch.random.fork_rng(devices=[]):torch.manual_seed(seed);weak=self.geom(im);strong=self.strong(weak.copy())
  return self.tensor(weak),self.tensor(strong),int(r['label_index']),r['image'],json.dumps({'image':r['image'],'domain':r['domain'],'epoch':self.epoch,'occurrence':occ,'original_M5_transform_seed':seed,'original_transform_definition':self.cfg['augmentation']},sort_keys=True)
def guard(out):
 lock=load(out/'00_Protocol/LOCK.json')
 for name,h in lock['sha256'].items():assert sha(out/name)==h,name
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();torch.set_num_threads(4)
def make(cfg):
 seed_all(42)
 if cfg['enhanced']:
  net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=4);state=torch.load(ROOT/cfg['student_start'],map_location='cpu',weights_only=True);assert sha(ROOT/cfg['student_start'])==cfg['student_sha256'];net.load_state_dict(state['state_dict'],strict=True)
 else:
  path=ROOT/cfg['pretrained_artifact'];record=load(path.parent/'provenance.json');assert sha(path)==record['sha256']==record['publisher_sha256'];net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=1000);net.load_state_dict(load_file(str(path)),strict=True);net.reset_classifier(4)
 actual=timm.data.resolve_model_data_config(net);assert all(np.allclose(actual[k],cfg['normalization'][k]) for k in ['mean','std']);return net

def optimizer(net,cfg,frozen):
 head=list(net.get_classifier().parameters());ids={id(p) for p in head}
 for p in net.parameters():p.requires_grad_(id(p) in ids or not frozen)
 groups=[{'params':head,'lr':cfg['frozen_head_lr'] if frozen else cfg['head_lr'],'role':'head'}]
 if not frozen:groups.append({'params':[p for p in net.parameters() if id(p) not in ids],'lr':cfg['backbone_lr'],'role':'backbone'})
 return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
def teacher(out,cfg):
 if not cfg['enhanced']:return None
 qualification=load(out/'00_Protocol/TEACHER_QUALIFICATION.json');assert qualification['qualified'];assert sha(ROOT/qualification['teacher_checkpoint'])==qualification['teacher_sha256'];net=timm.create_model(qualification['teacher_architecture'],pretrained=False,num_classes=4);s=torch.load(ROOT/qualification['teacher_checkpoint'],map_location='cpu',weights_only=True);net.load_state_dict(s['state_dict'],strict=True);return net.cuda().eval().requires_grad_(False)
def objective(net,t,w,s,y,weights,cfg):
 weak=net(w)
 if not cfg['enhanced']:loss=torch.nn.functional.cross_entropy(weak.float(),y,weight=weights);return loss,{'ce':loss.detach(),'kd':loss.detach()*0,'consistency':loss.detach()*0,'gate_acceptance':loss.detach()*0,'gate_mass':loss.detach()*0},weak
 strong=net(s)
 with torch.no_grad():tw,ts=t(w),t(s)
 loss,stats=m5_loss(weak,strong,tw,ts,y,weights,cfg);return loss,stats,weak

def smoke(out,key):
 guard(out);cfg=load(out/'00_Protocol/CONFIGURATIONS.json')[key];rows=[r for r in read(out/'01_Frozen_Data_Manifests/DEVELOPMENT_MANIFEST.csv') if r['study_split']=='train'];net=make(cfg).cuda();t=teacher(out,cfg);weights=torch.tensor(cfg['class_weights'],device='cuda');loader=DataLoader(Views(rows,cfg,1),batch_sampler=b.Balanced(rows,1,6),num_workers=4,pin_memory=True);opt=optimizer(net,cfg,False);b.trainmode(net);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();records=[]
 for w,s,y,ids,metas in loader:
  assert sum(json.loads(m)['domain']=='PAD' for m in metas)==8;w,s,y=w.cuda(),s.cuda(),y.cuda();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss,stats,logits=objective(net,t,w,s,y,weights,cfg)
  assert logits.shape==(16,4) and torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert norm>0 and torch.isfinite(norm) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
  if t is not None:assert not t.training and all(p.grad is None for p in t.parameters());assert abs(float(loss.detach())-(float(stats['ce'])+.5*float(stats['kd'])+.1*float(stats['consistency'])))<1e-5
  records.append({k:float(v) for k,v in stats.items() if k!='gate_mask'});opt.step()
 torch.cuda.synchronize();elapsed=time.perf_counter()-start;report={'status':'PASS','model':key,'batch':16,'effective_domain_ratio':'8ISIC+8PAD','benchmark_batches':6,'seconds':elapsed,'estimated_compute_epoch_seconds':elapsed/6*727,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'finite_losses_nonzero_gradients':True,'teacher_frozen_gradients_absent':True if t else None,'loss_components':records,'disposable_model':True,'full_run_seed_reset':True};dump(out/f'00_Protocol/SMOKE_{key}.json',report);print('SMOKE PASS',key,elapsed,report['peak_allocated_GiB'],flush=True);del net,t,opt;gc.collect();torch.cuda.empty_cache()
def validate(net,rows,cfg,path):
 result=b.validate(net,rows,cfg,path);records=read(path)
 for r in records:r['model']=cfg['id'];r['model_name']=cfg['name']
 write(path,records);return result

def execute(out,key):
 guard(out);assert load(out/f'00_Protocol/SMOKE_{key}.json')['status']=='PASS';cfg=load(out/'00_Protocol/CONFIGURATIONS.json')[key];run=out/'03_E2_Training'/key;run.mkdir(exist_ok=True);cpdir=run/'checkpoints';cpdir.mkdir(exist_ok=True)
 if (run/'TRAINING_COMPLETE.json').exists():assert sha(cpdir/'best.pt')==load(run/'TRAINING_COMPLETE.json')['best_checkpoint_sha256'];print('SKIP FINISHED',key,flush=True);return
 if cfg['enhanced']:q=load(out/'00_Protocol/TEACHER_QUALIFICATION.json');assert q['qualified'];cfg={**cfg,'teacher_checkpoint':q['teacher_checkpoint'],'teacher_sha256':q['teacher_sha256']}
 if (run/'resolved_configuration.json').exists():assert load(run/'resolved_configuration.json')==cfg
 else:dump(run/'resolved_configuration.json',cfg)
 data=read(out/'01_Frozen_Data_Manifests/DEVELOPMENT_MANIFEST.csv');training=[r for r in data if r['study_split']=='train'];padval=[r for r in data if r['study_split']=='validation' and r['domain']=='PAD'];isicval=[r for r in data if r['study_split']=='validation' and r['domain']=='ISIC'];net=make(cfg).cuda();t=teacher(out,cfg);weights=torch.tensor(cfg['class_weights'],device='cuda');best=-1.;bestepoch=0;stale=0;prev=None;last=0;accum=0.;oldpeak=0.;hist=[];opt=None;commits=sorted(run.glob('commit_*.json'))
 if commits:
  r=load(commits[-1]);assert sha(cpdir/r['checkpoint'])==r['sha256'];state=torch.load(cpdir/r['checkpoint'],map_location='cpu',weights_only=True);assert state['config']==cfg;net.load_state_dict(state['state_dict'],strict=True);ctl=state['control'];last=state['epoch'];best=ctl['best'];bestepoch=ctl['best_epoch'];stale=ctl['stale'];prev=ctl['frozen'];accum=ctl['seconds'];oldpeak=ctl['peak'];opt=optimizer(net,cfg,prev);opt.load_state_dict(state['optimizer']);b.restore(state['rng']);del state
  hist=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()]
 # Preserve and archive unfinished epoch artifacts, replay only from a verified committed boundary.
 uncommitted=[p for pattern in ['augmentation_epoch_*.jsonl','exposure_epoch_*.csv','PAD_validation_epoch_*.csv','ISIC_validation_epoch_*.csv'] for p in run.glob(pattern) if int(p.stem.split('_')[-1])>last]+[p for p in cpdir.glob('epoch_*.pt') if int(p.stem.split('_')[-1])>last]
 extra=[h for h in hist if h['epoch']>last]
 if uncommitted or extra:
  archive=run/'recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');archive.mkdir(parents=True)
  for p in uncommitted:shutil.move(str(p),archive/p.name)
  if extra:shutil.copyfile(run/'history.jsonl',archive/'uncommitted_history.jsonl');hist=[h for h in hist if h['epoch']<=last];(run/'history.jsonl').write_text(''.join(json.dumps(h)+'\n' for h in hist))
 assert len(hist)==last;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();maxepochs=cfg['frozen_epochs']+cfg['fine_epochs'];print('START',key,'after',last,flush=True)
 for epoch in ([] if last>=maxepochs or should_stop(last,cfg['frozen_epochs'],stale,cfg['patience']) else range(last+1,maxepochs+1)):
  frozen=epoch<=cfg['frozen_epochs']
  if prev is None or prev!=frozen:opt=optimizer(net,cfg,frozen);stale=0;prev=frozen
  b.trainmode(net);loader=DataLoader(Views(training,cfg,epoch),batch_sampler=b.Balanced(training,epoch),num_workers=4,pin_memory=True);total=0;n=0;ys=[];probs=[];components=collections.Counter();exposure=collections.Counter();order=hashlib.sha256();gates=[0]*4;classn=[0]*4
  with (run/f'augmentation_epoch_{epoch:03d}.jsonl').open('x') as log:
   for w,s,y,ids,metas in loader:
    for name,meta in zip(ids,metas):log.write(meta+'\n');exposure[name]+=1;order.update((name+'\n').encode())
    w,s,y=w.cuda(non_blocking=True),s.cuda(non_blocking=True),y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,stats,logits=objective(net,t,w,s,y,weights,cfg)
    assert torch.isfinite(loss);loss.backward();params=[p for p in net.parameters() if p.grad is not None];assert all(torch.isfinite(p.grad).all() for p in params);norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);opt.step();total+=float(loss.detach())*len(y);n+=len(y);ys.extend(y.cpu().tolist());probs.extend(logits.detach().float().softmax(1).cpu().tolist())
    for name,v in stats.items():
     if name!='gate_mask':components[name]+=float(v)*len(y)
    for j in range(4):mask=y==j;classn[j]+=int(mask.sum());gates[j]+=int(stats['gate_mask'][mask].sum()) if 'gate_mask' in stats else 0
  assert n==11632;pv=validate(net,padval,cfg,run/f'PAD_validation_epoch_{epoch:03d}.csv');iv=validate(net,isicval,cfg,run/f'ISIC_validation_epoch_{epoch:03d}.csv');score=pv['lesion']['macro_f1'];best,bestepoch,stale,improved=update(best,bestepoch,stale,score,epoch,frozen)
  elapsed=accum+time.perf_counter()-start;peak=max(oldpeak,torch.cuda.max_memory_allocated()/2**30);write(run/f'exposure_epoch_{epoch:03d}.csv',[{'image':r['image'],'domain':r['domain'],'class':r['label'],'presentations':exposure[r['image']]} for r in training]);record={'epoch':epoch,'stage':'head' if frozen else 'fine','training_weighted_CE':components['ce']/n,'training_total_loss':total/n,'loss_components':{k:v/n for k,v in components.items()},'train':b.metrics([{'true_index':a,**dict(zip(CLASSES,p))} for a,p in zip(ys,probs)]),'PAD_validation':pv,'ISIC_validation':iv,'learning_rates':{g['role']:g['lr'] for g in opt.param_groups},'stale':stale,'best_epoch_so_far':bestepoch,'elapsed_seconds':elapsed,'peak_allocated_GiB':peak,'optimizer_updates':727,'training_order_sha256':order.hexdigest(),'teacher_gate_by_class':{c:{'accepted':gates[j],'training_presentations':classn[j]} for j,c in enumerate(CLASSES)}};hist.append(record)
  with (run/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
  path=cpdir/f'epoch_{epoch:03d}.pt';payload={'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'optimizer':opt.state_dict(),'rng':b.rngstate(),'epoch':epoch,'config':cfg,'control':{'best':best,'best_epoch':bestepoch,'stale':stale,'frozen':frozen,'seconds':elapsed,'peak':peak},'provenance':{'artifact_kind':'six_model_joint_domain_E2','classes':CLASSES,'student_start_sha256':cfg.get('student_sha256'),'teacher_sha256':cfg.get('teacher_sha256'),'initialization':'joint M3 E2 parent' if cfg['enhanced'] else 'official ImageNet','selection':'PAD validation lesion macro-F1 only'}}
  with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
  dump(run/f'commit_{epoch:03d}.json',{'checkpoint':path.name,'sha256':sha(path)});b.alias(cpdir,'last.pt',path.name)
  if improved:b.alias(cpdir,'best.pt',path.name)
  b.atomic(run/'STATUS.json',{'status':'running','last_epoch':epoch,'best_epoch':bestepoch,'best_PAD_validation_lesion_F1':best,'training_seconds':elapsed});print(key,'epoch',epoch,'PADval',round(score,6),'ISICval',round(iv['image']['macro_f1'],6),'stale',stale,'seconds',round(elapsed,1),flush=True)
  if should_stop(epoch,cfg['frozen_epochs'],stale,cfg['patience']):
   print('EARLY STOP',key,epoch,'best',bestepoch,flush=True);break
 status={'status':'complete','stop_reason':'early_stopping_patience' if should_stop(hist[-1]['epoch'],cfg['frozen_epochs'],stale,cfg['patience']) else 'maximum_epochs','stop_epoch':hist[-1]['epoch'],'controller_revision':'engine_fixed_v1','model':key,'epochs':hist[-1]['epoch'],'best_epoch':bestepoch,'best_PAD_validation_lesion_F1':best,'training_seconds':accum+time.perf_counter()-start,'peak_allocated_GiB':max(oldpeak,torch.cuda.max_memory_allocated()/2**30),'best_checkpoint':str((cpdir/'best.pt').relative_to(ROOT)),'best_checkpoint_sha256':sha(cpdir/'best.pt'),'last_checkpoint_sha256':sha(cpdir/'last.pt'),'parameters':sum(p.numel() for p in net.parameters()),'teacher_checkpoint':cfg.get('teacher_checkpoint'),'teacher_sha256':cfg.get('teacher_sha256'),'student_start_sha256':cfg.get('student_sha256'),'budget_kind':cfg.get('budget_kind','up to5head+30fine epochs')};dump(run/'TRAINING_COMPLETE.json',status);dump(run/'SELECTION_LOCK.json',status);(run/'SELECTION_LOCK.json').chmod(0o444);print('COMPLETE',key,status['best_epoch'],status['training_seconds'],flush=True);del net,t,opt;gc.collect();torch.cuda.empty_cache()

def qualify(out):
 teacherstate=load(out/'03_E2_Training/M4/TRAINING_COMPLETE.json');student=load(out/'00_Protocol/REUSED_M3_E2.json');passed=teacherstate['best_PAD_validation_lesion_F1']>student['best_PAD_validation_lesion_F1'];q={'qualified':passed,'criterion':'M4 E2 PAD validation lesion macro-F1 strictly greater than M3 E2; zero margin, no test evidence','teacher_validation_F1':teacherstate['best_PAD_validation_lesion_F1'],'student_validation_F1':student['best_PAD_validation_lesion_F1'],'teacher_checkpoint':teacherstate['best_checkpoint'],'teacher_sha256':teacherstate['best_checkpoint_sha256'],'teacher_architecture':load(out/'00_Protocol/CONFIGURATIONS.json')['M4']['architecture'],'student_checkpoint':student['best_checkpoint'],'student_sha256':student['best_checkpoint_sha256']};dump(out/'00_Protocol/TEACHER_QUALIFICATION.json',q);(out/'00_Protocol/TEACHER_QUALIFICATION.json').chmod(0o444);assert passed,'Material blocker: original teacher qualification failed; do not substitute enhancement';print('TEACHER QUALIFIED',q['teacher_validation_F1'],flush=True)
def main(out):
 out=Path(out);guard(out)
 for key in ORDER:
  if key=='M5':qualify(out)
  run=out/'03_E2_Training'/key
  if not (run/'TRAINING_COMPLETE.json').exists():
   if not (out/f'00_Protocol/SMOKE_{key}.json').exists():smoke(out,key)
   execute(out,key)
  states={k:load(out/'03_E2_Training'/k/'TRAINING_COMPLETE.json') if (out/'03_E2_Training'/k/'TRAINING_COMPLETE.json').exists() else {'status':'pending'} for k in ORDER};b.atomic(out/'13_Reports/EXPERIMENT_SUMMARY.json',states)
 dump(out/'ALL_FIVE_NEW_COMPLETE.json',{'status':'PASS','new_models':ORDER,'M3_retrained':False,'seed':42,'test_access_for_training':False})
if __name__=='__main__':main(ROOT/sys.argv[1])
