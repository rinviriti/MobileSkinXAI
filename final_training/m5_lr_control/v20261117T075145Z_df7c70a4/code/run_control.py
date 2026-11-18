"""One smoke test in a disposable process, then exactly one authorized full run."""
import argparse,csv,hashlib,importlib.util,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
O=Path(__file__).resolve().parents[1];R=O.parents[2];P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9';D=R/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8'
sys.path.insert(0,str(P/'code'))
import torch
from core import model,config,seed_all,configure_stage,training_mode,loader,train_weights,autocast,precision,digest,optimization_loss
spec=importlib.util.spec_from_file_location('lr_variant_train',O/'code/train.py');trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
cfg=json.loads((O/'configs/M5.json').read_text())
def dump(name,obj):
 p=O/name;t=p.with_suffix('.next');t.write_text(json.dumps(obj,indent=2)+'\n');os.replace(t,p)
def gpu():
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in sandbox; no experiment or smoke started')
 assert torch.cuda.is_bf16_supported() and precision()==torch.bfloat16
 torch.set_num_threads(4)
def smoke():
 gpu();receipt=trainer.approval_check(cfg);seed_all(42)
 st,_=trainer.load_final_parent(cfg['initialization']['student_checkpoint'],'M3',receipt);tt,_=trainer.load_final_parent(cfg['initialization']['teacher_checkpoint'],'M4',receipt)
 net=model(cfg);net.load_state_dict(st['state_dict'],strict=True);teacher=model(config('M4')).cuda().eval().requires_grad_(False);teacher.load_state_dict(tt['state_dict'],strict=True);del st,tt;net=net.cuda();opt=configure_stage(net,cfg,False);training_mode(net,cfg,False)
 _,weights=train_weights(cfg);weights=weights.cuda();batches=loader(cfg,'train',True);batch=tuple(x.cuda(non_blocking=True) for x in next(iter(batches)))
 with autocast():loss,stats=optimization_loss(net,batch,weights,cfg,teacher)
 assert torch.isfinite(loss);loss.backward();head={id(x) for x in net.get_classifier().parameters()}
 head_norm=sum(float(x.grad.float().square().sum()) for x in net.parameters() if id(x) in head and x.grad is not None)**.5;back_norm=sum(float(x.grad.float().square().sum()) for x in net.parameters() if id(x) not in head and x.grad is not None)**.5
 assert head_norm>0 and back_norm>0 and all(torch.isfinite(x.grad).all() for x in net.parameters() if x.grad is not None);assert all(x.grad is None for x in teacher.parameters());assert not teacher.training
 assert opt.param_groups[0]['lr']==1e-5 and opt.param_groups[1]['lr']==1e-6
 observed={k:float(stats[k]) for k in ['ce','kd','consistency','gate_acceptance','gate_mass']};assert abs(float(loss)-(observed['ce']+.5*observed['kd']+.1*observed['consistency']))<1e-5
 dump('reports/smoke.json',{'status':'PASS','disposable_model_only':True,'optimizer_steps':0,'batch_size':len(batch[-1]),'total_loss':float(loss.detach()),'components':observed,'head_gradient_L2':head_norm,'backbone_gradient_L2':back_norm,'teacher_gradients_absent':True,'head_lr':opt.param_groups[0]['lr'],'backbone_lr':opt.param_groups[1]['lr'],'precision':str(precision()),'data':'approved training images only','learned_weights_saved':False,'full_run_seed_reset_in_separate_process':True});print(json.dumps({'smoke':'PASS','loss':float(loss.detach())}),flush=True)
def execute():
 gpu();assert not (O/'execution_status.json').exists() and not (O/'run/run_status.json').exists() and not list((O/'run/checkpoints').glob('*.pt'))
 snapshot=json.loads((O/'reports/original_artifact_snapshot.json').read_text())
 for p,h in snapshot.items():assert digest(R/p)==h,p
 receipt=trainer.approval_check(cfg);old=json.loads((P/'configs/M5.json').read_text());allowed=['head_lr','backbone_lr','checkpoint_dir'];assert {k:v for k,v in cfg.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed}
 original=json.loads((R/'final_training/phase5b/v20261117T063340Z_d20cbfaa/final_results.json').read_text());expected={r['model']:r['checkpoint_sha256'] for r in original['models']}
 for parent,name in [('M3','student_checkpoint'),('M4','teacher_checkpoint')]:assert digest(R/cfg['initialization'][name])==expected[parent]
 # Verify eligible train/validation bytes only; no test or PAD image is opened.
 with open(R/cfg['split_manifest'],newline='') as f:rows=[r for r in csv.DictReader(f) if r['split'] in ['train','validation']]
 for r in rows:assert digest(R/r['processed_path'])==r['processed_sha256'],r['image']
 free,total=torch.cuda.mem_get_info();x=torch.ones(8,8,device='cuda');assert (x@x).sum().item()==512
 dump('reports/preflight.json',{'status':'PASS','dataset_version':D.name,'manifest_sha256':receipt['manifest_sha256'],'student_M3_sha256':expected['M3'],'teacher_M4_sha256':expected['M4'],'original_M5_not_used_for_initialization':True,'original_artifacts_hash_checked':len(snapshot),'eligible_train_validation_prepared_hashes_checked':len(rows),'only_training_hyperparameter_changes':['head_lr:1e-4->1e-5','backbone_lr:1e-5->1e-6'],'GPU':torch.cuda.get_device_name(0),'total_GiB':total/2**30,'free_GiB':free/2**30,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'precision':str(precision()),'test_PAD_images_accessed':False})
 # Separate child process guarantees smoke cannot alter run RNG/model/optimizer/data-loader states.
 subprocess.run([str(R/'.venv/bin/python'),'-u',str(Path(__file__)),'--smoke'],check=True,cwd=R)
 start=time.perf_counter();dump('execution_status.json',{'status':'running','started_UTC':datetime.now(timezone.utc).isoformat(),'full_control_runs_started':1})
 with (O/'training_stdout.log').open('x') as log:
  proc=subprocess.Popen([str(R/'.venv/bin/python'),'-u',str(O/'code/train.py'),'M5','--execute'],cwd=R,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  for line in proc.stdout:
   log.write(line);log.flush()
   try:
    h=json.loads(line)
    if 'epoch' in h:print(json.dumps({'epoch':h['epoch'],'validation':h['validation'],'head_lr':h['head_lr'],'backbone_lr':h['backbone_lr'],'components':h['loss_components'],'elapsed_seconds':h['elapsed_seconds']}),flush=True)
   except json.JSONDecodeError:print(line.rstrip(),flush=True)
  code=proc.wait()
 status=json.loads((O/'run/run_status.json').read_text());status.update({'process_wall_seconds':time.perf_counter()-start,'finished_UTC':datetime.now(timezone.utc).isoformat(),'returncode':code,'full_control_runs_started':1});dump('execution_status.json',status)
 if code:raise RuntimeError('Single LR control failed; preserve outputs, no automatic new experiment')
 print(json.dumps(status,indent=2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();smoke() if a.smoke else execute()
