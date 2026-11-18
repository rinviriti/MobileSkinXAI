"""Sequential execution of exactly the five authorized frozen configurations."""
import argparse,csv,hashlib,json,os,subprocess,sys,time,uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9';D=R/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
 t=p.with_suffix('.next');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execution-dir',required=True);args=parser.parse_args();O=R/args.execution_dir
 assert O.is_dir();assert not (O/'sequence_status.json').exists(),'Never overwrite an existing execution'
 for folder,lockname in [(D,'RELEASE_LOCK.json'),(P,'PREPARATION_LOCK.json')]:
  lock=json.loads((folder/lockname).read_text());assert sha(folder/'checksums.sha256')==lock['checksums_sha256']
  for line in (folder/'checksums.sha256').read_text().splitlines():
   h,n=line.split('  ',1);assert sha(folder/n)==h,n
 assert json.loads((D/'data_approval.json').read_text())['approved'] is True
 assert len(list((P/'configs').glob('M*.json')))==5
 for m in ['M1','M2','M3','M4','M5']:
  cfg=json.loads((P/f'configs/{m}.json').read_text());cp=R/cfg['checkpoint_dir'];assert not list(cp.glob('*.pt')) and not (cp.parent/'run_status.json').exists(),m
 # Check every release-manifest source/prepared hash before the first update.
 rows=list(csv.DictReader(open(D/'dataset_manifest.csv')))
 def verify(r):
  assert sha(R/r['processed_path'])==r['processed_sha256'],r['image']
  assert sha(r['source'])==r['source_sha256'],r['image']
  return 1
 with ThreadPoolExecutor(max_workers=4) as pool:checked=sum(pool.map(verify,rows,buffersize=16))
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable inside sandbox; no training started')
 torch.set_num_threads(4);free,total=torch.cuda.mem_get_info();x=torch.ones(8,8,device='cuda');assert (x@x).sum().item()==512
 dump(O/'preflight.json',{'dataset_release':D.name,'preparation':P.name,'release_and_preparation_checksums':'PASS','manifest_rows_raw_and_prepared_checked':checked,'GPU':torch.cuda.get_device_name(0),'total_GiB':total/2**30,'free_GiB':free/2**30,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'bf16_supported':torch.cuda.is_bf16_supported(),'GPU_tensor_operation':'PASS','test_or_PAD_inference_performed':False})
 dump(P/'training_authorization.json',{'authorized':True,'authorized_UTC':datetime.now(timezone.utc).isoformat(),'user_instruction':'Approve and begin Phase5B: exactly five fresh final models sequentially; M5 conditional on fresh M4 teacher qualification','execution_directory':str(O.relative_to(R)),'dataset_version':D.name,'preparation_version':P.name,'main_configurations':['M1','M2','M3','M4','M5'],'M5_condition':'M4 best validation macro-F1 strictly greater than M3; otherwise stop without substitute'})
 status={'execution_id':O.name,'started_UTC':datetime.now(timezone.utc).isoformat(),'status':'running','dataset_version':D.name,'preparation_version':P.name,'models':{},'test_PAD_evaluated':False};dump(O/'sequence_status.json',status);start=time.perf_counter()
 for m in ['M1','M2','M3','M4','M5']:
  if m=='M5':
   s3=json.loads((P/'runs/M3_seed42/run_status.json').read_text());s4=json.loads((P/'runs/M4_seed42/run_status.json').read_text());cfg=json.loads((P/'configs/M5.json').read_text());margin=cfg['teacher_qualification']['min_macro_f1_improvement']
   qualified=s4['best_validation_macro_f1']>s3['best_validation_macro_f1']+margin
   gate={'M3_validation_macro_F1':s3['best_validation_macro_f1'],'M4_validation_macro_F1':s4['best_validation_macro_f1'],'required_margin':margin,'qualified':qualified,'basis':'Best validation macro-F1 only; no test/PAD evidence','M3_best_checkpoint_sha256':s3['best_checkpoint_sha256'],'M4_best_checkpoint_sha256':s4['best_checkpoint_sha256']};dump(O/'teacher_qualification.json',gate)
   if not qualified:
    status['models'][m]={'status':'blocked_teacher_unsuitable','epochs_completed':0,'reason':'Fresh M4 validation macro-F1 does not strictly exceed fresh M3'};status['status']='completed_M1_M4_M5_blocked';summary=json.loads((P/'experiment_summary.json').read_text());summary['models']['M5'].update(status['models'][m]);dump(P/'experiment_summary.json',summary);dump(O/'sequence_status.json',status);print(json.dumps({'teacher_gate':gate,'M5':'NOT_TRAINED'}),flush=True);break
  print(json.dumps({'starting':m,'UTC':datetime.now(timezone.utc).isoformat()}),flush=True)
  launched=time.perf_counter();status['models'][m]={'status':'running'};dump(O/'sequence_status.json',status)
  with (O/f'{m}_stdout.log').open('x') as log:
   proc=subprocess.Popen([str(R/'.venv/bin/python'),'-u',str(P/'code/train.py'),m,'--execute'],cwd=R,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
   for line in proc.stdout:
    log.write(line);log.flush()
    try:
     record=json.loads(line)
     if 'epoch' in record:print(json.dumps({'model':m,'epoch':record['epoch'],'stage':record['stage'],'validation':record['validation'],'elapsed_seconds':record['elapsed_seconds']}),flush=True)
    except (json.JSONDecodeError,TypeError):print(m+': '+line.rstrip(),flush=True)
   code=proc.wait()
  if code:
   status['models'][m]={'status':'failed','returncode':code};status['status']='stopped_on_failure';status['sequence_wall_seconds']=time.perf_counter()-start;dump(O/'sequence_status.json',status);raise RuntimeError(m+' failed; preserve outputs, no automatic extra experiment')
  run=json.loads((P/f'runs/{m}_seed42/run_status.json').read_text());run['process_wall_seconds']=time.perf_counter()-launched;run['checkpoint_path']=str((P/f'runs/{m}_seed42/checkpoints/best.pt').relative_to(R));status['models'][m]=run;dump(O/'sequence_status.json',status);print(json.dumps({'completed':m,**run}),flush=True)
 else:status['status']='complete_all_five'
 status['sequence_wall_seconds']=time.perf_counter()-start;status['finished_UTC']=datetime.now(timezone.utc).isoformat();dump(O/'sequence_status.json',status);print(json.dumps(status,indent=2),flush=True)
if __name__=='__main__':main()
