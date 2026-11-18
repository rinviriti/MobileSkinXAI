from engine_fixed_v1 import *
def main(out):
 out=Path(out);guard(out)
 fix=load(out/'00_Protocol/CONTROLLER_FIX_LOCK.json')
 for p,h in fix['sha256'].items():assert sha(out/p)==h,p
 assert load(out/'13_Reports/CONTROLLER_TEST_RESULTS.json')['status']=='PASS'
 assert load(out/'03_E2_Training/M1/TRAINING_COMPLETE.json')['status']=='recovered_prespecified_policy'
 assert sha(out/'03_E2_Training/M1/checkpoints/best.pt')==load(out/'03_E2_Training/M1/TRAINING_COMPLETE.json')['best_checkpoint_sha256']
 for key in ['M2','M4','M5','M5-LR']:
  run=out/'03_E2_Training'/key
  if (run/'TRAINING_COMPLETE.json').exists():
   assert sha(run/'checkpoints/best.pt')==load(run/'TRAINING_COMPLETE.json')['best_checkpoint_sha256'];continue
  if key=='M5':
   qp=out/'00_Protocol/TEACHER_QUALIFICATION.json'
   if qp.exists():
    q=load(qp);assert q['qualified'] and sha(ROOT/q['teacher_checkpoint'])==q['teacher_sha256'] and sha(ROOT/q['student_checkpoint'])==q['student_sha256']
   else:qualify(out)
  if run.exists() and not list(run.glob('commit_*.json')) and (run/'history.jsonl').exists():
   archive=run/'recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_first_epoch');archive.mkdir(parents=True);shutil.move(str(run/'history.jsonl'),archive/'uncommitted_history.jsonl')
  if not (out/f'00_Protocol/SMOKE_{key}.json').exists():smoke(out,key)
  execute(out,key)
  b.atomic(out/'13_Reports/EXPERIMENT_SUMMARY.json',{k:load(out/'03_E2_Training'/k/'TRAINING_COMPLETE.json') if (out/'03_E2_Training'/k/'TRAINING_COMPLETE.json').exists() else {'status':'pending'} for k in ORDER})
 dump(out/'ALL_FIVE_NEW_COMPLETE.json',{'status':'PASS','new_models':ORDER,'M1_recovered_without_retraining':True,'M3_retrained':False,'seed':42,'test_access_for_training':False,'controller_revision':'engine_fixed_v1'})
if __name__=='__main__':main(ROOT/sys.argv[1])
