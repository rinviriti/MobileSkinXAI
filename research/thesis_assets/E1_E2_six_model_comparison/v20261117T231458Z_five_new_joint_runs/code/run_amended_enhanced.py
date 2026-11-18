from engine_amended_v2 import *
def main(out):
 out=Path(out);guard(out)
 assert load(out/'13_Reports/CONTROLLER_TEST_RESULTS.json')['status']=='PASS'
 for k in ['M1','M2','M4']:
  s=load(out/'03_E2_Training'/k/'TRAINING_COMPLETE.json');assert sha(ROOT/s['best_checkpoint'])==s['best_checkpoint_sha256']
 for key in ['M5','M5-LR']:
  run=out/'03_E2_Training'/key
  if (run/'TRAINING_COMPLETE.json').exists():assert sha(run/'checkpoints/best.pt')==load(run/'TRAINING_COMPLETE.json')['best_checkpoint_sha256'];continue
  if run.exists() and not list(run.glob('commit_*.json')) and (run/'history.jsonl').exists():
   archive=run/'recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_first_epoch');archive.mkdir(parents=True);shutil.move(str(run/'history.jsonl'),archive/'uncommitted_history.jsonl')
  if not (out/f'00_Protocol/SMOKE_{key}.json').exists():smoke(out,key)
  execute(out,key)
  b.atomic(out/'13_Reports/AMENDED_EXPERIMENT_SUMMARY.json',{k:load(out/'03_E2_Training'/k/'TRAINING_COMPLETE.json') if (out/'03_E2_Training'/k/'TRAINING_COMPLETE.json').exists() else {'status':'pending'} for k in ORDER})
 q=amendment_authorization(out)
 dump(out/'ALL_FIVE_NEW_COMPLETE.json',{'status':'PASS','new_models':ORDER,'M1_recovered_without_retraining':True,'M3_retrained':False,'seed':42,'test_access_for_training':False,'protocol_amendment_id':q['amendment_id'],'original_teacher_qualified':False,'teacher_use':'explicitly authorized post-hoc exploratory use, not superiority-qualified'})
if __name__=='__main__':main(ROOT/sys.argv[1])
