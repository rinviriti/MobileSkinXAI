"""Epoch-exact recovery wrapper; never changes configuration or starts extra variants."""
from engine import *
def resume(out):
 out=Path(out);guard(out)
 for key in ORDER:
  run=out/'03_E2_Training'/key
  if (run/'TRAINING_COMPLETE.json').exists():assert sha(run/'checkpoints/best.pt')==load(run/'TRAINING_COMPLETE.json')['best_checkpoint_sha256'];continue
  if key=='M5':
   qpath=out/'00_Protocol/TEACHER_QUALIFICATION.json'
   if qpath.exists():
    q=load(qpath);assert q['qualified'] and sha(ROOT/q['teacher_checkpoint'])==q['teacher_sha256'] and sha(ROOT/q['student_checkpoint'])==q['student_sha256'];assert q['teacher_validation_F1']>q['student_validation_F1']
   else:qualify(out)
  if run.exists() and not list(run.glob('commit_*.json')) and (run/'history.jsonl').exists():
   archive=run/'recovery'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_first_epoch');archive.mkdir(parents=True);shutil.move(str(run/'history.jsonl'),archive/'uncommitted_history.jsonl')
  if not (out/f'00_Protocol/SMOKE_{key}.json').exists():smoke(out,key)
  execute(out,key)
  b.atomic(out/'13_Reports/EXPERIMENT_SUMMARY.json',{k:load(out/'03_E2_Training'/k/'TRAINING_COMPLETE.json') if (out/'03_E2_Training'/k/'TRAINING_COMPLETE.json').exists() else {'status':'pending'} for k in ORDER})
 dump(out/'ALL_FIVE_NEW_COMPLETE.json',{'status':'PASS','new_models':ORDER,'M3_retrained':False,'seed':42,'test_access_for_training':False})
if __name__=='__main__':resume(ROOT/sys.argv[1])
