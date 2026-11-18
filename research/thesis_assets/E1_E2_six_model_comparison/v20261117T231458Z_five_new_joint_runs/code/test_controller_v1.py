import unittest,json,copy,io,contextlib,hashlib
from pathlib import Path
from stopping_v1 import update,should_stop
OUT=Path(__file__).resolve().parents[1]
def replay(scores,frozen_epochs=0,patience=6,state=None,start=1):
 best,be,stale,prev=state or (-1.,0,0,None);rows=[]
 for epoch,score in enumerate(scores,start):
  if should_stop(epoch-1,frozen_epochs,stale,patience):break
  frozen=epoch<=frozen_epochs
  if prev is None or prev!=frozen:stale=0;prev=frozen
  best,be,stale,improved=update(best,be,stale,score,epoch,frozen);rows.append((epoch,best,be,stale,improved))
  if should_stop(epoch,frozen_epochs,stale,patience):break
 return rows,(best,be,stale,prev)
class ControllerTests(unittest.TestCase):
 def test_improvement_resets(self):
  r,s=replay([.5,.4,.6]);self.assertEqual(s[:3],(.6,3,0))
 def test_six_stale(self):
  r,s=replay([.5]+[.4]*20);self.assertEqual(r[-1][0],7);self.assertEqual(s[2],6)
 def test_later_improvement(self):
  r,s=replay([.5]+[.4]*5+[.6]+[.4]*10);self.assertEqual(r[-1][0],13);self.assertEqual(s[1],7)
 def test_highest_metric(self):
  r,s=replay([.4,.7,.5,.6]);self.assertEqual(s[1],2)
 def test_resume_equivalence(self):
  scores=[.5,.4,.4,.6,.5,.5,.5,.5,.5,.5,.8]
  full,s=replay(scores);a,t=replay(scores[:6]);b,u=replay(scores[6:],state=t,start=7)
  self.assertEqual(full,a+b);self.assertEqual(s,u)
 def test_no_updates_after_stop(self):
  import torch
  p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=.001);best=-1.;be=0;stale=0;updates=0
  for epoch in range(1,30):
   opt.zero_grad();(p*p).backward();opt.step();updates+=1
   best,be,stale,_=update(best,be,stale,.5,epoch,False)
   if should_stop(epoch,0,stale,6):break
  self.assertEqual(updates,7);self.assertEqual(int(opt.state[p]['step']),7);self.assertEqual(be,1)
 def test_frozen_stage(self):
  r,s=replay([.9]+[.5]*20,frozen_epochs=5);self.assertEqual(r[4][3],0);self.assertEqual(r[-1][0],11);self.assertEqual(s[1],1)
 def test_equality_and_index(self):
  r,s=replay([.5]*20);self.assertEqual(s[1],1);self.assertEqual(r[-1][0],7)
 def test_resume_at_boundary_no_updates(self):
  r,s=replay([.5]*7);t,u=replay([.9]*5,state=s,start=8);self.assertEqual(t,[]);self.assertEqual(s,u)
 def test_m1_replay(self):
  h=[json.loads(x) for x in (OUT/'03_E2_Training/M1/history.jsonl').read_text().splitlines()]
  r,s=replay([x['PAD_validation']['lesion']['macro_f1'] for x in h],5)
  self.assertEqual(s[1],10);self.assertEqual(r[-1][0],16)
 def test_actual_loop_contains_guard(self):
  s=(OUT/'code/engine_fixed_v1.py').read_text();self.assertIn("if should_stop(epoch,cfg['frozen_epochs'],stale,cfg['patience']):",s)
  self.assertLess(s.index("print('EARLY STOP'"),s.index("status={'status':'complete'"))
if __name__=='__main__':
 stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControllerTests));print(stream.getvalue())
 report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'engineering_smoke':'actual CPU AdamW loop; constant metric; exactly seven updates then stop, best epoch one','output':stream.getvalue()}
 # Publication tests report to stdout; original research reports remain immutable.
 print(json.dumps(report,indent=2))
 raise SystemExit(not result.wasSuccessful())
