"""Read-only integrity and loss-gradient contracts; never launches training."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv,json,math
import torch
from core import PROJECT,ROOT,config,digest,m5_loss
from train import approval_check

def main():
 release=ROOT/config('M1')['dataset_release'];rows=list(csv.DictReader(open(release/'dataset_manifest.csv')))
 def check(r):
  assert digest(ROOT/r['processed_path'])==r['processed_sha256'],r['image']
  assert digest(r['source'])==r['source_sha256'],r['image']
  return 1
 with ThreadPoolExecutor(max_workers=4) as p:checked=sum(p.map(check,rows,buffersize=16))
 receipts={m:approval_check(config(m)) for m in ['M1','M2','M3','M4','M5']}
 cfg=config('M5');weights=torch.tensor([.42527054694355076,1.1137495212562236,1.6164535853251807,7.572916666666667])
 y=torch.arange(4);sw=torch.zeros(4,4,requires_grad=True);ss=torch.zeros(4,4,requires_grad=True);tw=torch.zeros(4,4,requires_grad=True);ts=torch.zeros(4,4,requires_grad=True)
 loss,stats=m5_loss(sw,ss,tw,ts,y,weights,cfg);loss.backward()
 assert stats['gate_acceptance']==0 and stats['kd']==0 and stats['consistency']==0
 assert torch.isfinite(loss) and sw.grad is not None and ss.grad is not None and tw.grad is None and ts.grad is None
 logits=torch.eye(4)*10
 _,accepted=m5_loss(logits,logits,logits,logits,y,weights,cfg);assert accepted['gate_acceptance']==1
 assert all(not list((ROOT/config(m)['checkpoint_dir']).glob('*.pt')) for m in receipts)
 assert len(list((PROJECT/'configs').glob('M*.json')))==5
 report={'status':'PASS','all_release_manifest_raw_and_prepared_hashes_checked':checked*2,'manifest_rows_checked':checked,'all_five_training_approval_checks_pass':True,'zero_gate_finite_supervised_gradients':True,'teacher_logits_receive_no_gradients':True,'all_class_correct_confident_gate_contract':True,'final_trained_checkpoints_present':False,'exactly_five_main_configs':True,'data_receipts':receipts}
 (PROJECT/'reports/contract_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
