from common import *
import torch,timm
from stopping_v1 import update,should_stop
OUT=Path(__file__).resolve().parents[1];run=OUT/'03_E2_Training/M1';cp=run/'checkpoints'
assert load(OUT/'13_Reports/CONTROLLER_TEST_RESULTS.json')['status']=='PASS'
lock=load(OUT/'00_Protocol/LOCK.json')
for p,h in lock['sha256'].items():assert sha(OUT/p)==h,p
history=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()];cfg=load(run/'resolved_configuration.json');best=-1.;be=0;stale=0;prev=None
for h in history:
 e=h['epoch'];frozen=e<=cfg['frozen_epochs']
 if prev is None or prev!=frozen:stale=0;prev=frozen
 best,be,stale,_=update(best,be,stale,h['PAD_validation']['lesion']['macro_f1'],e,frozen)
 if should_stop(e,cfg['frozen_epochs'],stale,cfg['patience']):stop=e;break
assert be==10 and stop==16
for c in sorted(run.glob('commit_*.json')):
 x=load(c);assert sha(cp/x['checkpoint'])==x['sha256']
net=timm.create_model(cfg['architecture'],pretrained=False,num_classes=4)
for epoch in [10,16]:
 p=cp/f'epoch_{epoch:03d}.pt';state=torch.load(p,map_location='cpu',weights_only=True)
 assert state['epoch']==epoch and state['config']==cfg and state['provenance']['classes']==CLASSES
 assert state['control']['best_epoch']==10 and state['control']['best']==best
 net.load_state_dict(state['state_dict'],strict=True)
 assert all(torch.isfinite(v).all() for v in state['state_dict'].values() if v.is_floating_point())
 if epoch==16:assert state['control']['stale']==6
assert sha(cp/'best.pt')==sha(cp/'epoch_010.pt')
selected=read(run/'PAD_validation_epoch_010.csv')
from base_engine import metrics,primary
m=metrics(primary(selected));assert abs(m['macro_f1']-best)<1e-12
status={'status':'recovered_prespecified_policy','model':'M1','epochs':16,'stop_epoch':16,'best_epoch':10,'stop_reason':'early_stopping_patience_reconstructed','best_PAD_validation_lesion_F1':best,'training_seconds':history[-1]['elapsed_seconds'],'actual_completed_training_seconds':history[-1]['elapsed_seconds'],'prescribed_training_seconds':history[15]['elapsed_seconds'],'duration_note':'actual committed duration includes out-of-protocol epochs17-23; interrupted epoch24 duration unavailable; prescribed duration is epoch16 history time','actual_last_completed_epoch':history[-1]['epoch'],'excluded_selection_epochs':[h['epoch'] for h in history if h['epoch']>16],'peak_allocated_GiB':max(h['peak_allocated_GiB'] for h in history),'best_checkpoint':str((cp/'epoch_010.pt').relative_to(ROOT)),'best_checkpoint_sha256':sha(cp/'epoch_010.pt'),'last_checkpoint_sha256':sha(cp/'epoch_016.pt'),'prescribed_final_checkpoint':str((cp/'epoch_016.pt').relative_to(ROOT)),'actual_last_checkpoint':str((cp/'epoch_023.pt').relative_to(ROOT)),'actual_last_checkpoint_sha256':sha(cp/'epoch_023.pt'),'parameters':sum(p.numel() for p in net.parameters()),'teacher_checkpoint':None,'teacher_sha256':None,'student_start_sha256':None,'budget_kind':'up to5head+30fine epochs; recovered at prescribed patience6 boundary','no_retraining':True,'recovery_policy_authorized_by':'user instruction to fix stopping and recover M1 without retraining'}
for name in ['TRAINING_COMPLETE.json','SELECTION_LOCK.json']:
 p=run/name
 with p.open('x') as f:json.dump(status,f,indent=2)
 (p).chmod(0o444)
write(run/'history_prespecified.csv',[{'epoch':h['epoch'],'PAD_validation_lesion_macro_F1':h['PAD_validation']['lesion']['macro_f1'],'best_epoch':h['best_epoch_so_far'],'stale':h['stale'],'elapsed_seconds':h['elapsed_seconds']} for h in history[:16]])
text=f'''# M1 recovery report\n\nPASS: recovered without optimization or new inference. The original history replay selects epoch 10 and stops at epoch 16 under strict improvement and six consecutive non-improving fine-tuning epochs. The selected validation prediction CSV reproduces macro-F1 {best:.12f}. Epochs 10 and 16 load strictly into {cfg['architecture']} with finite weights and the correct four-class metadata; all 23 committed checkpoint hashes verify.\n\nSelected checkpoint: `{status['best_checkpoint']}`\nSHA256: `{status['best_checkpoint_sha256']}`\nPrescribed final checkpoint: epoch16, preserved directly; historical last.pt still points to epoch23 and is intentionally unchanged.\n\nPrescribed stopped-run duration: {status['prescribed_training_seconds']:.6f} seconds. Actual completed-epoch duration: {status['actual_completed_training_seconds']:.6f} seconds. Additional partial epoch24 consumed unmeasured compute; neither duration claims to include that partial epoch.\n\nAll original files, epoch17–23 checkpoints, partial epoch24 artifacts, logs, best/last symlinks and historical STATUS.json remain unchanged. New TRAINING_COMPLETE.json and SELECTION_LOCK.json explicitly supersede the stale operational status, document recovery and prohibit extra epochs from model selection. No test data were read for this recovery.\n'''
with (OUT/'13_Reports/M1_RECOVERY_REPORT.md').open('x') as f:f.write(text)
print(json.dumps(status,indent=2))
