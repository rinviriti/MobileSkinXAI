from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'research/thesis_assets/journal_development/CA_MACL_training/v20261117T132848Z_five_run_seed42'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert json.loads((out/'00_Experiment_Protocol/CUDA_PREREQUISITES.json').read_text())['status']=='PASS'
text=r'''# Immutable CA-MACL five-configuration execution protocol

MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.

Exactly C0,C1,C2,C3,C4; seed42. No teacher/KD/reliability gate; no original model retraining. Approved implementation v20261117T124247Z_CA_MACL and release v20261117T060846Z_608314c8. All five configuration files, common/model/loss/pipeline source and approved optimizer/stage/validation functions are byte-identical copies. Existing implementation authorization remains false and unchanged; this NEW execution authorization records the user's explicit approval. The new runner only adds persistent status, update counters, checkpoint commit hashes, complete RNG/control-state serialization and safe interrupted-epoch recovery. It reuses the approved optimizer, mode, validation, metrics and exact mathematical objective; no learning rule was changed.

## Fixed training/selection
Approved ISIC train11632/validation2835, class orderNV/MEL/BCC/SCC; original known lesions/identity/risk groups disjoint. Fresh local legitimate ImageNet publisher SHA7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4, not any prior lesion checkpoint.224RGB, raw EXIF/RGB→Lanczos gray114letterbox; original model normalization; RGB treatment only. Source-only augmentation and stateless RNG ranges unchanged. Batch16/workers4, BF16, moderated class weights clip(sqrt(N/(4Nc)),.5,3), AdamW decay1e-4, head-only5epochs LR1e-3, up to35fine epochs head/fusion/projector1e-4 and backbone1e-5. Fixed stage LRs, fresh optimizer on transition, no rollback, frozen BN running statistics, global gradient clip1.0. Auxiliaries disabled in head stage and linearly ramp over first5fine epochs. C3:0.05JS; C4:0.05JS+0.02class-aware SupCon, temperature0.1. Both-view CE averaged for EVERY config; no doubling updates. Exactly727updates/epoch(max29080/config), checked and persisted each epoch. C0 paired CE vs one-view gradient equivalence is exact in the precheck; BN buffers unchanged, dropout0. C4 training-only positives/negatives use training labels, paired positives for singleton SCC and differentiable zero for a single-class batch.

Global best clean validation image macro-F1 selects checkpoint; patience7applies only in fine stage against global best, stale reset at stage transition. Save all epoch payloads, last/best selectors, optimizer and RNG/control states, per-epoch validation raw softmax vectors and per-image augmentation metadata. No stress outcome, prior scores, PAD, independent ISIC test, external calibration or thresholds influences selection. Exactly five main configurations; no additional seed/variant, automatic winner replacement, final TFLite export or Flutter modification.

## Recovery
Exclusive-create epoch checkpoint plus SHA256 commit receipt marks completed boundaries. Resume validates commit hash, config and protocol, restores model/projector/optimizer/RNG/best/stale/stage/update controls, and replays only an interrupted epoch. Partial artifacts are moved to a versioned recovery archive, never deleted or overwritten. Completed models are verified/skipped. Any numerical/algorithmic failure stops execution for investigation rather than silently changing hyperparameters. Device/storage failures can be resumed with the same sealed configuration. Histories and checkpoint lineage distinguish replay from new experiments.

## Source-only synthetic robustness
For each validation-selected checkpoint, write read-only selection hash before the fixed8conditions from approved pipeline.STRESS: brightness.85/1.15, warm/cool(1.04,1,.96)/(.96,1,1.04), JPEG60, Gaussian radius.75, centered scale.85, horizontal illumination gradient amplitude.12. No crop or synthetic lesion/background; no PAD-derived parameters. Save every stress probability record, condition/class metrics and degradation=clean macro-F1−condition macro-F1. Aggregate robustness=equal mean macro-F1 across8conditions. Stress is diagnostic only and never changes the remaining locked runs. Clean and stress metrics are DEVELOPMENT results, not independent external evidence.

## Planned analysis
Report clean accuracy/macro-F1/balanced accuracy, all-class precision/recall/F1/support/specificity, MEL/SCC recall, NV→MEL/SCC→BCC counts, ROC/AP, calibration without fitting, selected/actual epochs, parameters/MACs, allocated peakVRAM and actual wall time. Adjacent component contrasts C0→C1→C2→C3→C4 are conditional incremental recipe effects; unequal selected/realized epochs prevent fixed-update mechanism claims. C4−C0 is bundled, C4−C3 adds projector+class-aware SupCon jointly, not class balancing vs ordinary SupCon.

Predeclared descriptive C4vsC0 criteria: mean stressF1strictly higher; cleanF1not more than.01lower; cleanMEL/SCC recall not more than.03lower. Point margins are not clinical/statistical noninferiority thresholds. Report all failures and tradeoffs. Known-lesion cluster bootstrap2000replicates/seed42, shared draws across models, percentile95% intervals for source image macro-F1/accuracy/balanced accuracy/MEL/SCC recall and paired differences. Intervals are conditional selected-validation estimates, do not remove checkpoint-selection bias, missing patient links, multiple comparisons or single-seed variability. No significance, established novelty or external-superiority claims.

## Actual-data figures and explanations
Only after actual outputs exist: individual loss/accuracy/macro-F1 curves and marked best epochs; raw/row-normalized and consistent side-by-side confusions; model/class/rare-class comparisons; OvRROC/PR from probabilities; fixed-condition robustness/degradation; complexity/runtime vs macro-F1; C4−C3 deltas. Training loss is weighted composite, validation loss unweightedCE, explicitly labeled. PNG300DPI plus SVG/PDF, captions, sourceCSV/JSON and exact scripts.
Grad-CAM uses actual clean-validation cases chosen deterministically nearest median top1confidence within model/true-class/correctness stratum, tie by imageID. Target predicted-class logit at actual spatial stage; preserve inference batch/context precision and verify probability parity. For multi-scale models, last-stage Grad-CAM explains that branch only; explicitly disclose that shallow residual contributions are not fully represented. Qualitative activation maps are not clinical fidelity or lesion masks. No placeholders if strata absent; record unavailable examples.

## Limits and independence
No held-out ISIC/PAD images, labels, statistics, predictions or historical test metrics are accessed for this study. Hashing approved release metadata is integrity verification only. No HIBA or additional collection. Original source-validation is reused development data; global ISIC patient independence unavailable. Synthetic acquisition tests cannot establish transfer to real smartphone photographs. Single seed and component interactions remain journal limitations. No actual results fabricated. All original checkpoints/reports/figures remain intact.
'''
(out/'00_Experiment_Protocol/EXPERIMENT_PROTOCOL.md').write_text(text)
paths=list((out/'code').glob('*.py'))+list((out/'configs').glob('*.json'))+[out/'source_data/approved_train_validation.csv',out/'source_data/train_class_statistics.json',out/'00_Experiment_Protocol/EXPERIMENT_PROTOCOL.md',out/'00_Experiment_Protocol/PREFLIGHT.json',out/'00_Experiment_Protocol/CUDA_PREREQUISITES.json']
dump(out/'00_Experiment_Protocol/PROTOCOL_LOCK.json',{'created_UTC':datetime.now(timezone.utc).isoformat(),'sha256':{str(p.relative_to(out)):sha(p) for p in paths},'exactly_five':['C0','C1','C2','C3','C4'],'authorized':True,'immutability':'SHA256 plus read-only permissions, owner-reversible'})
for p in paths+[out/'00_Experiment_Protocol/PROTOCOL_LOCK.json']:p.chmod(0o444)
print('PROTOCOL SEALED',out)
