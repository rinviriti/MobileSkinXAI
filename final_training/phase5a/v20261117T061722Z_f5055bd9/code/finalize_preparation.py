"""Write preparation reports from actual benchmark/export records; no training."""
import csv,json,math,shutil
from pathlib import Path
from core import PROJECT,ROOT,config,digest

def js(name,x):(PROJECT/name).write_text(json.dumps(x,indent=2)+'\n')
b=json.loads((PROJECT/'reports/benchmark.json').read_text());epoch=json.loads((PROJECT/'reports/bounded_epoch.json').read_text());export=json.loads((PROJECT/'reports/export_pretrained_tflite.json').read_text());contracts=json.loads((PROJECT/'reports/contract_checks.json').read_text())
assert len(epoch['cases'])==5 and len(b['cases'])==9 and export['status']==contracts['status']=='PASS'
rows=[];estimates={}
for e in epoch['cases']:
 m=e['model'];cfg=config(m);cs=[c for c in b['cases'] if c['model']==m];ft=next(c for c in cs if c['stage']=='finetune');head=next((c for c in cs if c['stage']=='head'),None)
 validation=math.ceil(2835/cfg['batch_size'])*ft['inference_batch_mean_seconds_training_images_only']
 fine_seconds=e['epoch_wall_seconds_including_loader_startup']+validation
 head_seconds=math.ceil(11632/cfg['batch_size'])*head['bounded_step_mean_seconds']+math.ceil(2835/cfg['batch_size'])*head['inference_batch_mean_seconds_training_images_only'] if head else 0
 nominal=(head_seconds*cfg['frozen_epochs']+fine_seconds*(cfg['max_total_epochs']-cfg['frozen_epochs']))/60
 estimates[m]={'batch_size':cfg['batch_size'],'measured_training_epoch_seconds':e['epoch_wall_seconds_including_loader_startup'],'validation_proxy_seconds':validation,'fine_epoch_including_validation_proxy_seconds':fine_seconds,'head_epoch_extrapolated_seconds':head_seconds,'nominal_max_budget_minutes':nominal,'planning_interval_minutes':[nominal*1.25,nominal*2],'budget':cfg['budget_kind'],'epochs_max':cfg['max_total_epochs']}
 rows.append({'model':m,'batch_size':cfg['batch_size'],'training_images_per_second':e['images_per_second'],'measured_training_epoch_seconds':e['epoch_wall_seconds_including_loader_startup'],'peak_allocated_GiB':max(e['peak_allocated_GiB'],ft['peak_allocated_GiB']),'peak_reserved_GiB':max(e['peak_reserved_GiB'],ft['peak_reserved_GiB']),'nominal_max_budget_minutes':nominal,'planning_min_minutes':nominal*1.25,'planning_max_minutes':nominal*2})
nominal=sum(x['nominal_max_budget_minutes'] for x in estimates.values())/60
report={'source':'Actual one-epoch timings on11632approved training images per architecture; frozen-head stage from96-update bounded samples; validation inference proxy from training images only','models':estimates,'total_nominal_max_budget_hours':nominal,'total_planning_interval_hours':[nominal*1.25,nominal*2],'early_stopping':'May shorten actual runs; patience6 validation macro-F1','included_compute':'M1-M4 up to5head+25finetune; M5 up to40 additional after fresh M3, including two frozen teacher and two student forwards','not_measured':'Final validation/checkpoint serialization costs, sustained desktop/thermal variation, final M5 gate efficacy; planning multipliers are heuristics, not statistical confidence intervals'}
js('reports/runtime_estimates.json',report)
with (PROJECT/'reports/benchmark_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
provenance={}
for m in ['M1','M2','M3','M4']:
 p=ROOT/f'final_training/pretrained/{m}/provenance.json';r=json.loads(p.read_text());assert r['sha256']==r['publisher_sha256']==digest(p.parent/'model.safetensors');provenance[m]=r
js('reports/official_initialization_provenance.json',provenance)
free=shutil.disk_usage(ROOT).free/2**30
js('reports/resource_summary.json',{'GPU':b['environment'],'free_disk_GiB':free,'checkpoint_storage_planning':'Reserve20GiB for immutable epoch checkpoint payloads and logs; exact sizes depend on optimizer tensors; ~917GiB currently free','requested_batch_sizes_fit':True,'OOM_events':sum(len(c['OOM_batch_attempts']) for c in b['cases']),'checkpoint_policy':'Exclusive-create epoch_###.pt payloads; best.pt and last.pt are mutable symlink selectors within this version; no payload overwritten'})
table='\n'.join(f"| {r['model']} | {r['batch_size']} | {r['training_images_per_second']:.1f} | {r['measured_training_epoch_seconds']:.2f} | {r['peak_allocated_GiB']:.3f} / {r['peak_reserved_GiB']:.3f} | {r['planning_min_minutes']:.1f}–{r['planning_max_minutes']:.1f} |" for r in rows)
(PROJECT/'RUNTIME_REPORT.md').write_text(f'''# Phase5A measured feasibility

RTX5070, {b['environment']['total_GiB']:.2f}GiB total / {b['environment']['free_GiB_before_benchmark']:.2f}GiB free before benchmark. Torch {b['environment']['torch']}, torchvision {b['environment']['torchvision']}, timm {b['environment']['timm']}; CUDA{b['environment']['cuda_runtime']}, BF16. CUDA matrix test passed and pip check reports no broken requirements. Requested batch sizes fit without fallback.

| Model | Batch | Actual train images/s | Actual single epoch (s) | Peak allocated/reserved GiB | Planned max-budget run (min) |
|---|---:|---:|---:|---:|---:|
{table}

Each epoch processed exactly11,632 training images, including augmentation/loader startup, transfers, forward/backward, AdamW and mixed precision. M1-M4 use official ImageNet initialization; M5 uses ImageNet student/teacher architecture proxies only. These are disposable compute probes, not final experiments or thesis accuracy evidence. Their weights were discarded. The additional9 stage probes use3warmups+20representative+96bounded updates each. No heldout image loader ran.

Total nominal maximum-budget estimate: {nominal:.2f}hours; planning interval **{nominal*1.25:.2f}–{nominal*2:.2f}hours**. Model intervals include5head+25fine epochs for M1-M4 and40additional epochs for M5. Early stopping can shorten runs. Validation costs are extrapolated from training-image inference; checkpoint I/O, desktop/thermal load and long-run variation are unmeasured. The1.25–2multipliers are practical allowances, not statistical intervals. A future low-VRAM batch fallback changes timings and must be logged. Reserve20GiB disk; currently{free:.1f}GiB free.

## Conversion result

**PASS:** legitimate ImageNet MobileNetV4 backbone with newly initialized four-class head, PyTorch -> official LiteRT Torch -> TFLite -> LiteRT CPU. Three synthetic parity inputs; max absolute error{export['max_abs_error']:.3g}, tolerances atol1e-4/rtol1e-3. Artifact{export['bytes']:,}bytes; no Flex ops. This is not a final lesion-model export, INT8 test, smartphone latency or Flutter integration.

Existing isolated CPU export environment: torch2.10.0+cpu, timm1.0.31, litert-torch0.10.0, ai-edge-litert2.3.0. Converter warned that optional C++ extensions expect torch>=2.11; conversion/runtime parity succeeded through the available graph path. Do not change the working CUDA environment to address this optional warning. Final trained export and actual phone delegation/latency still require later verification.
''')
plan=f'''# Phase5A final five-experiment plan

Preparation version `{PROJECT.name}`; dataset **v20261117T060846Z_608314c8 only**. Dataset is technically approved; full final training has not been started or authorized in this phase. All earlier lesion checkpoints/accuracy are demonstrations and are excluded. Official ImageNet artifacts are cached publisher-hash-verified weights reloaded into fresh instances, not demonstration-trained weights.

Exactly five configurations in configs/: M1 MobileNetV3-Small; M2 EfficientNet-Lite0; M3 MobileNetV4-Conv-Small; M4 ConvNeXt-Tiny reference/conditional teacher; M5 candidate enhanced MobileNetV4. Unique project ID and empty version-specific checkpoint directories prevent loading earlier skin-model experiments. Do not create extra seeds, tuning configurations or ablations under this budget.

Data: train11,632(NV6838/MEL2611/BCC1799/SCC384); validation2,835; internal test2,703; external PAD1,303. The canonical manifest, data approval, integrity checks, train-only weights and38release artifact checksums were verified; all20,795manifest rows' raw/prepared hashes were also rechecked (41,590file checks). Known/unresolved audited risks are separated by the frozen release policy. Biological patient independence remains unverified where identifiers are absent.

Configuration: input224x224RGB, class orderNV/MEL/BCC/SCC, seed42; AdamW decay1e-4. M1-M4 frozen backbone up to5epochs/headLR1e-3 then up to25epochs/headLR1e-4/backboneLR1e-5. M5 has no repeated head stage and up to40additional fine-tuning epochs. Batch16lightweight/M5, batch8ConvNeXt, measured BF16 mixed precision. Early stopping patience6 on validation macro-F1, including a stage transition reset as predeclared. Train-only weights N/(4*n_c): 0.425270547,1.113749521,1.616453585,7.572916667. No heldout augmentation, optimization, calibration, class weighting or teacher target-domain fitting; no quarantined image may enter any experimental loader. Test/PAD are not loaded by the trainer.

## M5 condition and loss

M5 must load this version's **newly completed** M3 and M4 best checkpoints, enforcing project/config/run IDs, completion status, official initialization hashes and identical data receipt. Fresh M4 validation macro-F1 must strictly exceed M3 (margin0); otherwise stop M5 and report, without replacement. Teacher is frozen/eval.

Exact equations and every configurable coefficient/gate are in M5_LOSS.md and configs/M5.json. L=lambdaCE*weightedCE(weak,strong)+lambdaKD*cross-view temperature-scaledKL+lambdaConsistency*studentJS. Default lambdas1/.5/.1, temperature4, class-weight power.5, confidence>=.7, normalized entropy<=.6, teacher viewJS<=.1, exp(-JS/.1), training-label correctness on both teacher views required. All teacher targets/reliability are detached. Auxiliary denominators retain all class-weight mass; zero acceptance yields zero auxiliary losses with finite supervised gradients. Contracts verified zero/all-class gates and no teacher gradients. Gate coverage must be logged per class during actual training. ImageNet-proxy benchmark acceptance is not final teacher-quality evidence.

M5 may receive up to70student epochs cumulatively(M3<=30+M5<=40), plus M4 teacher training. Any gain is confounded by additional optimization and compute; do not attribute it exclusively to the candidate objective. Budget-matched continuation and component ablations need separate future authorization; none are prepared as extra main configurations.

## Execution safeguards and outputs

Training implementation snapshot is code/train.py; default invocation only inspects a config. It rechecks the approved release lock before model construction, uses no test/PAD loader, records per-epoch train/validation loss, accuracy, macro-F1, balanced accuracy, MEL/SCC recall, validation probabilities and peakVRAM. Immutable epoch checkpoint payloads use exclusive creation; best.pt/last.pt are symlink selectors. Persistent experiment_summary.json updates after each run. No old final/demo checkpoint paths are permitted for M5 parents. No training checkpoint exists now.

Future authorized sequence: M1,M2,M3,M4,M5, using `.venv/bin/python {str((PROJECT/'code/train.py').relative_to(ROOT))} M1 --execute` (substitute one of exactlyfive IDs). This phase deliberately never runs that command with --execute. Code and configurations are version-isolated; the original dataset-release files and active historical config snapshots are unchanged.

## Review decision

**All five configurations are technically ready for approval; M5 execution remains conditional on fresh teacher evidence.** Targeted novelty review finds direct cross-view/confidence overlap(CRLD), class-aware KD overlap(BKD/CBD), confidence weak/strong overlap(FixMatch), and domain-consistency prior art(FixBi). LITERATURE_REVIEW.md gives primary links and the CFIA-KD full-text access limitation. Do not claim novelty or smartphone generalization from these ingredients alone. Training is not blocked on proving novelty, but thesis novelty claims require full-text comparison and later controlled evidence.

Reports: RUNTIME_REPORT.md, reports/runtime_estimates.json, reports/benchmark_summary.csv, reports/benchmark.json, reports/bounded_epoch.json, reports/environment_snapshot.json, both requirement locks, reports/export_pretrained_tflite.json, reports/contract_checks.json and reports/official_initialization_provenance.json. Dataset-only release figures remain unchanged; no methodology or model-performance figures were generated.
'''
(PROJECT/'EXPERIMENT_PLAN.md').write_text(plan)
summary=json.loads((PROJECT/'experiment_summary.json').read_text());summary['preflight_status']='PASS_ready_for_training_approval';summary['M5_condition']='fresh_M4_validation_macro_F1 > fresh_M3_validation_macro_F1; stop otherwise';summary['runtime_estimates']=report;summary['export_status']=export['status'];js('experiment_summary.json',summary)
js('training_authorization.json',{'authorized':False,'reason':'Phase5A prepares and benchmarks only; full five-model training requires a subsequent user instruction','five_configurations_ready_for_approval':True,'M5_conditional':True,'dataset_approval_is_separate':True})
print(json.dumps(report,indent=2))
