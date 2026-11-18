from common import *
import shutil,xml.etree.ElementTree as ET,importlib.metadata
from PIL import Image
from analyze import assess

def table(rows,cols):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(r.get(c,'')) for c in cols)+' |' for r in rows)+'\n'
def main(out):
 out=Path(out);sealcheck(out);receipt=load(out/'00_Protocol/FINAL_LOCK_RECEIPT.json');assert sha(out/'00_Protocol/FINAL_MODEL_SELECTION_LOCK.json')==receipt['sha256'];summary=read(out/'ABLATION_COMPARISON.csv');post=read(out/'07_Cross_Dataset_Exploratory/EXPLORATORY_RESULTS.csv');idx=read(out/'FIGURE_INDEX.csv');preserved=load(out/'11_Source_Data/original_preservation_stat_snapshot.json')
 for name,s in preserved.items():
  p=ROOT/name;assert p.stat().st_size==s['size'] and p.stat().st_mtime_ns==s['mtime_ns'],name
 checkfiles(PHASE6,'PACKAGE_CHECKSUMS.sha256');trainingpackage=ROOT/'research/thesis_assets/final_training_results/v20261117T091430Z_704ed3cc';checkfiles(trainingpackage,'PACKAGE_CHECKSUMS.sha256');assert len(read(trainingpackage/'FIGURE_INDEX.csv'))==104;checkfiles(RELEASE,'checksums.sha256')
 v=load(out/'00_Protocol/verification.json')
 for m in v['parents'].values():assert sha(ROOT/m['path'])==m['sha256']
 for r in summary:assert sha(ROOT/r['checkpoint'])==r['checkpoint_sha256']
 for k in KEYS:
  status=load(out/f'runs/{k}/status.json');rows=read(out/f'runs/{k}/validation_predictions_epoch_{status["best_epoch"]:03d}.csv');m,*_=assess(rows);assert abs(m['macro_f1']-float(next(r['macro_f1'] for r in summary if r['model']==k)))<1e-12
 for r in idx:
  for ext in ['png','svg','pdf']:
   p=out/r[ext];assert p.stat().st_size>100
   if ext=='png':
    with Image.open(p) as im:assert min(im.info.get('dpi',(0,0)))>=299;im.verify()
   elif ext=='svg':ET.parse(p)
   else:assert p.read_bytes().startswith(b'%PDF-')
  for p,h in json.loads(r['source_sha256']).items():assert sha(out/p)==h
  assert (out/r['generation_script']).is_file() or r['generation_script']=='code/analyze.py'
 # Copy exact plotting/analysis scripts without modifying any locked implementation.
 for p in Path(__file__).parent.glob('*.py'):
  dest=out/'code'/p.name
  if dest.exists():assert sha(dest)==sha(p)
  else:shutil.copyfile(p,dest)
 # Full per-class predictions/history already present, plus flat histories for convenience.
 for k in KEYS:
  shutil.copyfile(out/f'runs/{k}/history.jsonl',out/f'01_Training_Histories/{k}_history.jsonl')
  shutil.copyfile(out/f'runs/{k}/status.json',out/f'02_Validation_Results/{k}_status.json')
 histories={k:[json.loads(l) for l in (out/f'runs/{k}/history.jsonl').read_text().splitlines()] for k in KEYS};common=min(len(h) for h in histories.values());assert all(len({histories[k][j]['train_order_sha256'] for k in KEYS})==1 for j in range(common))
 shared=[]
 for k,hs in histories.items():
  for n in [1,7]:
   if len(hs)>=n:shared.append({'model':k,'window':f'first{n} additional epochs','maximum_validation_macro_f1':max(h['validation']['macro_f1'] for h in hs[:n]),'selected_for_primary_result':False,'interpretation':'predeclared descriptive matched-window statistic; no alternate model selected'})
 originalhs=[json.loads(l) for l in (PREP/'runs/M5_seed42/history.jsonl').read_text().splitlines()]
 for n in [1,7]:shared.append({'model':'M5','window':f'first{n} additional epochs','maximum_validation_macro_f1':max(h['validation']['macro_f1'] for h in originalhs[:n]),'selected_for_primary_result':False,'interpretation':'original7epoch trajectory; no additional training'})
 write(out/'11_Source_Data/matched_window_descriptive_results.csv',shared)
 st=[{'Model':r['model_name'],'Accuracy':f"{100*float(r['accuracy']):.2f}%",'Macro-F1':f"{float(r['macro_f1']):.4f}",'Balanced accuracy':f"{float(r['balanced_accuracy']):.4f}",'MEL recall':f"{float(r['MEL_recall']):.4f}",'SCC recall':f"{float(r['SCC_recall']):.4f}",'NV→MEL':r['NV_to_MEL'],'SCC→BCC':r['SCC_to_BCC'],'Best/total epochs':f"{r['best_epoch']}/{r['epochs']}",'Minutes':f"{float(r['training_seconds'])/60:.2f}"} for r in summary]
 dic={r['model']:r for r in summary};b1=float(dic['B1']['macro_f1']);b2=float(dic['B2']['macro_f1']);m5=float(dic['M5']['macro_f1']);m3=float(dic['M3']['macro_f1']);b3=float(dic['B3']['macro_f1'])
 clauses=f"Without reliability, validation macro-F1 differs from original full enhancement by {b1-m5:+.4f}. Under shared ungated consistency, adding teacher KD (B1 minus B2) changes macro-F1 by {b1-b2:+.4f}. The source-only robustness intervention differs from baseline by {b3-m3:+.4f} and original full enhancement by {b3-m5:+.4f}. These are recorded single-seed differences, not significance or mechanism proof."
 (out/'ABLATION_STUDY_REPORT.md').write_text(f'''# Three-run ablation and source-only robustness study

{TITLE}

Release `{RELEASE.name}`. Exactly three new full configurations completed. Pre-registered protocol and implementation seal: `00_Protocol/PRE_REGISTERED_ABLATION_PROTOCOL.md` and `PROTOCOL_LOCK.json`; smoke checks passed before full training. All three began from the same fresh final MobileNetV4-Conv-Small checkpoint, with a fresh optimizer. Original qualified ConvNeXt-Tiny is used only by B1. No demonstration weights used.

## Validation findings

{table(st,['Model','Accuracy','Macro-F1','Balanced accuracy','MEL recall','SCC recall','NV→MEL','SCC→BCC','Best/total epochs','Minutes'])}

M3 epochs include its original ImageNet head/fine-tuning; all other rows are additional epochs from M3. Runtime includes actual training/validation/checkpoint overhead and excludes discarded smoke tests. Full components, parameters, source checkpoints and hashes are in ABLATION_COMPARISON.csv. Per-class precision/recall/F1/support/AUROC/AP and actual raw probabilities are in02_Validation_Results. Loss/teacher diagnostics and optimizer LR states are retained per epoch.

{clauses}

## What is and is not isolated

Original reliability gates BOTH cross-view KD and student JS. B1 removes hard gate and continuous reliability together, keeping full-batch class-weight denominator and coefficients, so effective auxiliary strength changes intentionally. It does not isolate the hard threshold from continuous reliability, or gating from scale attenuation. B1 versus B2 changes only KD under the common ungated JS objective (and avoids teacher forward compute); their initial inputs and augmentations match and all three train image-order hashes match over shared completed epochs. Different weights follow naturally from the intended objectives. B2 is not teacher-guided: its JS has gradients in both student views. Original M5 versus B2 changes teacher KD and teacher-dependent consistency weighting jointly.

B3 uses two-view weighted CE with a mild, pre-fixed source-only acquisition policy; it removes teacher/JS while changing strong-view augmentation. It is a separate robustness intervention and cannot establish an augmentation-only causal effect. The source perturbations preserve full image geometry by avoiding cropping; clinical preservation of every diagnostic color feature has not been independently assessed.

Original M5's cap40 versus study25 differs administratively, but original stopped7 so a25cap would not alter its observed trajectory. Realized and selected epochs differ; descriptive epoch1/first7 windows are saved without choosing alternate checkpoints. There is no matched full-M5 restart and no seed replication, by explicit three-configuration budget. M3 comparisons also include extra optimization and two-view differences. Therefore these data do not establish that original M5's small gain is exclusively caused by either component.

## Implementation and smoke evidence

Exact original objective, confidence thresholds, entropy normalization, T=4 cross-view KL, gradient detach points, weighted reductions and augmentation details are in the protocol. No mismatch with the inspected implementation was found; the misleading shorthand that only JS was gated is corrected explicitly. B1/B2 first-batch CE and augmented tensors are identical. Both branches receive gradients; original teacher/gates are detached; all-rejected original gates zero auxiliary terms. Eight discarded backward batches per configuration provided local timings; no full fourth configuration or optimizer update occurred in smoke tests. BF16 CUDA was used for full training; BatchNorm running stats remain frozen. B2/B3 zero teacher diagnostic values are unused sentinels, not an actual teacher acceptance rate; B3's student JS is computed diagnostically but is not part of its loss.

## Exploratory external reuse and deployment

Read CROSS_DATASET_ANALYSIS.md. All checkpoint choices were sealed before any held-out access in this study. Reuse is POST-HOC EXPLORATORY because Phase6 was previously examined; no winner is replaced and no subsequent optimization was triggered. Primary external metrics aggregate declared lesion probabilities; validation metrics here are image-level and should not be compared as the same statistical unit. No clinical-safety or established novelty claim.

The inference architecture is unchanged4-output timm MobileNetV4-Conv-Small; there are no teacher/gate/augmentation modules in inference. This preserves the previously investigated lightweight conversion architecture, but new checkpoint TFLite correctness and Flutter integration were not tested or modified. No new export occurred.

## Artifacts

FIGURE_INDEX.csv lists{len(idx)}publication figures, eachPNG300DPI/SVG/PDF and mirrored in10_Publication_Ready. Exact source data/scripts are retained. Checkpoints are under runs/B1, runs/B2, runs/B3/checkpoints with every epoch, best and last selectors.09_GradCAM is intentionally documented as not generated: no additional XAI analysis was required and no fabricated examples were added.
''')
 pts=[]
 for r in post:
  if r['level']=='lesion_primary':pts.append({'Model':r['model_name'],'Dataset':r['split'],'n':r['n'],'Accuracy':f"{float(r['accuracy']):.4f}",'Macro-F1':f"{float(r['macro_f1']):.4f}",'95%CI':f"[{float(r['macro_f1_CI_low']):.4f},{float(r['macro_f1_CI_high']):.4f}]",'Balanced accuracy':f"{float(r['balanced_accuracy']):.4f}",'MEL recall':f"{float(r['MEL_recall']):.4f}",'SCC recall':f"{float(r['SCC_recall']):.4f}"})
 (out/'CROSS_DATASET_ANALYSIS.md').write_text(f'''# POST-HOC EXPLORATORY cross-dataset analysis

{TITLE}

Original Phase6 ISIC/PAD results were viewed before this research direction. This is exploratory reuse of exposed cohorts, not fresh independent confirmation or pre-exposure registration. The study protocol, exact hyperparameters and losses were sealed before training; all three validation-selected checkpoint hashes were sealed before held-out reads in this study. No training, calibration, threshold selection, pseudo-labels or adaptation used PAD. Original Phase6 and its roles remain unchanged.

Exactly2703eligibleISICtest and1303PAD images were inferred once per new checkpoint. Primary scoring uses1421ISIC and1026PAD declared lesions, averaging predeclared exact-capture representatives; image-level results are secondary in EXPLORATORY_RESULTS.csv. Same raw/unfitted probabilities and NV/MEL/BCC/SCC argmax convention. M3/M5 rows reuse frozen original Phase6 predictions, not new reference inference. These comparisons as a group are exploratory even though reference measurements were originally independent.

{table(pts,['Model','Dataset','n','Accuracy','Macro-F1','95%CI','Balanced accuracy','MEL recall','SCC recall'])}

Uncertainty:2000unstratified cluster bootstrap replicates, ISIC lesion/PAD patient, seeds42/43, percentile95bounds. PAD803patients can contribute multiple lesions/classes. Same draws enable paired contrasts; separate independent cohort draws quantify domain gaps. Bootstrap arrays, paired macro-F1 differences and internal-minus-external gaps are saved under07_Cross_Dataset_Exploratory. These are conditional single-model/cohort intervals, not seed-level or unknown identity uncertainty and not adjusted multiple-testing evidence. No statistical-significance claim is made.

PAD melanoma has only50images/34lesions/34patients, so recall is imprecise. Missing globally linkable ISIC patients, uncertain biological identity, finite duplicate-screen recall, modality and prevalence shifts, and source SCC/Bowen label definitions remain limitations. No truly independent new smartphone dataset was available or acquired; external superiority remains unconfirmed. Results must not select another winner or justify autonomous clinical use. The full policy and frozen quarantine are unchanged.
''')
 (out/'LIMITATIONS_AND_NOVELTY.md').write_text('''# Limitations and novelty boundaries

Single seed and no matched newly restarted full-M5 arm; unequal realized optimization, different administrative caps and validation-selected stopping; validation effect estimates have selection bias. B1 removes a bundled hard/continuous reliability mechanism and changes effective auxiliary scale; no scale-matched alternative is allowed within this budget. B1/B2 compares KD given ungated JS; originalM5/B2 jointly changes KD and weighting. B3 is a joint supervised-only/augmentation intervention, not an isolated augmentation experiment. OriginalM3 has less total optimization and a different initial training-view policy.

Prior exposure to Phase6/PAD informed direction. Every new held-out comparison is POST-HOC EXPLORATORY and cannot reset independence. No seed-level replication, independent external dataset or clinical validation. Bootstraps use known lesion/patient grouping only and cannot repair missing global patient IDs or uncertain biological identity. PAD melanoma support is small; imbalance/modality/diagnostic-label differences limit interpretation. No additional runs, target adaptation or threshold/calibration fitting were performed. No superiority, clinical safety or established novelty claim.

KD, consistency regularization and source augmentation have established prior-method overlap. See11_Source_Data/VERIFIED_PRIOR_WORK.md for checked primary references to Hinton etal., Mean Teacher and AugMix. The current method is not Mean Teacher or AugMix. This limited review is not an exhaustive confidence-aware distillation/domain-generalization novelty search. A paper needs systematic differentiation, independent external replication and stronger matched controls/multiple-seed evidence; these are limitations, not permission for more runs now.

Identical lightweight inference architecture is retained; no phone latency, new TFLite numerical equivalence or Flutter XAI integration is claimed. Augmentation ranges are a priori mild proxies, not validated smartphone acquisition physics. No Grad-CAM figures or fabricated examples are introduced in this study.
''')
 (out/'EXPERIMENT_PROVENANCE.md').write_text(f'''# Experiment provenance

Dataset release `{RELEASE.name}`, unchanged. The stream-filtered train/validation manifest and every input hash were checked; approved release metadata checksums verified. Source parent hashes and provenance are in00_Protocol/verification.json. Original source trainer/core are copied exactly. Pre-registered protocol/config/engine hashes are immutable-read-only and sealchecked before every run. No use of old demonstration weights.

New best checkpoints:

'''+table([{'Model':r['model_name'],'Checkpoint':r['checkpoint'],'SHA256':r['checkpoint_sha256']} for r in summary if r['model'] in KEYS],['Model','Checkpoint','SHA256'])+f'''\nActual run histories, optimizer payloads, complete per-epoch validation probability files, train image-order SHA256, best/last selectors, timings and peak allocated CUDA memory are retained. First-batch augmentation parity B1/B2 verified; data-order parity across all3 shared epochs verified. Original datasets, original final checkpoint hashes, combined original-checkpoint/Phase6 {len(preserved)} snapshot files/stat records, full Phase6 package checksums and all104training figure families passed preservation verification. No Flutter modifications or TFLite export. Versioned output `{out.relative_to(ROOT)}`.

Reproduce from repo using .venv/bin/python scripts/ablation_study/{'{prepare,engine,analyze,posthoc,finish}'}.py with documented arguments; prepare creates a NEW version, engine check then train takes relative output root. Training refuses overwrite/resume, all seals are verified. Analyze freezes selection, posthoc requires that lock before heldout reads, finish validates hashes and exports. Copy archived scripts back to the same relative script directory if reproducing elsewhere; common resolves repository ancestor. Reproduction of training means a separately authorized new study, not silently executing more full configurations.
''')
 (out/'09_GradCAM/README.md').write_text('No new Grad-CAM requested/generated in this ablation package. Existing Phase6 XAI is preserved. This folder deliberately contains no fabricated or reused-as-new examples.\n')
 (out/'README.md').write_text('Versioned three-run ablation study. Start with ABLATION_STUDY_REPORT.md and CROSS_DATASET_ANALYSIS.md.00_Protocol contains sealed preregistration/configs, checks and final selection lock;01histories;02validation metrics/probabilities;03confusions;04ROC/PR;05class metrics;06ablation comparisons;07explicitly exploratory exposed-cohort reuse;08parameters/runtime;09no new XAI;10mirrored publication figures;11source data and original references;12verification reports;code exact generation scripts;runs immutable epoch payloads and best/last selectors. All primary model roles unchanged.\n')
 dump(out/'11_Source_Data/environment_versions.json',{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']})
 dump(out/'12_Reports/VERIFICATION.json',{'status':'PASS','exactly_three_new_full_configurations':KEYS,'original_checkpoint_and_Phase6_preservation':'PASS','original_training_figures':104,'figures':len(idx),'exports':len(idx)*3,'new_models_same_architecture':True,'data_order_shared_epochs_match':common,'posthoc_labeling':True,'no_subsequent_training':True,'PNG_decode_DPI_SVG_XML_PDF_headers':'PASS','PDF_full_render_not_available':True,'clinical_novelty_superiority_claims':False})
 lines=[]
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name!='PACKAGE_CHECKSUMS.sha256':lines.append(f'{sha(p)}  {p.relative_to(out)}')
 (out/'PACKAGE_CHECKSUMS.sha256').write_text('\n'.join(lines)+'\n');print('FINAL',out,'figures',len(idx),flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
