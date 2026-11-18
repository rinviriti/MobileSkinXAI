from common import *
from analyze import KEYS,NAMES,assess
import shutil,xml.etree.ElementTree as ET,importlib.metadata as md
from PIL import Image
import torch

def table(rows,cols):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(r.get(c,'')) for c in cols)+' |' for r in rows)+'\n'
def main(out):
 out=Path(out);assert load(out/'ALL_FIVE_COMPLETE.json')['status']=='PASS';assert load(out/'12_Reports/GRADCAM_COMPLETE.json')['status']=='PASS';lock=load(out/'00_Experiment_Protocol/PROTOCOL_LOCK.json')
 for name,h in lock['sha256'].items():assert sha(out/name)==h,name
 protected=load(out/'11_Source_Data/protected_original_stat_snapshot.json')
 for name,value in protected.items():
  p=ROOT/name;assert p.exists() and p.stat().st_size==value['bytes'] and p.stat().st_mtime_ns==value['mtime_ns'],name
 # Verify the approved implementation package remains byte-for-byte unchanged, without opening heldout outcomes.
 approved=ROOT/'research/thesis_assets/journal_development/v20261117T124247Z_CA_MACL'
 for line in (approved/'PACKAGE_CHECKSUMS.sha256').read_text().splitlines():
  h,name=line.split('  ',1);assert sha(approved/name)==h
 for line in (RELEASE/'checksums.sha256').read_text().splitlines():
  h,name=line.split('  ',1);assert sha(RELEASE/name)==h
 dev=read(out/'source_data/approved_train_validation.csv')
 for r in dev:assert sha(Path(r['source']))==r['source_sha256'];assert sha(ROOT/r['processed_path'])==r['processed_sha256']
 summary=read(out/'02_Validation_Results/CA_MACL_COMPARISON.csv');classwise=read(out/'05_Classwise_Performance/classwise_metrics.csv');contrasts=read(out/'06_Ablation_Comparisons/paired_component_differences.csv');stress=read(out/'07_Synthetic_Shift_Robustness/condition_metrics.csv');indices=read(out/'FIGURE_INDEX.csv');hist={};verified=[];fusion=[]
 for r in summary:
  k=r['model'];run=out/'runs'/k;status=load(run/'status.json');assert sha(run/'checkpoints/best.pt')==r['checkpoint_sha256'];rows=read(run/f'validation_epoch_{int(r["best_epoch"]):03d}.csv');values=assess(rows)[-1]
  for name in ['accuracy','macro_f1','balanced_accuracy','MEL_recall','SCC_recall']:assert abs(float(r[name])-values[name])<1e-12
  hist[k]=[json.loads(l) for l in (run/'history.jsonl').read_text().splitlines()];assert len(hist[k])==int(r['epochs_completed']);assert all(h['optimizer_updates_this_epoch']==727 for h in hist[k]);assert int(status['global_updates'])==727*len(hist[k])
  for commit in sorted(run.glob('commit_epoch_*.json')):
   receipt=load(commit);assert sha(run/'checkpoints'/receipt['checkpoint'])==receipt['sha256']
  verified.append({'model':k,'metrics_from_probabilities':'PASS','epoch_commit_hashes':'PASS','updates_per_epoch':727,'epochs':len(hist[k])})
  if k in ['C2','C3','C4']:
   cp=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=True);s=cp['student'];alpha=s['scale_logits'].float().softmax(0).tolist();fusion.append({'model':k,'alpha_stride8':alpha[0],'alpha_stride16':alpha[1],'residual_gain':float(s['rho'].sigmoid()),'interpretation':'learned feature fusion coefficients, NOT teacher confidence/reliability gates'});del cp
 common=min(len(h) for h in hist.values())
 for i in range(common):assert len({hist[k][i]['training_order_sha256'] for k in KEYS})==1
 write(out/'11_Source_Data/numerical_verification.csv',verified);write(out/'08_Model_Complexity/selected_fusion_coefficients.csv',fusion)
 for r in indices:
  sources=r['source_data'].split(';');r['source_sha256']=json.dumps({s:sha(out/s) for s in sources})
  for ext in ['png','svg','pdf']:
   p=out/r[ext];assert p.exists() and p.stat().st_size>100
   if ext=='png':
    with Image.open(p) as im:assert min(im.info.get('dpi',(0,0)))>=299;im.verify()
   elif ext=='svg':ET.parse(p)
   else:assert p.read_bytes().startswith(b'%PDF-')
   assert sha(p)==sha(out/'10_Publication_Figures'/p.name)
  assert (out/r['generation_script']).exists()
 write(out/'FIGURE_INDEX.csv',indices)
 dump(out/'11_Source_Data/analysis_environment.json',{d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name']})
 best=max(summary,key=lambda r:float(r['macro_f1']));robust=max(summary,key=lambda r:float(r['mean_stress_macro_f1']));c4=next(r for r in summary if r['model']=='C4');c3=next(r for r in summary if r['model']=='C3');criteria=load(out/'07_Synthetic_Shift_Robustness/PREDECLARED_CRITERIA.json')
 resulttable=[{'Model':r['model_name'],'Accuracy':f"{float(r['accuracy']):.4f}",'Macro-F1':f"{float(r['macro_f1']):.4f}",'Balanced accuracy':f"{float(r['balanced_accuracy']):.4f}",'MEL recall':f"{float(r['MEL_recall']):.4f}",'SCC recall':f"{float(r['SCC_recall']):.4f}",'Mean stress F1':f"{float(r['mean_stress_macro_f1']):.4f}",'Degradation':f"{float(r['robustness_degradation']):+.4f}",'Best/total epochs':r['best_epoch']+'/'+r['epochs_completed'],'Train min':f"{float(r['training_seconds'])/60:.2f}",'Stress min':f"{float(r['stress_seconds'])/60:.2f}"} for r in summary]
 changes=[]
 for i in range(1,5):
  a,b=summary[i],summary[i-1];delta=float(a['macro_f1'])-float(b['macro_f1']);st=float(a['mean_stress_macro_f1'])-float(b['mean_stress_macro_f1']);ci=next(r for r in contrasts if r['contrast']==f'C{i} minus C{i-1}' and r['metric']=='macro_f1');changes.append({'Component':['Acquisition augmentation','Multi-scale fusion','JS consistency','Class-aware SupCon'][i-1],'Contrast':f'C{i}−C{i-1}','Clean F1 difference':f'{delta:+.4f}','Conditional95%CI':f"[{float(ci['CI_low']):+.4f},{float(ci['CI_high']):+.4f}]",'Mean stress F1 difference':f'{st:+.4f}','Interpretation':'positive clean point effect' if delta>0 else ('negative clean point effect' if delta<0 else 'no clean point difference')})
 title='MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.'
 report=f'''# CA-MACL five-configuration training and source-only robustness results

{title}

Exactly five new seed42configurations completed using approved ISIC train11632/development-validation2835 from releasev20261117T060846Z_608314c8. Model/loss/data/optimizer functions were preserved from approved implementationv20261117T124247Z_CA_MACL; fresh official ImageNet initialization only. No original M1–M5 was retrained, no teacher/KD introduced, no optional preprocessing treatment enabled. Original experiments and approved implementation passed preservation checks.

## Actual selected-checkpoint results

{table(resulttable,['Model','Accuracy','Macro-F1','Balanced accuracy','MEL recall','SCC recall','Mean stress F1','Degradation','Best/total epochs','Train min','Stress min'])}

Best clean validation macro-F1: **{best['model_name']} ({float(best['macro_f1']):.4f})**. Best equal-mean synthetic-shift macro-F1: **{robust['model_name']} ({float(robust['mean_stress_macro_f1']):.4f})**. Rankings are development point estimates, not independent clinical model selection or external superiority. All row metrics refer to the same clean-validation-selected checkpoint. Stress metrics never selected checkpoints or changed any locked run.

Training time includes actual optimization, clean validation, epoch checkpoint/logging; excludes preparation, initialization and post-training figure/XAI work. Source stress time is separate. All epochs have727optimizer updates, not1454despite two views. Shared train-order hashes match for all{common}epochs available in every configuration. Raw probabilities, seed/config/state hashes and checkpoint commit receipts are retained; class precision/recall/F1/specificity/support/AUROC/AP are in05_Classwise_Performance/classwise_metrics.csv. NLL/Brier/ECE10 are in the comparisonCSV and reliability bins in11_Source_Data; no calibration fitted.

## Incremental component findings

{table(changes,['Component','Contrast','Clean F1 difference','Conditional95%CI','Mean stress F1 difference','Interpretation'])}

“Helps” is metric-specific: read class recall, errors and stress alongside macro-F1. Positive/negative signs are observed single-seed conditional differences, not general causal or significance claims. Adjacent comparisons change one intended component package given earlier components. Fusion includes its projections/learned residual mixture; SupCon includes its training-only projector and class-aware objective. C4−C0 changes multiple factors; no factorial interactions or contrastive class-weighting-only contrast is identified.

**Full C4 versus C3:** clean macro-F1 difference{float(c4['macro_f1'])-float(c3['macro_f1']):+.4f}, mean stress difference{float(c4['mean_stress_macro_f1'])-float(c3['mean_stress_macro_f1']):+.4f}, MEL recall difference{float(c4['MEL_recall'])-float(c3['MEL_recall']):+.4f}, SCC recall difference{float(c4['SCC_recall'])-float(c3['SCC_recall']):+.4f}. This quantifies the contrastive addition under this exact recipe. It does not establish reproducible superiority, clinical safety, novelty or better smartphone transfer. Realized/selected epochs can differ despite shared caps/stopping, so these are training-policy contrasts rather than equal-update endpoint tests.

## Fixed source-only acquisition stress

Eight conditions from the immutable protocol are reported individually: brightness.85/1.15, warm/cool gains, JPEG60, Gaussian radius.75, scale.85, horizontal shading.12. Same eligible source validation images/labels, deterministic transforms, selected checkpoint locked first. Degradation=cleanF1−shiftF1, positive=worse. Mean robustness equally weights all8conditions. Smaller degradation alone can reflect a weaker clean model; inspect absolute clean AND shifted performance. Per-class MEL/SCC stress recalls and all conditions are saved; no favorable-only reporting.

C4vsC0predeclared development criteria: `{json.dumps(criteria)}`. These are descriptive point margins, not clinical thresholds or statistical noninferiority tests. No setting/model was changed on the basis of a stress outcome.

## Uncertainty, inference architecture and explanations

Known-lesion group bootstrap2000replicates/seed42with shared draws across models/conditions gives percentile95% intervals and paired source differences. Validation contains{load(out/'12_Reports/ANALYSIS_COMPLETE.json')['known_validation_lesions']}known lesion groups; SCC has95images, with repeated lesions. These intervals condition on selected checkpoints, reused validation and one training seed; they do not account for training-seed uncertainty, checkpoint-selection bias, multiplicity or missing patient linkage. No statistical-significance claim.

C0/C1inference parameters2,498,148; C2/C3/C4parameters2,587,815. Counted Conv/Linear MACs184,730,688vs184,817,728at224RGB, excluding pooling/normalization/activation/adds/memory traffic. C4training-only projector172,224parameters is excluded from inference footprint. PeakVRAMmeansallocatedCUDA memory, not total device/driver usage. No phone latency estimate is substituted.

Grad-CAM uses genuine evaluated validation cases selected by the predeclared median-confidence rule, with BF16original-batch probability parity. Fusion models show separate early/intermediate/deep maps; they are independently normalized and are not a combined causal explanation. Deep-stage maps omit direct shallow residual contributions. Missing strata or zero maps, if present, are recorded; no replacement cases are cherry-picked. No clinician/XAI-fidelity validation.

## Preservation, outputs and journal limitations

Original checkpoints/reports/figures, immutable release metadata and all development raw/prepared image hashes verified unchanged. Previous approved source-only implementation package remains byte-identical. No PAD or independent ISIC test files/outcomes were used; no HIBA/additional dataset, final trained-model TFLite export or Flutter change. Exactly five new full variants; no extra seeds or hidden controls. Epoch-boundary recovery archives, if any, preserve interrupted work and skip verified completed models.

This is a development study. Source-only synthetic corruption robustness cannot establish real dermoscopy-to-smartphone transfer. Source validation was already used in earlier research; unknown biological/patient independence, class imbalance, limited SCC/MEL supports, single seed, stopping-budget differences and component interactions constrain inference. An independently sourced clinical smartphone cohort, appropriate replicated/matched controls, trained-checkpoint export/phone measurements and quantitative explanation validation remain needed for stronger journal claims. CA-MACL is a working name, not established originality.

FIGURE_INDEX.csv lists{len(indices)}actual-data figure families with captions, source hashes andPNG300DPI/SVG/PDFpaths; mirrored in10_Publication_Figures. All checkpoints and resume state remain under runs/C0…C4/checkpoints; exact paths/hashes below.

{table([{'Model':r['model_name'],'Checkpoint':r['checkpoint'],'SHA256':r['checkpoint_sha256']} for r in summary],['Model','Checkpoint','SHA256'])}
'''
 (out/'12_Reports/CA_MACL_TRAINING_RESULTS.md').write_text(report)
 (out/'README.md').write_text('''# CA-MACL full training package

Start with12_Reports/CA_MACL_TRAINING_RESULTS.md and02_Validation_Results/CA_MACL_COMPARISON.csv.00_Experiment_Protocol holds immutable preregistration, original definitions and prechecks;01histories;02selected probabilities/metrics;03confusions;04ROC/PR;05class metrics;06paired ablations/C4vsC3;07fixed source-only stress;08complexity/resources;09actual validation Grad-CAM;10mirrored publication exports;11source/verification data;12reports. FIGURE_INDEX.csv provides captions, exact source paths/hashes and generation scripts. runs/C0…C4 contain immutable epoch payloads, best/last symlink selectors, optimizer/RNG/control states, every augmentation and prediction log. Exactly five configurations, no heldout evaluation/export/Flutter changes.

Training runner supports --resume only for failed/partial epochs, verifies commits/config/seal, archives uncommitted artifacts and skips complete models. Approved model/loss/pipeline/optimizer remain unchanged. Code files for training are sealed; postprocessing scripts are archived separately with final package hashes. All development scores have validation-selection and single-seed limitations. No external generalization or novelty established.
''')
 (out/'FIGURE_CAPTIONS.md').write_text('\n\n'.join(f"## {r['figure_id']}\n\n{r['title']}\n\n{r['caption']}" for r in indices)+'\n')
 dump(out/'12_Reports/POSTFLIGHT_VERIFICATION.json',{'status':'PASS','exactly_five_new_full_configurations':KEYS,'all_epoch_checkpoint_hashes_verified':True,'every_epoch_updates':727,'shared_order_epochs':common,'original_protected_file_stats_unchanged':len(protected),'approved_implementation_byte_identical':True,'release_and_all_development_input_hashes_unchanged':True,'metrics_recomputed_from_probabilities':True,'figure_families':len(indices),'canonical_exports':len(indices)*3,'PNG_decode_DPI_SVG_XML_PDF_header_checks':'PASS','full_PDF_renderer_used':False,'XAI':load(out/'12_Reports/GRADCAM_COMPLETE.json'),'no_heldout_evaluation':True,'final_TFLite_or_Flutter_modification':False,'results_are_development_only':True})
 lines=[f'{sha(p)}  {p.relative_to(out)}' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='PACKAGE_CHECKSUMS.sha256'];(out/'PACKAGE_CHECKSUMS.sha256').write_text('\n'.join(lines)+'\n');print('FINAL PACKAGE VERIFIED',out,'FIGURES',len(indices),flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
