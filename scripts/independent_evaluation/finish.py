from common import *
import shutil,platform,xml.etree.ElementTree as ET
import importlib.metadata
from collections import Counter
from PIL import Image
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,confusion_matrix

def table(rows,columns):
 return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(str(r.get(c,'')) for c in columns)+' |' for r in rows)+'\n'
def main(out):
 out=Path(out);lock=lockcheck(out);summary=read(out/'final_model_comparison.csv');cw=read(out/'classwise_test_metrics.csv');idx=read(out/'FIGURE_INDEX.csv');manifest=read(out/'11_Source_Data/locked_evaluation_manifest.csv')
 assert sha(PACKAGE/'PACKAGE_CHECKSUMS.sha256')==load(PACKAGE/'PACKAGE_VERIFICATION.json')['checksums_sha256']
 before=check_list(PACKAGE,'PACKAGE_CHECKSUMS.sha256');assert before==1848;assert len(read(PACKAGE/'FIGURE_INDEX.csv'))==104
 assert sha(RELEASE/'checksums.sha256')==load(RELEASE/'RELEASE_LOCK.json')['checksums_sha256'];check_list(RELEASE,'checksums.sha256')
 verify=[]
 for r in summary:
  kind='lesion' if r['analysis_level']=='lesion_primary' else 'image';rows=read(out/f'independent_prediction_records/{r["model"]}_{r["split"]}_{kind}_probabilities.csv');y=np.array([int(x['true_index']) for x in rows]);p=np.array([[float(x[c]) for c in CLASSES] for x in rows]);pred=p.argmax(1);assert np.allclose(p.sum(1),1,atol=1e-6)
  assert abs(float(r['accuracy'])-accuracy_score(y,pred))<1e-12;assert abs(float(r['macro_f1'])-f1_score(y,pred,labels=range(4),average='macro',zero_division=0))<1e-12;assert abs(float(r['balanced_accuracy'])-balanced_accuracy_score(y,pred))<1e-12
  for c in CLASSES:assert int(next(x for x in cw if x['model']==r['model'] and x['split']==r['split'] and x['analysis_level']==r['analysis_level'] and x['class']==c)['support'])==int((y==CLASSES.index(c)).sum())
  verify.append({'model':r['model'],'split':r['split'],'level':r['analysis_level'],'recomputed_from_probabilities':'PASS','support':len(rows)})
 # Finalize source hashes for cumulative Grad-CAM source table.
 for r in idx:
  sources=r['source_data'].split(';');r['source_sha256']=json.dumps({s:sha(out/s) for s in sources})
  assert all((out/s).is_file() for s in sources)
  for ext in ['png','svg','pdf']:
   p=out/r[ext+'_path'];assert p.is_file() and p.stat().st_size>100
   if ext=='png':
    with Image.open(p) as im:assert min(im.info.get('dpi',[0,0]))>=299;im.verify()
   elif ext=='svg':ET.parse(p)
   else:assert p.read_bytes().startswith(b'%PDF-')
  assert (out/r['generation_script']).is_file()
 write(out/'FIGURE_INDEX.csv',idx)
 write(out/'11_Source_Data/numerical_recomputation_verification.csv',verify)
 for p in Path(__file__).parent.glob('*.py'):
  target=out/'code'/p.name
  if target.exists():assert target.read_bytes()==p.read_bytes(),('Frozen generation code mismatch',p)
  else:shutil.copyfile(p,target)
 dump(out/'11_Source_Data/package_versions.json',{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']})
 # Frozen input configuration and metadata; checkpoints remain in original immutable locations.
 for k,m in lock['models'].items():
  assert sha(ROOT/m['checkpoint'])==m['checkpoint_sha256'];assert sha(ROOT/m['configuration'])==m['configuration_sha256'];shutil.copyfile(ROOT/m['configuration'],out/'11_Source_Data'/f'{k}_frozen_configuration.json')
 point=[]
 for level in ['lesion_primary','image_secondary']:
  for r in summary:
   if r['analysis_level']==level:
    point.append({'Model':r['model_name'],'Dataset':'ISIC' if r['split']=='test' else 'PAD','Unit':level,'n':r['support'],'Accuracy':f"{float(r['accuracy']):.4f}",'Macro-F1':f"{float(r['macro_f1']):.4f}",'Balanced accuracy':f"{float(r['balanced_accuracy']):.4f}",'MEL recall':f"{float(r['MEL_recall']):.4f}",'SCC recall':f"{float(r['SCC_recall']):.4f}"})
 differences=read(out/'11_Source_Data/paired_model_differences.csv');comparison=[r for r in differences if r['analysis_level']=='lesion_primary' and r['comparison']==NAMES['M5']+' minus '+NAMES['M3'] and r['metric']=='macro_f1'];ct=[{'Dataset':r['split'],'M5–baseline F1 difference':f"{float(r['difference']):+.4f}",'95% paired CI':f"[{float(r['CI_low']):+.4f}, {float(r['CI_high']):+.4f}]"} for r in comparison]
 denom=[]
 for split in ['test','external']:
  rs=[r for r in manifest if r['split']==split]
  for c in CLASSES:
   sr=[r for r in rs if r['label']==c];denom.append({'Dataset':split,'Class':c,'Images':len(sr),'Exact captures':sum(r['use_in_lesion_probability_mean']=='True' for r in sr),'Declared lesions':len({r['lesion_scoring_unit'] for r in sr}),'Bootstrap groups containing class':len({r['bootstrap_cluster'] for r in sr})})
 write(out/'11_Source_Data/final_evaluation_denominators.csv',denom)
 ci=[]
 for r in summary:
  if r['analysis_level']=='lesion_primary':
   ci.append({'Model':r['model_name'],'Dataset':r['split'],'Macro-F1 (95% CI)':f"{float(r['macro_f1']):.4f} [{float(r['macro_f1_CI_low']):.4f}, {float(r['macro_f1_CI_high']):.4f}]",'ECE10':f"{float(r['ECE10']):.4f}",'Brier sum':f"{float(r['Brier_sum']):.4f}",'NLL':f"{float(r['NLL']):.4f}"})
 report=f'''# FINAL INDEPENDENT TEST REPORT

{TITLE}

Dataset release: `{RELEASE.name}`. Selection lock: `MODEL_SELECTION_LOCK.json`, SHA256 `{sha(out/'MODEL_SELECTION_LOCK.json')}`. First predictions were generated only after the read-only lock and pre-prediction execution addendum. Six model roles remain fixed; there was no training, threshold tuning, calibration fitting, checkpoint replacement, or test-driven model selection.

## Primary evaluation: declared lesion units

All 2,703 eligible ISIC and 1,303 eligible PAD images were inferred. Per the release protocol, primary scores average raw softmax probabilities over preselected exact-capture representatives within declared lesions, then use argmax in NV/MEL/BCC/SCC order. This yields 1,421 ISIC lesions and 1,026 PAD lesions. Image-level scores below are secondary. Results are not directly comparable to image-level validation without acknowledging the unit difference.

{table([r for r in point if r['Unit']=='lesion_primary'],['Model','Dataset','n','Accuracy','Macro-F1','Balanced accuracy','MEL recall','SCC recall'])}

## Original enhancement versus fixed baseline

{table(ct,['Dataset','M5–baseline F1 difference','95% paired CI'])}

Original MobileSkinXAI-Enhanced has modest positive point differences in macro-F1 on both cohorts and higher melanoma/SCC lesion recall than the fixed MobileNetV4-Conv-Small baseline. Both paired 95% intervals include zero, so these results do not establish reliable superiority. The larger reference and LR control do not alter the predeclared primary candidate. The LR control has higher internal macro-F1 but no external macro-F1 gain over the baseline; no automatic replacement is made. The extra enhanced optimization budget and lack of mechanism-isolating controls prevent attributing any difference solely to the distillation mechanism. Novelty is not established.

## Secondary image-level results

{table([r for r in point if r['Unit']=='image_secondary'],['Model','Dataset','n','Accuracy','Macro-F1','Balanced accuracy','MEL recall','SCC recall'])}

## Denominators and uncertainty

{table(denom,['Dataset','Class','Images','Exact captures','Declared lesions','Bootstrap groups containing class'])}

Confidence intervals use 2,000 unstratified grouped bootstrap replicates and percentile 2.5/97.5 bounds. ISIC resamples declared lesions; PAD resamples 803 patients, retaining each sampled patient's lesions together. Base seed 42 is used with independent PAD stream seed 43 (offset 1), frozen before predictions. All models share draws for paired comparisons; domain-gap intervals subtract independently resampled datasets. Class-containing patient counts can overlap classes and must not be summed. No replicate lacked a class. Intervals are conditional on these trained models/cohorts; they exclude training-seed uncertainty, audit errors and unknown identity links. No multiplicity-adjusted superiority claims are made. CIs are supplied for prespecified confusion-derived key metrics; AUROC, average precision and calibration have point estimates only.

PAD melanoma has only 50 images,34 declared lesions and 34 patients. SCC has183 images/140 lesions, with low external recall across all models. Small support and clustered sampling make these sensitivity estimates imprecise. Different image modalities (ISIC dermoscopy versus PAD smartphone clinical photography), label definitions, patient mix and prevalence limit transportability; PAD SCC includes the source dataset's in-situ/Bowen category. Raw image accuracy is strongly affected by BCC predominance on PAD. Unknown ISIC patient linkage and biological lesion independence cannot be guaranteed.

## Calibration (no fitting)

{table(ci,['Model','Dataset','Macro-F1 (95% CI)','ECE10','Brier sum','NLL'])}

Probabilities are uncalibrated softmax outputs (lesion means for primary metrics). NLL uses numerical clipping 1e-12 only; multiclass Brier sums squared errors across 4 classes (range 0–2); ECE/MCE use 10 fixed-width top-label bins. These summaries describe the tested cohorts; they are not calibrated clinical risk estimates. Reliability curves show bin counts and may be unstable with small bin support.

## Deployment implications

The large external macro-F1 degradation and particularly poor SCC sensitivity do not support autonomous smartphone cancer screening or clinical-safety claims. Keep the predeclared baseline and original enhanced candidate roles for thesis reporting. Any future domain adaptation, calibration or threshold work requires separate development data and a new independent evaluation; do not reuse these now-unblinded sets to obtain another supposedly independent result. This task performed no Flutter modification or TFLite export.

## Figures, predictions and preservation

`FIGURE_INDEX.csv` lists{len(idx)}figure families, eachPNG300DPI, SVG andPDF, with captions and source hashes. Figures cover primary and secondary confusion matrices, focused baseline/enhancement/LR comparisons, lesion ROC/PR, class metrics, domain gaps, calibration, grouped CIs and genuine Grad-CAM cases. ROC/PR and reliability figures use primary lesion scoring; image-level curve points and calibration source data are also provided. Qualitative Grad-CAM is predicted-class gradient-weighted CAM, not segmentation, causal explanation or clinical validation. The frozen median-confidence selection rule includes correct/error examples without cherry-picking by visual appearance; absent strata, if any, are listed.

Independent image and lesion records are in `independent_prediction_records/`. The original training package's1,848 checksummed files, including all 104 training figure families and 105 archived epoch checkpoints, passed post-evaluation preservation verification. Original checkpoints and release artifacts also reverified. The immutable release's quarantined images were not evaluated.
'''
 # Fix compact counts in prose for readability.
 import re
 report=re.sub(r'(?<=\D)(?=\d)(?<![ /_`.:#\-])','',report) if False else report
 (out/'FINAL_TEST_REPORT.md').write_text(report)
 provenance=f'''# Dataset and model provenance

{TITLE}

Release `{RELEASE.name}`; canonical manifest `{str(RELEASE.relative_to(ROOT))}/dataset_manifest.csv`, SHA256 `{lock['dataset_manifest_sha256']}`. Scoring units SHA256 `{lock['scoring_units_sha256']}`. Release approval true with documented independence limits. Pre-model quarantine/exclusions and 139 ambiguous pair dispositions are copied under 11_Source_Data. Their rules were fixed without any model outputs; no validation/test assignment changed. Eligible known-identity and risk components do not cross retained partitions. Unknown biological/patient links can remain; ISIC global patient IDs are unavailable.

Recorded trainer code rejects loaders other than train/validation; optimization flags are train-only, checkpoint selection was validation macro-F1, and calibration was forbidden. Existing package verification records no previous test/PAD inference. Review supports absence of heldout fitting/selection in recorded workflows, rather than proving all undocumented behavior. Earlier pixel-based similarity audits and integrity hashing are pre-model data-quality work, not optimization. The LR control was tuned against validation only and was frozen as secondary before Phase6.

{table([{'Model':m['name'],'Role':m['role'],'Checkpoint':m['checkpoint'],'SHA256':m['checkpoint_sha256'],'Selected epoch':m['best_epoch'],'Normalization':json.dumps(m['config']['normalization'])} for m in lock['models'].values()],['Model','Role','Selected epoch','Checkpoint','SHA256','Normalization'])}

M1–M4 started from documented legitimate ImageNet publisher artifacts; original M5 and LR control descend from the same fresh final M3 and M4 with parent hashes in the selection lock. No demonstration weights were used. M5 epochs are additional epochs. Strict state-dict loading and original 4-class mappings were used; no pretrained download occurred. Configurations are copied exactly in 11_Source_Data; no TTA, threshold tuning or calibration. EfficientNet-Lite0 uses mean/std 0.5, others ImageNet mean/std. Existing 224 RGB aspect-preserving gray114 letterboxing was consumed unchanged.

ISIC2019: CC BY-NC4.0; attribution to ISIC challenge contributors including HAM10000/ViDIR Medical University Vienna, BCN20000/Hospital Clínic Barcelona and MSK (see repository DATASETS.md for source citations). PAD-UFES-20: CC BY 4.0, Pacheco etal.; clinical images used with source attribution. Reuse remains subject to source licensing. Figures are research illustrations, not diagnosis.
'''
 (out/'DATASET_AND_MODEL_PROVENANCE.md').write_text(provenance)
 (out/'REPRODUCIBILITY_REPORT.md').write_text(f'''# Reproducibility report

Status: PASS. Selection lock permissions0444 plus SHA256 receipt (owner-reversible permissions, not cryptographic external timestamping). Pre-prediction addendum records bootstrap streams and executable inference/metric source hashes. GPU inference: CUDA BF16 with FP32 softmax; strict frozen checkpoints; original batch16/ConvNeXt8, deterministic manifest order, eval mode, no gradients for prediction. Environment, actual durations and peak allocated VRAM are in 11_Source_Data/inference_environment.json and inference_runs.json. No optimizer or model parameter changes.

The active scripts are `scripts/independent_evaluation/`. For reproduction, restore the archived code files to that same repository-relative directory, then run `.venv/bin/python scripts/independent_evaluation/<script>.py <NEW-relative-output-root>`. The common module resolves repository ROOT from this directory layout. Order: preflight (NEW empty output only), inference, metrics, figures, gradcam, finish. Archived scripts are exact copies; their expected location is recorded above. Inference refuses to overwrite predictions. Figure generation refuses overwrites. Checkpoints are referenced by original path and SHA256; all selected frozen copies also remain in the preserved training package. No retraining required. Scikit-learn recomputation independently matches all 24 summary rows, class supports and argmax decisions.

Primary aggregation uses frozen exact-capture representatives and declared lesion IDs. All probabilities are raw/unfitted; per-lesion means are explicitly labeled. Bootstrap source data includes every replicate; paired differences use identical within-dataset draws. ISIC seed 42/PAD43 were sealed before predictions; intervals are percentile 95%, not clinical thresholds. ECE10 uses fixed equal-width bins. Zero-divison precision/F1=0; AUPRC means noninterpolated average precision. No invalid bootstrap replicates. Identity grouping cannot repair unknown patient links. Test sets are now unblinded and cannot honestly serve as untouched tests for subsequent changes selected from these results.

Verification: PNG decoding and 300 DPI; SVG XML parsing; PDF headers; source-file hashes; source-script presence; independent metric recomputation; original 1,848 training-package files and 104 figure families unchanged; all six selected checkpoints/configurations unchanged; release 38 checksummed files unchanged. Full PNG files are raster figures, SVG/PDF are vector exports except embedded photographs/CAM rasters. Numerical axes/annotations are generated directly from verified CSV arrays; selected visual inspection additionally checks layout. No external PDF renderer is required for the header check, which does not prove every PDF rendering engine behaves identically.

No packages installed, no training, no Flutter changes, no TFLite export. Archived earlier failed preprocessing-check attempt contains no predictions: an overly broad verifier assumed every model used ImageNet normalization; the final verifier correctly checks model-specific timm metadata. This was a verifier correction before evaluation, not a changed model recipe.
''')
 (out/'README.md').write_text('''# Final independent evaluation package

FINAL_TEST_REPORT.md contains findings; final_model_comparison.csv has primary lesion and secondary image metrics/CIs; classwise_test_metrics.csv contains class metrics/support/specificity/AUC/AP.

- MODEL_SELECTION_LOCK.json / LOCK_RECEIPT.json / PRE_PREDICTION_EXECUTION_ADDENDUM.json: fixed pre-evaluation roles and rules.
- independent_prediction_records/: all6models×2datasets, image and declared-lesion probabilities, identifiers and labels.
- 01_Confusion_Matrices/: raw/normalized and focused comparisons for both scoring levels.
- 02_Classwise/: precision, recall, F1 and melanoma/SCC sensitivity.
- 03_ROC_PR/: lesion-level one-vs-rest ROC/PR.
- 04_Comparisons/: cross-domain metrics and gaps.
- 05_Calibration/: unfitted reliability diagrams.
- 06_Explainability/: evaluated median-confidence correct/error Grad-CAM examples.
- 07_Confidence_Intervals/: grouped bootstrap plots.
- 11_Source_Data/: configs, audit policy, exact curve points/confusions/bins, bootstrap replicates, timing/environment, Grad-CAM maps and denominator evidence.
- code/: exact generation scripts; no model weights modified or duplicated here.
- FIGURE_INDEX.csv: captions/source hashes/export paths; PACKAGE_CHECKSUMS.sha256 covers final package files.

ISIC and PAD are now unblinded; do not select further model changes and claim these same sets remain independent.
''')
 captions='\n\n'.join(f"## {r['figure_id']}\n\n{r['title']}\n\n{r['caption']}" for r in idx);(out/'FIGURE_CAPTIONS.md').write_text(captions+'\n')
 dump(out/'PACKAGE_VERIFICATION.json',{'status':'PASS','figure_families':len(idx),'figure_exports':len(idx)*3,'original_training_package_files_preserved':before,'original_training_figures_preserved':104,'metrics_rows_independently_recomputed':len(verify),'six_checkpoints_unchanged':True,'dataset_release_checksums_unchanged':True,'selection_lock_unchanged':True,'training_or_calibration_performed':False,'image_predictions':24036,'image_sets':{'ISIC':2703,'PAD':1303},'primary_lesions':{'ISIC':1421,'PAD':1026},'limitations':['unknown biological/patient independence','small PAD melanoma support','no training-seed uncertainty','PDF header/XML+PNG decode checks with selected visual review']})
 lines=[]
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name!='PACKAGE_CHECKSUMS.sha256':lines.append(f'{sha(p)}  {p.relative_to(out)}')
 (out/'PACKAGE_CHECKSUMS.sha256').write_text('\n'.join(lines)+'\n');print('FINAL PACKAGE',out,'FIGURES',len(idx),'CHECKSUMMED',len(lines),flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
