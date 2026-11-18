"""Format verified results into thesis prose; does not recompute predictions or train."""
from pathlib import Path
import csv,json,hashlib,shutil
P=Path(__file__).resolve().parent.parent
ROOT=next(p for p in P.parents if (p/'final_training').is_dir())
def read(name):return list(csv.DictReader((P/name).open()))
def load(name):return json.loads((P/name).read_text())
def table(rows,cols):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(r.get(k,'')) for k in cols)+' |' for r in rows)+'\n'
status=load('03_E2_Training/TRAINING_COMPLETE.json');counts=read('01_PAD_Splits/SPLIT_COUNTS.csv');split=read('01_PAD_Splits/PAD_SPLIT.csv');side=read('E1_E2_SIDE_BY_SIDE_RESULTS.csv');summary=read('05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv');cw=read('CLASSWISE_RESULTS.csv');gaps=read('GENERALIZATION_COMPARISON.csv');paired=read('05_Cross_Dataset_Comparison/PAIRED_DIFFERENCES.csv');eff=read('MODEL_EFFICIENCY.csv');index=read('FIGURE_INDEX.csv');classes=['NV','MEL','BCC','SCC']
def point(k,d,m):return float(next(r for r in summary if r['model']==k and r['dataset']==d and r['analysis_level']=='lesion_primary')[m])
def cls(k,c):return next(r for r in cw if r['model']==k and r['dataset']=='PAD' and r['analysis_level']=='lesion_primary' and r['class']==c)
rows1=[]
for c in classes:rows1.append({'Class':c,**{s:int(next(r for r in counts if r['split']==s and r['class']==c)['images']) for s in ['train','validation','test']}})
rows1.append({'Class':'Total',**{s:sum(r[s] for r in rows1) for s in ['train','validation','test']}})
support=[]
for s in ['train','validation','test']:
 rs=[r for r in split if r['study_split']==s];mel=[r for r in rs if r['label']=='MEL'];support.append({'Partition':s,'Images':len(rs),'Patients':len({r['patient_id'] for r in rs}),'Declared lesions':len({(r['patient_id'],r['lesion_id']) for r in rs}),'MEL lesions / patients':str(len({r['lesion_id'] for r in mel}))+' / '+str(len({r['patient_id'] for r in mel}))})
labels={'accuracy':'Accuracy','macro_f1':'Macro-F1','weighted_f1':'Weighted-F1','balanced_accuracy':'Balanced accuracy','macro_precision':'Macro precision','macro_recall':'Macro recall','macro_AUROC':'Macro AUROC (OvR)','macro_AUPRC_AP':'Macro AUPRC (average precision)','NV_recall':'NV recall','MEL_recall':'MEL recall','BCC_recall':'BCC recall','SCC_recall':'SCC recall'}
rows2=[{'Metric':labels[r['Metric']],**{k:f'{float(v):.4f}' for k,v in r.items() if k!='Metric'}} for r in side]
rows3=[{'Model':r['model'],'ISIC − PAD macro-F1':f"{float(r['ISIC_minus_PAD_macro_F1']):+.4f}",'Grouped 95% interval':f"[{float(r['CI_low']):+.4f}, {float(r['CI_high']):+.4f}]",'Meaning':r['interpretation']} for r in gaps if r['analysis_level']=='lesion_primary']
gain=next(r for r in paired if r['dataset']=='PAD' and r['analysis_level']=='lesion_primary' and r['metric']=='macro_f1')
rows4=[{'Model':k,'Class':c,'Correct / total':cls(k,c)['correct']+' / '+cls(k,c)['support'],'Precision':f"{float(cls(k,c)['precision']):.4f}",'Recall':f"{float(cls(k,c)['recall']):.4f}",'F1':f"{float(cls(k,c)['f1']):.4f}"} for k in ['E1','E2'] for c in classes]
rows5=[{'Model':r['model'],'Parameters':f"{int(r['parameters']):,}",'MACs':f"{int(r['ConvLinear_MACs']):,}",'Train minutes':f"{float(r['training_seconds'])/60:.2f}",'Best / total epochs':r['best_epoch']+' / '+r['epochs'],'Full checkpoint bytes':r['checkpoint_bytes']} for r in eff]
exposure=[r for r in read('11_Source_Data/CUMULATIVE_IMAGE_EXPOSURE.csv') if r['domain']=='PAD'];values=[int(r['presentations']) for r in exposure]
text='''# E1 versus E2: supervised joint-domain training and held-out evaluation

**MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.**

All PAD findings are **POST-HOC EXPLORATORY** because this dataset was examined repeatedly in earlier project phases. The new patient-disjoint split does not make PAD a historically untouched external validation cohort.

E1 is the unchanged original final MobileNetV4-Conv-Small M3 checkpoint, selected using the original ISIC validation set. E2 is exactly one new seed-42 training run of the same inference architecture, initialized from official ImageNet weights and trained jointly on labeled ISIC and PAD development data. E1 measures zero-shot transfer; E2 measures supervised joint-domain adaptation. Standard joint training is not presented as a novel algorithm.

## Table 1. Fixed PAD split

'''+table(rows1,['Class','train','validation','test'])+'\n'+table(support,list(support[0]))+'''
The 781/261/261 image allocation approximates 60/20/20. Patient proportions differ because image/class balance was prioritized under patient and known-identity constraints. No patient or known identity component crosses partitions. Melanoma has 20/7/7 declared lesions from 20/7/7 patients. Biological independence cannot be proved beyond available metadata. Patient [private group], including its previously documented SCC capture/lesion ambiguity, is entirely in validation; no original records were merged, relabeled or removed. The immutable original release and Phase 6 manifests remain unchanged.

The original release's optimization/selection flags retain their historical source-only meaning. The separately authorized adaptation protocol and `study_split` field define PAD eligibility for this new study.

## Training and checkpoint selection

'''+f'''E2 completed **{status['epochs']} epochs**, selected **epoch {status['best_epoch']}**, and used **{status['training_seconds']:.2f} seconds ({status['training_seconds']/60:.2f} minutes)**. Peak allocated GPU memory was **{status['peak_allocated_GiB']:.3f} GiB**, excluding driver and other device allocations. The best PAD validation lesion macro-F1 was **{status['best_PAD_validation_lesion_F1']:.4f}**. ISIC validation was diagnostic only.

'''+f'''Each batch contained eight ISIC and eight PAD images. There were 727 updates per epoch, with uniform shuffled cycles inside each domain and no class-specific oversampling. During the actual run, each PAD training image appeared **{min(values)}–{max(values)} times** (mean {sum(values)/len(values):.2f}); exact per-image and per-epoch exposure logs are retained. This controlled repetition still creates memorization risk, particularly with only 20 melanoma training lesions. Class weights used training labels only.

'''+'''The protocol fixed five frozen-head epochs, up to 30 fine-tuning epochs, AdamW, weight decay 1e-4, head learning rates 1e-3 then 1e-4, backbone learning rate 1e-5, BF16, frozen BN running statistics and early-stopping patience six. Only PAD validation lesion macro-F1 selected the checkpoint. The engineering benchmark used a disposable model; the full run restarted from fresh official ImageNet initialization. No test metric changed training.

## Table 2. Primary declared-lesion results

'''+table(rows2,list(rows2[0]))+'''
Both models used exactly the same 1,421 ISIC test scoring units and 205 PAD test scoring units. PAD test contains 261 images from 126 patients, including seven melanoma lesions and 28 SCC lesions. Primary probabilities are arithmetic means over the release's frozen exact-capture representatives within each declared lesion, followed by argmax in NV/MEL/BCC/SCC order. Supplementary image-level results are reported separately in `05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv`.

## Table 3. Cross-dataset performance differences and PAD improvement

'''+table(rows3,list(rows3[0]))+f'''
The E2-minus-E1 PAD macro-F1 change is **{float(gain['E2_minus_E1']):+.4f} ({100*float(gain['E2_minus_E1']):+.2f} percentage points)**, with a paired patient-grouped percentile 95% interval of **[{float(gain['CI_low']):+.4f}, {float(gain['CI_high']):+.4f}]**. E2's arithmetic ISIC-minus-PAD difference is after labeled PAD exposure; it is not a zero-shot generalization measure.

'''+'''## Table 4. PAD class-wise precision, recall and F1

'''+table(rows4,list(rows4[0]))+'''\n## Table 5. Inference complexity and measured training budget

'''+table(rows5,list(rows5[0]))+'''
MACs count Conv/Linear multiply-accumulates for 224×224 RGB and exclude pooling, normalization, activations and additions. Full checkpoint sizes include training state, not just deployment weights. The inference graph is unchanged, with 2,498,148 parameters and 184,730,688 counted MACs. TFLite compatibility is an architecture-based expectation; this study performed no final conversion, phone latency measurement or Flutter modification.

## Findings and trade-offs

ISIC-only learning transferred poorly to the held-out PAD subset: macro-F1 was 0.4032, balanced accuracy 0.4378, melanoma recall 1/7 and SCC recall 2/28. Joint-domain training improved PAD macro-F1 to 0.7100, balanced accuracy to 0.6784, melanoma recognition to 5/7 and SCC recognition to 6/28. These are target-supervised gains on this fixed exploratory cohort, not independent cross-domain confirmation.

The accuracy increase corresponds to 40 additional correct lesions: 32 BCC, four melanoma and four SCC, with NV correct counts unchanged. Thus 80% of the correct-count gain came from majority BCC. However, macro-F1 improvement was not merely majority-class inflation: melanoma F1 rose from 0.0476 to 0.8333, NV from 0.7200 to 0.8675, BCC from 0.7203 to 0.8603 and SCC from 0.1250 to 0.2791. E2's observed melanoma precision of 1.0 reflects only five predicted melanoma lesions; it does not establish clinical predictive value.

SCC remained the clearest failure: 22/28 SCC lesions were missed by E2. SCC-to-BCC errors increased from 16 to 22 even though SCC recall improved, because SCC-to-NV/MEL errors decreased. The paired SCC-recall gain interval was [-3.23, 31.82] percentage points, so this gain remains uncertain.

Source retention deteriorated. ISIC macro-F1 fell from 0.6853 to 0.6121 (-7.32 percentage points); melanoma recall fell from 62.11% to 48.42%, and SCC recall from 31.58% to 15.79%. The corresponding paired ISIC lesion-bootstrap intervals were negative. Joint training therefore provided a substantial PAD benefit under this recipe while reducing minority-class source performance.

E1 and E2 differ in more than dataset exposure: the historical baseline used different class weights, augmentations, validation-selection units and training budget. Consequently, this comparison evaluates the complete supervised adaptation recipe and cannot isolate the causal effect of adding PAD alone. Only one new full run was authorized and performed; there was no E3, cross-validation, multi-seed study or hyperparameter sweep.

## Uncertainty, thesis contribution and journal limits

Paired uncertainty uses 2,000 valid grouped bootstrap draws, with ISIC declared-lesion groups and PAD patient groups. Images and lesions belonging to a PAD patient remain together. Absent-class candidate draws would be rejected and counted as predeclared; the actual count was zero for all analyses. Cohort intervals do not account for training-seed variation, historical PAD exposure, checkpoint-selection bias or unknown identity links.

With only seven melanoma test patients, one error changes recall by 14.29 percentage points. The small target validation set also makes checkpoint selection noisy. No unsupported significance, clinical-screening, algorithmic-novelty or prospective-validation claim is made.

The defensible thesis contribution is a reproducible frozen-source versus supervised joint-domain comparison, with patient-disjoint PAD partitions, transparent repeat-exposure accounting, lesion-level scoring, paired uncertainty and explicit source/target trade-offs. The unchanged inference cost is encouraging for a future mobile prototype, but practical device benefit and clinical adequacy remain unproven. A stronger journal submission would need genuinely unused clinical-image confirmation, stronger matched controls and replication, prospective validation, explanation-fidelity assessment and actual export/device measurements. Ordinary joint training alone is not a new method. No further training is automatically triggered by these results.

## Explainability, files and preservation

Existing verified original M3 ISIC-validation Grad-CAM is referenced with its source hashes. Eight new E2 Grad-CAM examples use PAD development-validation images only, covering correct and incorrect cases for all four classes. Selection is deterministic by median confidence; predicted-logit gradients target `blocks.4`, and probabilities are checked against the original BF16 batch context. There are no fabricated or blank maps. These qualitative examples are neither lesion segmentation nor Flutter on-device explanations.

The package contains 19 figure families in 300-DPI PNG, SVG and PDF, with source CSV/JSON, scripts and captions. `FIGURE_INDEX.csv` records each export and source hash. The root `Thesis Results` folder was absent, so no existing thesis collection was changed. `12_Reports/POSTFLIGHT.json` records successful preservation, image/checkpoint hash, identity-boundary, exposure and probability/metric reconciliation checks.

Exact frozen checkpoints:

'''
for r in eff:text+=f"- **{r['model']}**: `{r['checkpoint']}`; SHA-256 `{r['checkpoint_sha256']}`.\n"
text+='\nAll original datasets, releases, checkpoints, historical evaluations and figures remain unchanged. No calibration, threshold tuning, final TFLite export or Flutter modification was performed.\n'
archive=P/'12_Reports/INITIAL_ASSEMBLED_REPORT.md'
if not archive.exists():shutil.copyfile(P/'E1_E2_EXPERIMENT_REPORT.md',archive)
(P/'E1_E2_EXPERIMENT_REPORT.md').write_text(text)
(P/'README.md').write_text('# E1/E2 supervised joint-domain experiment\n\nStart with E1_E2_EXPERIMENT_REPORT.md. 00_Protocol holds frozen settings and checkpoint selection; 01_PAD_Splits holds the immutable patient/identity-disjoint split; 02_E1_Frozen_Evaluation and 04_E2_Evaluation contain image/lesion probabilities; 03_E2_Training retains all epoch checkpoints, optimizer/RNG states, histories and augmentation/exposure logs; 05_Cross_Dataset_Comparison contains primary and supplementary metrics and paired intervals; 06–08 hold confusion, ROC/PR and class-wise figures; 09_Explainability holds genuine validation Grad-CAM and prior E1 references; 10_Publication_Figures mirrors all 19 exports; 11_Source_Data contains reproducibility sources; 12_Reports contains verification and logs.\n\nE1 is a frozen zero-shot source baseline. E2 is one supervised joint-domain seed-42 run. All PAD findings are post-hoc exploratory. No E3, folds, multi-seed training, calibration or final TFLite/Flutter work. report.py assembles verified tables; polish_report.py formats the final thesis prose without changing numeric outcomes.\n')
# Preserve the declared PAD-gain columns when rerunning the existing analysis script.
f=P/'code/evaluate.py';s=f.read_text();needle=" for name,data in [('05_Cross_Dataset_Comparison/ALL_SETTING_METRICS.csv',summary)"
if 'for gaprow in gaps:' not in s:
 insert=" for gaprow in gaps:\n  gain=next(z for z in comparisons if z['dataset']=='PAD' and z['analysis_level']==gaprow['analysis_level'] and z['metric']=='macro_f1');gaprow['PAD_macro_F1_gain_vs_E1']=gain['E2_minus_E1'] if gaprow['model']=='E2' else 0.;gaprow['PAD_gain_CI_low']=gain['CI_low'] if gaprow['model']=='E2' else 0.;gaprow['PAD_gain_CI_high']=gain['CI_high'] if gaprow['model']=='E2' else 0.\n"
 assert needle in s;s=s.replace(needle,insert+needle);f.write_text(s)
# Seal after formatting; no probability/checkpoint/figure bytes are changed.
lines=[]
for f in sorted(P.rglob('*')):
 if f.is_file() and f.name!='PACKAGE_CHECKSUMS.sha256':
  with f.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  lines.append(f'{digest}  {f.relative_to(P)}')
(P/'PACKAGE_CHECKSUMS.sha256').write_text('\n'.join(lines)+'\n');print('THESIS REPORT FORMATTED AND PACKAGE RESEALED',len(lines),flush=True)
