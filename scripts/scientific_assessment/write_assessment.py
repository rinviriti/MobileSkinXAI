"""Read reports/configurations only; write a separate scientific assessment, never model scores."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
A=ROOT/'research/thesis_assets/cross_dataset_ablation/v20261117T105731Z_three_run_source_only'
T=ROOT/'research/thesis_assets/final_test_results/v20261117T103000Z_phase6_final'
P=ROOT/'final_training/phase5a/v20261117T061722Z_f5055bd9'
L=ROOT/'final_training/m5_lr_control/v20261117T075145Z_df7c70a4'
R=ROOT/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
inputs=[ROOT/'research/thesis_assets/final_training_results/v20261117T091430Z_704ed3cc/TRAINING_RESULTS_REPORT.md']+list((P/'configs').glob('M*.json'))+list((A/'00_Protocol').glob('B?.json'))+[L/'configs/M5.json',L/'CONTROL_RESULTS_v2.md',P/'code/core.py',P/'code/train.py',A/'code/engine.py',A/'ABLATION_STUDY_REPORT.md',A/'CROSS_DATASET_ANALYSIS.md',A/'ABLATION_COMPARISON.csv',A/'00_Protocol/PRE_REGISTERED_ABLATION_PROTOCOL.md',A/'11_Source_Data/PREDECLARED_COMPONENT_CONTRASTS.csv',A/'11_Source_Data/gate_scale_diagnostic.json',A/'11_Source_Data/original_M5_B1_gate_parity.json',A/'07_Cross_Dataset_Exploratory/EXPLORATORY_RESULTS.csv',T/'FINAL_TEST_REPORT.md',T/'final_model_comparison.csv',T/'MODEL_SELECTION_LOCK.json',R/'DATASET_PROTOCOL.json',R/'data_approval.json',R/'integrity_checks.json']
before={str(p.relative_to(ROOT)):sha(p) for p in inputs}
version='v'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_read_only_assessment'
out=ROOT/'research/thesis_assets/scientific_assessment'/version;out.mkdir(parents=True)
report=r'''# NOVELTY AND JOURNAL READINESS

**MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.**

**Verdict:** sufficient experimental scope for an undergraduate defense, subject to adviser/rubric requirements, if presented as a reproducible engineering and empirical evaluation study. **Not presently established as a novel, reliably superior reliability-gated method or a clinically validated smartphone screening system.** A negative component result is scientifically useful; it does not make the thesis a failure.

Scope: read-only assessment of recorded configurations, source code and aggregate reports. No training, checkpoint loading, held-out image access, prediction-vector reanalysis, score replacement, calibration, Flutter modification or dataset change. This report is stored outside every frozen package. Evidence paths/hashes are in `ASSESSMENT_PROVENANCE.json`.

## 1. What the controlled comparisons actually identify

The original student remains stock MobileNetV4-Conv-Small, 2,498,148 inference parameters. Its objective is
\[
L=CE_w+0.5\frac{\sum_i a_i r_i k_i}{\sum_i a_i}+0.1\frac{\sum_i a_i r_i j_i}{\sum_i a_i},\quad a_i=\sqrt{w_{y_i}},\;T=4.
\]
Here CE averages weighted weak/strong cross-entropies; k is bidirectional cross-view teacher-to-student KL; j is student-view JS. Detached r combines teacher correctness in both views, confidence>=0.7, mean entropy/log4<=0.6, teacher-view JS<=0.1, and continuous entropy/JS attenuation. **Both KD and JS are gated**, with a full-batch class-weight denominator, not accepted-sample normalization. Teacher confidence is a proxy, not validated clinical uncertainty.

| Contrast | Internally interpretable effect | Boundary/confounding |
|---|---|---|
| Original M5 vs B1 | Remove the whole hard+continuous reliability mechanism from both auxiliaries, conditional on this seed/parent/training policy | Same selected epoch1 and realized7 epochs; original40 vs B1 cap25 was nonbinding. Initial input parity and seven-epoch teacher-gate parity corroborate controls. Cannot isolate hard gating, continuous weighting, class effects, or informative selection **from attenuation of auxiliary strength**. |
| B1 vs B2 | Add teacher cross-view KD **given identical ungated student JS**, views, parent, learning rates and seed | Both7 epochs, selected1. This is the cleanest KD contrast in this study, not proof of teacher benefit under original gated JS or across seeds/cohorts. |
| Original M5 vs B2 | Joint removal of KD and teacher-dependent JS weighting | Not a KD-only ablation; B2 retains meaningful student-only JS. |
| M3 vs M5/B1/B2 | Whole continuation/training-recipe difference | Extra optimization from M3, optimizer restart, two-view training, changed augmentation and auxiliary objectives; not mechanism-only. |
| B3 vs M3/M5/B1/B2 | Whole source-only robustness intervention | Objective/view policy/augmentation and realized optimization differ. No matched CE-only original-view continuation; cannot attribute B3 gains solely to smartphone-style augmentation. B3 selected8/14 additional epochs. |
| M5 vs LR control | Joint10-fold LR reduction under the same stopping/selection policy | Other settings preserved; selected1/7 vs8/14. Different realized updates are an effect of that policy, not an independently controlled variable. Not an equal-update LR contrast or mechanism ablation. |
| M1–M4 | Descriptive architecture-plus-training comparisons | Backbone/pretrained initialization, model-specific normalization, batch sizes and selected/realized epochs differ. No general architecture-only causal ranking. |

“Controlled” means a within-seed computational intervention under documented assumptions, **not** population-level causal proof. A single seed does not erase an internally controlled contrast, but does prevent a robust training-variability claim. Validation was reused for checkpoint selection; uncertainty on selected validation results cannot remove that selection bias. Some exploratory grouped intervals exclude zero; they still do not establish cross-seed reproducibility, multiplicity-adjusted significance or new independent confirmation.

## 2. Contribution–evidence matrix

| Candidate contribution/claim | Recorded evidence | Defensible status |
|---|---|---|
| Audited reproducible lightweight research pipeline | Frozen release, quarantine, provenance, six original/control checkpoints and three ablations, documented budgets, group-aware evaluation | **Supported engineering contribution**; unknown biological/patient identity remains possible, so not “leakage eliminated.” |
| Teacher guidance helps this continuation | B1–B2 validation macro-F1 +0.0142; exploratory PAD +0.0240, conditional paired95% interval[+0.0025,+0.0455] | **Conditional single-seed evidence** for KD under ungated JS; not universal benefit. |
| Reliability gating is the beneficial innovation | M5–B1 validation +0.0007; exploratory PAD −0.0157, paired95% interval[−0.0357,+0.0043] | **Not demonstrated.** Similar validation scores and higher ungated external point score do not establish either general gating benefit or harm. |
| Reliability information rather than weaker regularization causes benefit | B1 removes both selection and continuous suppression; initial-batch KD/JS loss ratios ungated:gated1.51/1.65 | **Unidentified.** Ratios are batch diagnostics, not a constant gradient multiplier. |
| Source augmentation produces domain generalization | B3 PAD0.4105 vs M3 0.3833, but B3 changes objective and receives additional training; PAD already exposed | **Exploratory intervention result**, not isolated augmentation efficacy or confirmed domain generalization. |
| New lightweight architecture / new KD or XAI principle | Same timm student graph; established KD, weighting, consistency and Grad-CAM components | **Unsupported broad novelty claim.** A project-specific combination is implemented, not established methodological originality. |
| Explainability | Actual qualitative Grad-CAM examples | **Visualization demonstrated**; no explanation faithfulness, clinician utility or clinical reasoning validation. |
| Smartphone deployment and safety | Lightweight graph and prior conversion feasibility work | **Research feasibility only.** No new selected-checkpoint TFLite equivalence, real-phone latency/energy or end-to-end Flutter/XAI validation; poor external recall precludes a safety claim. |

Original M5 reduces validation NV→MEL errors relative to B1 (221vs240), but B1 has higher MEL/SCC recall. B1 vs B2 reduces NV→MEL321→240 while lowering MEL recall0.7782→0.7355 and SCC0.4211→0.3895. This is a tradeoff, not an across-metric improvement. The LR control improves selected validation macro-F1 but loses melanoma sensitivity and fails its joint development goals. None establishes necessity of the original full reliability mechanism.

## 3. Closest verified published overlap

Targeted primary-source review, checked2026-11-17; abstracts and accessible method excerpts, **not an exhaustive systematic novelty search or reproduced benchmark**. Published scores are not comparable across different class sets, splits, units or datasets.

| Published method | Real overlap | Difference / unresolved gap |
|---|---|---|
| Zhang et al., **CRLD: Cross-View Consistency Regularisation for Knowledge Distillation**, ACM Multimedia 2024 ([paper/method](https://arxiv.org/html/2412.16493v1), [author code](https://github.com/arcanienz/CRLD)) | **Closest objective-level overlap:** weak/strong views, bidirectional cross-view KD, confidence-based teacher soft-label selection, unchanged inference graph | CRLD also has within-view teacher–student KD. MobileSkinXAI instead adds student–student JS and joint correctness/entropy/teacher-JS reliability plus class factors. The core cross-view/confidence-selection idea is already published; exact variants do not establish novelty. |
| Wang etal., **SSD-KD**, Medical Image Analysis84 (2023),102693 ([publisher](https://doi.org/10.1016/j.media.2022.102693)) | Lightweight MobileNet skin-lesion KD on ISIC2019, softened outputs and weighted CE | Uses intra/inter-instance relational and self-supervised knowledge; not this reliability/JS objective. “First lightweight KD for skin lesions” is untenable. |
| Zhang, Chen & Wang, **Confidence-Aware Multi-Teacher KD**, ICASSP2022 ([author paper](https://arxiv.org/abs/2201.00007), [author implementation](https://github.com/Rorozhl/CA-MKD)) | Ground-truth-assisted sample-wise teacher reliability weighting | Multiple teachers/intermediate features; exact objective differs. Label-correctness/confidence weighting itself is prior art. |
| Zhang et al., **Balanced KD for Long-tailed Learning**, Neurocomputing 527 (2023),36–46 ([publisher](https://doi.org/10.1016/j.neucom.2023.01.063)) | Class-prior-aware distillation under imbalance | Instance-balanced CE plus class-balanced KD differs from current weighted CE/sqrt-class-weight auxiliaries; class awareness itself is prior art. |
| Guo etal., **Leveraging logit uncertainty for better KD**, Scientific Reports14,31249 (2024) ([publisher DOI](https://doi.org/10.1038/s41598-024-82647-6), [indexed abstract](https://pubmed.ncbi.nlm.nih.gov/39732943/)) | Teacher-confidence masks and category uncertainty weighting | Also correlation losses/adaptive temperature; not proof that this precise entropy/JS combination was published, but strong conceptual overlap. |
| Dakhli & Barhoumi, **CFIA-KD**, Applied Soft Computing199,115313 (2026) ([publisher](https://doi.org/10.1016/j.asoc.2026.115313)) | Confidence-aware lightweight skin-lesion distillation and explainability | Ensemble evidence fusion/Bayesian updates and feature-importance transfer; MobileSkinXAI uses one frozen teacher and post-hoc Grad-CAM. Not a directly matched smartphone evaluation. |
| Beyer etal., **A Good Teacher Is Patient and Consistent**, CVPR2022 ([proceedings](https://openaccess.thecvf.com/content/CVPR2022/html/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.html)) | KD sensitivity to augmentation/view alignment and training budget | Their recipe emphasizes same-view correspondence; current cross-view guidance is a testable difference, not automatically superior. No same-view KD control here. |
| **AugMix**, ICLR2020 ([paper](https://arxiv.org/abs/1912.02781)); **MixStyle**, ICLR2021 ([paper](https://arxiv.org/abs/2104.02008)) | Source perturbation/consistency and style-based domain generalization are established | B3 is neither AugMix nor feature-statistic mixing. Fixed mild acquisition augmentation is not by itself a new DG method. |

**Possible research gap, not a priority claim:** an audited, source-only, lightweight four-class dermoscopy-to-smartphone study that distinguishes per-sample reliability information from auxiliary-loss attenuation, reports rare-class failures, and verifies deployment. The evidence supports pursuing that empirical question; it does not establish that no prior paper addresses it. CRLD is a particularly important matched prior-method reference for any journal method claim; it was not reproduced in this study, so superiority to it is untested. Its published finding that cross-view and within-view KD can work jointly also means a failure of this cross-view-only teacher recipe must not be generalized to all cross-view distillation. No feature-importance distillation, target adaptation or novel XAI mechanism was implemented here.

## 4. Domain gap and rare-class failures

Primary unit: declared lesion means, fixed before Phase6; validation is image-level. Do not mix these units or choose the favorable one retrospectively.

| Model | ISIC lesion macro-F1 | PAD lesion macro-F1 | PAD MEL recall | PAD SCC recall |
|---|---:|---:|---:|---:|
| M3 baseline |0.6853|0.3833|8/34=0.2353|5/140=0.0357|
| Original M5 |0.6927|0.3977|13/34=0.3824|7/140=0.0500|
| B1 no reliability |0.6991|0.4133|15/34=0.4412|12/140=0.0857|
| B2 no teacher |0.6529|0.3893|16/34=0.4706|10/140=0.0714|
| B3 robustness intervention |0.6700|0.4105|15/34=0.4412|23/140=0.1643|

M3/M5 gaps are30.20/29.51 percentage points. A smaller gap alone is not success: B2/B3 also have lower internal point scores. B3 still misses117/140 PAD SCC lesions. PAD melanoma has only50images/34lesions/34patients. Known-patient bootstrap accounts for repeated images/lesions, not seed variance or unknown identity links. BCC predominance, modality/acquisition, case mix, label definitions including SCC/Bowen, and calibration shift are plausible contributors, **not individually identified causes**. Original M5 top-label ECE rises0.0448→0.1948. Source validation qualification of M4 did not guarantee target suitability (teacher PAD macro-F1 0.3502).

**Unit sensitivity:** original M5 exceeds M3 on primary ISIC lesion macro-F1 (0.6927vs0.6853), but trails on secondary image macro-F1 (0.6371vs0.6500). Its ISIC SCC recall also changes comparison direction by unit. Preserve the prespecified primary result and report both; do not claim unconditional superiority.

Original Phase6 remains the first locked evaluation of its then-frozen models. All ablation held-out comparisons are **POST-HOC EXPLORATORY**, because prior PAD results informed direction. This is not target-image training, but it is not fresh confirmation either. Closed four-class filtering leaves other lesions/conditions and rejection behavior untested; these results cannot validate general-population screening.

## 5. Thesis contribution and journal requirements

**Strongest undergraduate contribution:** an explainable, mobile-oriented research prototype plus an auditable experimental framework demonstrating that source-domain validation gains do not ensure smartphone transfer, and that KD/reliability choices cause class-specific tradeoffs. Present negative/ambiguous ablations as findings. Keep the official title, but distinguish the research screening objective from achieved clinical capability. Defense readiness is a judgement about scope/evidence, not a guarantee of an institutional pass.

**Realistic journal route:** a transparent empirical study of lightweight distillation and dermoscopy-to-clinical domain shift, rather than a claimed breakthrough gated architecture. Essential for a strong generalization claim: a genuinely untouched smartphone cohort with verified four-class diagnosis mapping, patient/lesion grouping, adequate MEL/SCC precision, permissions/provenance and overlap audit; frozen model roles and analysis before unblinding. Essential for a mechanism claim: matched objective/view/budget controls and multi-seed paired replication with training variability separated from patient sampling. Current single-seed positive grouped intervals cannot substitute for that replication. Subgroup and calibration assessment need adequate support. If retaining deployment/XAI claims, add actual exported-checkpoint numerical equivalence and phone measurements, and quantitative/clinician-informed XAI evaluation. These are evidence requirements, not authorization to run new experiments now. Not every empirical journal paper requires a new algorithm, but this package is not yet a convincing validated novel-method submission.

**Single most valuable next step:** secure and preregister an untouched, independently sourced smartphone cohort, then assess the already-frozen primary M3/M5 pair under the locked roles. No new PAD-based model search. Specify cohort eligibility, reference standards, class-wise CI precision and patient bootstrap before predictions; any additional candidates must be secondary and locked in advance.

**Availability assessment:** locally only ISIC2019/PAD-UFES-20 are present; no new qualifying cohort has been acquired or verified. Public candidates exist, but suitability is not established:
- [DDI](https://ddi-dataset.github.io/index.html) and [DDI-2](https://daneshjoulab.github.io/ddi2-dataset/index.html): pathology-confirmed clinical-image resources with patient information and access terms; smartphone-device provenance and sufficient four-class support require verification. Not automatically smartphone confirmation.
- [SCIN, official project](https://github.com/google-research-datasets/scin) and [Google Research description](https://research.google/blog/scin-a-new-resource-for-representative-dermatology-images/): crowdsourced conditions with dermatologist labels, largely inflammatory/infectious/allergic rather than this cancer taxonomy; not automatically an adequately supported pathology-confirmed NV/MEL/BCC/SCC test.
- [Derm7pt author code](https://github.com/jeremykawahara/derm7pt/blob/master/derm7pt/dataset.py): clinical images, but standard grouped categories do not provide the required separate SCC class.
- [SLICE-3D official data](https://challenge.isic-archive.com/data/): non-dermoscopic3D total-body crops, not equivalent to real smartphone acquisition; four-class labels, source overlap and patient independence need separate checks.
No dataset images or label files were downloaded or inspected for this assessment. Do not claim that no suitable dataset exists globally; **none has been verified suitable here**. A curated prospective or independently held clinical smartphone cohort may be necessary.

## 6. Exactly one conditional training proposal — not executed

Lower priority than independent confirmation. **Hypothesis:** per-image reliability alignment adds information beyond its class-wise attenuation distribution. This follows from the loss non-identifiability, not from optimizing PAD scores.

**One configuration: class-wise permuted-reliability control.** Same frozen M3/M4 parents, student graph, original M5 two-view transforms, class weights, denominator, T=4 and CE+0.5KD+0.1JS. Compute original r_i, then apply r_{pi(i)} to both auxiliary terms, where pi is a random permutation restricted to the same true class **within the current training batch**. Use a separate permutation RNG(seed42) that never consumes augmentation/dropout RNG. Since a_i is constant within class, sum_i a_i r_{pi(i)}=sum_i a_i r_i: class-wise weight mass and marginal reliability are preserved; only assignment to examples changes. This does not preserve loss or gradient norms exactly. Singleton/constant-reliability class groups are unchanged and must be counted as ineffective perturbations.

Fixed preset: seed42;224RGB/ImageNet normalization; batch16/workers4; CUDA BF16; fresh AdamW; weight decay1e-4; head LR1e-4/backbone1e-5; no head-only phase/scheduler; frozen BN running statistics; original data-order policy. **Exactly7 additional epochs, no early stopping and no checkpoint search for the primary contrast.** Compare epoch7 against the existing original-M5 epoch7, not against its selected epoch1. This matches updates and initialization; the earlier stopping bookkeeping did not alter original M5's7-epoch optimization. Primary endpoint: source validation image macro-F1 at fixed epoch7; secondary class recalls/errors and per-class reliability/mass diagnostics. Do not change official M5 scores or replace its selected checkpoint. No ISIC/PAD test use or external superiority inference in this mechanism pilot. It remains a one-seed diagnostic; it cannot establish journal-level reproducibility on its own. No configurations beyond this single conditional proposal are suggested; no training is authorized or performed in this assessment.

**Bottom-line claim boundary:** claim a reproducible empirical framework and documented tradeoffs/domain-shift failures; do not claim validated gating advantage, originality of confidence-aware KD, clinical safety or demonstrated smartphone deployment. The thesis is defensible on the former contribution without chasing another PAD score.
'''
# Spacing in prose only; source paths/formulas remain exact.
for a,b in [('epoch1','epoch 1'),('realized7','realized 7'),('Both7','Both 7'),('selected1','selected 1'),('Joint10','Joint 10'),('joint10','joint 10'),('vs8/14','vs 8/14'),('selected1/7','selected 1/7'),('paired95%','paired 95%'),('PAD0.4105','PAD 0.4105'),('ungated:gated1.51','ungated:gated 1.51'),('221vs240','221 vs 240'),('NV→MEL321→240','NV→MEL 321→240'),('recall0.7782','recall 0.7782'),('SCC0.4211','SCC 0.4211'),('checked2026','checked 2026'),('Wang etal.','Wang et al.'),('Guo etal.','Guo et al.'),('Beyer etal.','Beyer et al.'),('Analysis84','Analysis 84'),('(2023),102693','(2023), 102693'),('Computing199,115313','Computing 199, 115313'),('Reports14,31249','Reports 14, 31249'),('ICASSP2022','ICASSP 2022'),('CVPR2022','CVPR 2022'),('ICLR2020','ICLR 2020'),('ICLR2021','ICLR 2021'),('are30.20','are 30.20'),('misses117','misses 117'),('only50images/34lesions/34patients','only 50 images/34 lesions/34 patients'),('rises0.0448','rises 0.0448'),('0.6927vs0.6853','0.6927 vs 0.6853'),('0.6371vs0.6500','0.6371 vs 0.6500'),('non-dermoscopic3D','non-dermoscopic 3D'),('seed42','seed 42'),('epoch7','epoch 7'),('Exactly7','Exactly 7'),("M5's7-epoch","M5's 7-epoch"),('seed 42;224RGB','seed 42; 224 RGB'),('batch16/workers4','batch 16/workers 4'),('decay1e-4','decay 1e-4')]:report=report.replace(a,b)
(out/'NOVELTY_AND_JOURNAL_READINESS.md').write_text(report)
assert before=={str(p.relative_to(ROOT)):sha(p) for p in inputs}
provenance={'created_UTC':datetime.now(timezone.utc).isoformat(),'status':'read_only_scientific_assessment','reviewed_local_sources_sha256':before,'original_reviewed_files_unchanged':True,'heldout_images_opened':False,'prediction_vectors_reanalyzed':False,'training_or_checkpoint_loading':False,'new_results_generated':False,'new_file':'NOVELTY_AND_JOURNAL_READINESS.md','report_sha256':sha(out/'NOVELTY_AND_JOURNAL_READINESS.md'),'literature_scope':'targeted primary-source abstracts and accessible methods; no systematic completeness claim','external_dataset_access':'official public descriptive webpages only; no image/label downloads'}
(out/'ASSESSMENT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(out.relative_to(ROOT));print('Words:',len(report.split()));print('Reviewed source hashes unchanged:',len(inputs))
