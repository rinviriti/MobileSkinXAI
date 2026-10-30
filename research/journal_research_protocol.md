# MobileSkinXAI — Joint BSc Thesis and Journal Research Protocol

**Status:** Research planning document, 30 October 2026. Hypotheses and methodological proposals are not established findings. **Clinical use is not supported.**

## 1. Working goal and hypothesis

**Working title:** *MobileSkinXAI: Reliable and Efficient Four-Class Skin-Lesion Classification Through Controlled Teacher–Student Learning and External Reliability Auditing*.

**Main research question:** Can the accuracy/reliability–efficiency trade-off of a mobile-ready four-class skin-lesion classifier be improved over a reproducible MobileNetV4 baseline, and do internal improvements generalize beyond dermoscopic source images?

**Candidate hypotheses (to be tested, not asserted):**

- **H1 (capacity):** A pretrained higher-capacity ConvNeXt-Tiny reference increases internal validation macro-F1 relative to a MobileNetV4-Conv-Small student under the documented training and preprocessing protocol.
- **H2 (distillation, conditional):** If H1 is supported, label+soft-logit distillation improves a MobileNetV4 student over its non-distilled baseline without increasing inference complexity materially.
- **H3 (reliability):** The chosen student exhibits measurable calibration and risk–coverage trade-offs; these metrics may deteriorate on external smartphone images.

**Do not claim** a novel combination of KD + Grad-CAM + uncertainty. Prior research already covers that combination. A publishable method claim will require a specific, literature-differentiated mechanism and convincing ablation evidence. If no novel method is established, frame a rigorous reproducibility, efficiency, and cross-device reliability evaluation where journal scope permits.

## 2. What is already finished (single seed 42; 2,835 validation images)

| Architecture | Val accuracy | Val macro-F1 | Val balanced accuracy | MEL recall | SCC recall |
|---|---:|---:|---:|---:|---:|
| MobileNetV3-Small partial fine-tune | 0.7538 | 0.6354 | 0.6534 | 0.5922 | 0.4000 |
| EfficientNet-Lite0 partial fine-tune | 0.7474 | 0.6400 | 0.6489 | 0.6945 | 0.3579 |
| MobileNetV4-Conv-Small partial fine-tune | 0.7584 | **0.6520** | **0.6702** | 0.6877 | 0.3895 |

These are development-validation results, **not independent test results**. Models did not have exactly identical effective training budgets; do not present them as controlled architecture-only evidence without qualification.

## 3. Data integrity and inclusion criteria

- Original source: ISIC 2019 training collection; selected four one-hot labels: NV=0, MEL=1, BCC=2, SCC=3. Other diagnoses excluded from this **closed-set four-class task**, not conflated with negatives.
- Known lesion-ID development images: 19,462; excluded 1,886 images missing lesion IDs from primary development. **No patient ID** is available; never describe the split as proven patient-disjoint. Near-duplicate image exclusion is not verified.
- Split seed 42: train 13,686; validation 2,835; untouched internal test 2,941, with lesion-group disjointness as previously audited.
- Preprocessing: aspect-preserved RGB 224×224 letterboxing with gray 114; preserve raw source files. Training-only augmentation. Use per-model pretrained normalization.
- Class frequency corrections come from train counts ONLY. Maintain same labels/order in every model.
- **PAD-UFES-20**: smartphone-acquired, six diagnosis categories; any external evaluation requires a documented label harmonization protocol for the four common classes, coverage/exclusion table, patient/lesion identifier audit, no target-domain tuning before a locked assessment, and disclosure of dermoscopy-to-clinical modality shift.

## 4. Experiment 6: capacity feasibility and teacher qualification

Use the accompanying `scripts/experiment6_convnext_tiny_teacher.py`:

- Model `convnext_tiny.fb_in1k` (ImageNet-1K pretrained) at 224×224. Approximately 28.6 million parameters **before replacing the ImageNet head**; script records the actual four-class parameter count. Input normalization is resolved from timm.
- Stage A: freeze backbone; train four-class head; maximum eight epochs, early stopping patience four.
- Stage B: start from **best frozen checkpoint**; train final ConvNeXt block plus head (default memory-conscious). Maximum twelve epochs, patience four. Optional `--finetune-last-blocks 2` or `3` only as separately named follow-up experiments, with transparently documented extra budget.
- Weighted train cross-entropy as before; **unweighted validation loss** and macro-F1-selected checkpoint. Record weights and SHA-256 of the parent checkpoint.
- Target for a promising teacher: consistent improvement over MobileNetV4's 0.6520 validation macro-F1, ideally **at least 0.02 absolute macro-F1** without severe deterioration in melanoma or SCC; this is a *planning heuristic*, not a statistical threshold.
- If the reference is similar or worse, do not assume distillation helps: investigate preprocessing/lesion resolution, class-specific errors, training capacity, augmentation and label quality before adding KD.

## 5. Conditional follow-up: controlled student distillation

Only proceed if capacity reference justifies teacher–student learning. Compare:

1. MobileNetV4 baseline from identical initialization/training protocol.
2. Vanilla Hinton-style soft-logit distillation (hard CE + temperature-scaled KL, teacher frozen), with temperature and mixture coefficient tuned on validation only.
3. A single **clearly literature-differentiated** proposed method if the review reveals a meaningful gap, with an ablation removing only its distinctive component.

Implement no validation/test-label-derived weights, features or pseudo-label selection. Teacher logits may be computed on **training images only** for distillation. Student trained without test data. Preserve per-seed full checkpoints and prediction exports. Retain the non-distilled student baseline as a required comparison.

## 6. Evaluation protocol required for a journal

- **Primary:** internal validation macro-F1 during model development; once finalized, **one** locked independent test evaluation, with bootstrap confidence intervals (consider lesion-cluster bootstrap for correlated images).
- **Secondary:** accuracy, balanced accuracy, class-wise precision/recall/F1 and confusion matrix; one-vs-rest AUROC and AUPRC where class probabilities exist.
- **Calibration and referral:** expected calibration error, multiclass Brier score, reliability diagram, and selective risk versus coverage. Set referral rules using validation, not the test set; evaluate after locking. Do not claim clinically safe thresholds.
- **Robustness:** at least 3 full independent training seeds for finalists (frozen initialization, shuffling, augmentation and fine-tuning). Report mean ± SD of per-run metrics. Same validation images do not become independent just because seeds differ.
- **Efficiency:** parameters, serialized checkpoint/TFLite size, measured CPU/GPU latency under stated hardware and warmed-up measurement protocol, peak memory, conversion success or failure. Do not infer mobile runtime from parameters alone.
- **Interpretability:** Grad-CAM examples for correctly and incorrectly classified cases; quantify lesion-localization alignment *only* on datasets where appropriate reference masks exist. Grad-CAM alone is not a proof of faithful model reasoning.
- **External evaluation:** PAD-UFES-20 held out as a clinically different imaging domain; report per-class sample counts, label mismatch exclusions, confidence intervals, and performance degradation. Never treat external results as i.i.d. with source validation.
- **Limitations:** no patient-level split guarantee for ISIC; no verified near-duplicate audit; 95 SCC validation images, distribution shift, model-selection reuse of validation, demographic diversity limits, no clinical approval.

## 7. Literature anchors and novelty audit (verified links, October 2026)

| Study | Why it matters / overlap |
|---|---|
| Zheng et al., 2025. *Stepwise self-knowledge distillation for skin lesion image classification*. Scientific Reports. https://doi.org/10.1038/s41598-025-10717-4 | Skin-lesion self-KD already exists. |
| *Knowledge distillation via teacher-modeled sample relationships for skin cancer diagnosis*, 2025. Engineering Applications of AI. https://doi.org/10.1016/j.engappai.2025.112452 | Distillation with minority-class sample relationships on ISIC2019 already exists. |
| Dakhli & Barhoumi, 2026. *CFIA-KD: Confidence and feature importance aware knowledge distillation for effective skin lesion diagnosis from dermoscopic images*. Applied Soft Computing. https://doi.org/10.1016/j.asoc.2026.115313 | Confidence- and explanation-aware KD already exists. |
| Shakir et al., 2026. *Reliability-Audited Explainable AI for Multi-Class Skin-Lesion Triage Under Cross-Device Domain Shift*. Diagnostics. https://doi.org/10.3390/diagnostics16203368 | Cross-device, PAD external assessment, calibration and selective classification already studied. |
| Kline et al., 2025/2026. *Best Practices and Checklist for Reviewing Artificial Intelligence-Based Medical Imaging Papers: Classification*. Journal of Imaging Informatics in Medicine. https://doi.org/10.1007/s10278-025-01548-w | Medical imaging AI reporting quality checklist. |
| Pacheco et al., 2020. *PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones*. Data in Brief. https://doi.org/10.1016/j.dib.2020.106221 | Source for smartphone images, label metadata and dataset limitations. |

**Novelty decision:** Before writing abstract/title as a novel method, add 10–15 highly related 2023–2026 papers to a structured related-work matrix and identify what none of them actually measures or proposes. A novelty claim must be supported by exact comparison, not by the absence of a familiar buzzword.

## 8. Paper structure and submission gates

Proposed manuscript:

1. Introduction: clinical/edge motivation, concrete research gap and measurable contributions.
2. Related work: lightweight classifiers, imbalance, KD, calibration, cross-device validation; explicit comparisons.
3. Data and methods: sources, class mapping, lesion groups, exclusions, training, model selection, efficiency measurements.
4. Experiments: baseline architecture table; teacher comparison; distillation and ablation; repeats.
5. Reliability: calibration, referral/selective risk, explanation evidence, external smartphone-domain test.
6. Discussion: trade-offs, limitations, threats to validity, failure cases.
7. Conclusion, ethics and reproducibility statements.

**Submission gate:** No journal submission until data leakage limitations are audited, independent test is locked/assessed, external generalization is honestly characterized (if claiming cross-device work), runs can be reproduced, paper novelty is differentiated, and supervisor has approved the manuscript. Publication acceptance cannot be guaranteed.

Journal selection: make a shortlist only after we know the actual contribution and budget; verify each target journal's official scope, article type, current indexing/quartile, open-access/optional subscription charges, and policies. Do not infer an acceptance probability from quartile.
