# Teacher protocol amendment

Amendment ID: **TEACHER_AMENDMENT_20261118T003127Z**
Timestamp UTC: 2026-11-18T00:31:27.388140+00:00
Git HEAD: `55b155650f94e53c8c127f54e006031f88770d99` (experiment scripts may be untracked; exact content hashes are authoritative in AMENDMENT_LOCK.json).

## Original requirement and permanent outcome

The locked six-model protocol required M4 PAD validation declared-lesion macro-F1 to exceed M3's, margin zero. Actual M4: **0.629633181895**; M3: **0.723210247464**. Original qualification: **FAILED**. TEACHER_QUALIFICATION.json remains unchanged with qualified=false. Neither the original protocol nor its failure is rewritten.

## Explicit revised exploratory question

Investigate whether the fixed class-aware reliability-gated teacher-student training recipe can benefit from a teacher that does not outperform the baseline student on aggregate validation macro-F1. Aggregate superiority is sufficient neither to establish useful guidance nor necessary for every individual class/example; this is a rationale for an experiment, not evidence of benefit. The existing M4 checkpoint is permitted only after identity, graph, data provenance, validation selection, normalization and logits-interface verification. It is not superiority-qualified.

This is a **POST-HOC EXPLORATORY amendment**, authorized after seeing validation performance and **before any new final test evaluation**. M5 and M5-LR cannot be presented as satisfying or confirming the original superior-teacher hypothesis. All PAD results are also post-hoc exploratory because of earlier project exposure.

## Unchanged controls

Exactly M5 and M5-LR train once, sequentially, seed42. No M1/M2/M3/M4 retraining, no different teacher, no split changes, folds, sweeps, architecture changes, Flutter modifications or TFLite export. Frozen ISIC/PAD partitions and original capture-representative lesion probability averaging remain. Original M3 joint checkpoint initializes both students identically; the existing M4 teacher is frozen eval/no_grad.

M5: AdamW, WD1e-4, batch16,8ISIC+8PAD,727updates/epoch; headLR1e-4/backboneLR1e-5; no new head stage; maximum40additionalepochs; patience6 strict PAD validation lesion macro-F1. M5-LR changes only headLR1e-5/backboneLR1e-6. Class weights, original weak/strong transforms, losses, confidence/entropy/agreement/correctness gates and stopping semantics remain unchanged. Loss=weightedCE+0.5cross-viewKD(T4)+0.1reliability-gatedJS. Class/reliability weighting and its original full-batch denominator are retained, not renormalized by gate acceptance. Original teacher correctness gates use TRAIN labels only. The amendment waives only the aggregate-superiority prerequisite, never these training gates.

Final E1/E2 evaluation begins only after both enhanced selections and the validation-based lightweight candidate are frozen. Identical ISIC and261-image PADtest manifests,205declaredPADlesions, classorderNV/MEL/BCC/SCC, raw softmax probability means, and patient-grouped paired uncertainty remain. No test score can trigger another training run or a new winner.

## Attribution limitations

Even if M5 exceeds standard M3, the difference cannot establish a teacher-caused benefit: there is no matched joint-domain no-teacher ablation. Initialization, extra budget, weighting and training-view differences confound attribution. The M5-LR pair is a learning-rate sensitivity comparison, not a sixth architecture. No novelty, clinical safety or fresh independent external confirmation follows.

## Verified parents

Teacher: `research/thesis_assets/E1_E2_six_model_comparison/v20261117T231458Z_five_new_joint_runs/03_E2_Training/M4/checkpoints/best.pt`
SHA256: `3f9712101fa659d47497dcab22fa3f5d723e42c8a1e296a2fc30cabf4efd8cab`
Architecture: convnext_tiny.fb_in1k; best25, stopped31.
Student: `research/thesis_assets/E1_E2_joint_domain/v20261117T221855Z_supervised_joint_seed42/03_E2_Training/checkpoints/best.pt`
SHA256: `54abdab247118cd6801402932a1cc37fad9a0d81813013ce4ce0bd8e395b4afc`

## Exact source evidence

- `00_Protocol/LOCKED_PROTOCOL.md` SHA256 `2414d08ae5095dfae892ee1828e54ce91a1f00630e49b6d87498c8f66c0b5d97`
- `00_Protocol/CONFIGURATIONS.json` SHA256 `6efb77e0a5e600f2851d82ed670bc66e7b88bd52226d4a8fa2afc2526d7ea094`
- `00_Protocol/TEACHER_QUALIFICATION.json` SHA256 `32a70ef17c865d800a1477aaadeaa8887045915d79e76b3e7f8e2db21f5f7b14`
- `13_Reports/TRAINING_CONTROL_BLOCKER.md` SHA256 `c6ca4755fa24f8ac93ae60cbf42f1939262886964797bd808fefee67e97b2b63`
- `13_Reports/FINAL_E2_TRAINING_STATUS.md` SHA256 `3d860ee0ca257fbb53dbce18e4dbaa94aaab342097dc08d69576bc87ed0949cd`
- `13_Reports/RECOVERY_FINAL_AUDIT.json` SHA256 `c98ae0117f7c07ec89e2b3aa2319bdfbf762fd49334374aaa39bb98da227778a`
- `03_E2_Training/M4/TRAINING_COMPLETE.json` SHA256 `5494f244ebefb65c663e4567af4b037cf33d02f70af47016c60cf16c0e4144c4`
- `code/original_m5_loss.py` SHA256 `ca8277132c939ac68fdb63709edfe372bfe1d3d48c50a08e960207a0fc877100`
- Repository source `final_training/phase5a/v20261117T061722Z_f5055bd9/configs/M5.json` SHA256 `24fc1d35342f17f6c5acdf2466ef12c43a58a095b272084247b731988dd3d721`
