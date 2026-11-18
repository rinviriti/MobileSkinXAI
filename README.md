# MobileSkinXAI

**MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.**

Research prototype for NV, MEL, BCC and SCC classification. It is not a medical diagnosis or a clinically validated screening product. Genuine offline Grad-CAM figures are included; on-device Grad-CAM is NOT implemented.

## Start here

- [Deployment guide](docs/DEPLOYMENT_GUIDE.md): use the existing Flutter UI and the frozen, validation-selected trained model.
- [Results summary](docs/RESULTS_SUMMARY.md): measured six-model source-only/joint-domain comparisons.
- [Progress and verification](docs/PROGRESS_REPORT.md): what passed, what remains unverified.
- [Publication blockers](docs/PUBLICATION_BLOCKERS.md): GitHub write access and unexecuted mobile CI gates.
- [Data/reproducibility](docs/DATA_REPRODUCTION.md) and [licenses/attribution](docs/DATA_AND_ARTIFACT_LICENSES.md).

## Verified deployment candidate

**MobileSkinXAI-Enhanced (LR Control), E2**, selected on PAD development-validation macro-F1 before final test evaluation. It remains a single MobileNetV4-Conv-Small inference graph (2,498,148parameters); teacher/gates/losses are training-only. The ordinary MobileNetV4 reference's higher PAD test point estimate did not replace this selection.

Canonical model: `app/flutter/assets/models/e2_lr_control_v20261118/model_float32.tflite` — **10,073,864bytes**, SHA256 `8b6e19f3d8d0dd9c3b798a2cf99c0525079b9de4ea2f5d1708dd928a718412f3`.

Input:float32NHWC[1,224,224,3], normalized RGB. Output:float32[1,4] **logits**, in NV/MEL/BCC/SCC order. The app applies stable softmax once. Native libjpeg-turbo3.2.0 reproduces EXIF/RGB, Lanczos gray114letterboxing and the original JPEG95/4:4:4 roundtrip before ImageNet normalization. Simply dividing pixels by255 is wrong. Float16 exports failed fixed numerical tolerances and are NOT included for deployment; INT8 was not fitted.

Linux verification:32development-image exact preprocessing,26format/orientation fixtures, actual Flutter interpreter parity on the original four cases,5application tests. Android/iPhone physical execution and camera acquisition remain to be verified. A release XCFramework, when present, is a CI-built artifact rather than proof of physical-device or clinical performance. HEIC/HEIF is unsupported; there is no silent conversion fallback.

## Actual frozen results

E1=ISIC-only, zero-shot transfer to PAD. E2=supervised joint ISIC+PAD learning; **not zero-shot domain generalization**. Both use the same current PAD test partition:261images,205declared lesion units,126patients, including7MEL and28SCC units. All PAD findings are **POST-HOC EXPLORATORY** because earlier PAD outcomes influenced research direction.

|Model|E1 ISIC macro-F1|E1 fixed PAD macro-F1|E2 ISIC macro-F1|E2 fixed PAD macro-F1|
|---|---:|---:|---:|---:|
|MobileNetV3-Small|0.6541|0.3839|0.6380|0.6762|
|EfficientNet-Lite0|0.6583|0.3214|0.6403|0.7014|
|MobileNetV4-Conv-Small|0.6853|0.4032|0.6121|0.7100|
|ConvNeXt-Tiny|0.7168|0.3642|0.6813|0.6936|
|MobileSkinXAI-Enhanced|0.6927|0.4239|0.6791|0.6722|
|MobileSkinXAI-Enhanced (LR Control)|0.7103|0.4232|0.6639|0.6965|


These are lesion-level point estimates, not clinical reliability claims or evidence that one seed establishes significance. Historical full-PAD scores are separately tabulated and are not interchangeable with the fixed-subset scores. The original joint ConvNeXt teacher-superiority condition failed; enhanced runs used a documented post-hoc amendment. No enhanced model established PAD superiority over ordinary MobileNetV4; there is no matched E2 teacher-free ablation proving a teacher effect.

## Repository layout

- `app/flutter/`: original screen/theme/navigation structure plus real inference service, native codec and versioned model assets.
- `training/`: exact architecture mapping, sixE2configs, privacy-safe frozen assignment digests and prerequisite helpers.
- `final_training/`, `scripts/`, selected versioned `research/.../code/`: archived original guarded research implementations; no automatic training on import from README commands.
- `evaluation/`: aggregate tables/CSV, confusion matrices, curves, class-wise specificity/precision/recall/F1 and reports.
- `figures/`: actual architecture schematics, licensed anonymous examples, genuine Grad-CAM and headless app screenshots when generated.
- `deployment/verification/`: exact model provenance, inference-only PyTorch reference, preprocessing specification and parity tools/reports. The canonical TFLite binary is stored once in app assets.
- `docs/`: scientific limitations, deployment, source-data rights and progress.

No full datasets, patient/demographic records, training caches or virtual environments are distributed. Only the bounded, separately attributed anonymous fixture/example images described in the license inventory are included. The empty legacy model asset is preserved for repository history but is never registered or used; the complete release archive excludes it.

## Reproduce engineering checks

Use Python3.11 and [verification requirements](deployment/verification/requirements.txt) for CPU PyTorch/LiteRT parity. No pretrained download is needed for the supplied inference reference. From the repository root:

```sh
python scripts/release_integrity.py
python deployment/verification/prepare_goldens.py
python deployment/verification/verify_model.py
```

For Flutter/iPhone and native codec commands, follow the deployment guide. Research retraining additionally requires licensed datasets and locally reconstructed/frozen metadata; public hash-based membership records do not waive the original data-approval locks. No training is started by these verification commands.
