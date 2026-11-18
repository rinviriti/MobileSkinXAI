from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=list(csv.DictReader((ROOT/'evaluation/source_data/ALL_SETTING_METRICS.csv').open()));primary=[r for r in rows if r['model']=='M5-LR' and r['analysis_level']=='lesion_primary']
 def get(model,condition,dataset):return next(r for r in rows if r['model']==model and r['condition']==condition and r['dataset']==dataset and r['analysis_level']=='lesion_primary')
 names={'M1':'MobileNetV3-Small','M2':'EfficientNet-Lite0','M3':'MobileNetV4-Conv-Small','M4':'ConvNeXt-Tiny','M5':'MobileSkinXAI-Enhanced','M5-LR':'MobileSkinXAI-Enhanced (LR Control)'}
 table='|Model|E1 ISIC macro-F1|E1 fixed PAD macro-F1|E2 ISIC macro-F1|E2 fixed PAD macro-F1|\n|---|---:|---:|---:|---:|\n'
 for k,n in names.items():table+='|'+n+'|'+'|'.join(f'{float(get(k,c,d)["macro_f1"]):.4f}' for c,d in [('E1','ISIC'),('E1','PAD'),('E2','ISIC'),('E2','PAD')])+'|\n'
 (ROOT/'README.md').write_text('''# MobileSkinXAI

**MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.**

Research prototype for NV, MEL, BCC and SCC classification. It is not a medical diagnosis or a clinically validated screening product. Genuine offline Grad-CAM figures are included; on-device Grad-CAM is NOT implemented.

## Start here

- [Deployment guide](docs/DEPLOYMENT_GUIDE.md): use the existing Flutter UI and the frozen, validation-selected trained model.
- [Results summary](docs/RESULTS_SUMMARY.md): measured six-model source-only/joint-domain comparisons.
- [Progress and verification](docs/PROGRESS_REPORT.md): what passed, what remains unverified.
- [Data/reproducibility](docs/DATA_REPRODUCTION.md) and [licenses/attribution](docs/DATA_AND_ARTIFACT_LICENSES.md).

## Verified deployment candidate

**MobileSkinXAI-Enhanced (LR Control), E2**, selected on PAD development-validation macro-F1 before final test evaluation. It remains a single MobileNetV4-Conv-Small inference graph (2,498,148parameters); teacher/gates/losses are training-only. The ordinary MobileNetV4 reference's higher PAD test point estimate did not replace this selection.

Canonical model: `app/flutter/assets/models/e2_lr_control_v20261118/model_float32.tflite` — **10,073,864bytes**, SHA256 `8b6e19f3d8d0dd9c3b798a2cf99c0525079b9de4ea2f5d1708dd928a718412f3`.

Input:float32NHWC[1,224,224,3], normalized RGB. Output:float32[1,4] **logits**, in NV/MEL/BCC/SCC order. The app applies stable softmax once. Native libjpeg-turbo3.2.0 reproduces EXIF/RGB, Lanczos gray114letterboxing and the original JPEG95/4:4:4 roundtrip before ImageNet normalization. Simply dividing pixels by255 is wrong. Float16 exports failed fixed numerical tolerances and are NOT included for deployment; INT8 was not fitted.

Linux verification:32development-image exact preprocessing,26format/orientation fixtures, actual Flutter interpreter parity on the original four cases,5application tests. Android/iPhone physical execution and camera acquisition remain to be verified. A release XCFramework, when present, is a CI-built artifact rather than proof of physical-device or clinical performance. HEIC/HEIF is unsupported; there is no silent conversion fallback.

## Actual frozen results

E1=ISIC-only, zero-shot transfer to PAD. E2=supervised joint ISIC+PAD learning; **not zero-shot domain generalization**. Both use the same current PAD test partition:261images,205declared lesion units,126patients, including7MEL and28SCC units. All PAD findings are **POST-HOC EXPLORATORY** because earlier PAD outcomes influenced research direction.

'''+table+'''

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
''')
 (ROOT/'docs/RESULTS_SUMMARY.md').write_text('''# Frozen results summary

Primary scoring averages raw softmax probabilities over predeclared exact-capture representatives within each declared lesion, then uses argmax in NV/MEL/BCC/SCC order. ISIC1421declaredlesions/2703images; fixed PAD205declaredlesions/261images. Patient grouping limitations and the ambiguous validation-only SCC identity are preserved; known lesion independence is not biological patient independence when identifiers are unavailable.

'''+table+'''

The selected LR Control's E2 PAD accuracy is81.4634%, macro-F1 is0.6964749025 and balanced accuracy is64.4949%. MEL recall4/7(57.14%); SCC recall7/28(25.00%). Its ISIC accuracy is83.9550%, macro-F1 is0.6638767283 and balanced accuracy is65.7348%. Its PAD macro-F1 change versus its own E1 is+27.3315percentagepoints. E2 has labeled PAD exposure, so this is adaptation performance, not independent zero-shot confirmation.

MobileNetV4-Conv-Small has the highest E2 PAD macro-F1 point estimate0.7100417745, withMEL5/7 andSCC6/28. It was NOT retrospectively substituted for the development-selected candidate. Small melanoma support, poor SCC recognition, one seed in the E1/E2 study, previously examined PAD, unequal parent-training budgets and the failed-teacher amendment limit interpretation. Multi-seed CA-MACL evidence is a separate source-development study, not a controlled E2teacher ablation or proof of external superiority.

[All four-setting tables](../evaluation/metrics_tables/Table_05_Complete_Four_Setting_Classification_Comparison.md), [class-wise precision/recall/F1](../evaluation/metrics_tables/Table_07_PAD_Classwise_Precision_Recall_and_F1_Comparisons.md), [exact MEL/SCC counts](../evaluation/metrics_tables/Table_08_Melanoma_and_SCC_Exact_Detections.md), [model efficiency](../evaluation/metrics_tables/Table_09_Model_Efficiency_Training_and_Selected_Checkpoints.md), [figure index](../evaluation/FIGURE_INDEX.csv).

The full aggregate metrics/CMcells/classwise specificity/AUC/AP are in evaluation/source_data. Counts reconcile independently with48metric rows. Conditional patient-grouped bootstrap intervals are reported from original outputs, but no patient/group records are public; image bootstrap is not a substitute for training-seed replication. Historical full-PAD scores are inTable02and separately labeled comparative figures; only matched fixed-subset E1/E2 differences support the adaptation comparison.

Original model epochs, histories, measured runtimes and the MobileNetV3 controller recovery are retained. No new research training/test evaluation or score tuning was performed for publication. Methodological novelty, clinical safety and fresh independent smartphone superiority remain unproven.
''')
 # Resolve the exact classwise table stem without inventing a link.
 p=ROOT/'docs/RESULTS_SUMMARY.md';t=p.read_text();actual=next((ROOT/'evaluation/metrics_tables').glob('Table_07*.md')).name;t=t.replace('Table_07_PAD_Classwise_Precision_Recall_and_F1_Comparisons.md',actual);p.write_text(t)
 (ROOT/'docs/PROGRESS_REPORT.md').write_text('''# Progress snapshot — 2026-11-18

This snapshot integrates PREVIOUSLY COMPLETED training/evaluation and today's curation/export/codec/app engineering. It does not claim that new models were trained today.

Completed: five original final models, LRcontrol, source-only ablations/CA-MACL evidence and sixE1/E2comparisons preserved; aggregate tables and actual figures collected; validation-selected E2LRcontrol retained; genuine Float32TFLite and identical-tensor PyTorch reference packaged. Raw JPEG mismatch was isolated in both decode and historical encode boundaries; pinned native3.2.0codec and an odd-height orientation fix pass32development/26fixture checks. Actual Linux Flutter runtime and5app tests pass; original interface/style/navigation remains.

The teacher-superiority failure, post-hoc amendment, MobileNetV3controller recovery, additional enhanced optimization budgets, single-seed E1/E2limitations and post-hoc PAD exposure remain explicit. No clinical or novelty claim was added. Licensed bounded images were visually/metadata reviewed, and full datasets/patient records excluded.

Available local checks: Python AST and48-row metric/CM reconciliation; model/reference SHA256 checks; actual CPU PyTorch/LiteRT parity and Flutter/native goldens; Flutter pub get;5Flutter tests; pattern secret scan and detect-secrets review. Flutter analyze has0errors/0warnings but5existing Material-import deprecation infos underFlutter3.50 (default exits1). CI treats infos as nonfatal while retaining the diagnostics; it does not suppress warnings/errors or claim a UI migration.

Local Android APK attempt is blocked by absent SDK/NDK. macOS/Xcode and physical iPhone are unavailable in WSL. CI must independently verify iOS native framework and unsigned app build before a complete release is published. Physical-device/AOT acquisition/latency remains a separate gate even if CI builds pass. HEIC and on-device Grad-CAM are unsupported.

Git transport had no noninteractive credentials; the authenticated GitHub connector is the alternate publication path under review. No token/password is requested or stored. Remote protection/status checks must pass before a normal PR merge; no bypass or force-push is permitted. Consult final PR/release status rather than treating this progress document as proof of publication.
''')
 (ROOT/'docs/DEPLOYMENT_GUIDE.md').write_text('''# Deployment guide

Use the selected trained Float32 file already registered in app/flutter/assets/models/e2_lr_control_v20261118/. Model/input/class/normalization hashes and signatures are in its JSON companions and deployment/verification/MODEL_PROVENANCE.json. The canonical binary is stored once; deployment/tflite contains its manifest, not a duplicate or placeholder.

## Linux engineering verification

Use Flutter3.50.0/Dart3.14.0, CMake>=3.16, Ninja/Clang or a supported C compiler. From root run prepare_goldens.py, then build the native codec with `cmake -S app/flutter/packages/mobileskinxai_codec/src -B .private/codec -G Ninja` and `cmake --build .private/codec --target mobileskinxai_codec --parallel4` (use `--parallel 4`). tflite_flutter0.12.1 additionally needs its LinuxC runtime under the Flutter engine's blobs directory; CI downloads a pinned upstream Linux2.5binary for HOST tests only. It is not a mobile dependency. Native mobile dependencies are resolved by the plugin.

From app/flutter run flutter pub get, flutter analyze (or --no-fatal-infos for declared legacy import infos) and flutter test. To run all5tests including real inference, pass `--dart-define=MSX_GOLDEN_MANIFEST=<repo>/.private/golden_manifest.json` and `--dart-define=MSX_CODEC_LIBRARY=<repo>/.private/codec/libmobileskinxai_codec.so`. No private patient metadata is required for these licensed fixture tests. Without those definitions, the heavy host test visibly skips; do not count it as executed.

## macOS / iPhone

Install the recorded Flutter SDK, current compatible Xcode/CocoaPods/CMake and check flutter doctor. From app/flutter run `bash packages/mobileskinxai_codec/build_ios_codec.sh` BEFORE pod install. This emits anarm64device/simulator XCFramework; existing frameworks are preserved. A complete GitHub release ZIP may include the verified CI-built framework and therefore differs from the normal Source code ZIP.

Run flutter pub get; enter ios/and run pod install, then open Runner.xcworkspace. Runner/Podfile targetiOS15. Configure a development team/bundleID/signing and trusted physical iPhone with DeveloperMode. Camera/photo purpose strings are included. Run `flutter test integration_test/model_device_test.dart -d <iphone-id>` for actual packaged codec/interpreter goldens, then flutter run. Save output/device/runtime/model hashes and actual latency. No iPhone deployment or timing is claimed from WSL or an unsigned CI build.

The existing Analyze action returns four real uncalibrated softmax scores to the existing ResultsScreen. Preview remains; errors return no fabricated result. Heatmap stays absent. HEIC/HEIF is rejected: do not add invisible conversion. Verify the iPhone picker's delivered source bytes and representation, including orientation, rather than assuming Photos/camera output matches originalJPEG. 8-bitPNG/JPEG are the tested adapter paths; CMYK/YCCKJPEG,16-bitPNG and other unverified formats fail explicitly.

Never replace the native codec with a generic resize/JPEG pipeline, omit the historical JPEG roundtrip, divide only by255, apply a second softmax, choose a model on test outcomes or loosen thresholds. Float16failed and INT8was not calibrated. Details and original measured parity are in deployment/verification/PREPROCESSING_PARITY_FINAL.md and TFLITE_RUNTIME_PARITY.md.
''')
 (ROOT/'docs/DATA_REPRODUCTION.md').write_text('''# Data and research reproducibility

Acquire ISIC2019 and PAD-UFES-20 from their official publishers under their licenses; no archive/full dataset or demographic/patient manifest is bundled. Dataset preparation/audit/split scripts are provided and preserve raw data. Do NOT silently regenerate the final split. The source experiment used immutable releasev20261117T060846Z_608314c8:ISIC11632train/2835validation/2703test andPAD781train/261validation/261test.

training/configs/frozen_split_checksums.csv preserves18473frozen image memberships using content/checksum keys only, not filenames, patient/lesion identifiers or medical demographics. Local official labels/group IDs must remain in private local storage. `python training/scripts/restore_private_split.py --local-index <your-local-audit-index.csv>` joins dataset/image keys and verifies original source bytes, then writes a new private membership file without changing assignments. It does NOT grant data approval, regenerate a split, or reproduce original path-dependent JSON seals byte-for-byte.

training/configs contains exact sixE2settings. Phase5A and the original joint/six-model training implementations retain existing paths under final_training and research/thesis_assets; inspect configs and hashes BEFORE executing anything. `python training/scripts/check_prerequisites.py` is a fail-closed inspection only. Full retraining needs the licensed images, private metadata, original/reconstructed reviewed release locks, legitimate ImageNet artifacts and necessary fresh parent checkpoints; these are not magically included or downloaded by a demo command. Do not remove safeguards to make archived commands run.

For a private reconstruction, validate known lesion/patient disjointness locally, source/prepared checksums, inclusion/exclusion/capture rules, class mapping and train-only weights. Preserve validation/test assignments and new run IDs. Recreate stage-specific approval/config/code receipts only under a documented local reproduction protocol; never claim a path-rebased manifest is byte-identical to the original release. Public code has privacy-sensitive group literals replaced by MSX_PRIVATE_VALIDATION_GROUP supplied locally; absent configuration fails closed. The original research workspace is unchanged. This privacy redaction and private locks limit one-command public retraining; they are not hidden.

Original M1–M4 use legitimate ImageNet starts; original enhanced runs andE2enhanced controls have documented parent/teacher budgets. Use engine_amended_v2.py for the corrected final E2controller, not its archived superseded versions. Do not run setup/archival trainers inside existing experiment directories. Original final results are NOT demonstration scores, while older pre-final scripts remain explicitly historical. Reproduction can differ numerically across hardware and one seed is not a stability estimate.

Aggregate CSV/CM data support independent reconciliation of48metric rows and reproduction of presentation tables. Conditional bootstrap intervals require private grouping/prediction records to recompute; public aggregate data alone cannot recreate that uncertainty procedure. The bounded engineering fixtures and preserved PyTorch/TFLite weights independently reproduce deployment parity without requiring full research datasets. No held-out tests should drive preprocessing, calibration or further model selection.
''')
 (ROOT/'docs/DATA_AND_ARTIFACT_LICENSES.md').write_text('''# Data and artifact rights / privacy inventory

No full datasets, archives, patient/demographic records, original patient/lesion IDs, virtualenvs, credentials or WSL system files are published. Frozen split digests are non-identifying image integrity/membership keys; actual medical grouping data remains local. Source code names metadata fields but includes no patient records.

## ISIC development fixtures — CC BY-NC4.0

Exactly four already fixed, anonymous ISIC2019 development-validation photographs are embedded in the test-only golden_cases.dart. They are NOT held-out test cases, a training archive or production assets. All four have noEXIF/JPEGcomment metadata and were visually reviewed as lesion closeups without faces, names or other visible identifying information. Hashes/numeric reference outputs are listed in GOLDEN_REFERENCE.json. The original image bytes are retained for parity; no extra images or private metadata are embedded. Restrict these fixtures/derivatives to the source noncommercial/attribution terms; no blanket commercial permission is asserted.

Official source/license: [ISIC Challenge2019](https://challenge.isic-archive.com/data/#2019), [CC BY-NC4.0](https://creativecommons.org/licenses/by-nc/4.0/). Aggregate attribution: BCN20000/Department of Dermatology, Hospital Clínic de Barcelona; HAM10000/ViDIR Group, Medical University of Vienna; MSK dataset, credited as Anonymous by the publisher. Cite Tschandl/Rosendahl/Kittler(2018), HAM10000, Scientific Data5:180161, doi10.1038/sdata.2018.161; Codella etal.(2017), arXiv1710.05006; Hernández-Pérez etal.(2024), BCN20000, Scientific Data11:641. The official page provides the complete references.

## PAD training illustrations and genuine Grad-CAM — CC BY4.0

One metadata-stripped4-class training illustration grid and six existing development-validation Grad-CAM panels are included, not full clinical records. They were visually reviewed as lesion closeups without visible identifying text/faces. Public titles/filenames omit patientIDs. No identifier CSV, per-patient predictions or medical questionnaire is included. Changes:RGB letterboxing/224preparation and, for Grad-CAM, the actual trained-model map/overlay plus academic labels. These remain research illustrations, not clinical explanations or on-device Flutter Grad-CAM.

Attribution: Pacheco etal., PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones, Data in Brief32(2020)106221, doi10.1016/j.dib.2020.106221. [Official dataset](https://data.mendeley.com/datasets/zr7vgbcyr2/1), [publisher paper and open-data license](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/), [CC BY4.0](https://creativecommons.org/licenses/by/4.0/). Source studies supply consent/ethics statements; this project does not assert a new clinical consent or validation study.

## Code, weights and third parties

Native libjpeg-turbo3.2.0 retains its IJG/BSD/zlib notices; Pillow geometry retains its notice. See the native package LICENSE/THIRD_PARTY_NOTICES.md. Dependency licenses do not grant rights to unrelated project code/data. No new repository-wide permissive license is imposed. The project owner authorized distribution of the trained student artifacts; this is a nonclinical research release, not a grant of upstream data rights for commercial use.

The trained Float32 file and inference-only PyTorch reference preserve identical student weights; optimizer/RNG/training checkpoints are excluded. Failed Float16variants, uncalibratedINT8, extra weights and dataset archives are not shipped. Source clinical image licenses remain distinct from aggregate metric/plot and project-code rights. The local source research archive/Thesis Results is preserved; public copies omit private identifiers and raw metadata.
''')
 (ROOT/'deployment/tflite/README.md').write_text('''# Canonical trained mobile artifact

The genuine validated Float32 model is stored once at ../../app/flutter/assets/models/e2_lr_control_v20261118/model_float32.tflite. Its labels.json, preprocessing.json and model_metadata.json are adjacent. No symlink, placeholder or failedFloat16model is substituted. Use those files with the native codec/inference service described in ../../docs/DEPLOYMENT_GUIDE.md.

SHA256:8b6e19f3d8d0dd9c3b798a2cf99c0525079b9de4ea2f5d1708dd928a718412f3;10073864bytes;NV/MEL/BCC/SCC. See ../verification/MODEL_PROVENANCE.json and verify_model.py for the preserved trained PyTorch reference and independent numerical checks.
''')
 (ROOT/'docs/RELEASE_NOTES.md').write_text('''# MobileSkinXAI research integration

Includes the validation-selected trained E2EnhancedLRControlFloat32student, native JPEG preprocessing, existing Flutter interface integration, original research code/configs, aggregate six-model comparisons, licensed bounded examples/genuine Grad-CAM, and source/verification documentation. No training or held-out model selection was performed for release.

Research limitations remain: post-hoc exploratoryPAD, one E1/E2seed,7PADmelanoma/28SCClesions, poor minority-class recognition, failed original teacher-superiority condition and exploratory amendment. No clinical safety, established novelty or on-device Grad-CAM. Float16failed parity and is excluded. HEICunsupported. Physical iPhone/camera/AOTand device latency remain to be verified even after CI builds.

The complete release ZIP additionally includes the CI-built iOS native codec XCFramework; GitHub's ordinary Source code ZIP has source/model assets but requires building that framework. Neither ZIP includes full datasets, patient records, caches or credentials. Source ZIP retains the old empty unused asset for history; the curated complete archive omits it. Four anonymous licensed development fixtures are test-only; see data/third-party notices.
''')
 print('Public documentation generated from verified aggregate sources')
if __name__=='__main__':main()
