# Local release-candidate validation — 2026-11-18

Status: **LOCAL ENGINEERING PASS; PUBLICATION / MOBILE BUILD GATES BLOCKED**.

|Check|Result / evidence|
|---|---|
|Canonical trained model|Float32, 10,073,864 bytes, SHA256 `8b6e19f3d8d0dd9c3b798a2cf99c0525079b9de4ea2f5d1708dd928a718412f3`; four logits in NV/MEL/BCC/SCC order|
|Selected checkpoint|Original E2 Enhanced LR Control, epoch8, SHA256 `5bed9eb03a3f3ae3d17ea4fd5a81bd5b9795696b690ecff81096ee57c1eba060`; no new selection or optimization|
|PyTorch versus TFLite|PASS on four fixed development fixtures; max logit difference 0.000036657; max probability difference 0.000010014; top1 agreement4/4. Original thresholds retained|
|Real Flutter inference|PASS; all six tests, including original five app/inference checks plus an additional real home rendering, completed. See `deployment/verification/local_evidence/release_flutter_test.log`|
|Flutter analysis|PASS with `--no-fatal-infos`: no errors/warnings, five existing Material-import deprecation infos. No warnings/errors suppressed; original default analysis reports these infos as nonzero|
|Early stopping regression|11/11 PASS, including exact recorded MobileNetV3 history replay, state recovery and a tiny CPU optimizer loop. No research model training|
|Metric consistency|48 aggregate image/lesion result rows reconciled against confusion matrices, including 205 PAD lesion units with supports41/7/129/28|
|Syntax|All150 public Python files parsed; no GPU training command executed|
|Privacy / rights|Four non-identifying ISIC development fixtures and bounded PAD illustrations reviewed and attributed. Raw data, patient metadata, source split identity CSVs and caches excluded. Old notebook outputs sanitized only in the publication copy|
|Secret scan|detect-secrets1.5.0 entropy findings manually classified as public hashes, Xcode IDs and research path strings; no credential/private-key detector findings. Supplemental high-confidence patterns pass. See SECRET_SCAN_REVIEW.json|
|Asset/document checks|Canonical model/reference hashes pass; public README links resolve; PNGs decode and SVG XML parses; manually inspected sample, Grad-CAM and UI previews. File sizes below50MiB|
|Original artifacts|Local research workspace remains on codex/model-development; original checkpoint, dataset and experiment files untouched. All publication edits isolated in separate worktree|
|Android APK|NOT VERIFIED: local SDK/NDK unavailable|
|iOS / physical device|NOT VERIFIED: macOS/Xcode unavailable. Native XCFramework and unsigned app CI jobs supplied but unexecuted; no iPhone/latency claims|
|GitHub upload/PR/merge/release|BLOCKED: authenticated connector write endpoints return403; Git CLI lacks credentials. No remote mutation succeeded|

The complete mobile release must not be declared ready until the remaining CI/build gates pass. The local candidate archive is useful for review and Linux inference; its source includes the exact mobile build instructions but no fabricated iOS framework or failed Float16 export. See PUBLICATION_BLOCKERS.md and DEPLOYMENT_GUIDE.md.

PAD results remain post-hoc exploratory, with one E1/E2 seed and only seven melanoma test lesions. The teacher qualification failed and the enhanced experiments used a recorded exploratory amendment. No clinical reliability, causal teacher effect, independent external confirmation or algorithmic novelty is claimed.
