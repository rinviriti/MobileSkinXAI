from common import *
import shutil,importlib.metadata as md,xml.etree.ElementTree as ET
from PIL import Image
from model import Student

TITLE='MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.'
def table(rows,cols):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(r.get(c,'')) for c in cols)+' |' for r in rows)+'\n'
def main(out):
 out=Path(out);cpu=load(out/'reports/CPU_IMPLEMENTATION_CHECKS.json');gpu=load(out/'reports/GPU_SMOKE.json');steady=load(out/'reports/STEADY_PIPELINE_BENCHMARK.json');conversion=load(out/'reports/TFLITE_CONVERSION_SMOKE.json');complexity=load(out/'reports/model_complexity.json');contract=load(out/'reports/TRAINING_CONTRACT_CHECKS.json');assert all(x['status']=='PASS' for x in [cpu,gpu,steady,contract]+conversion);assert gpu['training_updates']==steady['optimizer_steps']==0;assert not load(out/'TRAINING_AUTHORIZATION.json')['authorized'];assert not any((out/'runs').iterdir())
 rows=read(out/'source_data/approved_train_validation.csv');stats=load(out/'source_data/train_class_statistics.json');cost=read(out/'reports/projected_training_cost.csv')
 configs=[load(out/f'configs/C{i}.json') for i in range(5)];assert len(configs)==5;assert all(c['preprocessing']['treatment']=='RGB' for c in configs)
 for i,c in enumerate(configs):assert [c['acquisition'],c['fusion'],c['consistency'],c['contrastive']]==[i>=1,i>=2,i>=3,i>=4]
 for i in range(1,5):
  prev,cur=configs[i-1],configs[i];allowed={1:{'acquisition'},2:{'fusion'},3:{'consistency','loss'},4:{'contrastive','loss'}}[i]
  differences={k for k in cur if cur[k]!=prev[k]}-{'id'};assert differences<=allowed,(i,differences)
 # A few engineering-only artifacts; do not generate any actual performance/result plots.
 figrows=[]
 for name,title,caption,sources in [
  ('source_class_distribution','ISIC approved source development distribution','Approved training and validation counts only; no external/test data or model results.',['source_data/approved_train_validation.csv']),
  ('source_preprocessing_audit','Source-only preprocessing and acquisition visual audit','16 fixed training images, four per class. ColumnsRGB/CC/luminanceCLAHE/weak/acquisition. No model inference or clinical-validation claim.',['source_data/QA_sample_selection.csv','source_data/preprocessing_QA_metrics.csv','source_data/augmentation_audit_parameters.json']),
  ('actual_multiscale_architecture','CA-MACL MobileNetV4 multi-scale residual inference head','Schematic abstraction derived from actual model code and measured feature stages, not a methodology diagram.',['source_data/architecture_stages.csv','reports/model_complexity.json'])]:
  for ext in ['png','svg','pdf']:
   p=out/f'figures/{name}.{ext}';assert p.exists()
   if ext=='png':
    with Image.open(p) as im:assert min(im.info.get('dpi',[0,0]))>=299;im.verify()
   elif ext=='svg':ET.parse(p)
   else:assert p.read_bytes().startswith(b'%PDF-')
  figrows.append({'figure':name,'title':title,'caption':caption,'split':'ISIC training/development metadata only','source_data':';'.join(sources),'generation_script':'code/check_cpu.py',**{ext:f'figures/{name}.{ext}' for ext in ['png','svg','pdf']}})
 write(out/'FIGURE_INDEX.csv',figrows)
 shutil.copyfile(PRETRAIN/'provenance.json',out/'source_data/official_ImageNet_provenance.json')
 (out/'source_data/fusion_model_definition.txt').write_text(str(Student(True,False)))
 dump(out/'reports/environment_versions.json',{d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name']})
 # Cost derivation archived here so projections are independently reproducible.
 base_compute=gpu['runs'][-1]['median_compute_seconds_per_batch'];derived=[]
 for r in gpu['runs']:
  b=max(r['median_compute_seconds_per_batch'],steady['mean_steady_seconds_per_batch']+r['median_compute_seconds_per_batch']-base_compute);epoch=727*b+2835/steady['steady_images_per_second']+2
  derived.append({'configuration':r['configuration'],'measured_GPU_batch_seconds':r['median_compute_seconds_per_batch'],'measured_peak_allocated_GiB_no_optimizer':r['peak_allocated_GiB'],'projected_epoch_seconds_low':epoch*1.15,'projected_epoch_seconds_high':epoch*1.75,'projected_max40_minutes_low':epoch*1.15*40/60,'projected_max40_minutes_high':epoch*1.75*40/60,'status':'projection from bounded smoke, NOT observed training duration'})
 write(out/'reports/projected_training_cost.csv',derived)
 full_names=['Standard MobileNetV4-Conv-Small','Acquisition-robust MobileNetV4-Conv-Small','Multi-scale acquisition-robust MobileNetV4-Conv-Small','Multi-scale MobileNetV4 with acquisition consistency','CA-MACL MobileNetV4 with class-aware contrastive learning']
 tab=[{'ID':c['id'],'Name':full_names[i],'Acquisition':c['acquisition'],'Fusion':c['fusion'],'JS coefficient':c['loss']['lambda_js'],'SupCon coefficient':c['loss']['lambda_supcon']} for i,c in enumerate(configs)]
 (out/'FIVE_EXPERIMENTS.md').write_text('# Exactly five new configurations\n\n'+TITLE+'\n\n'+table(tab,['ID','Name','Acquisition','Fusion','JS coefficient','SupCon coefficient'])+'''
The exact files are configs/C0.json…C4.json. All use224RGB, class orderNV/MEL/BCC/SCC, seed42, fresh verified ImageNet weights, original approved train/validation, identical moderated CE, two forward passes per image, batch16, matched data ordering/stateless geometry, AdamW decay1e-4, frozen BN running statistics and common schedule. C0's second view equals its weak view; with dropout0 and frozen BN this is the standard architecture/CE reference at matched two-pass compute. No new architecture comparator, LR variant, optional-preprocessing run or seed repetition is silently added.

Adjacent contrasts target exactly one component: C0→C1 acquisition policy; C1→C2 pooled residual fusion (including its projections/gates); C2→C3 student prediction JS; C3→C4 training-only projection/SupCon objective. These estimate effects conditional on earlier components, not a factorial interaction analysis. C4–C0 changes multiple components. C3–C4 does not isolate class-balanced reduction from ordinary SupCon; no such additional configuration is authorized. Realized stopping epochs may differ despite common limits, and that is a policy-level comparison, not fixed-update causal proof. Historical model scores are excluded from decisions and are not new controls here. Single seed limits journal conclusions; any future replicated seeds require explicit additional authorization, not automatic launch.
''')
 (out/'ARCHITECTURE_AND_LOSSES.md').write_text(r'''# Exact CA-MACL architecture and objective

'''+TITLE+r'''

## Single inference backbone
`mobilenetv4_conv_small.e2400_r224_in1k`, timm1.0.31. Common fresh ImageNet state is strict-loaded into1000class model, then reset to a seeded4class classifier. Publisher SHA and immutable revision are recorded. No prior skin-lesion checkpoint is loaded.

At224RGB: blocks.1 gives F8∈R^(64×28×28); blocks.2 gives F16∈R^(96×14×14); blocks.4 gives F32∈R^(960×7×7). The original pooled deep head gives d∈R^1280. For k∈{8,16}:
\[
u_k=W_{k2}\operatorname{ReLU6}(W_{k1}\operatorname{GAP}(F_k)+b_{k1})+b_{k2},\quad W_{k1}:C_k\to32,\ W_{k2}:32\to1280.
\]
\[
\alpha=\operatorname{softmax}(\theta),\quad h=d+\sigma(\rho)(\alpha_8u_8+\alpha_{16}u_{16}),\quad s=W_{cls}h+b_{cls}.
\]
Initial θ=(0,0), sigmoid(ρ)=0.1. Existing classifier/deep head are preserved; extra projections are initialized in an independent RNG context. C0/C1 set h=d; C2–C4 share the SAME fusion inference graph. No second backbone, teacher, lesion detector, ROI crop or inference contrastive projector. Global pooled multi-resolution features are not an explicit localized lesion segmentation.

Measured inference parameters: C0/C1=2,498,148; C2–C4=2,587,815 (+3.59%). Counted Conv/Linear MACs:184,730,688→184,817,728 (+0.047%). Pooling/nonlinear/normalization/add/memory traffic excluded; these are not full FLOPs or phone latency. The C4 training-only projector has172,224parameters and is excluded from export.

## Common moderated classification objective
Let N_c be retained train image counts and v_c=clip(sqrt(N/(4N_c)),0.5,3). Train-only weights and counts are in source_data/train_class_statistics.json. No sampler oversampling or target prior estimation.
\[
L_{CE}=\frac12\sum_{v\in\{w,s\}}\frac{\sum_i v_{y_i}[-\log p^v_{i,y_i}]}{\sum_i v_{y_i}}.
\]
The denominator is PyTorch's per-batch weighted CE reduction, identical in all five configurations.

## Cross-view prediction consistency
p_w=softmax(s_w), p_s=softmax(s_s), m=(p_w+p_s)/2:
\[
L_{JS}=B^{-1}\sum_i\tfrac12[KL(p^w_i\Vert m_i)+KL(p^s_i\Vert m_i)].
\]
Both branches receive gradients; no teacher, pseudo-label, confidence mask, correctness gate or detached target. Computation uses FP32/log safeguards1e-8. This is supervised source-view consistency, not cross-modality clinical supervision.

## Class-aware supervised contrastive term
C4 only: z=normalize(Linear64(ReLU6(Linear128(h)))), yielding2Bnormalized64-D vectors. P(a) contains other same-class views, excluding a itself. With τ=0.1,
\[
\ell_a=-|P(a)|^{-1}\sum_{p\in P(a)}\log\frac{\exp(z_a^Tz_p/\tau)}{\sum_{b\ne a}\exp(z_a^Tz_b/\tau)},\qquad
L_{CA}=|C_B|^{-1}\sum_{c\in C_B}|A_c|^{-1}\sum_{a\in A_c}\ell_a.
\]
Mean loss is balanced across classes present, not a class-completed negative denominator. It is NOT an implementation of published BCL. FP32 L2 normalization/logsumexp and diagonal masking are used. The paired view guarantees at least1positive even for a singleton SCC image; that case provides same-instance invariance, not cross-lesion class evidence. If the batch contains only1class, contrastive loss is differentiable zero. No synthetic SCC samples, stale queues, label hallucination or forced class oversampling. Per-batch class image/distinct-lesion counts, SCC single-lesion flags, skipped cases and positive minimums are logged. Class-balanced batch means can remain noisy for rare classes; small coefficient, warm-up and global clipping are safeguards, not proof of improved recall.

## Total loss
With e=fine-tuning epoch and r(e)=min(max(e,0)/5,1),
\[
L=L_{CE}+r(e)[\lambda_{JS}L_{JS}+\lambda_{CA}L_{CA}].
\]
C0/C1/C2 coefficients(0,0); C3(.05,0); C4(.05,.02). Auxiliaries are disabled during frozen-head epochs. Gradients pass through both views, fusion and active C4 projector. All losses/projector are training-only; inference returns four logits. These formulas and conservative coefficients are fixed before model optimization; no efficacy or novelty claim is implied.
''')
 (out/'PREPROCESSING_AND_AUGMENTATION.md').write_text('''# Preprocessing and source-only acquisition policy

Deterministic reference: raw approved image → EXIF transpose → RGB conversion → aspect-preserving Lanczos resize, target dimensions floor(original*scale+0.5), centered224×224gray114padding, left/top floor(padding/2). Pillow version is locked. RGB tensor=x/255 then(mean,std)=(.485,.456,.406)/(.229,.224,.225), NCHW FP32 before BF16 network autocast. All models are MobileNetV4, so model-specific normalization is identical. Embedded ICC profiles are not automatically color-calibrated; RGB is assumed sRGB-like. Existing release images/CSVs are not modified. Training prepares224on demand from approved raw sources;16QA RGB outputs and golden tensors are saved, not a newly rewritten full dataset.

Exclusive optional deterministic treatments, AFTER letterbox and BEFORE normalization:
- CC: on known content pixels with meanRGB intensity between.05and.95, compute per-channel means, gray-world gain=clip(globalmean/channelmean,.95,1.05), effective gain=1+.25*(gain−1). Round half-up to uint8. Padding is unchanged. If<=100valid pixels, return RGB unchanged. This is mild color constancy, not calibrated white balance.
- CLAHE: isolate valid rectangular content; OpenCV RGB→YCrCb; CLAHE on Y only, clipLimit1.5/tileGrid8×8; blendedY=.8Y+.2CLAHE(Y), half-up uint8; keep Cr/Cb unchanged in this color space; convert backRGB. Conversion/clipping can still change RGB chroma slightly. No treatment concatenation is accepted.
All five primary configs use RGB. Optional alternatives passed sample engineering QA but are NOT selected/combined automatically and no extra training configurations were launched. Any future treatment experiment requires a new locked protocol.

Shared geometry, all configurations: independent H/Vflip p=.5, rotation U[−10°,10°] with expand=True/bilinear, then whole-image letterbox. No lesion crop or background replacement. Weak image is that geometric view. C0 duplicates it. C1–C4 strong image applies the following source-only policy in order:

| Operation | Probability | Fixed range |
|---|---:|---|
|Brightness and contrast|.8|independent U[.9,1.1] each|
|RGB white-balance gains|.5|raw per-channel U[.97,1.03], divided by geometric mean; effective extrema about±4.1%|
|JPEG|.4|integer quality75–95, subsampling0|
|Gaussian blur|.2|Pillow radius U[.1,.6]|
|Scale/framing|.3|full-image scale U[.9,1], Lanczos; random placement inside224gray canvas, no crop|

No strong hue shift, grayscale, CutMix, lesion removal, artistic texture, other-patient background or PAD-derived distribution. Photometric/codec transforms operate on the prepared canvas; margins can vary before final framing. No anatomical background is replaced. Whole-image geometry preserves source field of view, but downsampling/blur cannot guarantee preservation of every diagnostic feature. Every sampled parameter is logged per image/epoch. Separate SHA256-derived RNG streams(seed,epoch,imageID,geometry/acquisition) give identical weak geometry across configurations/workers, unaffected by enabling losses or fusion.

QA:16fixed training images,4/class, exact selection CSV. Quantitative optional-treatment checks: content meanΔE76<=3, p95<=10, full-canvasSSIM>=.90, content edge-energy ratio[.75,1.25], clipping increase<=.01. All16/16CC and16/16CLAHE passed. These are engineering tolerances, not diagnostic equivalence tests. A downsampled preview was visually inspected: no obvious field-of-view cropping or extreme color alteration; borders/texture appear retained at displayed resolution. High-resolution panels remain for expert review. No clinical reviewer or lesion masks are available, so no lesion-specific mask-quality claim is made. Known content masks only track padding; no unverified ROI drives training.

See CONVERSION_AND_FLUTTER.md for exact portability limitations. Padding, channel order, orientation rules, gain formulas and normalization are reproducible in principle; byte-identical decoder/Lanczos/CLAHE results in Flutter require a matching implementation and golden tests. No Flutter code was changed and bitwise preprocessing parity is not claimed.
''')
 (out/'NOVELTY_OVERLAP_MATRIX.md').write_text('''# Novelty and overlap audit

CA-MACL is a working name, not established originality. Targeted review of published primary sources/abstracts and accessible method descriptions; no claim of exhaustive novelty clearance or reproduced external benchmark.

| Closest source | Overlap | Difference / remaining question |
|---|---|---|
|[Hu etal., Multi-scale feature fusion and class weight loss for skin lesion classification, ComputBiolMed2024](https://doi.org/10.1016/j.compbiomed.2024.108594)|Shallow/deep skin-lesion fusion plus imbalance handling|Current pooled residual head preserves a MobileNetV4deep branch with small measured overhead. This is an implementation distinction, not proof that fusion/weighting is new.|
|[Khosla etal., SupCon, NeurIPS2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)|Label-supervised view representation loss|Current mean is balanced over present-class anchors and has singleton/SCC diagnostics; broad supervised contrastive principle is prior art.|
|[Alzahrani, SkinLiTE, Current Medical Imaging2024](https://doi.org/10.2174/0115734056313837240612065845)|Lightweight supervised contrastive skin-lesion classification|SkinLiTE uses a two-phase encoder/projector then frozen-encoder classification recipe; current joint CE/JS/SupCon with residual multiscale head differs, but “first lightweight contrastive skin classifier” is unsupported.|
|[Zhu etal., Balanced Contrastive Learning,CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.html)|Class imbalance within contrastive training|Published BCL balances negatives and complements classes. Current implementation balances anchor-class means and does neither; no mathematical equivalence or priority claim.|
|[AugMix,ICLR2020](https://arxiv.org/abs/1912.02781);[MixStyle,ICLR2021](https://arxiv.org/abs/2104.02008)|Corruption/style augmentation and consistency for generalization|Current fixed image-level acquisition proxy does not mix styles, use target statistics or implement AugMix. No new DG principle established.|
|[Wang etal., Consistent representation via contrastive learning for skin lesion diagnosis,2023](https://doi.org/10.1016/j.cmpb.2023.107826)|Lesion representation consistency across imaging modalities|Uses multimodal/adversarial representation learning; current source-only transformed views are not real paired clinical/dermoscopy data.|
|[Jia etal., Frequency swapping and soft-mask disentanglement DG,ESWA2026](https://doi.org/10.1016/j.eswa.2025.131025)|Lesion-discriminative features, style disturbance and consistency|Current model uses neither Fourier swapping nor learned soft masks; preserved field of view is not learned lesion segmentation.|
|[CRLD,ACM Multimedia2024](https://arxiv.org/abs/2412.16493)|Cross-view learning and confidence-aware KD|CA-MACL uses student–student JS and supervised contrastive learning, no teacher or reliability-gated KD. This removes the prior mechanism, not prior consistency ideas.|

Possible defensible contribution: a precisely controlled, low-overhead, single-backbone integration and source-only evaluation of acquisition, fusion and class-aware consistency with measured mobile-operator feasibility. Whether it improves clinical transfer is unknown. The exact combination/head may be project-specific, but no evidence here establishes publication-level algorithmic novelty. The five-step ladder estimates conditional increments, not independent main effects/interactions, and does not isolate class-aware reduction from ordinary SupCon. New independent clinical evidence and replicated training are missing. Avoid claims of lesion-mask preservation, clinical safety or target-invariant representations before testing.
''')
 (out/'TRAINING_PROTOCOL.md').write_text('''# Frozen development protocol — no full training authorized

'''+TITLE+'''

Use only the approved immutable release's original11632ISIC train and2835development-validation images; no new split, original test, PAD, HIBA, extra ISIC collection or old skin-lesion weights. Known lesion/identity/risk groups do not cross these retained partitions. Source-validation has been used in earlier work and is development evidence, not fresh independent confirmation. Missing globally linkable ISIC patients remain a limitation. Raw and prepared input hashes were checked on all14467development images; release metadata checksums were verified. No held-out images, previous test reports or prediction vectors were consulted for design.

ExactlyC0–C4, seed42. Batch16, workers4. Fresh local official ImageNet SHA7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4. First5epochs freeze pretrained backbone (including BN affine), train classifier/fusion heads with AdamW LR1e-3, decay1e-4; no auxiliary loss. Continue up to35fine epochs, all student parameters trainable; classifier/fusion/projector LR1e-4, backbone1e-5, same decay. New optimizer state at stage transition; no scheduler or stage-end weight rollback. BN running statistics frozen throughout, affine parameters trainable in fine stage. BF16 autocast plus FP32 loss reductions; no GradScaler needed for BF16. Global gradient clip1.0all configurations. Deterministic cuDNN, fixed per-epoch shuffle generator seed42+epoch, stateless per-image transforms. Checkpoint the highest clean validation image macro-F1 across all epochs; no threshold tuning, stress-based selection or calibration fitting. Frozen stage is5epochs; patience7applies in fine stage against the global best, resetting stale counter at transition. Exact max40epochs and727train batches per epoch. No memory fallback is automatic; a required batch change must be documented and matched across all five, with contrastive-positive semantics reconsidered.

Trainer logs per-epoch train/validation losses, accuracy, macro-F1, balanced accuracy, class precision/recall/F1, confusion/error counts, gradient norms, class image/lesion supports, LRs, peak allocated VRAM and durations; per-batch contrastive rarity diagnostics; per-image augmentation parameters; every validation probability file and exclusive-create epoch checkpoint with optimizer/provenance. Best/last are selectors in NEW run directories. Refuse existing run directories. No published score from older experiments is replaced or used as a stopping target.

Source-only synthetic stress: fixed eight conditions applied independently to deterministic validation inputs: brightness.85; brightness1.15; warm gains(1.04,1,.96); cool(.96,1,1.04); JPEGquality60subsampling0; Gaussian radius.75; whole-image scale.85centered; horizontal illumination gradient amplitude.12. Never synthesize a lesion or replace an anatomical background. This grid is fixed independently of target data. Evaluate only after the clean-selected checkpoint hash is locked; no further training. Primary development selection remains clean macro-F1; report equal-weight mean of eight stress macro-F1 values and each class/condition, never only favorable cases.

Prespecified descriptive development criteria C4vsC0: mean stress macro-F1 strictly higher; clean macro-F1 no more than.01lower; clean MEL/SCC recall no more than.03lower each. These point-estimate margins are exploratory engineering criteria, not statistical noninferiority, clinical thresholds or evidence of smartphone transfer. If criteria fail, report failure; no automatic extra experiments, augmentation changes or test access. summarize_development.py is guarded until all five actual runs complete; no result table/figure is generated now.

Future evidence from ACTUAL data only: learning curves, clean/stress confusion matrices, ROC/PR, class-wise recall/specificity, NLL/Brier/ECE10 and reliability, paired differences and lesion-cluster bootstrap2000seed42. Validation selection bias and single-seed uncertainty must be disclosed; grouped resampling cannot substitute for multi-seed replication. No clinical cohort is included; source perturbations are proxies. A future independently sourced smartphone study needs separate approval, no HIBA/ISIC overlap, no test-based tuning and a frozen analysis plan. Previously exposed PAD can never be presented as fresh independent confirmation. Actual trained-weight TFLite parity/phone timing/XAI validation remain future tasks.

train.py defaults to configuration inspection. --execute additionally requires TRAINING_AUTHORIZATION.json authorized=true, a matching implementation seal, passing audit and CUDA BF16. This task leaves authorization FALSE and runs/ empty. No full training command was executed.
''')
 (out/'CONVERSION_AND_FLUTTER.md').write_text('''# Conversion and Flutter contract

Both random-weight architecture probes PASS via existing Python3.11LiteRT-Torch environment → TFLite → LiteRT CPU interpreter. No trained skin weights and no optimizer updates. Input[1,3,224,224]FP32normalizedRGB; output[1,4]logits, fixedNV/MEL/BCC/SCC. Host-side softmax if probabilities needed. One image, one backbone. C0/C1 graph and C2/C3/C4fusion graph tested on3random tensors, atol1e-4/rtol1e-3; max absolute errors2.10e-5/2.67e-5. No Flex or custom ops; see exact reports/operator lists. DELEGATE denotes interpreter delegation, not a custom model operator. Conversion is an untrained smoke test, not final model efficacy/export, quantization proof or Android delegate support. Repeat after training and any quantization; benchmark real devices before deployment claims.

| Deterministic step | Flutter reproduction requirement |
|---|---|
|EXIF orientation|Apply all8orientation cases before resize; synthetic corner fixtures pass exact Python pixel tests. Flutter decoder orientation behavior must be checked to prevent double rotation.|
|RGB decoding|Match RGB channel order, alpha behavior and sRGB-like assumptions. Different JPEG decoders can differ by1LSB; bit-exact decoder parity not tested.|
|Aspect-preserving resize|Match floor(dimension*scale+.5), PIL Lanczos filter/edge/rounding semantics. Dart image-library default interpolation is NOT asserted equivalent; use a conformance-tested implementation/native equivalent.|
|Letterbox|Canvas224×224, RGB(114,114,114), left/top floor(totalpadding/2). Integer padding/copy rules are exactly reproducible once resized pixels match.|
|Normalization/layout|FP32(x/255−mean)/std, NCHW. Golden PNG, .npy and rawFP32tensors supplied. Compare before inference, not only final labels.|
|OptionalCC|Port exact valid-pixel thresholds, float mean/gain clamp/blend/rounding; compare golden pixels. Not enabled in main experiments.|
|OptionalCLAHE|Requires matching OpenCV YCrCb conversions/CLAHEversion and parameters, likely native FFI. A visually similar Dart CLAHE is not identical. Disabled in main configs.|

Only padding/channel/layout/arithmetic rules are specified exactly; no claim of existing end-to-end bit-identical Flutter preprocessing. Golden tensor tolerance should be preregistered(at most1LSB inRGB and corresponding normalized error if decoder differs), with logit parity on trained models, before shipping. No Flutter file was modified, no app XAI was implemented, no phone latency/energy measured. Augmentations, contrastive projector and loss modules are never needed at inference. Native float fusion weights learned during training remain in the student graph.
''')
 costtable=[{'ID':r['configuration'],'GPU batch ms':f"{1000*r['measured_GPU_batch_seconds']:.1f}",'Smoke allocated GiB':f"{r['measured_peak_allocated_GiB_no_optimizer']:.3f}",'Projected epoch sec':f"{r['projected_epoch_seconds_low']:.0f}–{r['projected_epoch_seconds_high']:.0f}",'Projected max40min':f"{r['projected_max40_minutes_low']:.0f}–{r['projected_max40_minutes_high']:.0f}"} for r in derived]
 (out/'RESOURCE_ESTIMATE.md').write_text('# Measured smoke costs and projected training cost\n\n'+table(costtable,['ID','GPU batch ms','Smoke allocated GiB','Projected epoch sec','Projected max40min'])+f'''
Measured RTX5070, CUDA BF16, two views/batch16,2warm-up+6timed backward batches per config, no optimizer steps. C4raw-source steady-state benchmark:256random source images;4warm-up+12timed batches; mean{steady['mean_steady_seconds_per_batch']:.4f}s, median{steady['median_steady_seconds_per_batch']:.4f}s, p95{steady['p95_steady_seconds_per_batch']:.4f}s per batch;{steady['steady_images_per_second']:.1f}images/s. Includes decoding/augmentation/two forwards/loss/backward/clipping, excludes AdamWstate updates and validation/checkpoint IO. A separate128imageforward-only test included worker startup and is not extrapolated as steady throughput.

Projection formula: adjust C4mean steady batch time by each config's compute-median difference; floor at that config's compute time; multiply727batches, add validation2835/steady_images_per_second and2secworker overhead, then multiply1.15–1.75 for unmeasured optimizer/IO/cache/long-run uncertainty. Conservatively use fine-stage costs for frozen stage too. This is a bounded-sample projection, NOT actual training durations. Maximum40epoch all-five total≈{sum(r['projected_max40_minutes_low'] for r in derived)/60:.1f}–{sum(r['projected_max40_minutes_high'] for r in derived)/60:.1f}hours, plus roughly6–15minutes final synthetic stress/summary; early stopping can shorten this but cannot be predicted from new results that do not yet exist.

Observed allocated peak0.306–0.330GiBexcludes AdamWmoments, full CUDA driver/display/reserved memory and long-run behavior. New optimizer moments add about20–22MBplus training projector buffers. Plan1–2GiBPyTorch headroom and retain severalGiBdriver/display margin on the12GBcard; this is a reservation estimate, not a measured full-training peak. Batch16was feasible in every smoke test. Retain approximately8–10GiBdisk for up to200epoch checkpoints, augmentation/prediction logs and reports. No old checkpoints are overwritten, and no full alternative preprocessed dataset was materialized. CPU timing CSV measures raw pipeline treatments separately. Real phone latency cannot be inferred from MACs or these GPU numbers.
''')
 (out/'REPRODUCIBILITY_REPORT.md').write_text('''# Reproducibility and preservation

All code/configs in this version; source_data/approved_train_validation.csv retains approved IDs/source hashes and no held-out rows. Train-only class weights are explicit. Preflight reverified immutable release metadata and28,934raw+prepared development image hashes. Raw images and old prepared outputs were read, never modified. No original skin-lesion checkpoint, prior independent predictions/results, PAD image/label/outcome or HIBA collection was used. Existing ImageNet publisher artifact is loaded freshly in memory, with strict state dict, revision/hash verification and deterministic new classifier; it is not a lesion-trained M3 checkpoint.

Model initialization uses seed42; projection/projector initializations use isolated RNG contexts+101/+202 so common backbone/classifier are identical. CUDA deterministic settings and stateless hash-derived per-image/epoch/stream RNG make geometry and sample order comparable. Workers4, per-epoch shuffle seed42+epoch, fixed normalization and frozen BN statistics. Source files/manifest/config hashes are sealed; training cannot execute until separately authorized. No automatic resume or additional full configuration.

Commands from repo, using saved code paths (common.py resolves repo ancestor): prepare.py creates a NEW version; check_cpu.py builds source QA/model profile; check_contracts.py verifies stage groups/EXIF/input guards; check_gpu.py runs8discarded backward batches/config; check_gpu.py --pipeline-only runs bounded raw pipeline timing; convert_probe.py uses .venv-export-py311 CPU; finalize.py checks and archives. train.py without --execute inspects only. No full training, package install, dataset download, phone modification or old-artifact write was performed. CPU and GPU check results and both runtime environments are saved. A checker import omission was fixed before the final CPU smoke; final archived code passes syntax/contracts.

Only3source/architecture figure families exist,9canonicalPNG300DPI/SVG/PDFexports. No trained-model curves, ROC/PR, confusion matrices, efficacy examples or methodology diagrams were fabricated. The16training QA images carry original ISIC dataset licensing/attribution obligations; refer to repositoryDATASETS.md and source release provenance before public reuse. Colored corner PNGs are explicitly synthetic EXIF unit fixtures, not invented lesions.

Limits: engineering QA is not clinical adjudication, no segmentation masks/ROI are used, complete Flutter preprocessing parity and trained-weight conversion remain unverified, one training seed does not support journal reproducibility claims, and no independent clinical smartphone cohort is included. Source-only synthetic stress is not external confirmation. Original approved validation is reused development data with missing global patient identifiers.
''')
 # Archive scripts and define lock excluding mutable authorization and result/report files.
 for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,out/'code'/p.name)
 lockpaths=list((out/'code').glob('*.py'))+list((out/'configs').glob('*.json'))+[out/'source_data/approved_train_validation.csv',out/'source_data/train_class_statistics.json']
 dump(out/'IMPLEMENTATION_LOCK.json',{'created_UTC':datetime.now(timezone.utc).isoformat(),'sha256':{str(p.relative_to(out)):sha(p) for p in lockpaths},'training_authorized':False,'scope':'exactlyC0–C4; implementation and data/config integrity only'})
 # Explicitly verify the implemented training guard rejects execution before any optimizer/GPU setup.
 from train import guard
 try:guard(out,configs[0])
 except AssertionError as e:assert 'FULL TRAINING DISABLED' in str(e);blocked=True
 else:raise AssertionError('Training authorization guard failed')
 audit={'status':'PASS','technical_ready_for_RGB_training':True,'full_training_authorized':False,'training_updates':0,'dataset_release_unchanged_hash_check':True,'five_config_incremental_diff_check':True,'official_ImageNet_not_skin_checkpoint':True,'standard_graph_parity':True,'fusion_and_contrastive_gradients_finite_nonzero':True,'contrastive_singleton_and_singleclass_guards':True,'optimizer_groups_stage_contracts':True,'source_partition_guards':True,'authorization_guard':blocked,'CPU_QA_optional_treatments':'16/16each; not enabled','untrained_builtin_TFLite_graphs':'both PASS','models_inference_parameters':[2498148,2587815],'external_efficacy_evaluated':False,'HIBA_or_additional_ISIC_collection_used':False,'heldout_PAD_images_metrics_prediction_files_used':False,'results_figures_generated':False,'engineering_visual_preview_review':True,'remaining_scientific_limits':['candidate method novelty unestablished','no clinical diagnostic-preservation review','no independent clinical smartphone cohort','no multi-seed evidence or new efficacy results','no trained-weight/quantized-device parity or Flutter pixel-conformance tests'],'mac_convention':'Conv/Linear only','source_figures':3}
 dump(out/'reports/IMPLEMENTATION_AUDIT.json',audit)
 (out/'IMPLEMENTATION_AUDIT.md').write_text('''# Implementation readiness verdict

**PASS for technical preparation of the RGB C0–C4 study. Full training remains disabled.** All required pretraining integrity/shape/finite-gradient/stage-optimizer/partition checks passed; both inference graphs converted and ran with built-in TFLite operators. Five incremental config diffs checked. Authorize future training separately; no permission is requested as part of this implementation-only deliverable.

Measured results and numerical tolerances are in reports/*.json; losses/architecture in ARCHITECTURE_AND_LOSSES.md; preprocessor/augmentation in PREPROCESSING_AND_AUGMENTATION.md. All5configs use RGB; optional CC/CLAHE engineering QA does not enable them. No clinical ROI/mask is claimed. No original source dataset, checkpoint, figure or old experiment was targeted for writes. No held-out outcomes informed these settings.

This is technical readiness, not scientific success, a novelty clearance or deployment approval. Long-run optimization and generalization have not been tested. Single seed, reused source validation, missing global patient identifiers, unvalidated diagnostic color preservation, no new independent smartphone cohort and pending trained-weight/Flutter parity remain material research limitations. See NOVELTY_OVERLAP_MATRIX.md and CONVERSION_AND_FLUTTER.md. No HIBA is used.
''')
 (out/'README.md').write_text('''# CA-MACL implementation package

Start with IMPLEMENTATION_AUDIT.md. configs/C0.json…C4.json are the only5new experiment definitions. code/ contains exact implementation, guards, smoke/benchmark/conversion scripts and future source-only summary. runs/ is intentionally empty. TRAINING_AUTHORIZATION.json is FALSE.

ARCHITECTURE_AND_LOSSES.md; FIVE_EXPERIMENTS.md; TRAINING_PROTOCOL.md; PREPROCESSING_AND_AUGMENTATION.md; NOVELTY_OVERLAP_MATRIX.md; CONVERSION_AND_FLUTTER.md; RESOURCE_ESTIMATE.md; REPRODUCIBILITY_REPORT.md document design and limitations. source_data/ includes approved train/validation metadata, source QA selections/golden images+tensors, augmentation logs, stage shapes and train-only statistics. reports/ contains actual engineering checks and projected costs. conversion/ contains UNTRAINED architecture-only TFLite artifacts and console/environment logs. figures/ and FIGURE_INDEX.csv include source/architecture evidence only. No performance figures exist yet.

Training implementation/configuration seal excludes the authorization file so a later explicit authorization can be recorded without editing model/loss/data definitions. Any source/config changes require a new version and re-verification. A new clinical source and external evaluation are outside this package. No prior study is replaced.
''')
 for p in lockpaths+[out/'IMPLEMENTATION_LOCK.json']:p.chmod(0o444)
 # Recheck release metadata integrity at the end without loading held-out labels/pixels for development.
 for line in (RELEASE/'checksums.sha256').read_text().splitlines():
  h,p=line.split('  ',1);assert sha(RELEASE/p)==h
 for name,h in load(out/'IMPLEMENTATION_LOCK.json')['sha256'].items():assert sha(out/name)==h
 lines=[f'{sha(p)}  {p.relative_to(out)}' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='PACKAGE_CHECKSUMS.sha256'];(out/'PACKAGE_CHECKSUMS.sha256').write_text('\n'.join(lines)+'\n')
 print('IMPLEMENTATION COMPLETE',out,'FILES',len(lines),'TRAINING DISABLED',flush=True)
if __name__=='__main__':main(ROOT/sys.argv[1])
