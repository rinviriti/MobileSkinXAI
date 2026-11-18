from common import *
from collections import Counter
import shutil,torch
from PIL import Image
version='v'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_three_run_source_only'
out=ROOT/'research/thesis_assets/cross_dataset_ablation'/version;out.mkdir(parents=True)
for d in ['00_Protocol','01_Training_Histories','02_Validation_Results','03_Confusion_Matrices','04_ROC_PR','05_Classwise_Performance','06_Ablation_Comparisons','07_Cross_Dataset_Exploratory','08_Complexity','09_GradCAM','10_Publication_Ready','11_Source_Data','12_Reports','code']:(out/d).mkdir()
assert sha(RELEASE/'checksums.sha256')==load(RELEASE/'RELEASE_LOCK.json')['checksums_sha256'];checkfiles(RELEASE,'checksums.sha256');assert load(RELEASE/'data_approval.json')['approved']
# Stream/filter on partition first: held-out labels/images/predictions/metrics are never used in development.
with (RELEASE/'dataset_manifest.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['split'] in ['train','validation'] and r['release_eligible']=='True']
assert Counter(r['split'] for r in rows)=={'train':11632,'validation':2835}
for r in rows:
 assert CLASSES[int(r['label_index'])]==r['label'];assert sha(ROOT/r['processed_path'])==r['processed_sha256']
 with Image.open(ROOT/r['processed_path']) as im:assert im.mode=='RGB' and im.size==(224,224);im.verify()
write(out/'11_Source_Data/train_validation_only_manifest.csv',rows)
counts=[sum(r['split']=='train' and r['label']==c for r in rows) for c in CLASSES];assert dict(zip(CLASSES,counts))==load(RELEASE/'class_weights.json')['counts']
parents={}
expected={'M3':'e351e4d0a66604a57a9bb493f1d1d769ee152acd00471a214c6dfedab9f677de','M4':'2f2d0735a6c51ac813d05e4147851c6d23b535e36d38df77194c8b0b3030b95a','M5':'d096ab7fff62c6cb34504530cba8a295d076a9bcb255b16c4d506e9f8c88d883'}
for k in ['M3','M4','M5']:
 p=PREP/f'runs/{k}_seed42/checkpoints/best.pt';assert sha(p)==expected[k];cp=torch.load(p,map_location='cpu',weights_only=True);assert cp['provenance']['classes']==CLASSES
 parents[k]={'path':str(p.relative_to(ROOT)),'sha256':expected[k],'epoch':cp['epoch'],'provenance':cp['provenance'],'validation_macro_f1':cp['validation_macro_f1']};del cp
assert parents['M4']['validation_macro_f1']>parents['M3']['validation_macro_f1']
original=load(PREP/'configs/M5.json');assert original['loss']=={'lambda_ce':1.,'lambda_kd':.5,'lambda_consistency':.1,'temperature':4.,'class_weight_power':.5,'confidence_threshold':.7,'max_normalized_entropy':.6,'max_teacher_js':.1,'reliability_js_scale':.1,'require_teacher_correct_both_views':True,'epsilon':1e-8}
for k in KEYS:
 cfg=copy.deepcopy(original);cfg.update({'study_id':version,'study_key':k,'name':NAMES[k],'split_manifest':str((out/'11_Source_Data/train_validation_only_manifest.csv').relative_to(ROOT)),'max_total_epochs':25,'max_finetune_epochs':25,'checkpoint_dir':str((out/f'runs/{k}/checkpoints').relative_to(ROOT)),'study_intervention':{'B1':'reliability replaced by1 in KD and JS; class weighting/denominator unchanged','B2':'KD removed; student-only ungated class-aware JS retained at0.1','B3':'CE on shared-geometry weak and acquisition-perturbed strong; no teacher terms'}[k]})
 cfg['smartphone_augmentation']={'brightness_contrast_range':[.9,1.1],'white_balance_channel_gain_range':[.95,1.05],'white_balance_gain_geometric_mean_normalized':True,'JPEG_probability':.5,'JPEG_quality_integer_range':[75,95],'blur_probability':.2,'blur_radius_range':[.1,.6],'framing_probability':.3,'scale_range':[.9,1.0],'framing':'resize full image, no crop; random placement within gray114 canvas; all lesion content retained','policy_source':'a priori source-only mild ranges; no PAD statistics or visual tuning'} if k=='B3' else None
 dump(out/f'00_Protocol/{k}.json',cfg)
originalhist=[json.loads(x) for x in (PREP/'runs/M5_seed42/history.jsonl').read_text().splitlines()];assert len(originalhist)==7
# Preserve without reading any previous Phase6 metric or prediction contents.
snapshot={}
for p in list(PREP.rglob('*.pt'))+list(PHASE6.rglob('*')):
 if p.is_file():snapshot[str(p.relative_to(ROOT))]={'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
dump(out/'11_Source_Data/original_preservation_stat_snapshot.json',snapshot)
dump(out/'00_Protocol/verification.json',{'status':'PASS','dataset_release_checksums':'PASS','train_validation_image_hashes_decodes':len(rows),'counts':dict(zip(CLASSES,counts)),'class_weights':[len([r for r in rows if r['split']=='train'])/(4*n) for n in counts],'parents':parents,'original_M5_epochs':7,'original_M5_cap':40,'new_cap':25,'budget_interpretation':'Original stopped7<25, so administrative cap does not truncate its observed trajectory. New arms may realize unequal extra updates; report selected and total epochs and matched epoch1/first7 summaries descriptively. No restarted full-M5 comparator permitted.','development_access':'train/validation images+labels only; release checksum bytes and original checkpoint metadata verified; no Phase6 metric/prediction/image reads','no_additional_confirmatory_dataset':True})
(out/'00_Protocol/PRE_REGISTERED_ABLATION_PROTOCOL.md').write_text(r'''# Pre-registered three-run source-only ablation and robustness protocol

MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.

## Scope, hypotheses and frozen roles
Exactly three new full configurations, seed42, single sequential execution each. No hyperparameter search/restarts selected by outcomes. Original M3 baseline and M5 enhanced remain references, with original qualified M4 teacher. All old checkpoints/results and immutable release preserved. Development uses only11632ISIC train and2835validation images, class order NV/MEL/BCC/SCC. Original Phase6 has been viewed; any subsequent reuse is explicitly POST-HOC EXPLORATORY and not independent confirmation. No PAD-derived statistics, image previews, predictions or metrics inform these choices. Earlier exposure informed the broad research direction, which cannot be undone by this registration.

H1 (descriptive): removing reliability may alter validation macro-F1 and class errors by admitting unreliable teacher targets. H2: removing KD under a shared ungated student consistency rule changes performance (compare B1/B2). H3: predefined mild source-only acquisition perturbations may improve robustness; this is a separate intervention, not an isolated M5 ablation. No directional success is assumed. Report negative findings, tradeoffs and all three configurations.

## Verified original implementation and precise objective
For batch size n, class weights w_c=N/(4N_c), a_i=sqrt(w_yi), D=sum_i a_i, weak/strong student logits s_w,s_s and detached frozen teacher logits t_w,t_s. Let p=softmax(t_w),q=softmax(t_s), h_i=[H(p_i)+H(q_i)]/(2 log4), d_i=JS(p_i,q_i). The hard gate is
G_i=1[min(max p_i,max q_i)>=.7, h_i<=.6, d_i<=.1, argmax p_i=argmax q_i=y_i].
Reliability r_i=stopgrad[G_i*(1-h_i)*exp(-d_i/.1)]. JS(P,Q)=.5 KL(P||(P+Q)/2)+.5 KL(Q||(P+Q)/2).
CE=.5[sum_i w_yi CE(s_wi,y_i)/sum_i w_yi + sum_i w_yi CE(s_si,y_i)/sum_i w_yi].
k_i=(T²/2)[KL(softmax(t_wi/T)||softmax(s_si/T))+KL(softmax(t_si/T)||softmax(s_wi/T))], T=4.
j_i=JS(softmax(s_wi),softmax(s_si)).
Original L=CE+.5*(sum_i a_i r_i k_i/D)+.1*(sum_i a_i r_i j_i/D).
Both KD and student JS are gated. Denominator is full class-weight mass, NOT accepted-weight mass. Rejecting all cases makes both auxiliaries zero; batch CE retains PyTorch weighted-mean reduction. Weak view has shared geometric transforms only (no mild ColorJitter); strong adds ColorJitter(.2,.2,.15,.05) and RandomGrayscale(.1). Shared geometry: independent horizontal/vertical flips p=.5 and bilinear rotation uniform[-15,15] gray114fill. Teacher sees both views. Teacher certainty is computed at T=1; KD uses T=4. Teacher correct-both-views gate uses training labels only. Both student branches receive gradients; teacher/gate do not. BatchNorm running stats frozen. This matches actual code; shorthand 'gated JS' alone would incorrectly omit KD gating.

## Exactly three new objectives
B1 (MobileSkinXAI-Enhanced without reliability gate): L=CE+.5 sum_i a_i k_i/D+.1 sum_i a_i j_i/D. Set r=1 for BOTH auxiliaries; retain class factors, full denominator, coefficients and temperature. This removes hard selection AND continuous reliability as one bundled component, not either separately. Auxiliary gradient scale increases relative to original by an amount depending on batch r and per-case loss. No renormalization or compensating coefficient tuning. Record original gate acceptance/reliability mass, counterfactual gated auxiliary values and weighted mass; gradient norms/scale on the bounded smoke batch.
B2 (MobileNetV4 with student-only consistency): L=CE+.1 sum_i a_i j_i/D. No teacher forward, target, gate or KD. Student JS is meaningful without a teacher and propagates to both branches. B1 versus B2 isolates KD conditional on ungated JS; original M5 versus B2 removes teacher guidance plus its reliability selection and is NOT a pure single-component contrast. Original M5 versus B1 tests the bundled reliability mechanism with both auxiliary losses otherwise matched.
B3 (Domain-robust MobileNetV4-Conv-Small): L=CE on weak shared-geometry and strong acquisition-perturbed views; no teacher/KD/JS. Strong view: independent brightness/contrast factors U(.9,1.1), RGB gains U(.95,1.05) normalized by geometric mean (white-balance perturbation), JPEG p=.5 quality integer75..95, Gaussian blur p=.2 radius U(.1,.6), framing p=.3 full-image scale U(.9,1), randomly placed within224gray114canvas without cropping. Processing order brightness, contrast, white balance, JPEG, blur, framing. No PAD statistics, no random grayscale or strong hue shift. This is a prespecified proxy policy; preservation of medical utility is NOT clinically verified. No visual tuning after outcomes. B3 bundles augmentation and supervised-only objective changes and is a robustness intervention, not a causal augmentation-only ablation.

## Controls, budgets and stopping
All initialize STRICTLY from same fresh final M3 checkpoint; B1 uses same frozen qualified M4. All architectures remain identical4-output MobileNetV4. Original initialization RNG order is matched: construct student and reference-teacher architecture under seed42 for every arm; B2/B3 teacher is never forwarded and is discarded. Fresh AdamW (no parent optimizer), head LR1e-4/backbone1e-5, decay1e-4, no scheduler, no frozen-head stage, batch16, workers4, CUDA BF16, max25additional epochs, patience6/min_delta0, strict validation image-level macro-F1 improvement, no baseline-score floor, seed42. Train-only class weights unchanged. Train data order generator and worker policies match original; strong augmentation differs only in B3 by design. Train BatchNorm running stats frozen as original. Standard224RGB preprocessing and ImageNet normalization unchanged. Validation has no augmentation and unweighted CE.
Original M5 max40, stopped7(best1). Because7<25 the cap difference would not alter its observed trajectory. It does limit new arms; record it. Report realized optimization budgets and descriptive shared-epoch/first7 checkpoint histories without selecting another model. Single-seed differences do not establish causal/statistical superiority; no additional full-M5 control/repetition is authorized. No clean augmentation-only comparator exists for B3. No full-reference restart will be silently added.

## Pretraining checks and execution acceptance
Verify release metadata checksums, train/validation input hashes and decoding, class mappings/counts; parent hashes/provenance; train-only loader guard. Smoke test all three: finite tensor shapes/losses, nonzero student gradients (both views for consistency), no teacher gradients, gate unit tests, exact original-loss parity, deterministic order/augmentation parity B1/B2, no optimizer state carryover. Bounded representative GPU backward benchmark with discarded state estimates duration, not a fourth full configuration. Full runs begin only if every required check passes. No automatic retries on failed runs; stop/report. Exactly three unique run directories; retain every epoch, best/last immutable payloads and histories. Read-only protocol/hash seal before smoke/full training.

## Analysis and checkpoint freeze
Primary development endpoint validation image-level macro-F1, report accuracy/balanced accuracy and4-class precision/recall/F1, MEL/SCC recall, NV→MEL and SCC→BCC errors, loss components, gate statistics, parameter count/runtime/VRAM. Pairwise differences descriptive; validation was used for stopping/selection, so bootstrap intervals on validation do not remove selection bias. No significance/clinical/novelty claim. Avoid presenting B1/B2 as fully causal under budget/seed limitations. Selected checkpoints use only validation macro-F1; no score-based replacement of original M5.

## Optional held-out reuse, explicitly permitted after final lock
Once all3runs and validation-selected hashes are frozen, write read-only final selection/analysis lock. Then evaluate each selected arm ONCE on released2703ISICtest and1303PAD images; original Phase6 remains frozen and comparisons are labeled POST-HOC EXPLORATORY. No quarantined images, no fitting/calibration/threshold selection or subsequent training. Raw softmax argmax, released lesion means over prechosen exact-capture representatives primary; image metrics secondary. Cluster bootstrap2000replicates, base seed42 (ISIC42/PAD43), ISIC lesion/PAD patient, percentile95CI; identical draws across arms/reference comparisons. Accuracy, macro-F1, balanced accuracy, MEL/SCC recall and internal-minus-external gaps. Previously recorded baseline/originalM5 predictions may be read ONLY after this freeze for paired exploratory comparison. Report small PAD melanoma support50images/34lesions and missing global ISIC patient identifiers. All tests are already exposed; external superiority remains unconfirmed without a genuinely independent smartphone dataset. No winner chosen from PAD.

## Figures and reporting
Actual source CSV/JSON only, full model names, accessible palette, PNG300DPI+SVG/PDF, captions and script hashes. No methodology diagrams. Original M3/M5 validation references only before final lock. No TFLite export/Flutter changes in this study; identical inference architecture supports the prior conversion path but new checkpoints are not exported/verified here. No novelty assertion; overlap with standard KD, consistency regularization and source augmentation must be distinguished in a paper.
''')
for p in [PREP/'code/core.py',PREP/'code/train.py']:
 shutil.copyfile(p,out/'11_Source_Data'/('original_'+p.name))
dump(out/'00_Protocol/READY_FOR_IMPLEMENTATION_CHECKS.json',{'status':'protocol_saved_not_yet_training','created_UTC':datetime.now(timezone.utc).isoformat(),'output':str(out.relative_to(ROOT))})
print(out.relative_to(ROOT),flush=True)
