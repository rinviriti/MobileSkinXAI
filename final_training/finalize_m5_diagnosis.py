"""Add evidence-led interpretation and ONE unexecuted proposal; no image reads."""
import csv,hashlib,json,math,shutil,sys
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];P=R/'final_training/phase5a/v20261117T061722Z_f5055bd9';D=R/'research/thesis_assets/final_dataset_release/v20261117T060846Z_608314c8';O=R/'research/thesis_assets/M5_validation_diagnosis/v20261117T073553Z_99a1fb04'
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(n,o):(O/n).write_text(json.dumps(o,indent=2)+'\n')
C=['NV','MEL','BCC','SCC'];cfg=json.loads((P/'configs/M5.json').read_text());wr=json.loads((D/'class_weights.json').read_text());w=np.array([wr['weights'][c] for c in C]);n=np.array([wr['counts'][c] for c in C]);N=int(n.sum())
# These are class-count-only expectations, not reconstruction of historical batch gates.
rng=np.random.default_rng(42);batch_counts=rng.multivariate_hypergeometric(n,16,size=200000);mass=batch_counts*w;shares=(mass/mass.sum(1,keepdims=True)).mean(0);kmass=batch_counts*np.sqrt(w);kshares=(kmass/kmass.sum(1,keepdims=True)).mean(0)
rows=[]
manifest=read(D/'dataset_manifest.csv')
for j,c in enumerate(C):
 train=[r for r in manifest if r['dataset']=='ISIC2019' and r['split']=='train' and r['release_eligible']=='True' and r['label']==c];val=[r for r in manifest if r['dataset']=='ISIC2019' and r['split']=='validation' and r['release_eligible']=='True' and r['label']==c]
 rows.append({'class':c,'train_images':len(train),'train_declared_lesions':len({r['lesion_id'] for r in train}),'validation_images':len(val),'validation_declared_lesions':len({r['lesion_id'] for r in val}),'CE_weight':float(w[j]),'KD_class_factor':float(np.sqrt(w[j])),'natural_image_fraction':float(n[j]/N),'dataset_population_CE_weight_mass_fraction':float(n[j]*w[j]/sum(n*w)),'dataset_population_KD_class_mass_before_gates':float(n[j]*np.sqrt(w[j])/sum(n*np.sqrt(w))),'simulated_batch_normalized_CE_mass_fraction':float(shares[j]),'simulated_batch_normalized_KD_mass_fraction_before_gates':float(kshares[j]),'exact_probability_class_absent_batch16':float(np.prod([(N-int(n[j])-k)/(N-k) for k in range(16)]))})
write(O/'source_data/class_weight_and_sampling_diagnostics.csv',rows)
# Additional gate edge cases, still synthetic-only and without optimizer construction.
sys.path.insert(0,str(P/'code'));from core import m5_loss
sw=torch.zeros(4,4,requires_grad=True);ss=torch.ones(4,4,requires_grad=True);target=torch.arange(4);wt=torch.tensor(w,dtype=torch.float32)
loss,st=m5_loss(sw,ss,torch.zeros(4,4),torch.zeros(4,4),target,wt,cfg);grad=torch.autograd.grad(loss,[sw,ss]);assert all(torch.isfinite(x).all() and x.abs().sum()>0 for x in grad)
wrong=torch.roll(torch.eye(4)*10,1,1);_,st=m5_loss(sw,ss,wrong,wrong,target,wt,cfg);assert st['gate_acceptance']==0
correct=torch.eye(4)*10;badstrong=torch.roll(correct,1,1);_,st=m5_loss(sw,ss,correct,badstrong,target,wt,cfg);assert st['gate_acceptance']==0
js('additional_gate_contracts.json',{'status':'PASS','zero_gate_student_gradients_finite_and_nonzero':True,'confident_but_wrong_teacher_rejected':True,'teacher_disagreement_between_views_rejected':True,'real_parameter_updates':0})
proposal={'name':'M5_LR01_single_factor_control','status':'RECOMMENDED_ONLY_NOT_AUTHORIZED_OR_RUN','additional_training_configurations_recommended':1,'dataset_release':D.name,'initialization':cfg['initialization'],'initialization_rationale':'Restart from the SAME fresh M3 best checkpoint and SAME M4 teacher as original M5. Starting from M5 best would confound the LR change with a different parent and extra training.','only_changed_training_factor':{'head_lr':{'original':1e-4,'proposed':1e-5},'backbone_lr':{'original':1e-5,'proposed':1e-6},'common_multiplier':0.1},'input_size':[3,224,224],'classes':C,'batch_size':16,'seed':42,'optimizer':'AdamW','weight_decay':1e-4,'betas':[.9,.999],'eps':1e-8,'amsgrad':False,'optimizer_state':'Reset AdamW moments exactly as original M5; do not resume M3 or M5 optimizer','frozen_epochs':0,'maximum_additional_epochs':40,'scheduler':'none_fixed_stage_learning_rates','early_stopping':cfg['early_stopping'],'precision':cfg['mixed_precision'],'freeze_bn_running_stats':True,'normalization':cfg['normalization'],'augmentation':cfg['augmentation'],'train_class_weights':wr['weights'],'loss':cfg['loss'],'teacher_frozen_eval':True,'prediction_rule':'Unmodified argmax, class order NV,MEL,BCC,SCC. No calibration or threshold fitting.','checkpoint_selection':'Highest validation macro-F1, first maximum on ties, patience6; do not retrospectively choose another epoch for a different metric.','predeclared_success_guard':{'macro_f1_strictly_greater_than':0.6911747422941596,'SCC_true_positives_at_least':36,'SCC_validation_support':95,'MEL_true_positives_at_least':423,'MEL_validation_support':586,'otherwise':'Retain original M5 as the comparator; report followup as not meeting the joint target. No further automatic configurations or tuning.'},'future_logging':'Per-epoch CE,KD,JS and their weighted contributions, per-class hard gate acceptance and mean reliability mass, separate head/backbone gradient norms and exact LR; train batches only for those diagnostics.','claims':'Hypothesis only; improvement or melanoma noninferiority is not guaranteed. Validation is reused adaptively; independent locked evaluation is still required later. No novelty claim.'}
js('PROPOSED_SINGLE_CONTROL.json',proposal)
formula=r'''# One controlled follow-up recommendation — NOT RUN

Recommend **M5_LR01**, a repeat of the original M5 recipe from the **same final M3 best checkpoint (epoch18)** and the **same final M4 teacher (epoch13)**, changing only a common learning-rate multiplier from1 to0.1. This is source-training continuation from M3, not a resume of M5. No model or optimizer is created for this proposal.

Head LR **1e-5**, backbone LR **1e-6**, fixed for the entire run; AdamW weight decay1e-4, betas(0.9,0.999), eps1e-8. Reset optimizer moments exactly as original M5. Batch16, seed42, BF16 on the verified GPU, no frozen-head stage, max40additional epochs, patience6 on validation macro-F1. Preserve all original normalization, geometry, two-view augmentation, frozen BN running statistics, teacher eligibility/frozen-eval state, class weights, temperature and gate settings. Stop without replacement if parent provenance or dataset checks fail. No checkpoint/output directory is activated by this recommendation.

Rationale: M5's best macro-F1 occurred after one additional epoch, but later macro-F1 deteriorated while training loss fell. This motivates a smaller-update control; it does **not prove** the original LR was excessive. Epoch7 validation CE is actually lower than epoch1, so simple global overfitting or an LR-scheduler failure is not established. Lower LR is a conservative single-factor test that preserves the existing melanoma/SCC weighting rather than weakening minority supervision or introducing an unvalidated loss. It may fail to correct high-confidence SCC errors.

## Exact objective

For batchB of training pairs, y_i in NV,MEL,BCC,SCC order, let w_c=N_train/(4n_c), N_train=11632, n=(6838,2611,1799,384). The exact weights are (0.42527054694355076,1.1137495212562236,1.6164535853251807,7.572916666666667). Weak and strong views share geometry; only strong has strong color jitter/grayscale. Teacher logits are detached and teacher is frozen/eval. All following y_i and statistics are **training-only**.

Let p_S^v=softmax(z_S^v), p_T^v=softmax(z_T^v), v in {w,s}, and p_{S,T,t}^v=softmax(z_{S,T}^v/t). JS uses natural logarithms. Define

$$h_i={H(p_T^w)+H(p_T^s)\over2\log4},\qquad d_i=JS(p_T^w,p_T^s),$$
$$g_i=\mathbf1[\min_v\max_c p_{T,c}^v\ge0.7]\,
\mathbf1[h_i\le0.6]\,\mathbf1[d_i\le0.1]\,
\mathbf1[\arg\max p_T^w=\arg\max p_T^s=y_i],$$
$$r_i=\operatorname{stopgrad}[g_i(1-h_i)e^{-d_i/0.1}],\quad a_i=w_{y_i}^{0.5},\quad D=\max(\sum_i a_i,10^{-8}),\quad t=4.$$

$$L_{CE}={\sum_i w_{y_i}[-\log p_{S,y_i}^w-\log p_{S,y_i}^s]\over2\sum_iw_{y_i}},$$
$$L_{KD}={t^2\over2D}\sum_i a_ir_i\{KL(p_{T,t}^w\Vert p_{S,t}^s)+KL(p_{T,t}^s\Vert p_{S,t}^w)\},$$
$$L_{JS}={1\over D}\sum_i a_ir_i JS(p_S^w,p_S^s),$$
$$\boxed{L=L_{CE}+0.5L_{KD}+0.1L_{JS}.}$$

This is the existing M5 objective, unchanged. The only proposed intervention is LR scaling0.1; no new loss is claimed. LambdaCE1, lambdaKD0.5, lambdaJS0.1, temperature4, beta0.5, confidence0.7, entropy0.6, teacherJS0.1, reliability decay0.1, epsilon1e-8, both-view correctness required.

## Checkpoint selection and success rule

Select the highest validation macro-F1 checkpoint, first maximum on ties; use patience6 exactly as before. Then report success only if that selected checkpoint jointly has macro-F1>0.6911747423, SCC true positives>=36/95 (strictly above original35/95), and melanoma true positives>=423/586 (original72.1843%). Do not select a different checkpoint retrospectively to satisfy one endpoint. Keep the original M5 immutable as the comparator if any condition fails. The constraints are acceptance criteria, **not a guarantee** that the optimizer will meet them. Do not fit class thresholds, temperatures or gate parameters on validation.

One additional configuration only, subject to later explicit authorization; no repeats, second control or automatic followup. Same maximum budget and stopping rule are preserved, but actual stopping epochs may differ. Future training must log CE/KD/JS weighted terms, per-class reliability mass, gradient norms and LR to close the current observability gap. The already reused validation set is development data; any favorable result remains adaptively selected until later locked evaluation.

## Alternatives compared, not additional recommended runs

| Strategy | Precise interpretation | Evidence and reason not to combine now |
|---|---|---|
| Lower-LR continuation control | Existing M5 objective and parents, head1e-5/backbone1e-6 | Recommended: clean single-factor probe of early boundary drift, preserves minority weights; benefit uncertain. |
| Moderated class weighting | Replace CE w_c by sqrt(w_c), with optional common normalization (which cancels under weighted-mean CE); leave KD factors/gates unchanged | Reduces MEL:NV ratio2.619 to1.618 and SCC:NV17.807 to4.220. May reduce NV false-MEL pressure but risks the two minority-sensitivity targets; no empirical result available. |
| Training-only confusion auxiliary | L'=L+0.1 L_pair; q(NV)=MEL,q(MEL)=NV,q(BCC)=SCC,q(SCC)=BCC; L_pair= [sum_i w_y sum_{v=w,s} softplus(0.5+z_{S,q(y)}^v-z_{S,y}^v)]/[2sum_iw_y] | Explicit pairwise logistic margin0.5, temperature1, training labels only, no teacher/validation labels inside the loss. Symmetric MEL term avoids intentionally ignoring melanoma false negatives. Directly targets both pairs, but may amplify memorization: train SCC recall already89–96% versus validation26–48%. It introduces unvalidated gradient competition with KD and one more loss scale; do not combine with the LR control. This is a conceptual alternative, not a prepared runnable experiment. |

Pairwise cost-sensitive/logistic margins, imbalance reweighting and confidence/cross-view KD have prior art. This auxiliary is not claimed identical to LDAM or the 2018 Pairwise Confusion paper (the latter deliberately moves distributions closer, a different mechanism). No novel-method claim is made for any option.
'''
(O/'RECOMMENDATION.md').write_text(formula)
confidence=read(O/'source_data/per_class_confidence.csv');errors=read(O/'source_data/targeted_confusion_confidence.csv');lr=read(O/'source_data/M5_exact_LR_and_history.csv');boot=json.loads((O/'paired_validation_bootstrap.json').read_text())
table='\n'.join('| '+r['model']+' | '+r['true_class']+' | '+f"{float(r['recall']):.3f}"+' | '+f"{float(r['mean_true_class_probability']):.3f}"+' | '+f"{float(r['error_mean_confidence']):.3f}"+' | '+f"{float(r['ECE_10_equal_width']):.3f}"+' |' for r in confidence)
report=f'''# Read-only final M5 improvement diagnosis

**No material implementation or gradient defect was found in the inspected paths and synthetic contracts.** Recommend one unexecuted LR-only M5 control, not immediate loss redesign. Original checkpoints, configurations, validation predictions and approved release artifacts are hash-preserved. All predictions already existed; no model forward, optimizer step or image read was needed. Locked ISIC test/PAD images and predictions were not accessed.

## Implementation and optimization audit

The executed frozen code is final_training/phase5a/{P.name}/code/core.py and code/train.py. KD direction is teacher||student with t² scaling; two-view CE normalization matches the documented formula; teacher outputs/reliability are detached; teacher is no-grad/eval; optimizer includes student parameters only. Independent scalar and gradient comparisons passed. Confident but wrong/disagreeing teacher synthetic cases are rejected, zero-acceptance leaves finite nonzero supervised gradients, and no class-order mismatch was found.

Train weights are NV0.425270547, MEL1.113749521, BCC1.616453585, SCC7.572916667. They are identical in M3 and M5; a weight change cannot explain the M3→M5 shift. Their relative costs (MEL:NV2.619, SCC:NV17.807) plausibly influence the tradeoff jointly with the new objective, extra optimization and altered views, but there is no causal ablation. M5 weak view has geometry only, whereas M3 also has mild color jitter; strong M5 view has stronger jitter and10% grayscale. Optimizer moments reset at M5 start. These are additional differences beyond a bare KD coefficient.

Gate acceptance is0.833–0.840 overall over7epochs, MEL0.682–0.702 and SCC0.766–0.844; no hard-gate minority starvation. Accepted mass is further multiplied by entropy/JS reliability and sqrt(class weight); actual per-class reliability mass was **not logged**. Only total training objective was saved, not CE/KD/JS components or gradient norms. Consequently exact historical term dominance cannot be reconstructed; no exploding/nonfinite objective was observed, but hidden KD/CE imbalance cannot be ruled out from scalar totals. t²=16 alone is not evidence of a scale bug. With r<=1 and denominator including all class mass, the weighted consistency contribution is bounded by0.1log2≈0.0693; it cannot arbitrarily amplify a tiny accepted subset.

## Exact LR, selection and stopping

All7 saved M5 optimizer states contain head LR1e-4/backbone LR1e-5 and decay1e-4; no scheduler, warmup or LR decay ran. Adam state steps727,1454,...,5089 confirm one fresh optimizer continued throughout seven727-update epochs. Additional-epoch macro-F1 values are0.691175,0.659864,0.670875,0.680047,0.674193,0.686738,0.687964. Epoch1 is the strict maximum; epochs2–7 are six consecutive non-improvements, so patience6 correctly stops at7. This is not an off-by-one stopping bug.

Training objective fell0.8014→0.5652. Validation CE was0.5791 at1 and0.5635 at7, despite lower macro-F1 at7. Therefore global CE overfitting or an excessive LR is **not proven**. Selection emphasizes macro-F1 rather than CE, accuracy, or isolated SCC recall. Epoch4 had SCC recall46/95=48.42% but macro-F1 only0.6800; it was correctly not selected. The initial selection best=-1 also means the untouched M3 parent is not a candidate; this did not harm this run because M5 epoch1 exceeded its parent.

## Confusions and teacher guidance

MEL true positives increased363→423 (+60), while NV true positives decreased1437→1366 (-71). NV→MEL rose139→221, MEL→NV fell170→126, and MEL precision fell0.6624→0.6060. This is a decision tradeoff, not uniform improvement. Of97 previously correct NV examples flipped to MEL, the teacher was actually correct on68 and predicted MEL on29. Thus copying incorrect teacher hard labels does not explain most newly spoiled NV examples.

SCC correct counts changed33→35 (+2), while SCC→BCC changed35→38 (+3). Thirty of the38 errors were already SCC→BCC in M3. Of5 previously correct SCC cases newly sent to BCC, M4 was correct on4. This again weakens a simple wrong-teacher explanation. M4 itself misses51/95 SCC examples, including26 as BCC, despite passing the global teacher macro-F1 rule; that rule does not certify reliable class-wise or per-example guidance. Correctness gating protects observed training labels, not unseen validation identities.

M5 NV→MEL errors have mean predicted confidence0.739 and mean top-two margin0.495 (20/221 margins<=0.1;47 errors confidence>=0.9). M5 SCC→BCC errors have mean BCC confidence0.866 and margin0.767 (3/38 margins<=0.1;22 confidence>=0.9). These are often confident representation/classification failures, not simply threshold ties. Confidence is not calibrated correctness. At diagnostic teacher temperature4, mean SCC/BCC soft probabilities over SCC validation images are0.315/0.282, compared with0.412/0.295 at temperature1. Temperature smoothing can weaken pair separation, but these unaugmented validation values do not reproduce historical train-view gates/targets and do not prove harm from KD.

## Per-class confidence (existing validation only)

| Model | True class | Recall | Mean true-class probability | Mean confidence when wrong | Conditional top-label ECE10 |
|---|---|---:|---:|---:|---:|
{table}

ECE uses10equal-width bins conditional on true class and the prediction's top-label correctness; it is a diagnostic, not classwise one-vs-rest ECE or a fitted calibration procedure. Exact NLL, Brier, quantiles/margins and paired prediction transitions are in source_data/. SCC statistics use95images, not95independent patients.

## Data and uncertainty

Only384 retained training images are SCC. A random batch16 has exact probability{rows[3]['exact_probability_class_absent_batch16']:.3f} of containing no SCC. The full-dataset CE weight mass is equal by design, while sqrt-weight KD still favors classes with more images in aggregate before gates; per-batch weighted-mean normalization changes expected shares (see class_weight_and_sampling_diagnostics.csv). This is a standard stochastic-weighting effect, not evidence of incorrect code. Actual batch masses/gate correlations were not saved.

SCC training recall is89–96% across M5 epochs versus validation26–48%, supporting a substantial generalization gap; it does not identify its cause. Cohort namespaces show NV→MEL increases in BCN84→123, HAM46→79 and MSK4 9→19. SCC→BCC increases are concentrated in BCN30→35, while HAM5→3. These metadata associations do not establish domain causes or clinical similarity; no image pixels, original external metadata or patient-level provenance were newly inspected.

M5–M3 macro-F1 difference0.00796 has paired lesion-cluster diagnostic95% percentile interval[{boot['paired_lesion_cluster_percentile95_interval'][0]:.5f},{boot['paired_lesion_cluster_percentile95_interval'][1]:.5f}] from2000seed42resamples of{boot['lesions']}declared validation lesions. It crosses zero. This is not a selection-adjusted significance test or independent generalization interval: both checkpoints were chosen using this validation set, global patient IDs are absent, and follow-up recommendations reuse it adaptively.

## Recommendation and outputs

RECOMMENDATION.md gives the exact unchanged M5 loss and one LR-only control: head1e-5/backbone1e-6 from the **same M3 parent/M4 teacher**, all other settings fixed, max40additional epochs and patience6. Select highest validation macro-F1; label success only if that selected checkpoint exceeds original M5 macro-F1 and has SCC>=36/95 and MEL>=423/586 correct. This targets improvement without accepting melanoma regression, but cannot guarantee it. PROPOSED_SINGLE_CONTROL.json is descriptive only; no execution directory or training authorization was created. Alternatives are compared mathematically without recommending or launching additional configurations.

figures/ contains seven confusion figure layouts, all with full architecture names, count and row-percent views, common color scales for comparisons, PNG300DPI plus SVG/PDF. source_data/ includes exact confusion tables and all confidence/margin/LR evidence. preservation_check.json records unchanged original input hashes. No model, checkpoint, release, Flutter file or live training configuration was modified.
'''
# Improve numerical prose spacing without editing any frozen original artifact.
for a,b in [('NV0.','NV 0.'),('MEL1.','MEL 1.'),('BCC1.','BCC 1.'),('SCC7.','SCC 7.'),('and10%','and 10%'),('is0.','is 0.'),('over7epochs','over 7 epochs'),('MEL0.','MEL 0.'),('SCC0.','SCC 0.'),('by0.1log2','by 0.1 log 2'),('All7','All 7'),('LR1e','LR 1e'),('decay1e','decay 1e'),('steps727','steps 727'),('seven727','seven 727'),('are0.','are 0.'),('Epoch1','Epoch 1'),('epochs2','epochs 2'),('at7','at 7'),('patience6','patience 6'),('fell0.','fell 0.'),('was0.','was 0.'),('at1','at 1'),('at7','at 7'),('at7','at 7'),('at4','at 4'),('Epoch4','Epoch 4'),('recall46','recall 46'),('only0.','only 0.'),('increased363','increased 363'),('decreased1437','decreased 1437'),('rose139','rose 139'),('fell170','fell 170'),('Of97','Of 97'),('on68','on 68'),('on29','on 29'),('changed33','changed 33'),('changed35','changed 35'),('the38','the 38'),('M3. Of5','M3. Of 5'),('on4.','on 4.'),('misses51','misses 51'),('including26','including 26'),('confidence0.','confidence 0.'),('margin0.','margin 0.'),(';47','; 47'),(';22','; 22'),('temperature4','temperature 4'),('are0.','are 0.'),('with0.','with 0.'),('temperature1','temperature 1'),('uses10equal','uses 10 equal'),('use95images','use 95 images'),('not95independent','not 95 independent'),('Only384','Only 384'),('batch16','batch 16'),('is89','is 89'),('validation26','validation 26'),('BCN84','BCN 84'),('HAM46','HAM 46'),('MSK4 9','MSK4 9'),('BCN30','BCN 30'),('HAM5','HAM 5'),('difference0.','difference 0.'),('diagnostic95','diagnostic 95'),('from2000seed42resamples','from 2000 seed-42 resamples'),('of1500declared','of 1500 declared'),('head1e','head 1e'),('max40additional','max 40 additional'),('PNG300DPI','PNG 300 DPI')]:report=report.replace(a,b)
(O/'ANALYSIS_REPORT.md').write_text(report)
lit='''# Prior-method overlap (primary records checked 2026-11-17)

- [Cross-View Consistency Regularisation for Knowledge Distillation](https://arxiv.org/abs/2412.16493): within/cross-view regularization and confidence-based soft-label mining directly overlap with central M5 ingredients.
- [Balanced Knowledge Distillation for Long-tailed Learning](https://arxiv.org/abs/2104.10510): class-prior-aware distillation is established; M5's weighting is not by itself novelty.
- [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555): established alternatives to raw inverse-frequency weighting. The hypothetical square-root rule here is not asserted identical to effective-number weighting.
- [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/abs/1906.07413): established class-dependent margin/imbalance treatment; a constant pairwise softplus margin is a different formulation and is not thereby novel.
- [Apportioned Margin Approach for Cost Sensitive Large Margin Classifiers](https://arxiv.org/abs/2002.01408): cost-sensitive margin allocation predates the suggested conceptual confusion penalty.
- [Improving Fine-Grained Visual Classification using Pairwise Confusion](https://openaccess.thecvf.com/content_ECCV_2018/papers/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.pdf): title similarity is not equivalence. That paper deliberately brings probability distributions closer, whereas the unselected pair-margin alternative here pushes the true/competing logits apart.

The recommended single LR control is an optimization comparison, not a new loss or novelty claim. This is a targeted overlap check, not an exhaustive review or mathematical priority clearance. No proposed follow-up results exist.
'''
(O/'PRIOR_METHOD_OVERLAP.md').write_text(lit)
# Figure manifest for exact export paths/DPI; verify sources and immutable inputs again.
from PIL import Image
figrows=[]
for p in sorted((O/'figures').glob('*.png')):
 with Image.open(p) as im:dpi=im.info.get('dpi');assert dpi and abs(dpi[0]-300)<1
 figrows.append({'figure':p.stem,'PNG':str(p.relative_to(O)),'SVG':str(p.with_suffix('.svg').relative_to(O)),'PDF':str(p.with_suffix('.pdf').relative_to(O)),'DPI':300})
write(O/'FIGURE_INDEX.csv',figrows)
snap=json.loads((O/'analysis_inputs.json').read_text());assert all(sha(R/p)==h for p,h in snap.items())
shutil.copyfile(Path(__file__),O/'finalize_analysis_source.py');shutil.copyfile(R/'final_training/analyze_m5_readonly.py',O/'analysis_source.py')
files=sorted(p for p in O.rglob('*') if p.is_file());(O/'checksums.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(O))+'\n' for p in files))
print(json.dumps({'report':str(O/'ANALYSIS_REPORT.md'),'proposal':str(O/'PROPOSED_SINGLE_CONTROL.json'),'figures':len(figrows),'SCC_no_sample_batch_probability':rows[3]['exact_probability_class_absent_batch16'],'class_data':rows,'input_files_preserved':len(snap),'checkpoints_changed':False,'training_performed':False},indent=2))
