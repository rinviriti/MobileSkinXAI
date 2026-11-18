# Six-model E1/E2 final results

**MobileSkinXAI: An Explainable Cross-Dataset Lightweight Deep Learning Framework for Mobile-Based Skin Cancer Screening.**

Status: completed under the explicitly authorized post-hoc weaker-teacher amendment, with documented M1 controller deviation and verified recovery. Original teacher-superiority qualification FAILED (M4 0.629633 versus M3 0.723210) and remains false. M5/M5-LR are exploratory and do not confirm the original superior-teacher hypothesis. No matched E2 no-teacher ablation exists, so any enhanced improvement cannot be attributed to distillation. Four remaining E2 configurations trained once; M1 recovered without retraining; completed M3 E2 and all six E1 checkpoints reused unchanged. All PAD findings are **POST-HOC EXPLORATORY**. E2 is supervised adaptation, not zero-shot generalization.

## Development selection versus test description

Top PAD validation configuration overall: MobileSkinXAI-Enhanced LR Control (MobileNetV4) (0.730528). The predeclared lightweight deployment candidate is **MobileSkinXAI-Enhanced LR Control (MobileNetV4)**, selected and locked before new test inference. Highest PAD test point estimate is MobileNetV4-Conv-Small (0.710042); that retrospective ranking does not replace the development-selected candidate. Highest E1 ISIC macro-F1: ConvNeXt-Tiny; highest E1 PAD macro-F1: MobileSkinXAI-Enhanced (MobileNetV4). Largest PAD improvement: EfficientNet-Lite0 (+38.00 pp).

## Adaptation and source-domain trade-offs

- MobileNetV3-Small: PAD macro-F1 0.3839 → 0.6762 (+29.23 pp), accuracy +30.73 pp, balanced accuracy +13.76 pp; MEL recall +0.00 pp, SCC recall +14.29 pp. ISIC macro-F1 changes -1.60 pp. Net correctly classified PAD lesions by NV/MEL/BCC/SCC: [-3, 0, 62, 4]; this separates majority-class accuracy gains from minority-class performance.
- EfficientNet-Lite0: PAD macro-F1 0.3214 → 0.7014 (+38.00 pp), accuracy +48.78 pp, balanced accuracy +24.41 pp; MEL recall +0.00 pp, SCC recall +21.43 pp. ISIC macro-F1 changes -1.79 pp. Net correctly classified PAD lesions by NV/MEL/BCC/SCC: [2, 0, 92, 6]; this separates majority-class accuracy gains from minority-class performance.
- MobileNetV4-Conv-Small: PAD macro-F1 0.4032 → 0.7100 (+30.68 pp), accuracy +19.51 pp, balanced accuracy +24.06 pp; MEL recall +57.14 pp, SCC recall +14.29 pp. ISIC macro-F1 changes -7.32 pp. Net correctly classified PAD lesions by NV/MEL/BCC/SCC: [0, 4, 32, 4]; this separates majority-class accuracy gains from minority-class performance.
- ConvNeXt-Tiny: PAD macro-F1 0.3642 → 0.6936 (+32.94 pp), accuracy +32.20 pp, balanced accuracy +19.15 pp; MEL recall +28.57 pp, SCC recall +10.71 pp. ISIC macro-F1 changes -3.55 pp. Net correctly classified PAD lesions by NV/MEL/BCC/SCC: [-6, 2, 67, 3]; this separates majority-class accuracy gains from minority-class performance.
- MobileSkinXAI-Enhanced (MobileNetV4): PAD macro-F1 0.4239 → 0.6722 (+24.84 pp), accuracy +19.02 pp, balanced accuracy +16.88 pp; MEL recall +42.86 pp, SCC recall +10.71 pp. ISIC macro-F1 changes -1.36 pp. Net correctly classified PAD lesions by NV/MEL/BCC/SCC: [-7, 3, 40, 3]; this separates majority-class accuracy gains from minority-class performance.
- MobileSkinXAI-Enhanced LR Control (MobileNetV4): PAD macro-F1 0.4232 → 0.6965 (+27.33 pp), accuracy +22.93 pp, balanced accuracy +12.79 pp; MEL recall +14.29 pp, SCC recall +14.29 pp. ISIC macro-F1 changes -4.64 pp. Net correctly classified PAD lesions by NV/MEL/BCC/SCC: [-6, 1, 48, 4]; this separates majority-class accuracy gains from minority-class performance.

## Failure modes

- MobileNetV3-Small: E1: largest off-diagonal BCC→MEL 44 lesions; E2: largest off-diagonal SCC→BCC 23 lesions.
- EfficientNet-Lite0: E1: largest off-diagonal BCC→MEL 73 lesions; E2: largest off-diagonal SCC→BCC 19 lesions.
- MobileNetV4-Conv-Small: E1: largest off-diagonal BCC→MEL 26 lesions; E2: largest off-diagonal SCC→BCC 22 lesions.
- ConvNeXt-Tiny: E1: largest off-diagonal BCC→NV 52 lesions; E2: largest off-diagonal SCC→BCC 22 lesions.
- MobileSkinXAI-Enhanced (MobileNetV4): E1: largest off-diagonal BCC→MEL 28 lesions; E2: largest off-diagonal SCC→BCC 23 lesions.
- MobileSkinXAI-Enhanced LR Control (MobileNetV4): E1: largest off-diagonal BCC→MEL 29 lesions; E2: largest off-diagonal SCC→BCC 21 lesions.

## Enhanced model and LR control

See ENHANCED_MODEL_TEACHER_ANALYSIS.md and ENHANCED_MODEL_TEACHER_ANALYSIS_ADDENDUM.md for the failed original qualification, amended rationale, actual gate/loss measurements and non-causal interpretation. Tables A–H in05_Four_Setting_Comparisons separately distinguish historical full-PAD and current fixed-subset evaluation. Historical scores cannot enter matched adaptation gains. No matched E2 no-teacher ablation exists.

The following secondary paired comparisons are descriptive, conditional on frozen models and pre-existing protocol differences. Intervals are patient-grouped on PAD and lesion-grouped on ISIC; no claim of statistical significance or causal novelty follows.

| comparison | dataset | analysis_level | metric | difference | CI_low | CI_high | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M5 minus M3 | PAD | lesion_primary | macro_f1 | -0.03781537515341882 | -0.10859831249638173 | 0.015537586934012877 | secondary exploratory fixed-model comparison; initialization/loss/budget confounding acknowledged |
| M5-LR minus M5 | PAD | lesion_primary | macro_f1 | 0.02424850316560856 | -0.07181119502117879 | 0.1166597055430877 | LR-only intended difference; identical parents/loss/views/sampling/seed; realized stopping budgets differ |


MobileSkinXAI-Enhanced (MobileNetV4) versus standard joint MobileNetV4: PAD test macro-F1 change -3.78 pp; PAD validation selection macro-F1 change -6.95 pp; additional own training 827.15s plus shared M3/M4 parent costs. These observations alone do not isolate enhancement causality or establish that the additional training complexity is justified.

MobileSkinXAI-Enhanced LR Control (MobileNetV4) versus standard joint MobileNetV4: PAD test macro-F1 change -1.36 pp; PAD validation selection macro-F1 change +0.73 pp; additional own training 713.19s plus shared M3/M4 parent costs. These observations alone do not isolate enhancement causality or establish that the additional training complexity is justified.

MobileNetV3-Small E2 lowest PAD recalls: SCC 5/28 (0.1786), MEL 4/7 (0.5714).

EfficientNet-Lite0 E2 lowest PAD recalls: SCC 9/28 (0.3214), MEL 3/7 (0.4286).

MobileNetV4-Conv-Small E2 lowest PAD recalls: SCC 6/28 (0.2143), MEL 5/7 (0.7143).

ConvNeXt-Tiny E2 lowest PAD recalls: SCC 6/28 (0.2143), MEL 5/7 (0.7143).

MobileSkinXAI-Enhanced (MobileNetV4) E2 lowest PAD recalls: SCC 5/28 (0.1786), MEL 5/7 (0.7143).

MobileSkinXAI-Enhanced LR Control (MobileNetV4) E2 lowest PAD recalls: SCC 7/28 (0.2500), MEL 4/7 (0.5714).

Descriptive parameter/MAC-versus-PAD-test point-estimate Pareto configurations: MobileNetV3-Small, MobileNetV4-Conv-Small. This is an exploratory efficiency description, not the development selection rule or a phone-latency measurement.

## Table 1. Fixed PAD split

| split | class | images | declared_lesions | patients_with_class | all_partition_patients |
| --- | --- | --- | --- | --- | --- |
| train | NV | 139 | 124 | 116 | 550 |
| train | MEL | 30 | 20 | 20 | 550 |
| train | BCC | 503 | 388 | 344 | 550 |
| train | SCC | 109 | 84 | 80 | 550 |
| validation | NV | 47 | 41 | 32 | 127 |
| validation | MEL | 10 | 7 | 7 | 127 |
| validation | BCC | 167 | 129 | 81 | 127 |
| validation | SCC | 37 | 28 | 25 | 127 |
| test | NV | 47 | 41 | 28 | 126 |
| test | MEL | 10 | 7 | 7 | 126 |
| test | BCC | 167 | 129 | 82 | 126 |
| test | SCC | 37 | 28 | 23 | 126 |


## Table 2. E1 source-only comparison

| Model | ISIC accuracy | ISIC macro-F1 | PAD accuracy | PAD macro-F1 | PAD balanced accuracy |
| --- | --- | --- | --- | --- | --- |
| MobileNetV3-Small | 0.8424 | 0.6541 | 0.5122 | 0.3839 | 0.4995 |
| EfficientNet-Lite0 | 0.8156 | 0.6583 | 0.3463 | 0.3214 | 0.4013 |
| MobileNetV4-Conv-Small | 0.8522 | 0.6853 | 0.6049 | 0.4032 | 0.4378 |
| ConvNeXt-Tiny | 0.8670 | 0.7168 | 0.4927 | 0.3642 | 0.4886 |
| MobileSkinXAI-Enhanced (MobileNetV4) | 0.8466 | 0.6927 | 0.6146 | 0.4239 | 0.4838 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | 0.8649 | 0.7103 | 0.5854 | 0.4232 | 0.5171 |


## Table 3. E2 joint-domain comparison

| Model | ISIC accuracy | ISIC macro-F1 | PAD accuracy | PAD macro-F1 | PAD balanced accuracy |
| --- | --- | --- | --- | --- | --- |
| MobileNetV3-Small | 0.8332 | 0.6380 | 0.8195 | 0.6762 | 0.6371 |
| EfficientNet-Lite0 | 0.8290 | 0.6403 | 0.8341 | 0.7014 | 0.6454 |
| MobileNetV4-Conv-Small | 0.8410 | 0.6121 | 0.8000 | 0.7100 | 0.6784 |
| ConvNeXt-Tiny | 0.8649 | 0.6813 | 0.8146 | 0.6936 | 0.6801 |
| MobileSkinXAI-Enhanced (MobileNetV4) | 0.8557 | 0.6791 | 0.8049 | 0.6722 | 0.6525 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | 0.8395 | 0.6639 | 0.8146 | 0.6965 | 0.6449 |


## Table 4. Four-setting comparison

| Model | configuration | Metric | E1 ISIC -> ISIC | E1 ISIC -> PAD | E2 ISIC+PAD -> ISIC | E2 ISIC+PAD -> PAD |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Small | M1 | accuracy | 0.8423645320197044 | 0.5121951219512195 | 0.8332160450387052 | 0.8195121951219512 |
| MobileNetV3-Small | M1 | macro_f1 | 0.6540523087742536 | 0.3838747014669557 | 0.6380152554101753 | 0.6761885775043669 |
| MobileNetV3-Small | M1 | weighted_f1 | 0.8439775811345769 | 0.5454534913618653 | 0.8289950219901278 | 0.789317714093067 |
| MobileNetV3-Small | M1 | balanced_accuracy | 0.6561634908576006 | 0.49948849363908926 | 0.6372677190377294 | 0.6370651351862355 |
| MobileNetV3-Small | M1 | macro_precision | 0.6930739859205359 | 0.4083749990781915 | 0.6518168087104014 | 0.7926816239316239 |
| MobileNetV3-Small | M1 | macro_recall | 0.6561634908576006 | 0.49948849363908926 | 0.6372677190377294 | 0.6370651351862355 |
| MobileNetV3-Small | M1 | MEL_recall | 0.7157894736842105 | 0.5714285714285714 | 0.49473684210526314 | 0.5714285714285714 |
| MobileNetV3-Small | M1 | SCC_recall | 0.21052631578947367 | 0.03571428571428571 | 0.2894736842105263 | 0.17857142857142858 |
| MobileNetV3-Small | M1 | NV_recall | 0.8972332015810277 | 0.9024390243902439 | 0.9140316205533597 | 0.8292682926829268 |
| MobileNetV3-Small | M1 | BCC_recall | 0.8011049723756906 | 0.4883720930232558 | 0.850828729281768 | 0.9689922480620154 |
| MobileNetV3-Small | M1 | macro_AUROC | 0.9470956689286963 | 0.7673757740913119 | 0.9276909168112286 | 0.863347783381772 |
| MobileNetV3-Small | M1 | macro_AUPRC_AP | 0.7367158200739877 | 0.4950633987091489 | 0.6566492041583146 | 0.7667640383171134 |
| EfficientNet-Lite0 | M2 | accuracy | 0.8156228008444757 | 0.3463414634146341 | 0.82899366643209 | 0.8341463414634146 |
| EfficientNet-Lite0 | M2 | macro_f1 | 0.6582769533421798 | 0.32143467631566314 | 0.6403379474119159 | 0.701386740014899 |
| EfficientNet-Lite0 | M2 | weighted_f1 | 0.8245387139511999 | 0.39728987868569376 | 0.8296615530739317 | 0.8156469333646406 |
| EfficientNet-Lite0 | M2 | balanced_accuracy | 0.6851935396929202 | 0.4013231639791481 | 0.6455806995323348 | 0.6453842881452071 |
| EfficientNet-Lite0 | M2 | macro_precision | 0.6444432461625951 | 0.43815882528102434 | 0.6699038665285145 | 0.843484555984556 |
| EfficientNet-Lite0 | M2 | macro_recall | 0.6851935396929202 | 0.4013231639791481 | 0.6455806995323348 | 0.6453842881452071 |
| EfficientNet-Lite0 | M2 | MEL_recall | 0.6736842105263158 | 0.42857142857142855 | 0.6421052631578947 | 0.42857142857142855 |
| EfficientNet-Lite0 | M2 | SCC_recall | 0.5 | 0.10714285714285714 | 0.21052631578947367 | 0.32142857142857145 |
| EfficientNet-Lite0 | M2 | NV_recall | 0.8764822134387352 | 0.8292682926829268 | 0.8843873517786561 | 0.8780487804878049 |
| EfficientNet-Lite0 | M2 | BCC_recall | 0.6906077348066298 | 0.24031007751937986 | 0.8453038674033149 | 0.9534883720930233 |
| EfficientNet-Lite0 | M2 | macro_AUROC | 0.9386996516788648 | 0.6935359684477679 | 0.9382796107727238 | 0.895044536676803 |
| EfficientNet-Lite0 | M2 | macro_AUPRC_AP | 0.7177290610535124 | 0.4724606050750603 | 0.697789693112768 | 0.7964579323581624 |
| MobileNetV4-Conv-Small | M3 | accuracy | 0.8522167487684729 | 0.6048780487804878 | 0.8409570724841661 | 0.8 |
| MobileNetV4-Conv-Small | M3 | macro_f1 | 0.6853367128965776 | 0.40323950766747374 | 0.612104262188053 | 0.7100417744850812 |
| MobileNetV4-Conv-Small | M3 | weighted_f1 | 0.850932382905336 | 0.6159856690092325 | 0.8308486314594226 | 0.7814219173581314 |
| MobileNetV4-Conv-Small | M3 | balanced_accuracy | 0.6747855050243258 | 0.43781230586392245 | 0.5925778848065724 | 0.67839923831132 |
| MobileNetV4-Conv-Small | M3 | macro_precision | 0.7242332547875856 | 0.4832833608653346 | 0.6664129025035599 | 0.7688311688311689 |
| MobileNetV4-Conv-Small | M3 | macro_recall | 0.6747855050243258 | 0.43781230586392245 | 0.5925778848065724 | 0.67839923831132 |
| MobileNetV4-Conv-Small | M3 | MEL_recall | 0.6210526315789474 | 0.14285714285714285 | 0.4842105263157895 | 0.7142857142857143 |
| MobileNetV4-Conv-Small | M3 | SCC_recall | 0.3157894736842105 | 0.07142857142857142 | 0.15789473684210525 | 0.21428571428571427 |
| MobileNetV4-Conv-Small | M3 | NV_recall | 0.9169960474308301 | 0.8780487804878049 | 0.9436758893280632 | 0.8780487804878049 |
| MobileNetV4-Conv-Small | M3 | BCC_recall | 0.8453038674033149 | 0.6589147286821705 | 0.7845303867403315 | 0.9069767441860465 |
| MobileNetV4-Conv-Small | M3 | macro_AUROC | 0.943017476254147 | 0.7185246044363349 | 0.9183813031045367 | 0.8911381667321281 |
| MobileNetV4-Conv-Small | M3 | macro_AUPRC_AP | 0.7397644972374211 | 0.46686253380259174 | 0.6546944676947789 | 0.7854770503850407 |
| ConvNeXt-Tiny | M4 | accuracy | 0.8669950738916257 | 0.4926829268292683 | 0.8648838845883181 | 0.8146341463414634 |
| ConvNeXt-Tiny | M4 | macro_f1 | 0.7168371316148643 | 0.3641819971231736 | 0.6812924315835279 | 0.693561872909699 |
| ConvNeXt-Tiny | M4 | weighted_f1 | 0.8641795468854423 | 0.4985557592845972 | 0.8611959317143205 | 0.7946515485221741 |
| ConvNeXt-Tiny | M4 | balanced_accuracy | 0.6896422775487405 | 0.48857973421926915 | 0.6690481451428453 | 0.6800536153202064 |
| ConvNeXt-Tiny | M4 | macro_precision | 0.784474250389661 | 0.46903669724770647 | 0.7222190855381185 | 0.7279830851259422 |
| ConvNeXt-Tiny | M4 | macro_recall | 0.6896422775487405 | 0.48857973421926915 | 0.6690481451428453 | 0.6800536153202064 |
| ConvNeXt-Tiny | M4 | MEL_recall | 0.6263157894736842 | 0.42857142857142855 | 0.6421052631578947 | 0.7142857142857143 |
| ConvNeXt-Tiny | M4 | SCC_recall | 0.34210526315789475 | 0.10714285714285714 | 0.23684210526315788 | 0.21428571428571427 |
| ConvNeXt-Tiny | M4 | NV_recall | 0.9337944664031621 | 1.0 | 0.9298418972332015 | 0.8536585365853658 |
| ConvNeXt-Tiny | M4 | BCC_recall | 0.856353591160221 | 0.4186046511627907 | 0.8674033149171271 | 0.937984496124031 |
| ConvNeXt-Tiny | M4 | macro_AUROC | 0.9573006700834447 | 0.7925347482215171 | 0.9482438742583814 | 0.9104924779406371 |
| ConvNeXt-Tiny | M4 | macro_AUPRC_AP | 0.7710164981985899 | 0.5012817964622012 | 0.7424331490758975 | 0.7930894704805935 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | accuracy | 0.8465869106263195 | 0.6146341463414634 | 0.8557353976073188 | 0.8048780487804879 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | macro_f1 | 0.6927236378496588 | 0.4238603425559947 | 0.6791463888237683 | 0.6722263993316624 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | weighted_f1 | 0.850561889540726 | 0.6291747164111958 | 0.8523371914091306 | 0.7752377692197975 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | balanced_accuracy | 0.6991705667494572 | 0.48378372903330363 | 0.6732487842890261 | 0.6525487534775679 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | macro_precision | 0.6978555843671878 | 0.47971981491423166 | 0.6880160621383105 | 0.7614805537599655 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | macro_recall | 0.6991705667494572 | 0.48378372903330363 | 0.6732487842890261 | 0.6525487534775679 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | MEL_recall | 0.7052631578947368 | 0.2857142857142857 | 0.5473684210526316 | 0.7142857142857143 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | SCC_recall | 0.3684210526315789 | 0.07142857142857142 | 0.39473684210526316 | 0.17857142857142858 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | NV_recall | 0.8942687747035574 | 0.926829268292683 | 0.9387351778656127 | 0.7560975609756098 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | BCC_recall | 0.8287292817679558 | 0.6511627906976745 | 0.8121546961325967 | 0.9612403100775194 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | macro_AUROC | 0.9490105491276909 | 0.7791253415429893 | 0.9442683956869357 | 0.8913980205156091 |
| MobileSkinXAI-Enhanced (MobileNetV4) | M5 | macro_AUPRC_AP | 0.7531784759249316 | 0.5059278885064131 | 0.7211258931910385 | 0.77338546565346 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | accuracy | 0.8648838845883181 | 0.5853658536585366 | 0.8395496129486277 | 0.8146341463414634 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | macro_f1 | 0.7102706624263553 | 0.4231598628088457 | 0.6638767283132739 | 0.696474902497271 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | weighted_f1 | 0.8639668114386581 | 0.6013918155072192 | 0.8399658291931766 | 0.7933999255682839 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | balanced_accuracy | 0.7014341424281119 | 0.5170822866866542 | 0.6573479111378779 | 0.6449487480755206 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | macro_precision | 0.7263696873931575 | 0.47633349796271146 | 0.7103477324858695 | 0.8067206477732793 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | macro_recall | 0.7014341424281119 | 0.5170822866866542 | 0.6573479111378779 | 0.6449487480755206 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | MEL_recall | 0.6368421052631579 | 0.42857142857142855 | 0.6473684210526316 | 0.5714285714285714 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | SCC_recall | 0.39473684210526316 | 0.10714285714285714 | 0.2631578947368421 | 0.25 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | NV_recall | 0.9288537549407114 | 0.9512195121951219 | 0.9011857707509882 | 0.8048780487804879 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | BCC_recall | 0.8453038674033149 | 0.5813953488372093 | 0.8176795580110497 | 0.9534883720930233 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | macro_AUROC | 0.9499909094289742 | 0.7800078999240178 | 0.9357294079531255 | 0.8995494003618675 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | M5-LR | macro_AUPRC_AP | 0.7565944611178379 | 0.5124108368447862 | 0.7075093383767856 | 0.7890221487113569 |


## Table 5. Source-target differences

| Model | E1 ISIC F1 | E1 PAD F1 | E1 gap pp | E2 ISIC F1 | E2 PAD F1 | E2 difference pp | PAD gain pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Small | 0.6541 | 0.3839 | 27.0178 | 0.6380 | 0.6762 | -3.8173 | 29.2314 |
| EfficientNet-Lite0 | 0.6583 | 0.3214 | 33.6842 | 0.6403 | 0.7014 | -6.1049 | 37.9952 |
| MobileNetV4-Conv-Small | 0.6853 | 0.4032 | 28.2097 | 0.6121 | 0.7100 | -9.7938 | 30.6802 |
| ConvNeXt-Tiny | 0.7168 | 0.3642 | 35.2655 | 0.6813 | 0.6936 | -1.2269 | 32.9380 |
| MobileSkinXAI-Enhanced (MobileNetV4) | 0.6927 | 0.4239 | 26.8863 | 0.6791 | 0.6722 | 0.6920 | 24.8366 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | 0.7103 | 0.4232 | 28.7111 | 0.6639 | 0.6965 | -3.2598 | 27.3315 |


## Table 6. PAD improvements and paired uncertainty

| Model | PAD F1 gain pp | 95% patient CI pp | Accuracy gain pp | Balanced accuracy gain pp |
| --- | --- | --- | --- | --- |
| MobileNetV3-Small | 29.2314 | [15.93, 39.19] | 30.7317 | 13.7577 |
| EfficientNet-Lite0 | 37.9952 | [22.96, 49.88] | 48.7805 | 24.4061 |
| MobileNetV4-Conv-Small | 30.6802 | [19.26, 39.78] | 19.5122 | 24.0587 |
| ConvNeXt-Tiny | 32.9380 | [21.85, 44.03] | 32.1951 | 19.1474 |
| MobileSkinXAI-Enhanced (MobileNetV4) | 24.8366 | [12.74, 34.81] | 19.0244 | 16.8765 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | 27.3315 | [15.38, 36.94] | 22.9268 | 12.7866 |


## Table 7. PAD melanoma/SCC supports and performance

| Model | Condition | Class | correct | support | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Small | E1 | MEL | 4 | 7 | 0.06557377049180328 | 0.5714285714285714 | 0.11764705882352942 |
| MobileNetV3-Small | E1 | SCC | 1 | 28 | 0.1111111111111111 | 0.03571428571428571 | 0.05405405405405406 |
| MobileNetV3-Small | E2 | MEL | 4 | 7 | 0.8 | 0.5714285714285714 | 0.6666666666666666 |
| MobileNetV3-Small | E2 | SCC | 5 | 28 | 0.625 | 0.17857142857142858 | 0.2777777777777778 |
| EfficientNet-Lite0 | E1 | MEL | 3 | 7 | 0.0297029702970297 | 0.42857142857142855 | 0.05555555555555556 |
| EfficientNet-Lite0 | E1 | SCC | 3 | 28 | 0.3 | 0.10714285714285714 | 0.15789473684210525 |
| EfficientNet-Lite0 | E2 | MEL | 3 | 7 | 1.0 | 0.42857142857142855 | 0.6 |
| EfficientNet-Lite0 | E2 | SCC | 9 | 28 | 0.6428571428571429 | 0.32142857142857145 | 0.4285714285714286 |
| MobileNetV4-Conv-Small | E1 | MEL | 1 | 7 | 0.02857142857142857 | 0.14285714285714285 | 0.04761904761904761 |
| MobileNetV4-Conv-Small | E1 | SCC | 2 | 28 | 0.5 | 0.07142857142857142 | 0.125 |
| MobileNetV4-Conv-Small | E2 | MEL | 5 | 7 | 1.0 | 0.7142857142857143 | 0.8333333333333333 |
| MobileNetV4-Conv-Small | E2 | SCC | 6 | 28 | 0.4 | 0.21428571428571427 | 0.27906976744186046 |
| ConvNeXt-Tiny | E1 | MEL | 3 | 7 | 0.1 | 0.42857142857142855 | 0.16216216216216217 |
| ConvNeXt-Tiny | E1 | SCC | 3 | 28 | 0.5 | 0.10714285714285714 | 0.17647058823529413 |
| ConvNeXt-Tiny | E2 | MEL | 5 | 7 | 0.7142857142857143 | 0.7142857142857143 | 0.7142857142857143 |
| ConvNeXt-Tiny | E2 | SCC | 6 | 28 | 0.42857142857142855 | 0.21428571428571427 | 0.2857142857142857 |
| MobileSkinXAI-Enhanced (MobileNetV4) | E1 | MEL | 2 | 7 | 0.05128205128205128 | 0.2857142857142857 | 0.08695652173913043 |
| MobileSkinXAI-Enhanced (MobileNetV4) | E1 | SCC | 2 | 28 | 0.4 | 0.07142857142857142 | 0.12121212121212122 |
| MobileSkinXAI-Enhanced (MobileNetV4) | E2 | MEL | 5 | 7 | 0.7142857142857143 | 0.7142857142857143 | 0.7142857142857143 |
| MobileSkinXAI-Enhanced (MobileNetV4) | E2 | SCC | 5 | 28 | 0.625 | 0.17857142857142858 | 0.2777777777777778 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | E1 | MEL | 3 | 7 | 0.07692307692307693 | 0.42857142857142855 | 0.13043478260869565 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | E1 | SCC | 3 | 28 | 0.42857142857142855 | 0.10714285714285714 | 0.17142857142857143 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | E2 | MEL | 4 | 7 | 1.0 | 0.5714285714285714 | 0.7272727272727273 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | E2 | SCC | 7 | 28 | 0.5384615384615384 | 0.25 | 0.3414634146341463 |


## Table 8. Complexity and training resource accounting

| Model | Parameters M | MAC M | Own training s | Prescribed training s | Shared parent s | Best/stop epoch |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Small | 1.5220 | 55.4905 | 731.3308 | 483.0973 | 0.0000 | 10/16 |
| EfficientNet-Lite0 | 3.3761 | 383.9127 | 623.8306 | 623.8306 | 0.0000 | 13/19 |
| MobileNetV4-Conv-Small | 2.4981 | 184.7307 | 335.2489 | 335.2489 | 0.0000 | 7/13 |
| ConvNeXt-Tiny | 27.8232 | 4454.7663 | 1385.3049 | 1385.3049 | 0.0000 | 25/31 |
| MobileSkinXAI-Enhanced (MobileNetV4) | 2.4981 | 184.7307 | 827.1533 | 827.1533 | 1720.5539 | 10/16 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | 2.4981 | 184.7307 | 713.1870 | 713.1870 | 1720.5539 | 8/14 |


## Table 9. Original enhancement versus LR control

| Model | Experiment | ISIC F1 | PAD F1 | PAD MEL recall | PAD SCC recall |
| --- | --- | --- | --- | --- | --- |
| MobileSkinXAI-Enhanced (MobileNetV4) | E1 | 0.6927 | 0.4239 | 0.2857 | 0.0714 |
| MobileSkinXAI-Enhanced (MobileNetV4) | E2 | 0.6791 | 0.6722 | 0.7143 | 0.1786 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | E1 | 0.7103 | 0.4232 | 0.4286 | 0.1071 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | E2 | 0.6639 | 0.6965 | 0.5714 | 0.2500 |


## Table 10. Supported findings and limitations

| Claim | Evidence | Limit |
| --- | --- | --- |
| Joint-domain adaptation on this fixed PAD population | Paired E1/E2 point estimates and patient-grouped intervals | Post-hoc, one seed, no untouched external replication |
| Zero-shot transfer | Only E1 qualifies | E2 has labeled PAD training exposure |
| Novel algorithm or clinical screening safety | Not established | Standard supervised joint training; only seven PAD test MEL lesion units |
| Enhanced inference efficiency | Same MobileNetV4 graph; teacher training-only | Additional parent and own training costs; no final TFLite export or phone benchmark |


## Interpretation and deployment

Joint-domain exposure provides evidence of performance on held-out patients from the same previously observed target dataset, with explicit class-wise and source-forgetting trade-offs above. It does not establish performance on a new smartphone cohort or clinical screening utility. A large BCC support can dominate accuracy; macro-F1 and minority counts must be considered alongside it. The validation-selected lightweight candidate remains fixed; test rankings only describe outcomes. ConvNeXt is the larger reference. Enhanced inference excludes its teacher but training costs include student/teacher parents separately. Conv/Linear MACs count one multiply-accumulate as one MAC for one224×224image; norm, activation, pooling, softmax, memory and runtime effects are excluded. Checkpoint sizes include training-state content and are not TFLite model sizes. No phone latency is inferred.

## Integrity, figures and reproducibility

All final audit checks passed with the documented M1 deviation. Exact checkpoint paths/hashes are in00_Protocol/FINAL_EVALUATION_LOCK.json andMODEL_COMPLEXITY.csv. The approved frozen manifests remain byte-identical. Primary scoring averages raw uncalibrated softmax across predeclared capture representatives within each declared lesion; image-level supplements are separate. No calibration was fitted. Undefined per-class precision with no predictions is zero; all four true classes are present, so AUROC/AP are computed. No historical full-PAD score is substituted for the fixed20%test result.

FIGURE_INDEX.csv contains titles, captions, source hashes and paths for every new figure family; each has300DPI PNG,SVG,PDF. Reused E1 figures and their original case evidence are listed separately in09_Explainability/E1_REUSED_PROVENANCE.csv. Plot data and scripts are retained;06_Confusion_Matrices and07_ROC_PR contain actual measured plots. Use the shortlist rather than repeating appendix panels.

See13_Reports/M1_RECOVERY_REPORT.md,13_Reports/TRAINING_CONTROLLER_FIX_REPORT.md,13_Reports/FINAL_E2_TRAINING_STATUS.md,13_Reports/FINAL_INTEGRITY_AUDIT.csv,EXPERIMENT_PROVENANCE.md andLIMITATIONS_AND_NOVELTY.md.
