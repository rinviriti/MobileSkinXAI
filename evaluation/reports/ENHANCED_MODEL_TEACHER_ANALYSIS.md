# Enhanced-model teacher analysis

Amendment: TEACHER_AMENDMENT_20261118T003127Z. Original qualification **FAILED**: frozen M4 validation macro-F1 0.629633 versus joint M3 0.723210. The teacher is permitted under an explicit post-hoc exploratory amendment; it never qualified under the original superior-teacher hypothesis.

| Model | Best additional epoch | Stop additional epoch | PAD validation F1 | PAD test F1 | PAD test minus joint M3 F1 pp | Training seconds | Best-epoch CE | Best-epoch weighted KD | Best-epoch weighted JS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileSkinXAI-Enhanced (MobileNetV4) | 10 | 16 | 0.6537 | 0.6722 | -3.7815 | 827.1533 | 0.2420 | 0.6152 | 0.0005 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | 8 | 14 | 0.7305 | 0.6965 | -1.3567 | 713.1870 | 0.3494 | 0.8598 | 0.0009 |

## Actual training guidance

| Model | Class | Accepted presentations | Total presentations | Acceptance |
| --- | --- | --- | --- | --- |
| MobileSkinXAI-Enhanced (MobileNetV4) | NV | 67038 | 71519 | 0.9373 |
| MobileSkinXAI-Enhanced (MobileNetV4) | MEL | 19759 | 24297 | 0.8132 |
| MobileSkinXAI-Enhanced (MobileNetV4) | BCC | 70921 | 74236 | 0.9553 |
| MobileSkinXAI-Enhanced (MobileNetV4) | SCC | 13649 | 16060 | 0.8499 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | NV | 58688 | 62623 | 0.9372 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | MEL | 17218 | 21224 | 0.8113 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | BCC | 62074 | 64944 | 0.9558 |
| MobileSkinXAI-Enhanced LR Control (MobileNetV4) | SCC | 11943 | 14057 | 0.8496 |

Gate acceptance uses only training views and labels with the original confidence, entropy, agreement and correctness tests. The CSV also records actual KD/JS scales and mean reliability mass by epoch; low or zero accepted mass is reported as measured. Acceptance or nonzero gradients do not prove useful causal guidance. The same teacher checkpoint was frozen for both runs.

## Controlled comparison and attribution

| comparison | dataset | analysis_level | metric | difference | CI_low | CI_high | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M5-LR minus M5 | PAD | lesion_primary | macro_f1 | 0.02424850316560856 | -0.07181119502117879 | 0.1166597055430877 | secondary exploratory fixed-model comparison; initialization/loss/budget confounding acknowledged |

The fixed standard-versus-enhanced supervised weighting differs as recorded below. All values use training data only. Domain-balanced resampling changes effective presentation frequencies; these weights cannot be interpreted as an isolated teacher intervention.

| Class | Standard E2 moderated weight | Enhanced inverse-frequency weight |
| --- | --- | --- |
| NV | 0.8080 | 0.4448 |
| MEL | 1.3791 | 1.1750 |
| BCC | 0.7912 | 1.3481 |
| SCC | 1.7021 | 6.2946 |

The M5-LR sensitivity control changes head/backbone rates from1e-4/1e-5to1e-5/1e-6. Parents, exact views/order, class weights, gates, losses, seed and selection rule match; realized stopping budgets can differ. There is **no matched joint-domain M5-without-teacher ablation**. Any M5-versus-M3 gain or loss is confounded by additional training, two views, different supervised weights and initialization policy. Neither a positive metric difference nor accepted teacher guidance proves the teacher caused an improvement. Negative results remain part of the study.

All PAD findings are POST-HOC EXPLORATORY. Full-PAD historical source-only scores use a different population and are not substituted into paired current-test effects. Seven melanoma and28SCCdeclared testlesions limit sensitivity estimates. These experiments do not establish novelty, external independence or clinical suitability.
