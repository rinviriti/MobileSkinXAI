# Prior_Ablations_and_Teacher_Qualification

These are separate studies, not one pooled experiment. Original source-only teacher qualification and the failed joint-study qualification are different events. No new method superiority or clinical safety is established.

| Study | Model | Configuration | Population | Measure | Value | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Original enhancement ablation | MobileNetV4-Conv-Small | Original baseline | ISIC image-level development validation | Macro-F1 | 0.6832 | Single-seed policy contrast; component and realized-budget limitations apply |
| Original enhancement ablation | MobileSkinXAI-Enhanced | Original full recipe | ISIC image-level development validation | Macro-F1 | 0.6912 | Single-seed policy contrast; component and realized-budget limitations apply |
| Original enhancement ablation | MobileSkinXAI-Enhanced | Without hard/continuous reliability weighting | ISIC image-level development validation | Macro-F1 | 0.6905 | Single-seed policy contrast; component and realized-budget limitations apply |
| Original enhancement ablation | MobileNetV4-Conv-Small | Teacher-free student consistency | ISIC image-level development validation | Macro-F1 | 0.6763 | Single-seed policy contrast; component and realized-budget limitations apply |
| Original enhancement ablation | MobileNetV4-Conv-Small | Separate source-augmentation intervention | ISIC image-level development validation | Macro-F1 | 0.6845 | Single-seed policy contrast; component and realized-budget limitations apply |
| CA-MACL single-seed development | MobileNetV4-Conv-Small | Reference | ISIC image-level development validation | Clean / mean synthetic-shift macro-F1 | 0.6730 / 0.6635 | Separate study; synthetic shifts do not establish smartphone generalization |
| CA-MACL single-seed development | MobileNetV4-Conv-Small | Acquisition augmentation | ISIC image-level development validation | Clean / mean synthetic-shift macro-F1 | 0.6677 / 0.6628 | Separate study; synthetic shifts do not establish smartphone generalization |
| CA-MACL single-seed development | MobileNetV4-Conv-Small | Acquisition + multi-scale fusion | ISIC image-level development validation | Clean / mean synthetic-shift macro-F1 | 0.6659 / 0.6530 | Separate study; synthetic shifts do not establish smartphone generalization |
| CA-MACL single-seed development | MobileNetV4-Conv-Small | Acquisition + fusion + JS consistency | ISIC image-level development validation | Clean / mean synthetic-shift macro-F1 | 0.6687 / 0.6589 | Separate study; synthetic shifts do not establish smartphone generalization |
| CA-MACL single-seed development | MobileNetV4-Conv-Small | Acquisition + fusion + JS + class-aware contrastive | ISIC image-level development validation | Clean / mean synthetic-shift macro-F1 | 0.6862 / 0.6705 | Separate study; synthetic shifts do not establish smartphone generalization |
| CA-MACL matched three seeds | MobileNetV4-Conv-Small | Reference | ISIC image-level development validation | Mean clean macro-F1 ± sample SD | 0.6680 ± 0.0052 | All seeds retained; three seeds do not establish statistical significance |
| CA-MACL matched three seeds | MobileNetV4-Conv-Small | Acquisition + fusion + JS consistency | ISIC image-level development validation | Mean clean macro-F1 ± sample SD | 0.6653 ± 0.0086 | All seeds retained; three seeds do not establish statistical significance |
| CA-MACL matched three seeds | MobileNetV4-Conv-Small | Acquisition + fusion + JS + class-aware contrastive | ISIC image-level development validation | Mean clean macro-F1 ± sample SD | 0.6758 ± 0.0102 | All seeds retained; three seeds do not establish statistical significance |
| Joint-study teacher qualification | ConvNeXt-Tiny | Frozen teacher, before exploratory amendment | PAD declared-unit development validation | Teacher / baseline validation macro-F1 | 0.629633 / 0.723210 | FAILED original superiority rule; enhanced joint recipes used an explicit post-hoc exploratory amendment |

