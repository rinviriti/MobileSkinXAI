# Final_E1_Source_Only_Test_Results

E1: frozen ISIC-only training; PAD is zero-shot transfer. Same 1,421 ISIC and 205 PAD declared test units across all configurations.

| Model | Evaluation dataset | Declared units | Accuracy (%) | Macro-F1 | Balanced accuracy (%) | NV recall (%) | MEL recall (%) | BCC recall (%) | SCC recall (%) | Macro-AUROC | Macro-AUPRC (AP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Small | ISIC | 1421 | 84.24 | 0.6541 | 65.62 | 89.72 | 71.58 | 80.11 | 21.05 | 0.9471 | 0.7367 |
| MobileNetV3-Small | PAD | 205 | 51.22 | 0.3839 | 49.95 | 90.24 | 57.14 | 48.84 | 3.57 | 0.7674 | 0.4951 |
| EfficientNet-Lite0 | ISIC | 1421 | 81.56 | 0.6583 | 68.52 | 87.65 | 67.37 | 69.06 | 50.00 | 0.9387 | 0.7177 |
| EfficientNet-Lite0 | PAD | 205 | 34.63 | 0.3214 | 40.13 | 82.93 | 42.86 | 24.03 | 10.71 | 0.6935 | 0.4725 |
| MobileNetV4-Conv-Small | ISIC | 1421 | 85.22 | 0.6853 | 67.48 | 91.70 | 62.11 | 84.53 | 31.58 | 0.9430 | 0.7398 |
| MobileNetV4-Conv-Small | PAD | 205 | 60.49 | 0.4032 | 43.78 | 87.80 | 14.29 | 65.89 | 7.14 | 0.7185 | 0.4669 |
| ConvNeXt-Tiny | ISIC | 1421 | 86.70 | 0.7168 | 68.96 | 93.38 | 62.63 | 85.64 | 34.21 | 0.9573 | 0.7710 |
| ConvNeXt-Tiny | PAD | 205 | 49.27 | 0.3642 | 48.86 | 100.00 | 42.86 | 41.86 | 10.71 | 0.7925 | 0.5013 |
| MobileSkinXAI-Enhanced | ISIC | 1421 | 84.66 | 0.6927 | 69.92 | 89.43 | 70.53 | 82.87 | 36.84 | 0.9490 | 0.7532 |
| MobileSkinXAI-Enhanced | PAD | 205 | 61.46 | 0.4239 | 48.38 | 92.68 | 28.57 | 65.12 | 7.14 | 0.7791 | 0.5059 |
| MobileSkinXAI-Enhanced (LR Control) | ISIC | 1421 | 86.49 | 0.7103 | 70.14 | 92.89 | 63.68 | 84.53 | 39.47 | 0.9500 | 0.7566 |
| MobileSkinXAI-Enhanced (LR Control) | PAD | 205 | 58.54 | 0.4232 | 51.71 | 95.12 | 42.86 | 58.14 | 10.71 | 0.7800 | 0.5124 |

