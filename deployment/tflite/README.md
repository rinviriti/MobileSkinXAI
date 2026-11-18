# Canonical trained mobile artifact

The genuine validated Float32 model is stored once at ../../app/flutter/assets/models/e2_lr_control_v20261118/model_float32.tflite. Its labels.json, preprocessing.json and model_metadata.json are adjacent. No symlink, placeholder or failedFloat16model is substituted. Use those files with the native codec/inference service described in ../../docs/DEPLOYMENT_GUIDE.md.

SHA256:8b6e19f3d8d0dd9c3b798a2cf99c0525079b9de4ea2f5d1708dd928a718412f3;10073864bytes;NV/MEL/BCC/SCC. See ../verification/MODEL_PROVENANCE.json and verify_model.py for the preserved trained PyTorch reference and independent numerical checks.
