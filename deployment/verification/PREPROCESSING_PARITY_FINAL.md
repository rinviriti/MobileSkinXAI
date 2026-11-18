# Final preprocessing parity — PASS on tested development inputs

Exactly the original four source files were retained, including the melanoma flip case. Then28additional frozen development examples and26declared engineering fixtures were tested. No held-out test images or outcome-based sample replacement was used. Fixtures cover8EXIForientations,3chroma-subsampling modes, progressive JPEG, valid embedded sRGBICC, grayscale/RGBJPEG and portrait/landscape dimensions. Original identifiers/hashes are in the preserved prior golden manifest and additional_development_manifest.json; source/fixture hashes and image metadata are saved.

|Check|Before|After|
|---|---|---|
|Original four prepared RGB|maxerror15|Exact pixel identity, max/mean0|
|Normalized tensors|maxerror0.261438|Exact float32identity, max/mean0|
|Original four top-1|3/4|4/4|
|Original four maximum probability error|0.075583079|Actual Flutter:3.751010681e-06|
|Extended real development set|Not previously certified|32/32exact final pixels/tensors;100%top-1|
|Orientation/format engineering fixtures|Initial odd-height orientation3failure documented|26/26exact pixels/tensors after explicit mapping|

On32cases, original PyTorch logits versus Python LiteRT with Dart/native tensors: maxlogiterror3.910064697e-05, meanlogiterror1.108394645e-05, maxprobabilityerror9.975328723e-06, meanprobabilityerror5.444829101e-07. All original bounds pass.

Unchanged bounds: max/mean logit0.001/0.0001; max/mean probability0.0001/0.00001;100%top-1. Exact raw-to-prepared pixels are also required and observed. No tolerance was relaxed, no difficult case dropped and no calibration/model adjustment performed. The reference is CPUfloat32saved PyTorch logits and stable softmax; these are engineering comparisons, not new research scores.

The true-SCC golden case remains predicted BCC as in original PyTorch. Parity is not classification correctness. A pass on32source-development images/26codec fixtures does not prove compatibility on every camera/OS/format. Physical mobile AOT/native-interpreter and acquisition-path parity remain required.
