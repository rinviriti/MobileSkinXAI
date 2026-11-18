# Frozen student export parity

Engineering results on32fixed development-validation images, not new classification-performance evidence. Reference: original PyTorch eval CPU float32 logits. Both default LiteRT CPU and undelegated built-in reference kernels were invoked on every image.

|Model|Weights|Max absolute logit error|Mean absolute logit error|Max probability error|Mean probability error|Top-1 agreement|Status|
|---|---|---:|---:|---:|---:|---:|---|
|MobileSkinXAI-Enhanced (LR Control)|float32|3.9100647e-05|1.1083946e-05|1.001358e-05|5.4146886e-07|100.0%|PASS|
|MobileSkinXAI-Enhanced (LR Control)|float16|0.075113297|0.015718967|0.010718763|0.00099044933|100.0%|FAIL|
|MobileNetV4-Conv-Small|float32|2.1457672e-05|6.3973639e-06|5.8710575e-06|6.6261038e-07|100.0%|PASS|
|MobileNetV4-Conv-Small|float16|0.034018517|0.0093248999|0.012632936|0.0012443676|100.0%|FAIL|

Exact thresholds were fixed in parity_protocol.json before predictions. Per-image logits, probabilities and margins: parity_report.csv. All32raw images reproduced frozen prepared RGB pixels exactly, including historical JPEG round trip.

Mobile Android/iOS interpreter and native preprocessing parity have NOT been exercised. CPU success is not an on-device claim. FP16 means stored weights; public input/output and activation contract remain float32. No clinical equivalence, full-validation performance or new research metrics were computed.
