# Actual Flutter interpreter parity — Linux PASS

This was NOT only a Dart CLI or Python test. Flutter3.50.0's flutter_test engine loaded the real bundled trained model using tflite_flutter0.12.1, checked float32NHWC[1,224,224,3] and float32logits[1,4], invoked native JPEG preprocessing and the actual interpreter, and met all fixed maximum/mean logit/probability bounds on the original four cases. The original melanoma example predictsMEL again.

The test host C runtime is TfLite2.5.0, from the original plugin maintainer's published Linux release; URL/SHA256 are saved in LINUX_RUNTIME_PROVENANCE.json. This binary is a TEMPORARY HOST-TEST dependency only, not shipped with Android/iOS. The pinned mobile plugin resolves TensorFlowLite2.12.0 natively; those mobile binaries have not been exercised here. Linux success is not an iPhone compatibility claim.

Maximum Flutter logit error:2.670288086e-05; maximum Flutter probability error:3.751010681e-06. All four argmax predictions agree. Source checkpoint/model hashes, exact model asset loading and class order are verified. No unsupported Flex/custom ops, teacher, reliability gate, softmax layer or calibration is added to the model.

|Image|Actual Flutter probabilities NV/MEL/BCC/SCC|Max probability error|
|---|---|---:|
|ISIC_0012660_downsampled|0.99674727, 0.00325217, 0.00000044, 0.00000011|1.90316627e-08|
|ISIC_0013258_downsampled|0.47324287, 0.52652585, 0.00021582, 0.00001546|4.79495814e-07|
|ISIC_0024504|0.00970985, 0.00255022, 0.72025251, 0.26748743|3.75101068e-06|
|ISIC_0025811|0.00021437, 0.00018185, 0.57900015, 0.42060363|3.02636211e-06|


Logs: flutter_runtime_test_v4.log (isolated harness PASS); app_test_final.log (actual app's production service and production-package codec build PASS,5tests). Unit coverage includes stable softmax/mapping, normalization, empty/disposed calls, valid trained inference, malformed/unsupported image rejection and original home/theme smoke test. The5tests do not include a physical mobile run.

Run the app suite from app/flutter with MSX_GOLDEN_MANIFEST pointing to the preserved original four-image manifest and MSX_CODEC_LIBRARY pointing to the built native library. Linux tflite_flutter loads its C runtime under the Flutter engine's blobs directory. Without the two dart-defines, the heavyweight host golden test explicitly skips instead of fabricating a pass.

Device test prepared at integration_test/model_device_test.dart. It contains the SAME original four JPEG byte sequences (hash-checked), invokes the actual packaged library/interpreter with no host-path override, and checks original tolerances. It is NOT RUN in WSL. Test-only golden_cases.dart is not imported by lib/main.dart or registered as production image assets.
