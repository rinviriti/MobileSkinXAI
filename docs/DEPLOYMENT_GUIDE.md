# Deployment guide

Use the selected trained Float32 file already registered in app/flutter/assets/models/e2_lr_control_v20261118/. Model/input/class/normalization hashes and signatures are in its JSON companions and deployment/verification/MODEL_PROVENANCE.json. The canonical binary is stored once; deployment/tflite contains its manifest, not a duplicate or placeholder.

## Linux engineering verification

Use Flutter3.50.0/Dart3.14.0, CMake>=3.16, Ninja/Clang or a supported C compiler. From root run prepare_goldens.py, then build the native codec with `cmake -S app/flutter/packages/mobileskinxai_codec/src -B .private/codec -G Ninja` and `cmake --build .private/codec --target mobileskinxai_codec --parallel4` (use `--parallel 4`). tflite_flutter0.12.1 additionally needs its LinuxC runtime under the Flutter engine's blobs directory; CI downloads a pinned upstream Linux2.5binary for HOST tests only. It is not a mobile dependency. Native mobile dependencies are resolved by the plugin.

From app/flutter run flutter pub get, flutter analyze (or --no-fatal-infos for declared legacy import infos) and flutter test. To run all5tests including real inference, pass `--dart-define=MSX_GOLDEN_MANIFEST=<repo>/.private/golden_manifest.json` and `--dart-define=MSX_CODEC_LIBRARY=<repo>/.private/codec/libmobileskinxai_codec.so`. No private patient metadata is required for these licensed fixture tests. Without those definitions, the heavy host test visibly skips; do not count it as executed.

## macOS / iPhone

Install the recorded Flutter SDK, current compatible Xcode/CocoaPods/CMake and check flutter doctor. From app/flutter run `bash packages/mobileskinxai_codec/build_ios_codec.sh` BEFORE pod install. This emits anarm64device/simulator XCFramework; existing frameworks are preserved. A complete GitHub release ZIP may include the verified CI-built framework and therefore differs from the normal Source code ZIP.

Run flutter pub get; enter ios/and run pod install, then open Runner.xcworkspace. Runner/Podfile targetiOS15. Configure a development team/bundleID/signing and trusted physical iPhone with DeveloperMode. Camera/photo purpose strings are included. Run `flutter test integration_test/model_device_test.dart -d <iphone-id>` for actual packaged codec/interpreter goldens, then flutter run. Save output/device/runtime/model hashes and actual latency. No iPhone deployment or timing is claimed from WSL or an unsigned CI build.

The existing Analyze action returns four real uncalibrated softmax scores to the existing ResultsScreen. Preview remains; errors return no fabricated result. Heatmap stays absent. HEIC/HEIF is rejected: do not add invisible conversion. Verify the iPhone picker's delivered source bytes and representation, including orientation, rather than assuming Photos/camera output matches originalJPEG. 8-bitPNG/JPEG are the tested adapter paths; CMYK/YCCKJPEG,16-bitPNG and other unverified formats fail explicitly.

Never replace the native codec with a generic resize/JPEG pipeline, omit the historical JPEG roundtrip, divide only by255, apply a second softmax, choose a model on test outcomes or loosen thresholds. Float16failed and INT8was not calibrated. Details and original measured parity are in deployment/verification/PREPROCESSING_PARITY_FINAL.md and TFLITE_RUNTIME_PARITY.md.
