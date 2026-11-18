# Android/iOS compatibility status

|Platform/check|Status|
|---|---|
|WSL/Linux native codec|Built successfully, exact development-pixel parity|
|Actual Linux Flutter/tflite_flutter invocation|PASS, trained model and original four cases|
|Android Gradle/CMake linkage source|Configured, not NDK-built|
|Android APK build|BLOCKED: no Android SDK/NDK after required Flutter artifact downloads|
|iOS native codec XCFramework source/build script|Prepared; bash syntax checked; NOT Xcode-built|
|iOS podspec and symbol retention|Configured, not CocoaPods/Xcode verified|
|Physical iPhone/Android native interpreter|NOT RUN|
|HEIC/HEIF camera decoding|UNSUPPORTED; rejected, not silently converted by this adapter|

Flutter3.50.0/Dart3.14.0 were used for Linux checks. iOS Runner/Podfile target15.0; native build script emitsarm64device/simulator XCFrameworks on macOS. Intel simulator slices are not prepared. Install a current Xcode supported by the Mac/device and verify flutter doctor; do not infer a tested Xcode/device combination from Linux. The pinned tflite_flutter0.12.1 iOS podspec uses TensorFlowLiteSwift/Metal/CoreML2.12.0; Android resolves TensorFlowLite2.12.0. No accelerator is enabled by app code. CPU first, then any separate delegate verification.

The native package's source/license/configuration are portable C, but that is not proof of successful Android/iOS linkage or AOT numerical identity. Android16KB alignment is requested for the codec; the plugin's native TFLite dependency must also be checked on relevant devices. No Windows native packaging/build is certified; Windows development here means WSL/Linux.

The adapter accepts original JPEG and bounded8-bit PNG bytes. CMYK/YCCKJPEG,16-bit PNG, HEIC/HEIF and unsupported/corrupt inputs fail explicitly. The forced picker imageQuality option was removed; system photo-picker representation/conversion still needs actual iPhone verification. Do not assume that a JPEG returned by an OS picker proves the original photo was JPEG. No HEIC converter was added, and no HEIC fidelity claim is made.

Mobile runtime latency/memory/energy, actual camera/gallery acquisition fidelity, CocoaPods resolution, code signing and release-symbol retention remain unmeasured. Do not deploy clinically or claim on-device Grad-CAM.
