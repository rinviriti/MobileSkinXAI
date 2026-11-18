# Native preprocessing third-party notices

libjpeg-turbo3.2.0 source commit c85e6b905bf237038faa936dab160ebfc5da0344 is vendored under src/vendor/libjpeg-turbo. Preserve LICENSE.md, README.ijg and all original source notices (IJG/BSD/zlib components). SIMD is disabled in this build. No encoder, color or model tuning is performed.

The Dart Lanczos algorithm follows Pillow Resample.c and ImageOps.contain/pad; the complete Pillow permission notice is retained in PILLOW_LICENSE.txt. Geometry and normalization originate from the project's verified engineering probe. Dart image4.10.1 and ffi2.2.0 remain pinned package dependencies with their package licenses; original library source is not modified. JPEG decoding/encoding does not use Dart image's JPEG codec.

The old upstream Linux TFLiteC2.5.0 binary is a temporary host-test dependency only, outside this app; it is NOT bundled for Android/iOS. Mobile builds use the TensorFlow plugin's own resolved native dependencies and must be tested on actual target devices.
