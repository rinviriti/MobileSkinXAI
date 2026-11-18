# Modified application files

- `app/flutter/analysis_options.yaml`
- `app/flutter/android/app/build.gradle.kts`
- `app/flutter/ios/Podfile`
- `app/flutter/ios/Runner.xcodeproj/project.pbxproj`
- `app/flutter/ios/Runner/Info.plist`
- `app/flutter/lib/models/prediction.dart`
- `app/flutter/lib/screens/analysis_screen.dart`
- `app/flutter/lib/screens/home_screen.dart`
- `app/flutter/lib/screens/results_screen.dart`
- `app/flutter/lib/services/skin_classifier.dart`
- `app/flutter/linux/flutter/generated_plugins.cmake`
- `app/flutter/pubspec.lock`
- `app/flutter/pubspec.yaml`
- `app/flutter/test/widget_test.dart`

New files/directories:

- `app/flutter/packages/mobileskinxai_codec/`
- `app/flutter/assets/models/e2_lr_control_v20261118/`
- `app/flutter/test/classifier_test.dart`
- `app/flutter/integration_test/`

Original lib/test/pubspec/iOS backups are in original_flutter_backup/. main.dart/theme is byte-identical. ResultsScreen only changes the NV display name. HomeScreen only removes forced quality recompression. AnalysisScreen changes state/logic/status text and binds existing fields; it does not introduce new screens or styles. SDK-generated analyzer/platform registration changes and lockfile updates are listed explicitly. No existing model, image, research directory or figure was overwritten.
