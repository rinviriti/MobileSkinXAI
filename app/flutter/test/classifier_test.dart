import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobileskinxai/services/skin_classifier.dart';
import 'package:mobileskinxai_codec/mobileskinxai_codec.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('Stable softmax is applied once, with exact fixed mapping', () {
    final scores = SkinClassifier.probabilitiesFromLogits([
      1000,
      1001,
      999,
      998,
    ]);
    expect(scores.reduce((a, b) => a + b), closeTo(1, 1e-12));
    expect(scores.indexOf(scores.reduce(math.max)), 1);
    expect(SkinClassifier.classCodes, ['NV', 'MEL', 'BCC', 'SCC']);
    expect(
      () => SkinClassifier.probabilitiesFromLogits([double.nan, 0, 0, 0]),
      throwsStateError,
    );
  });
  test('Float32 normalization preserves mean/std and RGB order', () {
    final rgb = Uint8List(224 * 224 * 3);
    final actual = normalizeRgb(rgb);
    final expected = Float32List.fromList([
      -0.485 / 0.229,
      -0.456 / 0.224,
      -0.406 / 0.225,
    ]);
    for (var c = 0; c < 3; c++) {
      expect(actual[c], closeTo(expected[c], 3e-7));
    }
  });
  test(
    'Empty input and disposed sessions never return fake predictions',
    () async {
      final classifier = SkinClassifier();
      await expectLater(
        classifier.predictBytes(Uint8List(0)),
        throwsFormatException,
      );
      classifier.dispose();
      await expectLater(
        classifier.predictBytes(Uint8List(1)),
        throwsStateError,
      );
    },
  );
  const manifest = String.fromEnvironment('MSX_GOLDEN_MANIFEST');
  const library = String.fromEnvironment('MSX_CODEC_LIBRARY');
  test(
    'Application service: real asset + native raw preprocessing + real TFLite',
    () async {
      final classifier = SkinClassifier(codecLibraryPath: library);
      final rows = jsonDecode(File(manifest).readAsStringSync()) as List;
      try {
        await classifier.loadModel();
        for (final r in rows.cast<Map<String, dynamic>>()) {
          final actual = await classifier.predictBytes(
            File(r['source'] as String).readAsBytesSync(),
          );
          final expected = SkinClassifier.probabilitiesFromLogits(
            (r['original_logits'] as List)
                .cast<num>()
                .map((x) => x.toDouble())
                .toList(),
          );
          final originalLogits = (r['original_logits'] as List).cast<num>();
          final logitDifference = List.generate(
            4, (i) => (actual.logits[i] - originalLogits[i]).abs(),
          );
          expect(logitDifference.reduce(math.max), lessThanOrEqualTo(.001));
          expect(logitDifference.reduce((a,b) => a+b)/4, lessThanOrEqualTo(.0001));
          final difference = List.generate(
            4,
            (i) =>
                (actual.probabilities[SkinClassifier.labels[i]]! - expected[i])
                    .abs(),
          );
          expect(difference.reduce(math.max), lessThanOrEqualTo(.0001));
          expect(
            difference.reduce((a, b) => a + b) / 4,
            lessThanOrEqualTo(.00001),
          );
          expect(
            actual.label,
            SkinClassifier.labels[expected.indexOf(expected.reduce(math.max))],
          );
          expect(actual.inferenceTimeMs, greaterThan(0));
        }
        await expectLater(
          classifier.predictBytes(Uint8List.fromList([1, 2, 3, 4])),
          throwsFormatException,
        );
        await expectLater(
          classifier.predictBytes(Uint8List.fromList([255, 216, 255, 217])),
          throwsFormatException,
        );
      } finally {
        classifier.dispose();
      }
    },
    skip: manifest.isEmpty || library.isEmpty
        ? 'Requires fixed development manifest and built native codec; pass MSX_GOLDEN_MANIFEST and MSX_CODEC_LIBRARY dart-defines.'
        : false,
  );
}
