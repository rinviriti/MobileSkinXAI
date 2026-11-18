import 'dart:convert';
import 'dart:math' as math;
import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mobileskinxai/services/skin_classifier.dart';
import 'golden_cases.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  testWidgets(
    'Actual packaged mobile codec, asset and interpreter golden parity',
    (tester) async {
      final classifier =
          SkinClassifier(); // Actual packaged library, no host-path override.
      final results = <Map<String, Object>>[];
      try {
        for (final item in goldenCases) {
          final raw = base64Decode(item['jpeg_base64'] as String);
          expect(sha256.convert(raw).toString(), item['sha256']);
          final prediction = await classifier.predictBytes(raw);
          final expected = SkinClassifier.probabilitiesFromLogits(
            (item['logits'] as List)
                .cast<num>()
                .map((x) => x.toDouble())
                .toList(),
          );
          final original = (item['logits'] as List).cast<num>();
          final logitDifferences = List.generate(
            4,
            (i) => (prediction.logits[i] - original[i]).abs(),
          );
          expect(logitDifferences.reduce(math.max), lessThanOrEqualTo(.001));
          expect(
            logitDifferences.reduce((a, b) => a + b) / 4,
            lessThanOrEqualTo(.0001),
          );
          final differences = List.generate(
            4,
            (i) =>
                (prediction.probabilities[SkinClassifier.labels[i]]! -
                        expected[i])
                    .abs(),
          );
          expect(differences.reduce(math.max), lessThanOrEqualTo(.0001));
          expect(
            differences.reduce((a, b) => a + b) / 4,
            lessThanOrEqualTo(.00001),
          );
          expect(
            prediction.label,
            SkinClassifier.labels[expected.indexOf(expected.reduce(math.max))],
          );
          results.add({
            'image': item['image'] as String,
            'probabilities': prediction.probabilities,
            'inference_ms': prediction.inferenceTimeMs,
            'max_probability_error': differences.reduce(math.max),
          });
        }
        // Retain actual device evidence in the integration-test result JSON.
        IntegrationTestWidgetsFlutterBinding.instance.reportData = {
          'golden_results': results,
        };
      } finally {
        classifier.dispose();
      }
    },
  );
}
