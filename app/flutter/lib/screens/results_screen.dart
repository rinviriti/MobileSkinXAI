import 'dart:typed_data';

import 'package:flutter/material.dart';

class ResultsScreen extends StatelessWidget {
  const ResultsScreen({
    super.key,
    required this.imageBytes,
    required this.imageName,
    this.probabilities,
    this.predictedClass,
    this.heatmapBytes,
    this.inferenceTimeMs,
    this.modelVersion,
  });

  final Uint8List imageBytes;
  final String imageName;
  final Map<String, double>? probabilities;
  final String? predictedClass;
  final Uint8List? heatmapBytes;
  final double? inferenceTimeMs;
  final String? modelVersion;

  static const Color primary = Color(0xFF176B65);
  static const Color heading = Color(0xFF213A38);
  static const Color muted = Color(0xFF647773);
  static const Color border = Color(0xFFDCE7E4);
  static const Color background = Color(0xFFF8FAF9);

  static const List<String> classes = [
    'Benign Nevus',
    'Melanoma',
    'Basal Cell Carcinoma',
    'Squamous Cell Carcinoma',
  ];

  bool get hasValidResults {
    final scores = probabilities;

    if (scores == null ||
        predictedClass == null ||
        !classes.contains(predictedClass) ||
        scores.length != classes.length) {
      return false;
    }

    double total = 0;

    for (final label in classes) {
      final score = scores[label];

      if (score == null || !score.isFinite || score < 0 || score > 1) {
        return false;
      }

      total += score;
    }

    if ((total - 1).abs() > 0.01) return false;

    final selectedScore = scores[predictedClass]!;

    return classes.every((label) => scores[label]! <= selectedScore + 0.000001);
  }

  Widget sectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 17,
          fontWeight: FontWeight.w700,
          color: heading,
        ),
      ),
    );
  }

  Widget card({required Widget child}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: border),
      ),
      child: child,
    );
  }

  Widget statusCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: hasValidResults
            ? const Color(0xFFEAF4F1)
            : const Color(0xFFFFF4E8),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            hasValidResults
                ? Icons.check_circle_outline
                : Icons.hourglass_empty_outlined,
            color: hasValidResults ? primary : const Color(0xFF93651D),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  hasValidResults
                      ? 'Model output available'
                      : 'Awaiting trained model',
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: heading,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  hasValidResults
                      ? 'Classification outputs have been '
                            'provided to this screen.'
                      : 'Classification results will appear '
                            'after model training, validation, '
                            'and TensorFlow Lite integration.',
                  style: const TextStyle(
                    fontSize: 13,
                    height: 1.6,
                    color: heading,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget imageCard() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: border),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        children: [
          Container(
            height: 230,
            width: double.infinity,
            color: const Color(0xFFF1F5F4),
            child: Image.memory(
              imageBytes,
              fit: BoxFit.contain,
              errorBuilder: (_, error, stackTrace) {
                return const Center(
                  child: Icon(
                    Icons.broken_image_outlined,
                    size: 36,
                    color: muted,
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(15),
            child: Row(
              children: [
                const Icon(Icons.image_outlined, color: primary, size: 20),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    imageName,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      fontWeight: FontWeight.w600,
                      color: heading,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget classificationCard() {
    if (!hasValidResults) {
      return card(
        child: Column(
          children: [
            const Icon(Icons.analytics_outlined, size: 36, color: muted),
            const SizedBox(height: 14),
            const Text(
              'No classification available',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: heading,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'The four lesion categories will appear '
              'once the trained model is connected.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 13, height: 1.5, color: muted),
            ),
            const SizedBox(height: 24),
            for (final label in classes)
              Padding(
                padding: const EdgeInsets.only(bottom: 15),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        label,
                        style: const TextStyle(fontSize: 13, color: heading),
                      ),
                    ),
                    const Text('—', style: TextStyle(color: muted)),
                  ],
                ),
              ),
          ],
        ),
      );
    }

    return card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Highest-scoring category',
            style: TextStyle(color: muted, fontSize: 12),
          ),
          const SizedBox(height: 8),
          Text(
            predictedClass!,
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w700,
              color: heading,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '${(probabilities![predictedClass!]! * 100).toStringAsFixed(1)}% model score',
            style: const TextStyle(fontWeight: FontWeight.w600, color: primary),
          ),
          const SizedBox(height: 24),
          for (final label in classes) ...[
            Row(
              children: [
                Expanded(
                  child: Text(
                    label,
                    style: const TextStyle(fontSize: 13, color: heading),
                  ),
                ),
                Text(
                  '${(probabilities![label]! * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    color: heading,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            LinearProgressIndicator(
              value: probabilities![label],
              minHeight: 7,
              borderRadius: BorderRadius.circular(8),
              backgroundColor: const Color(0xFFEAF0EE),
              color: primary,
            ),
            const SizedBox(height: 18),
          ],
          const Text(
            'Model scores are not validated clinical '
            'risk estimates or diagnostic probabilities.',
            style: TextStyle(fontSize: 12, height: 1.5, color: muted),
          ),
        ],
      ),
    );
  }

  Widget explanationCard() {
    if (!hasValidResults || heatmapBytes == null) {
      return card(
        child: const Column(
          children: [
            Icon(Icons.layers_outlined, size: 36, color: muted),
            SizedBox(height: 14),
            Text(
              'Visual explanation unavailable',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: heading,
              ),
            ),
            SizedBox(height: 8),
            Text(
              'Grad-CAM visualizations will appear '
              'after the explanation pipeline is '
              'implemented and validated.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 13, height: 1.6, color: muted),
            ),
          ],
        ),
      );
    }

    return card(
      child: Column(
        children: [
          Image.memory(
            heatmapBytes!,
            height: 250,
            width: double.infinity,
            fit: BoxFit.contain,
            errorBuilder: (_, error, stackTrace) {
              return const Text('Unable to display explanation image');
            },
          ),
          const SizedBox(height: 14),
          const Text(
            'Highlighted regions represent model '
            'attention, not clinically verified '
            'lesion boundaries.',
            style: TextStyle(fontSize: 12, height: 1.5, color: muted),
          ),
        ],
      ),
    );
  }

  Widget detailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 9),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: const TextStyle(color: muted, fontSize: 13),
            ),
          ),
          Flexible(
            child: Text(
              value,
              textAlign: TextAlign.end,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: heading,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget modelDetails() {
    return card(
      child: Column(
        children: [
          detailRow('Model', modelVersion ?? 'Not installed'),
          const Divider(),
          detailRow(
            'Inference',
            hasValidResults &&
                    inferenceTimeMs != null &&
                    inferenceTimeMs!.isFinite &&
                    inferenceTimeMs! >= 0
                ? '${inferenceTimeMs!.toStringAsFixed(1)} ms'
                : 'Not measured',
          ),
          const Divider(),
          detailRow(
            'Classification',
            hasValidResults ? 'Output received' : 'Not available',
          ),
          const Divider(),
          detailRow(
            'Explainability',
            hasValidResults && heatmapBytes != null
                ? 'Output received'
                : 'Not available',
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: background,
      appBar: AppBar(
        backgroundColor: background,
        title: const Text(
          'Assessment results',
          style: TextStyle(
            color: heading,
            fontSize: 19,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 520),
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 36),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Assessment overview',
                    style: TextStyle(
                      fontSize: 27,
                      fontWeight: FontWeight.w700,
                      color: heading,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Review the selected image, '
                    'classification output, and '
                    'model explanation.',
                    style: TextStyle(color: muted, fontSize: 14, height: 1.5),
                  ),
                  const SizedBox(height: 24),
                  statusCard(),
                  const SizedBox(height: 26),
                  sectionTitle('Selected image'),
                  imageCard(),
                  const SizedBox(height: 26),
                  sectionTitle('Classification results'),
                  classificationCard(),
                  const SizedBox(height: 26),
                  sectionTitle('Visual explanation'),
                  explanationCard(),
                  const SizedBox(height: 26),
                  sectionTitle('Model information'),
                  modelDetails(),
                  const SizedBox(height: 26),
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: const Color(0xFFEAF4F1),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: const Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.health_and_safety_outlined, color: primary),
                        SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            'For research and educational '
                            'evaluation only. MobileSkinXAI '
                            'does not diagnose skin cancer '
                            'or replace professional '
                            'medical assessment.',
                            style: TextStyle(
                              color: heading,
                              fontSize: 12,
                              height: 1.6,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
