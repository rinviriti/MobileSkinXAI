import 'dart:typed_data';

import 'package:flutter/material.dart';

/// Results are intentionally separated from model inference.
/// This screen can display real outputs after the model is validated.
class ResultsScreen extends StatelessWidget {
  const ResultsScreen({
    super.key,
    required this.imageBytes,
    required this.imageName,
    this.probabilities,
    this.predictedClass,
    this.inferenceTimeMs,
    this.heatmapBytes,
    this.modelVersion,
  });

  final Uint8List imageBytes;
  final String imageName;

  /// Only pass probabilities produced by the trained model.
  /// Expected keys must match the class labels below.
  final Map<String, double>? probabilities;

  /// Predicted class from the model's verified label mapping.
  final String? predictedClass;

  /// Actual measured inference duration in milliseconds.
  final double? inferenceTimeMs;

  /// Genuine explanation image produced by the XAI pipeline.
  final Uint8List? heatmapBytes;

  final String? modelVersion;

  static const Color _primary = Color(0xFF176B65);
  static const Color _heading = Color(0xFF213A38);
  static const Color _muted = Color(0xFF647773);
  static const Color _border = Color(0xFFDCE7E4);
  static const Color _background = Color(0xFFF8FAF9);

  static const List<String> _classes = [
    'Benign Nevus',
    'Melanoma',
    'Basal Cell Carcinoma',
    'Squamous Cell Carcinoma',
  ];

  bool get _hasValidResults {
    if (probabilities == null ||
        predictedClass == null ||
        !_classes.contains(predictedClass)) {
      return false;
    }

    if (probabilities!.length != _classes.length) {
      return false;
    }

    var sum = 0.0;

    for (final label in _classes) {
      final value = probabilities![label];

      if (value == null || !value.isFinite || value < 0 || value > 1) {
        return false;
      }

      sum += value;
    }

    final maximum = probabilities![predictedClass!]!;

    if ((sum - 1.0).abs() > 0.01) {
      return false;
    }

    for (final value in probabilities!.values) {
      if (value > maximum + 0.000001) {
        return false;
      }
    }

    return true;
  }

  Widget _sectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 17,
        fontWeight: FontWeight.w700,
        color: _heading,
      ),
    );
  }

  Widget _statusCard() {
    final ready = _hasValidResults;

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: ready ? const Color(0xFFEAF4F1) : const Color(0xFFFFF5E8),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            ready ? Icons.check_circle_outline : Icons.hourglass_empty_outlined,
            color: ready ? _primary : const Color(0xFF93651D),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  ready ? 'Model output available' : 'Awaiting trained model',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: ready ? _heading : const Color(0xFF805715),
                  ),
                ),
                const SizedBox(height: 7),
                Text(
                  ready
                      ? 'The following results were provided '
                            'by the classification pipeline.'
                      : 'Classification results will appear '
                            'here after model training, validation, '
                            'and TensorFlow Lite integration.',
                  style: const TextStyle(
                    fontSize: 13,
                    height: 1.6,
                    color: _heading,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _imageCard() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _border),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        children: [
          Container(
            height: 220,
            width: double.infinity,
            color: const Color(0xFFF1F5F4),
            child: Image.memory(
              imageBytes,
              fit: BoxFit.contain,
              errorBuilder: (context, error, stackTrace) {
                return const Center(
                  child: Icon(
                    Icons.broken_image_outlined,
                    size: 36,
                    color: _muted,
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(14),
            child: Row(
              children: [
                const Icon(Icons.image_outlined, size: 19, color: _primary),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    imageName,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      fontWeight: FontWeight.w600,
                      color: _heading,
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

  Widget _predictionCard() {
    if (!_hasValidResults) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: _border),
        ),
        child: Column(
          children: [
            const Icon(Icons.analytics_outlined, color: _muted, size: 36),
            const SizedBox(height: 12),
            const Text(
              'No classification available',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: _heading,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'The four lesion categories will be shown '
              'once the trained model is connected.',
              textAlign: TextAlign.center,
              style: TextStyle(color: _muted, fontSize: 13, height: 1.5),
            ),
            const SizedBox(height: 22),
            for (final label in _classes)
              Padding(
                padding: const EdgeInsets.only(bottom: 14),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        label,
                        style: const TextStyle(fontSize: 13, color: _heading),
                      ),
                    ),
                    const Text('—', style: TextStyle(color: _muted)),
                  ],
                ),
              ),
          ],
        ),
      );
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Highest-scoring category',
            style: TextStyle(color: _muted, fontSize: 12),
          ),
          const SizedBox(height: 8),
          Text(
            predictedClass!,
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w700,
              color: _heading,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '${(probabilities![predictedClass!]! * 100).toStringAsFixed(1)}% model score',
            style: const TextStyle(
              fontSize: 14,
              color: _primary,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 24),
          for (final label in _classes) ...[
            Row(
              children: [
                Expanded(
                  child: Text(
                    label,
                    style: const TextStyle(fontSize: 13, color: _heading),
                  ),
                ),
                Text(
                  '${(probabilities![label]! * 100).toStringAsFixed(1)}%',
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: _heading,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 7),
            LinearProgressIndicator(
              value: probabilities![label],
              minHeight: 7,
              borderRadius: BorderRadius.circular(8),
              backgroundColor: const Color(0xFFEAF0EE),
              color: _primary,
            ),
            const SizedBox(height: 16),
          ],
          const Text(
            'Model scores are not calibrated clinical '
            'risk estimates or diagnostic probabilities.',
            style: TextStyle(fontSize: 12, height: 1.5, color: _muted),
          ),
        ],
      ),
    );
  }

  Widget _explanationCard() {
    final available = _hasValidResults && heatmapBytes != null;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _border),
      ),
      child: available
          ? Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.memory(
                    heatmapBytes!,
                    width: double.infinity,
                    height: 250,
                    fit: BoxFit.contain,
                    errorBuilder: (context, error, stackTrace) {
                      return const Center(
                        child: Text('Explanation image unavailable'),
                      );
                    },
                  ),
                ),
                const SizedBox(height: 12),
                const Text(
                  'Visual explanation supplied by the '
                  'explainability pipeline. Highlighted '
                  'regions are not verified lesion boundaries.',
                  style: TextStyle(color: _muted, fontSize: 12, height: 1.5),
                ),
              ],
            )
          : const Column(
              children: [
                Icon(Icons.layers_outlined, color: _muted, size: 36),
                SizedBox(height: 12),
                Text(
                  'Visual explanation unavailable',
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: _heading,
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  'Grad-CAM visualizations will be displayed '
                  'after the explanation pipeline has been '
                  'implemented and validated.',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 13, height: 1.5, color: _muted),
                ),
              ],
            ),
    );
  }

  Widget _modelDetails() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _border),
      ),
      child: Column(
        children: [
          _detail('Model', modelVersion ?? 'Not installed'),
          const Divider(height: 22),
          _detail(
            'Inference',
            _hasValidResults && inferenceTimeMs != null
                ? '${inferenceTimeMs!.toStringAsFixed(1)} ms'
                : 'Not measured',
          ),
          const Divider(height: 22),
          _detail(
            'Classification',
            _hasValidResults ? 'Output received' : 'Not available',
          ),
          const Divider(height: 22),
          _detail(
            'Explainability',
            _hasValidResults && heatmapBytes != null
                ? 'Output received'
                : 'Not available',
          ),
        ],
      ),
    );
  }

  Widget _detail(String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Text(
            label,
            style: const TextStyle(color: _muted, fontSize: 13),
          ),
        ),
        Flexible(
          child: Text(
            value,
            textAlign: TextAlign.end,
            style: const TextStyle(
              color: _heading,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _background,
      appBar: AppBar(
        backgroundColor: _background,
        title: const Text(
          'Assessment results',
          style: TextStyle(
            color: _heading,
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
                      letterSpacing: -0.5,
                      color: _heading,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Review the image, classification '
                    'output, and model explanation.',
                    style: TextStyle(fontSize: 14, height: 1.5, color: _muted),
                  ),
                  const SizedBox(height: 24),
                  _statusCard(),
                  const SizedBox(height: 24),
                  _sectionTitle('Selected image'),
                  const SizedBox(height: 12),
                  _imageCard(),
                  const SizedBox(height: 26),
                  _sectionTitle('Classification results'),
                  const SizedBox(height: 12),
                  _predictionCard(),
                  const SizedBox(height: 26),
                  _sectionTitle('Visual explanation'),
                  const SizedBox(height: 12),
                  _explanationCard(),
                  const SizedBox(height: 26),
                  _sectionTitle('Model information'),
                  const SizedBox(height: 12),
                  _modelDetails(),
                  const SizedBox(height: 26),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: const Color(0xFFEAF4F1),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: const Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.health_and_safety_outlined, color: _primary),
                        SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            'For research and educational '
                            'evaluation only. This application '
                            'cannot diagnose skin cancer. '
                            'Consult a qualified healthcare '
                            'professional for medical concerns.',
                            style: TextStyle(
                              fontSize: 12,
                              height: 1.6,
                              color: _heading,
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
