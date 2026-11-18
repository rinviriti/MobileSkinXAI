class Prediction {
  final String label;
  final double confidence;
  final Map<String, double> probabilities;
  final double inferenceTimeMs;
  final String modelVersion;
  final List<double> logits;

  Prediction({
    required this.label,
    required this.confidence,
    this.probabilities = const {},
    this.inferenceTimeMs = 0,
    this.modelVersion = '',
    this.logits = const [],
  });
}
