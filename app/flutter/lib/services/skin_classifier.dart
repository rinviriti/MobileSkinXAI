import 'dart:io';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/prediction.dart';

class SkinClassifier {
  Interpreter? _interpreter;

  static const int inputSize = 224;

  final List<String> labels = const [
    'Benign Nevus',
    'Melanoma',
    'Basal Cell Carcinoma',
    'Squamous Cell Carcinoma',
  ];

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset(
      'assets/models/mobileskinxai.tflite',
    );

    final inputTensor = _interpreter!.getInputTensor(0);
    final outputTensor = _interpreter!.getOutputTensor(0);

    print('Input shape: ${inputTensor.shape}');
    print('Input type: ${inputTensor.type}');
    print('Output shape: ${outputTensor.shape}');
    print('Output type: ${outputTensor.type}');
  }

  Future<Prediction> predict(File imageFile) async {
    if (_interpreter == null) {
      throw Exception('Model has not been loaded.');
    }

    final bytes = await imageFile.readAsBytes();

    final decodedImage = img.decodeImage(bytes);

    if (decodedImage == null) {
      throw Exception('Unable to decode image.');
    }

    final resizedImage = img.copyResize(
      decodedImage,
      width: inputSize,
      height: inputSize,
    );

    final input = [
      List.generate(
        inputSize,
        (y) => List.generate(inputSize, (x) {
          final pixel = resizedImage.getPixel(x, y);

          return [
            pixel.r.toDouble() / 255.0,
            pixel.g.toDouble() / 255.0,
            pixel.b.toDouble() / 255.0,
          ];
        }),
      ),
    ];

    final output = [List<double>.filled(labels.length, 0.0)];

    _interpreter!.run(input, output);

    final scores = output[0];

    int bestIndex = 0;

    for (int i = 1; i < scores.length; i++) {
      if (scores[i] > scores[bestIndex]) {
        bestIndex = i;
      }
    }

    return Prediction(label: labels[bestIndex], confidence: scores[bestIndex]);
  }

  void dispose() {
    _interpreter?.close();
  }
}
