import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:mobileskinxai_codec/mobileskinxai_codec.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/prediction.dart';

Float32List _prepare(Map<String, Object?> request) => NativeJpegCodec(
  libraryPath: request['libraryPath'] as String?,
).preprocess(request['bytes'] as Uint8List);

class SkinClassifier {
  SkinClassifier({this.codecLibraryPath, AssetBundle? assets})
    : _assets = assets ?? rootBundle;
  static const assetRoot = 'assets/models/e2_lr_control_v20261118';
  static const modelHash =
      '8b6e19f3d8d0dd9c3b798a2cf99c0525079b9de4ea2f5d1708dd928a718412f3';
  static const classCodes = ['NV', 'MEL', 'BCC', 'SCC'];
  static const labels = [
    'Melanocytic Nevus',
    'Melanoma',
    'Basal Cell Carcinoma',
    'Squamous Cell Carcinoma',
  ];
  final String? codecLibraryPath;
  final AssetBundle _assets;
  Interpreter? _interpreter;
  Future<void>? _loading;
  bool _disposed = false;
  bool _busy = false;
  String _version = '';

  Future<void> loadModel() => _loading ??= _load().catchError((Object error) {
    _loading = null;
    throw error;
  });
  Future<void> _load() async {
    if (_disposed) throw StateError('Classifier has been disposed.');
    final data = await _assets.load('$assetRoot/model_float32.tflite');
    final bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    if (sha256.convert(bytes).toString() != modelHash) {
      throw StateError('Model integrity check failed.');
    }
    final codes =
        jsonDecode(await _assets.loadString('$assetRoot/labels.json')) as List;
    if (!listEquals(codes, classCodes)) {
      throw StateError('Model class mapping mismatch.');
    }
    final meta =
        jsonDecode(await _assets.loadString('$assetRoot/model_metadata.json'))
            as Map<String, dynamic>;
    if (meta['tflite_sha256'] != modelHash || meta['output'] != 'logits') {
      throw StateError('Model metadata mismatch.');
    }
    final options = InterpreterOptions()..threads = 2;
    Interpreter? interpreter;
    try {
      interpreter = Interpreter.fromBuffer(bytes, options: options);
      final input = interpreter.getInputTensor(0),
          output = interpreter.getOutputTensor(0);
      if (!listEquals(input.shape, [1, 224, 224, 3]) ||
          !listEquals(output.shape, [1, 4]) ||
          input.type != TensorType.float32 ||
          output.type != TensorType.float32) {
        throw StateError('Unexpected model tensor signature.');
      }
      if (_disposed) throw StateError('Classifier was closed while loading.');
      _version = meta['model_version'] as String;
      _interpreter = interpreter;
    } catch (_) {
      interpreter?.close();
      rethrow;
    } finally {
      options.delete();
    }
  }

  static List<double> probabilitiesFromLogits(List<double> logits) {
    if (logits.length != 4 || logits.any((v) => !v.isFinite)) {
      throw StateError('Invalid model output.');
    }
    final maximum = logits.reduce(math.max);
    final e = logits.map((v) => math.exp(v - maximum)).toList();
    final total = e.reduce((a, b) => a + b);
    return e.map((v) => v / total).toList(growable: false);
  }

  Future<Prediction> predict(File image) async =>
      predictBytes(await image.readAsBytes());
  Future<Prediction> predictBytes(Uint8List bytes) async {
    if (_disposed) throw StateError('Classifier has been disposed.');
    if (_busy) throw StateError('An image is already being analyzed.');
    if (bytes.isEmpty || bytes.length > 64 * 1024 * 1024) {
      throw const FormatException('Invalid or oversized image.');
    }
    _busy = true;
    try {
      await loadModel();
      final tensor = await compute(_prepare, <String, Object?>{
        'bytes': bytes,
        'libraryPath': codecLibraryPath,
      });
      if (_disposed) throw StateError('Analysis was cancelled.');
      final input = [
        List.generate(
          224,
          (y) => List.generate(
            224,
            (x) => List.generate(3, (c) => tensor[(y * 224 + x) * 3 + c]),
          ),
        ),
      ];
      final output = [List<double>.filled(4, 0)];
      final timer = Stopwatch()..start();
      _interpreter!.run(input, output);
      timer.stop();
      final p = probabilitiesFromLogits(output[0]);
      final best = p.indexOf(p.reduce(math.max));
      return Prediction(
        label: labels[best],
        confidence: p[best],
        probabilities: Map.unmodifiable({
          for (var i = 0; i < 4; i++) labels[i]: p[i],
        }),
        inferenceTimeMs: timer.elapsedMicroseconds / 1000,
        modelVersion: _version,
        logits: List.unmodifiable(output[0]),
      );
    } finally {
      _busy = false;
    }
  }

  void dispose() {
    _disposed = true;
    _interpreter?.close();
    _interpreter = null;
  }
}
