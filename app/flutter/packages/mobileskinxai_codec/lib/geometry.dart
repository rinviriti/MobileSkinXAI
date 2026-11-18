// Geometry/normalization retained byte-for-byte from the verified previous probe.
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;

int roundEven(double n) {
  final f = n.floor();
  final d = n - f;
  return d < .5 ? f : (d > .5 ? f + 1 : (f.isEven ? f : f + 1));
}

List<(int, List<int>)> coefficients(int input, int output) {
  final scale = input / output;
  final filterScale = math.max(1.0, scale);
  final support = 3 * filterScale;
  double sinc(double x) => x == 0 ? 1 : math.sin(x * math.pi) / (x * math.pi);
  return List.generate(output, (i) {
    final center = (i + .5) * scale;
    final lo = math.max(0, (center - support + .5).toInt());
    final hi = math.min(input, (center + support + .5).toInt());
    final weights = List.generate(hi - lo, (j) {
      final x = (j + lo - center + .5) / filterScale;
      return x < -3 || x >= 3 ? 0.0 : sinc(x) * sinc(x / 3);
    });
    final total = weights.fold(0.0, (a, b) => a + b);
    return (
      lo,
      weights.map((v) {
        final w = total == 0 ? v : v / total;
        final fixed = w * (1 << 22);
        return (fixed < 0 ? fixed - .5 : fixed + .5).toInt();
      }).toList(),
    );
  });
}

Uint8List imageRgb(img.Image image) {
  final rgb = Uint8List(image.width * image.height * 3);
  for (var y = 0; y < image.height; y++) {
    for (var x = 0; x < image.width; x++) {
      final p = image.getPixel(x, y);
      final offset = (y * image.width + x) * 3;
      rgb[offset] = p.r.toInt();
      rgb[offset + 1] = p.g.toInt();
      rgb[offset + 2] = p.b.toInt();
    }
  }
  return rgb;
}

Uint8List resizeLanczos(Uint8List rgb, int width, int height, int ow, int oh) {
  var input = rgb;
  if (width != ow) {
    final coeff = coefficients(width, ow);
    final horizontal = Uint8List(ow * height * 3);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < ow; x++) {
        final (lo, weights) = coeff[x];
        for (var c = 0; c < 3; c++) {
          var sum = 1 << 21;
          for (var j = 0; j < weights.length; j++) {
            sum += input[(y * width + lo + j) * 3 + c] * weights[j];
          }
          horizontal[(y * ow + x) * 3 + c] = (sum >> 22).clamp(0, 255);
        }
      }
    }
    input = horizontal;
  }
  if (height == oh) return input;
  final vertical = Uint8List(ow * oh * 3);
  final coeff = coefficients(height, oh);
  for (var y = 0; y < oh; y++) {
    final (lo, weights) = coeff[y];
    for (var x = 0; x < ow; x++) {
      for (var c = 0; c < 3; c++) {
        var sum = 1 << 21;
        for (var j = 0; j < weights.length; j++) {
          sum += input[((lo + j) * ow + x) * 3 + c] * weights[j];
        }
        vertical[(y * ow + x) * 3 + c] = (sum >> 22).clamp(0, 255);
      }
    }
  }
  return vertical;
}

Uint8List letterbox(img.Image oriented) {
  var w = 224;
  var h = 224;
  if (oriented.width > oriented.height) {
    h = roundEven(oriented.height / oriented.width * 224);
  } else if (oriented.width < oriented.height) {
    w = roundEven(oriented.width / oriented.height * 224);
  }
  if (w < 1 || h < 1) throw FormatException('Invalid image aspect ratio');
  final resized = resizeLanczos(
    imageRgb(oriented),
    oriented.width,
    oriented.height,
    w,
    h,
  );
  final result = Uint8List(224 * 224 * 3)..fillRange(0, 224 * 224 * 3, 114);
  final x = roundEven((224 - w) * .5);
  final y = roundEven((224 - h) * .5);
  for (var row = 0; row < h; row++) {
    result.setRange(
      ((row + y) * 224 + x) * 3,
      ((row + y) * 224 + x + w) * 3,
      resized,
      row * w * 3,
    );
  }
  return result;
}

Float32List normalizeRgb(Uint8List rgb) {
  final mean = Float32List.fromList([.485, .456, .406]);
  final std = Float32List.fromList([.229, .224, .225]);
  final result = Float32List(rgb.length);
  for (var i = 0; i < rgb.length; i++) {
    result[i] = rgb[i] / 255;
    result[i] = result[i] - mean[i % 3];
    result[i] = result[i] / std[i % 3];
  }
  return result;
}
