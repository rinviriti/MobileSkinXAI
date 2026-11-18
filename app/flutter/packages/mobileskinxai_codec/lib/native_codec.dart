import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:image/image.dart' as img;
import 'geometry.dart';

typedef _DecodeC =
    Pointer<Uint8> Function(
      Pointer<Uint8>,
      Size,
      Pointer<Int32>,
      Pointer<Int32>,
    );
typedef _DecodeD =
    Pointer<Uint8> Function(
      Pointer<Uint8>,
      int,
      Pointer<Int32>,
      Pointer<Int32>,
    );
typedef _RoundC = Pointer<Uint8> Function(Pointer<Uint8>, Int32, Int32);
typedef _RoundD = Pointer<Uint8> Function(Pointer<Uint8>, int, int);
typedef _FreeC = Void Function(Pointer<Void>);
typedef _FreeD = void Function(Pointer<Void>);
typedef _ErrorC = Pointer<Utf8> Function();
typedef _ErrorD = Pointer<Utf8> Function();

class NativeJpegCodec {
  NativeJpegCodec({String? libraryPath}) {
    final lib = libraryPath != null
        ? DynamicLibrary.open(libraryPath)
        : (Platform.isIOS
              ? DynamicLibrary.process()
              : DynamicLibrary.open(
                  Platform.isWindows
                      ? 'mobileskinxai_codec.dll'
                      : Platform.isMacOS
                      ? 'libmobileskinxai_codec.dylib'
                      : 'libmobileskinxai_codec.so',
                ));
    _decode = lib.lookupFunction<_DecodeC, _DecodeD>('msx_decode');
    _roundtrip = lib.lookupFunction<_RoundC, _RoundD>('msx_roundtrip');
    _free = lib.lookupFunction<_FreeC, _FreeD>('msx_free');
    _error = lib.lookupFunction<_ErrorC, _ErrorD>('msx_error');
  }
  late final _DecodeD _decode;
  late final _RoundD _roundtrip;
  late final _FreeD _free;
  late final _ErrorD _error;

  img.Image decode(Uint8List bytes) {
    if (bytes.length > 64 * 1024 * 1024) {
      throw const FormatException('Image exceeds64MiB');
    }
    if (bytes.length >= 26 &&
        bytes[0] == 137 &&
        bytes[1] == 80 &&
        bytes[2] == 78 &&
        bytes[3] == 71) {
      final header = ByteData.sublistView(bytes);
      final width = header.getUint32(16);
      final height = header.getUint32(20);
      if (width < 1 ||
          height < 1 ||
          width * height > 50000000 ||
          bytes[24] != 8) {
        throw const FormatException(
          'Only bounded 8-bit PNG images are supported',
        );
      }
      final png = img.decodePng(bytes);
      if (png == null) throw const FormatException('Unable to decode PNG');
      return orientRgb(png, png.exif.imageIfd.orientation ?? 1);
    }
    if (bytes.length < 4 ||
        bytes[0] != 255 ||
        bytes[1] != 216 ||
        bytes.length > 64 * 1024 * 1024) {
      throw const FormatException(
        'Select an original JPEG or 8-bit PNG. HEIC conversion is not verified.',
      );
    }
    final input = calloc<Uint8>(bytes.length);
    final width = calloc<Int32>();
    final height = calloc<Int32>();
    Pointer<Uint8> pixels = nullptr;
    try {
      input.asTypedList(bytes.length).setAll(0, bytes);
      pixels = _decode(input, bytes.length, width, height);
      if (pixels == nullptr) throw FormatException(_error().toDartString());
      final rgb = Uint8List.fromList(
        pixels.asTypedList(width.value * height.value * 3),
      );
      final image = img.Image.fromBytes(
        width: width.value,
        height: height.value,
        bytes: rgb.buffer,
        numChannels: 3,
      );
      return orientRgb(image, exifOrientation(bytes));
    } finally {
      if (pixels != nullptr) _free(pixels.cast());
      calloc.free(input);
      calloc.free(width);
      calloc.free(height);
    }
  }

  Uint8List jpegRoundtrip(Uint8List rgb) {
    if (rgb.length != 224 * 224 * 3) throw ArgumentError('Expected224 RGB');
    final input = calloc<Uint8>(rgb.length);
    Pointer<Uint8> output = nullptr;
    try {
      input.asTypedList(rgb.length).setAll(0, rgb);
      output = _roundtrip(input, 224, 224);
      if (output == nullptr) throw FormatException(_error().toDartString());
      return Uint8List.fromList(output.asTypedList(rgb.length));
    } finally {
      if (output != nullptr) _free(output.cast());
      calloc.free(input);
    }
  }

  Uint8List prepareRgb(Uint8List bytes) =>
      jpegRoundtrip(letterbox(decode(bytes)));
  Float32List preprocess(Uint8List bytes) => normalizeRgb(prepareRgb(bytes));
}

// Explicit integer-coordinate mapping: image4.10.1's flip-both leaves the
// middle row unflipped for odd heights (caught by the orientation3 fixture).
img.Image orientRgb(img.Image source, int orientation) {
  if (orientation == 1) return source;
  final w = source.width;
  final h = source.height;
  final swap = orientation >= 5;
  final result = img.Image(
    width: swap ? h : w,
    height: swap ? w : h,
    numChannels: 3,
  );
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final (ox, oy) = switch (orientation) {
        2 => (w - 1 - x, y),
        3 => (w - 1 - x, h - 1 - y),
        4 => (x, h - 1 - y),
        5 => (y, x),
        6 => (h - 1 - y, x),
        7 => (h - 1 - y, w - 1 - x),
        8 => (y, w - 1 - x),
        _ => throw const FormatException('Invalid orientation'),
      };
      final pixel = source.getPixel(x, y);
      result.setPixelRgb(ox, oy, pixel.r, pixel.g, pixel.b);
    }
  }
  return result;
}

// JPEG APP1 TIFF orientation; no ICC transform. All offsets are bounded.
int exifOrientation(Uint8List bytes) {
  var pos = 2;
  while (pos + 4 <= bytes.length) {
    if (bytes[pos] != 255) break;
    while (pos < bytes.length && bytes[pos] == 255) {
      pos++;
    }
    if (pos >= bytes.length) break;
    final marker = bytes[pos++];
    if (marker == 218 || marker == 217) break;
    if (marker == 1 || (marker >= 208 && marker <= 215)) continue;
    if (pos + 2 > bytes.length) break;
    final size = bytes[pos] * 256 + bytes[pos + 1];
    if (size < 2 || pos + size > bytes.length) {
      throw const FormatException('Malformed JPEG marker');
    }
    final start = pos + 2;
    final end = pos + size;
    if (marker == 225 &&
        size >= 16 &&
        bytes[start] == 69 &&
        bytes[start + 1] == 120 &&
        bytes[start + 2] == 105 &&
        bytes[start + 3] == 102 &&
        bytes[start + 4] == 0 &&
        bytes[start + 5] == 0) {
      final tiff = start + 6;
      final view = ByteData.sublistView(bytes);
      final endian = bytes[tiff] == 73 && bytes[tiff + 1] == 73
          ? Endian.little
          : bytes[tiff] == 77 && bytes[tiff + 1] == 77
          ? Endian.big
          : null;
      if (endian == null || view.getUint16(tiff + 2, endian) != 42) {
        throw const FormatException('Malformed EXIF');
      }
      final ifd = tiff + view.getUint32(tiff + 4, endian);
      if (ifd < tiff || ifd + 2 > end) {
        throw const FormatException('Malformed EXIF offset');
      }
      final count = view.getUint16(ifd, endian);
      if (ifd + 2 + count * 12 > end) {
        throw const FormatException('Truncated EXIF entries');
      }
      for (var j = 0; j < count; j++) {
        final entry = ifd + 2 + j * 12;
        if (view.getUint16(entry, endian) == 274) {
          if (view.getUint16(entry + 2, endian) != 3 ||
              view.getUint32(entry + 4, endian) != 1) {
            throw const FormatException('Invalid EXIF orientation type');
          }
          final orientation = view.getUint16(entry + 8, endian);
          if (orientation < 1 || orientation > 8) {
            throw const FormatException('Invalid EXIF orientation');
          }
          return orientation;
        }
      }
      return 1;
    }
    pos = end;
  }
  return 1;
}
