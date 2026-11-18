# JPEG codec diagnosis

The original failure was **numeric incompatibility in BOTH JPEG decoding and the historical JPEG re-encoding step**, not just a resize or divide-by255 error. Pillow13.0.0 reports JPEG ABI6.2, implemented by libjpeg-turbo3.2.0. Original prepare_image applies EXIF once, converts RGB, Lanczos-pads, saves JPEGquality95/subsampling0, and the loader decodes that JPEG before normalization. No ImageCms/ICC conversion or draft decode is used. Full-resolution default libjpeg decoding is accurate integer IDCT (ISLOW) with fancy/smooth chroma upsampling. The encoder uses quality95/4:4:4, ordinary tables/non-optimized Huffman coding and accurate integer DCT.

The old Dart image4.10.1 implementation has its own JPEG decode/color/IDCT and encode/DCT/quantization paths. Identical original source SHA256 values were checked. On the four original cases, decoded RGB differs by3–4levels before resizing; the final prepared input differs by11–15. Normalization and same-decoded-pixel Lanczos geometry were already exact.

## Controlled codec crossover evidence

Decoding the EXACT SAME Pillow-prepared JPEG bytes with Dart still differs by3levels. Encoding the EXACT SAME correct padded RGB pixels with Dart, then decoding with Pillow, differs by6–8levels. Thus decoder replacement alone would leave a genuine encoder mismatch. See CODEC_CROSS_OVER_DIAGNOSIS.csv for mean errors and differing-pixel percentages. This isolates codec numeric paths; it does not claim that one unmeasured IDCT/upsampling coefficient accounts for every residual. ICC and EXIF cannot explain the original four failures: those inputs have orientation1and no applied color management; channel order and shape match.

Replacing both lossy codec boundaries with pinned libjpeg-turbo3.2.0 makes all raw/prepared pixels and normalized tensors exact on32development inputs. Geometry and normalization were not altered. The historical JPEG round trip is retained exactly once after padding; no extra recompression is added to the selected raw input. The image picker's optional forced quality95transcode was removed.

Extended tests exposed a SEPARATE image4.10.1 orientation3bug: its flip-both operation leaves the middle row unflipped for odd heights. The initial fixture result is preserved in FIXTURE_GATE_initial.json. Explicit integer-coordinate EXIF mappings replaced that operation, and all8orientations now pass across4:4:4/4:2:2/4:2:0 fixtures.

PIXEL_DIFFERENCES.csv contains raw/resized/prepared max, mean, percent differing and per-channel statistics before/after. SPATIAL_DIFFERENCES.csv provides8×8tile diagnostics. Four300DPI PNG/SVG diagnostic panels use actual development inputs; they are not new thesis model-performance results.

Sources: [Pillow13decoder](https://raw.githubusercontent.com/python-pillow/Pillow/13.0.0/src/libImaging/JpegDecode.c), [Pillow13encoder](https://raw.githubusercontent.com/python-pillow/Pillow/13.0.0/src/libImaging/JpegEncode.c), and pinned local libjpeg-turbo source. No ICC transform is applied; profiles are not treated as an instruction to change RGB values.
