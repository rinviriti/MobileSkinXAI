import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'analysis_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();

  Uint8List? _imageBytes;
  String? _imageName;
  String? _imageSource;
  bool _isPicking = false;

  static const Color primary = Color(0xFF176B65);
  static const Color heading = Color(0xFF213A38);
  static const Color muted = Color(0xFF647773);
  static const Color border = Color(0xFFDCE7E4);

  Future<void> _pickImage(ImageSource source) async {
    if (_isPicking) return;

    setState(() => _isPicking = true);

    try {
      final pickedFile = await _picker.pickImage(
        source: source,
        imageQuality: 95,
        requestFullMetadata: false,
      );

      if (pickedFile == null) return;

      final bytes = await pickedFile.readAsBytes();

      if (!mounted) return;

      setState(() {
        _imageBytes = bytes;
        _imageName = pickedFile.name;
        _imageSource = source == ImageSource.camera
            ? 'Camera'
            : 'Photo library';
      });
    } catch (_) {
      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Unable to open the image. Please try again.'),
        ),
      );
    } finally {
      if (mounted) {
        setState(() => _isPicking = false);
      }
    }
  }

  void _removeImage() {
    setState(() {
      _imageBytes = null;
      _imageName = null;
      _imageSource = null;
    });
  }

  void _openAnalysis() {
    final bytes = _imageBytes;
    final name = _imageName;
    final source = _imageSource;

    if (bytes == null || name == null || source == null) {
      return;
    }

    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (context) => AnalysisScreen(
          imageBytes: bytes,
          imageName: name,
          imageSource: source,
        ),
      ),
    );
  }

  Widget _buildImagePreview() {
    return Container(
      width: double.infinity,
      height: 260,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: border),
      ),
      clipBehavior: Clip.antiAlias,
      child: _imageBytes == null
          ? const Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.add_photo_alternate_outlined,
                  size: 48,
                  color: primary,
                ),
                SizedBox(height: 14),
                Text(
                  'No image selected',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: heading,
                  ),
                ),
                SizedBox(height: 8),
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 24),
                  child: Text(
                    'Take a photo or choose a skin lesion image '
                    'from your gallery.',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 13, height: 1.5, color: muted),
                  ),
                ),
              ],
            )
          : Image.memory(
              _imageBytes!,
              width: double.infinity,
              height: double.infinity,
              fit: BoxFit.contain,
              errorBuilder: (context, error, stackTrace) {
                return const Center(child: Text('Image preview unavailable'));
              },
            ),
    );
  }

  Widget _buildNotice() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFEAF4F1),
        borderRadius: BorderRadius.circular(14),
      ),
      child: const Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, size: 22, color: primary),
          SizedBox(width: 12),
          Expanded(
            child: Text(
              'MobileSkinXAI is a research prototype. '
              'Image review is available, while AI '
              'classification will be enabled after '
              'the trained model is integrated.',
              style: TextStyle(fontSize: 13, height: 1.5, color: heading),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'MobileSkinXAI',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: heading,
          ),
        ),
        actions: [
          IconButton(
            tooltip: 'About this prototype',
            icon: const Icon(Icons.info_outline),
            onPressed: () {
              showAboutDialog(
                context: context,
                applicationName: 'MobileSkinXAI',
                applicationVersion: 'Research Prototype',
                applicationLegalese: 'For research and educational use only.',
                children: [
                  const Text(
                    'MobileSkinXAI explores lightweight, '
                    'explainable deep learning for skin lesion '
                    'image classification. It is not a medical '
                    'diagnostic device.',
                  ),
                ],
              );
            },
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 480),
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(22, 20, 22, 32),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 7,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xFFEAF4F1),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.science_outlined, size: 16, color: primary),
                        SizedBox(width: 6),
                        Text(
                          'RESEARCH PROTOTYPE',
                          style: TextStyle(
                            fontSize: 11,
                            letterSpacing: 0.8,
                            fontWeight: FontWeight.w600,
                            color: primary,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 20),
                  const Text(
                    'Skin image assessment',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.w700,
                      letterSpacing: -0.6,
                      color: heading,
                    ),
                  ),
                  const SizedBox(height: 10),
                  const Text(
                    'Capture or upload a skin lesion image '
                    'to prepare it for analysis.',
                    style: TextStyle(fontSize: 15, height: 1.5, color: muted),
                  ),
                  const SizedBox(height: 28),
                  const Text(
                    'Your image',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: heading,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _buildImagePreview(),
                  if (_imageName != null) ...[
                    const SizedBox(height: 10),
                    Text(
                      _imageName!,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(fontSize: 12, color: muted),
                    ),
                  ],
                  const SizedBox(height: 22),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      onPressed: _isPicking
                          ? null
                          : () => _pickImage(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt_outlined),
                      label: const Text('Take a photo'),
                    ),
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: _isPicking
                          ? null
                          : () => _pickImage(ImageSource.gallery),
                      icon: const Icon(Icons.photo_library_outlined),
                      label: const Text('Choose from gallery'),
                      style: OutlinedButton.styleFrom(
                        minimumSize: const Size.fromHeight(52),
                        foregroundColor: primary,
                        side: const BorderSide(color: border),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14),
                        ),
                      ),
                    ),
                  ),
                  if (_imageBytes != null) ...[
                    const SizedBox(height: 18),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: _openAnalysis,
                        icon: const Icon(Icons.arrow_forward_outlined),
                        label: const Text('Review image'),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Center(
                      child: TextButton.icon(
                        onPressed: _removeImage,
                        icon: const Icon(Icons.close, size: 18),
                        label: const Text('Remove image'),
                        style: TextButton.styleFrom(foregroundColor: muted),
                      ),
                    ),
                  ],
                  const SizedBox(height: 28),
                  _buildNotice(),
                  const SizedBox(height: 24),
                  const Text(
                    'This application is intended for research '
                    'and educational evaluation only. It does '
                    'not diagnose skin cancer or replace '
                    'professional medical advice.',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 12, height: 1.6, color: muted),
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
