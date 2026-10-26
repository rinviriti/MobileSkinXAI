import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'results_screen.dart';

class AnalysisScreen extends StatelessWidget {
  const AnalysisScreen({
    super.key,
    required this.imageBytes,
    required this.imageName,
    required this.imageSource,
  });

  final Uint8List imageBytes;
  final String imageName;
  final String imageSource;

  static const Color primary = Color(0xFF176B65);
  static const Color heading = Color(0xFF213A38);
  static const Color muted = Color(0xFF647773);
  static const Color border = Color(0xFFDCE7E4);
  static const Color background = Color(0xFFF8FAF9);

  String get imageSize {
    final bytes = imageBytes.lengthInBytes;

    if (bytes < 1024) {
      return '$bytes B';
    }

    if (bytes < 1024 * 1024) {
      return '${(bytes / 1024).toStringAsFixed(1)} KB';
    }

    return '${(bytes / (1024 * 1024)).toStringAsFixed(2)} MB';
  }

  void openResultsPreview(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) =>
            ResultsScreen(imageBytes: imageBytes, imageName: imageName),
      ),
    );
  }

  Widget sectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 17,
        fontWeight: FontWeight.w700,
        color: heading,
      ),
    );
  }

  Widget detailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 11),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 95,
            child: Text(
              label,
              style: const TextStyle(fontSize: 13, color: muted),
            ),
          ),
          Expanded(
            child: Text(
              value,
              textAlign: TextAlign.end,
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w500,
                color: heading,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget imageCard() {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: border),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        children: [
          Container(
            height: 310,
            width: double.infinity,
            color: const Color(0xFFF1F5F4),
            child: Image.memory(
              imageBytes,
              fit: BoxFit.contain,
              errorBuilder: (_, error, stackTrace) {
                return const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.broken_image_outlined, size: 36, color: muted),
                      SizedBox(height: 8),
                      Text('Unable to display image'),
                    ],
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
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
                const SizedBox(width: 8),
                const Icon(
                  Icons.check_circle_outline,
                  color: primary,
                  size: 20,
                  semanticLabel: 'Image loaded',
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget modelStatus() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(color: border),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.memory_outlined, color: primary, size: 22),
              SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Classification model',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: heading,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: const Color(0xFFFFF4E5),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.schedule_outlined,
                  color: Color(0xFF926117),
                  size: 16,
                ),
                SizedBox(width: 7),
                Text(
                  'Not yet available',
                  style: TextStyle(
                    color: Color(0xFF926117),
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'The classification model is still being '
            'developed. Analysis will become available '
            'after training, validation, and mobile integration.',
            style: TextStyle(color: muted, fontSize: 13, height: 1.6),
          ),
        ],
      ),
    );
  }

  Widget privacyNotice() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFEAF4F1),
        borderRadius: BorderRadius.circular(14),
      ),
      child: const Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.shield_outlined, color: primary, size: 21),
          SizedBox(width: 12),
          Expanded(
            child: Text(
              'This prototype displays the selected image '
              'locally. It does not upload the image to a '
              'server or perform medical analysis.',
              style: TextStyle(color: heading, fontSize: 12, height: 1.6),
            ),
          ),
        ],
      ),
    );
  }

  Widget actionPanel(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 14, 20, 18),
      decoration: const BoxDecoration(
        color: Colors.white,
        border: Border(top: BorderSide(color: border)),
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () => openResultsPreview(context),
                icon: const Icon(Icons.visibility_outlined),
                label: const Text('Preview results layout'),
                style: OutlinedButton.styleFrom(
                  minimumSize: const Size.fromHeight(50),
                  foregroundColor: primary,
                  side: const BorderSide(color: primary),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: null,
                icon: const Icon(Icons.analytics_outlined),
                label: const Text('Analyze image'),
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Analysis will be available after model integration.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 11, color: muted),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: background,
      appBar: AppBar(
        title: const Text(
          'Image review',
          style: TextStyle(
            color: heading,
            fontSize: 19,
            fontWeight: FontWeight.w700,
          ),
        ),
        backgroundColor: background,
        leading: IconButton(
          tooltip: 'Back',
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 520),
            child: Column(
              children: [
                Expanded(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.fromLTRB(20, 18, 20, 24),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Review your image',
                          style: TextStyle(
                            fontSize: 27,
                            fontWeight: FontWeight.w700,
                            color: heading,
                            letterSpacing: -0.5,
                          ),
                        ),
                        const SizedBox(height: 8),
                        const Text(
                          'Check the selected image before '
                          'continuing to model analysis.',
                          style: TextStyle(
                            fontSize: 14,
                            color: muted,
                            height: 1.5,
                          ),
                        ),
                        const SizedBox(height: 24),
                        imageCard(),
                        const SizedBox(height: 26),
                        sectionTitle('Image details'),
                        const SizedBox(height: 10),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(16),
                            border: Border.all(color: border),
                          ),
                          child: Column(
                            children: [
                              detailRow('Source', imageSource),
                              const Divider(height: 1),
                              detailRow('File size', imageSize),
                              const Divider(height: 1),
                              detailRow('Status', 'Image loaded'),
                            ],
                          ),
                        ),
                        const SizedBox(height: 26),
                        sectionTitle('Model readiness'),
                        const SizedBox(height: 12),
                        modelStatus(),
                        const SizedBox(height: 22),
                        privacyNotice(),
                        const SizedBox(height: 22),
                        const Text(
                          'For research and educational '
                          'evaluation only. This application '
                          'does not provide a medical diagnosis.',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 12,
                            height: 1.5,
                            color: muted,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                actionPanel(context),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
