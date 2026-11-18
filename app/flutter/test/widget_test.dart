import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobileskinxai/main.dart';

void main() {
  testWidgets('Original app home and theme remain available', (tester) async {
    await tester.pumpWidget(const MobileSkinXAIApp());
    expect(find.byType(MaterialApp), findsOneWidget);
    expect(find.byType(Scaffold), findsOneWidget);
    final app = tester.widget<MaterialApp>(find.byType(MaterialApp));
    expect(app.title, 'MobileSkinXAI');
    expect(app.theme!.colorScheme.primary, const Color(0xFF176B65));
    expect(tester.takeException(), isNull);
  });
}
