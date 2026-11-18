"""Untrained MobileNetV4 export feasibility only, never a final thesis model."""
import argparse
import importlib.metadata
import json
import time

import numpy as np
import torch
from core import PROJECT, config, digest, model, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('format', choices=['onnx', 'tflite'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed_all()
    net = model(config('M3'), pretrained=False).cpu().eval()
    example = torch.randn(1, 3, 224, 224)
    start = time.perf_counter()
    report = {'purpose': 'UNTRAINED architecture feasibility probe, NOT a final model',
              'architecture': config('M3')['architecture'], 'pretrained': False, 'classes': 4,
              'shape': [1, 3, 224, 224], 'dtype': 'float32', 'format': args.format,
              'input_convention': 'NCHW normalized RGB tensor; preprocessing stays outside model',
              'versions': {'torch': torch.__version__, 'timm': importlib.metadata.version('timm')}}
    path = PROJECT / 'preflight' / f'mobilenetv4_UNTRAINED_probe.{args.format}'
    try:
        if args.format == 'onnx':
            import onnx
            import onnxruntime as ort
            torch.onnx.export(net, (example,), str(path), dynamo=True, opset_version=18,
                              input_names=['normalized_rgb'], output_names=['logits'],
                              external_data=False, verify=True)
            onnx.checker.check_model(onnx.load(path))
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            execute = lambda value: session.run(None, {'normalized_rgb': value})[0]
            report['versions'].update(onnx=onnx.__version__, onnxruntime=ort.__version__)
            report['path'] = 'PyTorch -> torch.export-based ONNX exporter -> ONNX Runtime CPU'
        else:
            import litert_torch
            from ai_edge_litert.interpreter import Interpreter
            converted = litert_torch.convert(net, (example,))
            converted.export(str(path))
            interpreter = Interpreter(model_path=str(path), num_threads=2)
            interpreter.allocate_tensors()
            inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError('Unexpected converter IO signature')
            def execute(value):
                if list(inputs[0]['shape']) == [1, 224, 224, 3]:
                    value = value.transpose(0, 2, 3, 1)
                interpreter.set_tensor(inputs[0]['index'], np.ascontiguousarray(value))
                interpreter.invoke()
                return interpreter.get_tensor(outputs[0]['index'])
            operations = sorted({op['op_name'] for op in interpreter._get_ops_details()})
            report['operator_names'] = operations
            report['requires_flex_ops'] = any('Flex' in name for name in operations)
            report['versions'].update({name: importlib.metadata.version(name) for name in ['litert-torch', 'ai-edge-litert', 'litert-converter']})
            report['path'] = 'PyTorch -> official Google LiteRT Torch converter -> .tflite -> LiteRT CPU interpreter'
        errors = []
        for seed in [42, 43, 44]:
            generator = torch.Generator().manual_seed(seed)
            value = torch.randn((1, 3, 224, 224), generator=generator)
            with torch.no_grad():
                expected = net(value).numpy()
            actual = execute(value.numpy())
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
            errors.append(float(np.max(np.abs(actual - expected))))
        report.update(status='PASS', artifact=str(path.relative_to(PROJECT.parent)),
                      sha256=digest(path), artifact_bytes=path.stat().st_size,
                      parity_inputs=3, max_abs_error=max(errors), atol=1e-4, rtol=1e-3)
    except Exception as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - start
        (PROJECT / 'reports' / f'export_{args.format}.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
