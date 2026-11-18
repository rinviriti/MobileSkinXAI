"""Offline environment checks; no datasets, weights, or training are loaded."""
import importlib
import json
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/mobileskinxai-matplotlib'
import torch
from torchvision.models import mobilenet_v3_small


def main():
    torch.set_num_threads(2)
    result = {
        'python': sys.version,
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'scripts': {},
        'libraries': {},
        'models': {},
    }
    for name in ['numpy', 'pandas', 'matplotlib', 'sklearn', 'cv2', 'PIL',
                 'tqdm', 'seaborn', 'torchvision', 'timm']:
        try:
            module = importlib.import_module(name)
            result['libraries'][name] = getattr(module, '__version__', 'import OK')
        except Exception as exc:
            result['libraries'][name] = f'FAIL: {exc}'
    for path in sorted(Path(__file__).parent.glob('*.py')):
        if path.stem == 'verify_environment':
            continue
        try:
            importlib.import_module(path.stem)
            result['scripts'][path.stem] = 'PASS'
        except (Exception, SystemExit) as exc:
            result['scripts'][path.stem] = f'FAIL: {exc}'
    if result['cuda_available']:
        result['gpu'] = torch.cuda.get_device_name(0)
        result['capability'] = torch.cuda.get_device_capability(0)
        x = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4, 4)
        actual = (x @ x.T).cpu()
        expected = x.cpu() @ x.cpu().T
        torch.testing.assert_close(actual, expected)
        torch.cuda.synchronize()
        result['gpu_tensor_operation'] = 'PASS: 4x4 float32 matrix multiplication matches CPU'
    else:
        result['gpu_tensor_operation'] = 'FAIL: CUDA unavailable'
    factories = {'mobilenet_v3_small': lambda: mobilenet_v3_small(weights=None, num_classes=4)}
    for name in ['experiment4_efficientnet_lite0', 'experiment5_mobilenetv4_conv_small',
                 'experiment6_convnext_tiny_teacher', 'experiment7_mobilenetv4_rgb320']:
        if result['scripts'].get(name) == 'PASS':
            module = importlib.import_module(name)
            factories[name] = lambda module=module: module.make_model(pretrained=False)
    for name, factory in factories.items():
        try:
            model = factory().eval()
            result['models'][name] = f'PASS: random initialization, {sum(p.numel() for p in model.parameters())} parameters'
            del model
        except Exception as exc:
            result['models'][name] = f'FAIL: {exc}'
    output = json.dumps(result, indent=2)
    Path('environment-verification.json').write_text(output + '\n')
    print(output)
    failed = not result['cuda_available'] or any(
        str(value).startswith('FAIL:')
        for section in ['scripts', 'libraries', 'models']
        for value in result[section].values()
    )
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
