"""Download new official timm ImageNet artifacts, pinned to publisher revisions."""
import json
from pathlib import Path
import shutil
import urllib.request

from core import PROJECT, config, digest, model, seed_all


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'MobileSkinXAI-final-preflight'}), timeout=60) as response:
        return json.load(response)


def main():
    report = {}
    for key in ['M1', 'M2', 'M3', 'M4']:
        cfg = config(key)
        repo = cfg['initialization']['official_repo']
        info = get_json(f'https://huggingface.co/api/models/{repo}?blobs=true')
        revision = info['sha']
        entry = next(row for row in info['siblings'] if row['rfilename'] == 'model.safetensors')
        official_hash = entry.get('lfs', {}).get('sha256')
        folder = PROJECT / 'pretrained' / key
        folder.mkdir(exist_ok=True, parents=True)
        artifact = folder / 'model.safetensors'
        url = f'https://huggingface.co/{repo}/resolve/{revision}/model.safetensors'
        if not artifact.exists():
            partial = artifact.with_suffix('.partial')
            with urllib.request.urlopen(url, timeout=180) as source, partial.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            partial.rename(artifact)
        checksum = digest(artifact)
        if official_hash and official_hash != checksum:
            raise ValueError(f'Publisher hash mismatch: {repo}')
        provenance = {'purpose': 'fresh official ImageNet initialization; never a demonstration checkpoint',
                      'repo_id': repo, 'revision': revision, 'url': url, 'sha256': checksum,
                      'publisher_sha256': official_hash, 'bytes': artifact.stat().st_size,
                      'license_from_model_card': info.get('cardData', {}).get('license'),
                      'dataset_from_model_card': info.get('cardData', {}).get('datasets')}
        (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        seed_all()
        net = model(cfg, pretrained=True)
        report[key] = dict(provenance, strict_state_load='PASS', four_class_parameters=sum(p.numel() for p in net.parameters()))
        print(f'{key}: strict ImageNet load and fresh four-class head PASS', flush=True)
        del net
    (PROJECT / 'reports/pretrained_provenance.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
