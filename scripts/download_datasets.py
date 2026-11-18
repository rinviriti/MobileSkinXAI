"""Download frozen official releases only; preserve existing files and raw archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / 'dataset'
ISIC = 'https://isic-archive.s3.amazonaws.com/challenges/2019/'


def download(url, destination, expected_hash=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        size = int(response.headers.get('Content-Length', 0))
        resolved_url = response.url
        etag = response.headers.get('ETag')
    if not size:
        raise ValueError(f'Cannot verify size before download: {url}')
    free = shutil.disk_usage(destination.parent).free
    print(f'{destination.name}: {size:,} bytes; free {free:,} bytes', flush=True)
    if free < size * 3 + 5_000_000_000:
        raise OSError('Insufficient reserve for archive, extraction and prepared images')
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError(f'Existing file size mismatch; preserving it: {destination}')
        print(f'Reusing existing {destination}', flush=True)
    else:
        partial = destination.with_name(destination.name + '.partial')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f'Oversized partial file: {partial}')
        def fetch(bounds):
            start, end = bounds
            for attempt in range(4):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'}
                    if etag:
                        headers['If-Match'] = etag
                    req = urllib.request.Request(resolved_url, headers=headers)
                    with urllib.request.urlopen(req, timeout=120) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Server did not honor the exact byte range')
                        data = response.read()
                    if len(data) != end - start + 1:
                        raise ValueError('Incomplete range')
                    return data
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        if size > 100_000_000:
            block = 8 * 1024 * 1024
            ranges = [(start, min(start + block, size) - 1) for start in range(offset, size, block)]
            with ThreadPoolExecutor(max_workers=32) as pool, partial.open('ab') as output:
                copied = offset
                last = time.monotonic()
                for chunk in pool.map(fetch, ranges, buffersize=32):
                    output.write(chunk)
                    copied += len(chunk)
                    if time.monotonic() - last > 20:
                        print(f'{destination.name}: {copied:,}/{size:,} bytes', flush=True)
                        last = time.monotonic()
        else:
            if offset:
                raise FileExistsError(f'Inspect incomplete small file before retry: {partial}')
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as response, partial.open('xb') as output:
                shutil.copyfileobj(response, output)
        if partial.stat().st_size != size:
            raise ValueError(f'Incomplete download: {partial}')
        partial.rename(destination)
    digest = hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest()
    if expected_hash and digest != expected_hash:
        raise ValueError(f'Official checksum mismatch: {destination}')
    record = {'url': url, 'size': size, 'sha256': digest, 'official_sha256': expected_hash, 'etag': etag}
    destination.with_name(destination.name + '.source.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Verified {destination.name}: {digest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['isic', 'pad'])
    args = parser.parse_args()
    if args.dataset == 'isic':
        for name in ['ISIC_2019_Training_GroundTruth.csv', 'ISIC_2019_Training_Metadata.csv',
                     'ISIC_2019_Training_Input.zip']:
            download(ISIC + name, ROOT / 'ISIC2019' / name)
    else:
        base = 'https://data.mendeley.com/public-api/datasets/zr7vgbcyr2/'
        files = []
        for folder in ['root', '6839ab16-69b8-4a2f-ba12-a3ccda4c89f5']:
            req = urllib.request.Request(base + f'files?folder_id={folder}&version=1',
                                         headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                files.extend(json.load(response))
        target = ROOT / 'PADUFES20/raw'
        target.mkdir(parents=True, exist_ok=True)
        (target / 'official_file_listing.json').write_text(json.dumps(files, indent=2) + '\n')
        total = sum(row['content_details']['size'] for row in files)
        if shutil.disk_usage(target).free < total * 3 + 5_000_000_000:
            raise OSError('Insufficient disk reserve for complete PAD release')
        print(f'PAD official release: {total:,} bytes across {len(files)} files', flush=True)
        for row in files:
            details = row['content_details']
            download(details['download_url'], target / row['filename'], details['sha256_hash'])


if __name__ == '__main__':
    main()
