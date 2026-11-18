"""Extract official archives without overwriting any existing source file."""
import argparse
from pathlib import Path
import shutil
import stat
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1] / 'dataset'


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate archive paths: {archive}')
        total = sum(member.file_size for member in members)
        if shutil.disk_usage(destination).free < total + 5_000_000_000:
            raise OSError('Insufficient free space for extraction')
        print(f'{archive}: {len(members)} entries, {total:,} expanded bytes', flush=True)
        for number, member in enumerate(members, 1):
            target = destination / member.filename
            if not target.resolve().is_relative_to(destination.resolve()) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f'Unsafe archive member: {member.filename}')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                checksum = 0
                with target.open('rb') as source:
                    while chunk := source.read(1024 * 1024):
                        checksum = zlib.crc32(chunk, checksum)
                if target.stat().st_size != member.file_size or checksum != member.CRC:
                    raise ValueError(f'Existing file differs; preserving it: {target}')
                continue
            # ZipFile verifies the member CRC as it is read to EOF.
            with bundle.open(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
            if number % 2000 == 0:
                print(f'{archive.name}: {number}/{len(members)}', flush=True)
    print(f'Extraction and CRC checks complete: {archive}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['isic', 'pad'])
    args = parser.parse_args()
    if args.dataset == 'isic':
        extract(ROOT / 'ISIC2019/ISIC_2019_Training_Input.zip', ROOT / 'ISIC2019/images')
    else:
        raw = ROOT / 'PADUFES20/raw'
        outer = ROOT / 'PADUFES20/zr7vgbcyr2-1.zip'
        if outer.exists():
            extract(outer, raw)
        for archive in sorted(raw.rglob('*.zip')):
            extract(archive, archive.parent / archive.stem)


if __name__ == '__main__':
    main()
