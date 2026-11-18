"""Audit raw ISIC/PAD files, frozen assignments and visual duplicate candidates.

Never changes labels, source files or split assignments. Perceptual hashes are
screening evidence only and are not a guarantee of visual independence.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from preprocess_isic2019 import CLASSES, prepare_image

ROOT = Path(__file__).resolve().parents[1] / 'dataset'
REPORT = ROOT / 'audit'


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def unique_index(rows, key, issues, source):
    output = {}
    for row in rows:
        value = row[key].strip()
        if not value or value in output:
            issues.append({'dataset': source, 'image': value, 'reason': f'duplicate_or_empty_{key}'})
        output[value] = row
    return output


def build_records():
    issues, records = [], []
    data = ROOT / 'ISIC2019'
    labels = read_csv(data / 'ISIC_2019_Training_GroundTruth.csv')
    metadata_rows = read_csv(data / 'ISIC_2019_Training_Metadata.csv')
    unique_index(labels, 'image', issues, 'ISIC2019_labels')
    metadata = unique_index(metadata_rows, 'image', issues, 'ISIC2019_metadata')
    assignments = {}
    split_dir = data / 'splits/lesion_stratified_seed42'
    for split in ['train', 'validation', 'test']:
        for row in read_csv(split_dir / f'{split}.csv'):
            if row['image'] in assignments:
                issues.append({'dataset': 'ISIC2019', 'image': row['image'], 'reason': 'duplicate_split_image'})
            assignments[row['image']] = dict(row, split=split)
    for row in labels:
        image = row['image']
        values = {key: float(value) for key, value in row.items() if key != 'image'}
        valid = all(value in (0, 1) for value in values.values()) and sum(values.values()) == 1
        label = next((key for key, value in values.items() if value == 1), '')
        meta = metadata.get(image, {})
        lesion = (meta.get('lesion_id') or '').strip()
        exclusion = ''
        if not valid:
            exclusion = 'invalid_one_hot_label'
        elif label not in CLASSES:
            exclusion = 'outside_four_classes'
        elif image not in metadata:
            exclusion = 'missing_metadata'
        elif not lesion:
            exclusion = 'missing_lesion_id'
        assignment = assignments.get(image, {})
        if not exclusion and not assignment:
            issues.append({'dataset': 'ISIC2019', 'image': image, 'reason': 'eligible_image_missing_from_split'})
        if assignment and (assignment['label'] != label or assignment['lesion_id'] != lesion or exclusion):
            issues.append({'dataset': 'ISIC2019', 'image': image, 'reason': 'split_metadata_mismatch'})
        records.append({'dataset': 'ISIC2019', 'image': image, 'label': label, 'original_label': label,
                        'lesion_id': lesion, 'patient_id': '', 'split': assignment.get('split', 'excluded'),
                        'exclusion': exclusion, 'source': str(data / 'images/ISIC_2019_Training_Input' / f'{image}.jpg')})
    pad = ROOT / 'PADUFES20/raw'
    files = list(pad.rglob('metadata.csv'))
    if len(files) != 1:
        raise ValueError(f'Expected one official PAD metadata.csv, found {files}')
    pad_rows = read_csv(files[0])
    unique_index(pad_rows, 'img_id', issues, 'PADUFES20_metadata')
    image_paths = defaultdict(list)
    for path in pad.rglob('*.png'):
        image_paths[path.name].append(path)
    for row in pad_rows:
        image = row['img_id'].strip()
        paths = image_paths.get(image, [])
        if len(paths) > 1:
            issues.append({'dataset': 'PADUFES20', 'image': image, 'reason': 'multiple_source_files_for_id'})
        original = row['diagnostic'].strip()
        label = 'NV' if original == 'NEV' else original
        records.append({'dataset': 'PADUFES20', 'image': image, 'label': label, 'original_label': original,
                        'lesion_id': row.get('lesion_id', '').strip(),
                        'patient_id': row.get('patient_id', '').strip(), 'split': 'external',
                        'exclusion': '' if label in CLASSES else 'outside_four_classes',
                        'source': str(paths[0]) if paths else str(pad / 'MISSING' / image)})
    # Detect lesion-label inconsistencies without assuming one diagnosis per patient.
    groups = defaultdict(list)
    for row in records:
        if row['lesion_id']:
            patient = row['patient_id'] if row['dataset'] == 'PADUFES20' else ''
            groups[(row['dataset'], patient, row['lesion_id'])].append(row)
    for (dataset, patient, lesion), rows in groups.items():
        if len({row['label'] for row in rows}) > 1:
            for row in rows:
                issues.append({'dataset': dataset, 'image': row['image'], 'reason': f'conflicting_labels_for_lesion:{patient}:{lesion}'})
                row['exclusion'] = 'conflicting_lesion_labels'
        if dataset == 'ISIC2019' and len({row['split'] for row in rows if row['split'] != 'excluded'}) > 1:
            issues.append({'dataset': dataset, 'image': lesion, 'reason': 'lesion_overlap_between_splits'})
    expected_isic = {Path(row['source']).name for row in records if row['dataset'] == 'ISIC2019'}
    expected_pad = {row['image'] for row in records if row['dataset'] == 'PADUFES20'}
    for dataset, actual, expected in [
        ('ISIC2019', {p.name for p in (data / 'images/ISIC_2019_Training_Input').glob('*.jpg')}, expected_isic),
        ('PADUFES20', set(image_paths), expected_pad),
    ]:
        for name in sorted(actual - expected):
            issues.append({'dataset': dataset, 'image': name, 'reason': 'unreferenced_image'})
    return records, issues


def scan(row):
    row = dict(row)
    path = Path(row['source'])
    row['status'] = 'ok'
    try:
        with path.open('rb') as handle:
            row['sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            image = ImageOps.exif_transpose(image).convert('RGB')
            row['width'], row['height'] = image.size
            row['pixel_sha256'] = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
            low = dctn(gray, type=2, norm='ortho')[:8, :8].flatten()
            bits = low > np.median(low[1:])
            bits[0] = False
            row['phash'] = f'{int.from_bytes(np.packbits(bits).tobytes()):016x}'
            small = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
            row['dhash'] = f'{int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes()):016x}'
    except (Exception, Image.DecompressionBombError) as exc:
        row['status'] = 'missing' if not path.exists() else 'corrupt_or_unreadable'
        row['exclusion'] = row['status']
        row['error'] = str(exc)
    return row


class BKTree:
    def __init__(self):
        self.root = None

    def query(self, value, threshold):
        todo = [self.root] if self.root else []
        while todo:
            node = todo.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                for index in node[1]:
                    yield index, distance
            todo.extend(child for edge, child in node[2].items()
                        if distance - threshold <= edge <= distance + threshold)

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]


def duplicate_audit(records):
    tree, pairs = BKTree(), []
    for i, row in enumerate(records):
        if row['status'] != 'ok':
            continue
        value = int(row['phash'], 16)
        for j, distance in tree.query(value, 6):
            other = records[j]
            kind = ('exact_file' if row['sha256'] == other['sha256'] else
                    'exact_pixels' if row['pixel_sha256'] == other['pixel_sha256'] else 'suspected_near_duplicate')
            boundary = row['dataset'] != other['dataset'] or row['split'] != other['split']
            included = not row['exclusion'] and not other['exclusion']
            pairs.append({'dataset_a': other['dataset'], 'image_a': other['image'], 'label_a': other['label'],
                          'split_a': other['split'], 'lesion_a': other['lesion_id'],
                          'dataset_b': row['dataset'], 'image_b': row['image'], 'label_b': row['label'],
                          'split_b': row['split'], 'lesion_b': row['lesion_id'], 'kind': kind,
                          'phash_distance': distance,
                          'dhash_distance': (int(row['dhash'], 16) ^ int(other['dhash'], 16)).bit_count(),
                          'cross_boundary': boundary, 'both_included': included,
                          'label_conflict': row['label'] != other['label']})
        tree.add(value, i)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if (REPORT / 'summary.json').exists():
        raise FileExistsError('Preserving completed audit; use its manifests rather than overwriting')
    input_paths = list((ROOT / 'ISIC2019/splits/lesion_stratified_seed42').glob('*.csv'))
    input_paths += [ROOT / 'ISIC2019/ISIC_2019_Training_GroundTruth.csv',
                    ROOT / 'ISIC2019/ISIC_2019_Training_Metadata.csv']
    input_paths += list((ROOT / 'PADUFES20/raw').rglob('metadata.csv'))
    input_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in input_paths}
    records, issues = build_records()
    scanned = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, row in enumerate(pool.map(scan, records, buffersize=args.workers * 2), 1):
            scanned.append(row)
            if number % 500 == 0:
                print(f'Integrity and hashes: {number}/{len(records)}', flush=True)
    fields = ['dataset', 'image', 'label', 'original_label', 'lesion_id', 'patient_id', 'split',
              'exclusion', 'source', 'status', 'width', 'height', 'sha256', 'pixel_sha256', 'phash', 'dhash', 'error']
    write_csv(REPORT / 'raw_manifest.csv', scanned, fields)
    write_csv(REPORT / 'exclusions.csv', [r for r in scanned if r['exclusion']], fields)
    write_csv(REPORT / 'metadata_issues.csv', issues, ['dataset', 'image', 'reason'])
    print('Searching perceptual hash neighbors (64-bit pHash distance <=6)...', flush=True)
    pairs = duplicate_audit(scanned)
    pair_fields = ['dataset_a', 'image_a', 'label_a', 'split_a', 'lesion_a', 'dataset_b', 'image_b',
                   'label_b', 'split_b', 'lesion_b', 'kind', 'phash_distance', 'dhash_distance',
                   'cross_boundary', 'both_included', 'label_conflict']
    write_csv(REPORT / 'duplicate_candidates.csv', pairs, pair_fields)
    summary = {'raw_images': len(scanned), 'metadata_issues': len(issues), 'datasets': {},
               'input_csv_sha256_before_preparation': input_hashes,
               'duplicate_pairs': dict(Counter(p['kind'] for p in pairs)),
               'included_cross_boundary_pairs': dict(Counter(p['kind'] for p in pairs if p['both_included'] and p['cross_boundary'])),
               'limitations': ['ISIC patient identifiers unavailable in challenge metadata; patient independence unverified.',
                               'pHash distance <=6 is a heuristic; candidates need visual review. Crops, rotations, and other near duplicates can escape detection.',
                               'No split assignments changed. Suspected near duplicates were not silently excluded.',
                               'PAD patient and lesion IDs are dataset-local; cross-dataset identity cannot be established from IDs.']}
    stats = []
    for dataset in ['ISIC2019', 'PADUFES20']:
        subset = [r for r in scanned if r['dataset'] == dataset]
        included = [r for r in subset if not r['exclusion']]
        summary['datasets'][dataset] = {
            'raw': len(subset), 'raw_class_counts': dict(Counter(r['original_label'] for r in subset)),
            'included': len(included), 'included_class_counts': dict(Counter(r['label'] for r in included)),
            'exclusions': dict(Counter(r['exclusion'] for r in subset if r['exclusion'])),
            'file_status': dict(Counter(r['status'] for r in subset)),
            'lesions_included': len({r['lesion_id'] for r in included if r['lesion_id']}),
            'patient_lesion_pairs_included': len({(r['patient_id'], r['lesion_id']) for r in included if r['lesion_id']}),
            'patients_included': len({r['patient_id'] for r in included if r['patient_id']}),
        }
        if dataset == 'PADUFES20':
            owners = defaultdict(set)
            for row in subset:
                owners[row['lesion_id']].add(row['patient_id'])
            summary['datasets'][dataset]['bare_lesion_ids_reused_across_patients'] = sum(len(v) > 1 for v in owners.values())
            summary['limitations'].append('PAD bare lesion IDs repeat across patients; consistency uses (patient_id, lesion_id), both source fields retained. Published lesion count uses bare IDs.')
        for (split, label), count in sorted(Counter((r['split'], r['label']) for r in included).items()):
            stats.append({'dataset': dataset, 'split': split, 'label': label, 'images': count})
    write_csv(REPORT / 'class_statistics.csv', stats, ['dataset', 'split', 'label', 'images'])
    (REPORT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
