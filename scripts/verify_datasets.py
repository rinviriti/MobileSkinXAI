"""Verify preparation against raw audit manifests and summarize remaining concerns."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from PIL import Image

from audit_datasets import ROOT, REPORT, read_csv


def check_raw(row):
    if row['status'] != 'ok':
        return None
    with Path(row['source']).open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != row['sha256']:
        raise ValueError(f'Source changed after audit: {row["source"]}')


def main():
    raw = read_csv(REPORT / 'raw_manifest.csv')
    summary = json.loads((REPORT / 'summary.json').read_text())
    for name, digest in summary['input_csv_sha256_before_preparation'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input metadata or split changed: {name}')
    isic = ROOT / 'ISIC2019/preprocessed/rgb224_letterbox_gray114_v1/full'
    pad = ROOT / 'PADUFES20/preprocessed/rgb224_letterbox_gray114_v1'
    prepared_counts = {}
    for dataset, folder in [('ISIC2019', isic), ('PADUFES20', pad)]:
        manifest = read_csv(folder / 'manifest.csv')
        expected = {row['image']: row for row in raw if row['dataset'] == dataset and not row['exclusion']}
        seen = set()
        for row in manifest:
            image = row['image']
            if image in seen or image not in expected:
                raise ValueError(f'Duplicate or unknown prepared image: {image}')
            seen.add(image)
            original = expected[image]
            for field in ['label', 'split', 'lesion_id']:
                if row[field] != original[field]:
                    raise ValueError(f'Prepared manifest mismatch: {image}/{field}')
            if dataset == 'PADUFES20' and row['split'] != 'external':
                raise ValueError('PAD image leaked into development partition')
            with Image.open(folder / row['processed_relative_path']) as image_file:
                image_file.load()
                if image_file.size != (224, 224) or image_file.mode != 'RGB':
                    raise ValueError(f'Invalid prepared image: {image}')
        if seen != set(expected):
            raise ValueError(f'Incomplete prepared coverage: {dataset}')
        if len(list(folder.rglob('*.jpg'))) != len(manifest):
            raise ValueError(f'Unexpected JPEG files outside manifest: {folder}')
        prepared_counts[dataset] = dict(Counter(row['split'] + '/' + row['label'] for row in manifest))
    split_dir = ROOT / 'ISIC2019/splits/lesion_stratified_seed42'
    recorded = json.loads((isic / 'summary.json').read_text())['split_sha256']
    for name, digest in recorded.items():
        if hashlib.sha256((split_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen split changed: {name}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check_raw, raw, buffersize=8))
    summary['final_verification'] = {
        'prepared_images_rgb224_and_complete': True,
        'raw_sha256_unchanged_after_preparation': True,
        'split_sha256_unchanged_after_preparation': True,
        'source_metadata_sha256_unchanged_after_preparation': True,
        'pad_external_only': True,
        'augmentation_applied': False,
        'prepared_counts': prepared_counts,
        'split_provenance': 'New split explicitly authorized by user; historical CSVs unavailable',
        'training_clearance': 'Not granted: review duplicate candidate report and metadata issues before training',
    }
    if (REPORT / 'visual_review.json').exists():
        summary['visual_review'] = json.loads((REPORT / 'visual_review.json').read_text())
    (REPORT / 'final_verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# MobileSkinXAI dataset verification', '',
             'ISIC split: newly created seed 42 with explicit user authorization; historical CSVs were unavailable.', '',
             '| Dataset / partition | NV | MEL | BCC | SCC | Total |',
             '|---|---:|---:|---:|---:|---:|']
    for dataset, split in [('ISIC2019', 'train'), ('ISIC2019', 'validation'),
                           ('ISIC2019', 'test'), ('PADUFES20', 'external')]:
        counts = prepared_counts[dataset]
        values = [counts.get(split + '/' + label, 0) for label in ['NV', 'MEL', 'BCC', 'SCC']]
        lines.append('| ' + dataset + ' / ' + split + ' | ' + ' | '.join(map(str, values + [sum(values)])) + ' |')
    lines += ['', 'All prepared images decode as 224x224 RGB. Raw image, source metadata and frozen split hashes are unchanged.',
              'No augmentation, model training or pretrained-weight download occurred. PAD is external-only.', '',
              '## Integrity and exclusions', '']
    for dataset, stats in summary['datasets'].items():
        lines += [f'- {dataset}: {stats["raw"]:,} raw records; file status {stats["file_status"]}; exclusions {stats["exclusions"]}.']
    lines += [f'- Metadata issues: {summary["metadata_issues"]}.', '',
              '## Unresolved leakage concerns', '',
              f'- Duplicate candidate pairs: {summary["duplicate_pairs"]}.',
              f'- Pairs across partitions or datasets with both images included: {summary["included_cross_boundary_pairs"]}.',
              '- These are candidate pairs, not a count of confirmed duplicate images. Review dataset/audit/duplicate_candidates.csv before training.',
              '- No split or image was silently reassigned to resolve duplicate candidates.',
              *['- ' + limitation for limitation in summary['limitations']], '',
              '## Artifacts', '',
              '- [Complete verification](dataset/audit/final_verification.json)',
              '- [Raw manifest and hashes](dataset/audit/raw_manifest.csv)',
              '- [Exclusions](dataset/audit/exclusions.csv)',
              '- [Per-class statistics](dataset/audit/class_statistics.csv)',
              '- [Metadata issues](dataset/audit/metadata_issues.csv)',
              '- [Duplicate review candidates](dataset/audit/duplicate_candidates.csv)',
              '- [Source licenses, paths and reproduction](DATASETS.md)', '']
    if 'visual_review' in summary:
        lines += ['## Visual screening', '',
                  'Two visually near-identical pairs under different lesion IDs cross train/validation and train/test. These remain unresolved.',
                  'Candidate counts include false positives and are not confirmed duplicate totals.',
                  '[Screening notes](dataset/audit/visual_review.json) and [18-pair contact sheet](dataset/audit/strong_candidate_contact_sheet.jpg).', '']
    (ROOT.parent / 'dataset-report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
