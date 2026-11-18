"""Create a versioned two-image exclusion proposal without changing any dataset.

This is a review artifact, not dataset approval and not a training launcher.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from core import ROOT, PROJECT, CLASSES, config, digest

REMOVED = {
    'ISIC_0012887_downsampled': {
        'label': 'MEL', 'counterpart': 'ISIC_0012989_downsampled', 'counterpart_split': 'validation'},
    'ISIC_0013163_downsampled': {
        'label': 'NV', 'counterpart': 'ISIC_0013109_downsampled', 'counterpart_split': 'test'},
}


def save_csv(path, rows, fields):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = ROOT / config('M1')['split_manifest']
    source_hash = digest(source)
    with source.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    excluded = [row for row in rows if row['image'] in REMOVED]
    if len(excluded) != 2 or {row['image'] for row in excluded} != set(REMOVED):
        raise ValueError('Expected both proposed exclusion IDs exactly once')
    for row in excluded:
        if row['split'] != 'train' or row['label'] != REMOVED[row['image']]['label']:
            raise ValueError('Refusing to exclude any held-out record or mismatched label')
    filtered = [row for row in rows if row['image'] not in REMOVED]
    for split in ['validation', 'test']:
        if [row for row in filtered if row['split'] == split] != [row for row in rows if row['split'] == split]:
            raise ValueError('Held-out records changed')
    if len({r['image'] for r in filtered}) != len(filtered):
        raise ValueError('Duplicate image IDs in proposal')
    lesion_owner = {}
    for row in filtered:
        if lesion_owner.setdefault(row['lesion_id'], row['split']) != row['split']:
            raise ValueError('Known lesion overlap in proposal')
        if not (ROOT / config('M1')['data_root'] / row['processed_relative_path']).is_file():
            raise FileNotFoundError(row['processed_relative_path'])
    version = datetime.now(timezone.utc).strftime('v%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    output = PROJECT / 'runs' / version
    output.mkdir(exist_ok=False)
    (output / 'data').mkdir()
    (output / 'configs').mkdir()
    proposed = output / 'data/manifest.csv'
    save_csv(proposed, filtered, fields)
    excluded_fields = fields + ['reason', 'retained_counterpart', 'retained_counterpart_split']
    save_csv(output / 'data/proposed_exclusions.csv', [dict(row,
        reason='visually_near_identical_to_held_out_image_under_different_lesion_id',
        retained_counterpart=REMOVED[row['image']]['counterpart'],
        retained_counterpart_split=REMOVED[row['image']]['counterpart_split']) for row in excluded], excluded_fields)
    statistics = []
    for split in ['train', 'validation', 'test']:
        counts = Counter(row['label'] for row in filtered if row['split'] == split)
        statistics.append({'split': split, **{c: counts[c] for c in CLASSES}, 'total': sum(counts.values())})
    save_csv(output / 'data/class_counts.csv', statistics, ['split', *CLASSES, 'total'])
    candidates_path = ROOT / 'dataset/audit/duplicate_candidates.csv'
    with candidates_path.open(newline='') as handle:
        pairs = [row for row in csv.DictReader(handle)
                 if row['both_included'] == 'True' and row['cross_boundary'] == 'True']
    remaining = [row for row in pairs if row['image_a'] not in REMOVED and row['image_b'] not in REMOVED]
    counts = Counter((row['dataset_a'], row['dataset_b']) for row in remaining)
    approval = {'approved': False, 'status': 'awaiting_user_review',
                'reason': 'User requested exclusions prepared for review before any training',
                'review_document': str((output / 'EXCLUSION_REVIEW.md').relative_to(ROOT)),
                'manifest_sha256': digest(proposed), 'original_manifest_sha256': source_hash,
                'known_lesion_disjoint': True, 'clean_visual_independence_established': False,
                'remaining_isic_cross_split_candidates': counts[('ISIC2019', 'ISIC2019')],
                'remaining_isic_pad_candidates': counts[('ISIC2019', 'PADUFES20')],
                'original_data_approval_file_unchanged': True}
    (output / 'data_approval.json').write_text(json.dumps(approval, indent=2) + '\n')
    summary = {'campaign_version': version, 'status': 'awaiting_manifest_review',
               'created_utc': datetime.now(timezone.utc).isoformat(),
               'hard_full_configuration_budget': 5, 'full_configurations_started': 0,
               'data_approval': approval, 'sample_counts': statistics,
               'note': 'No smoke optimization, full training, checkpoint loading or predictions produced.',
               'models': {}}
    for key in ['M1', 'M2', 'M3', 'M4', 'M5']:
        cfg = config(key)
        cfg['split_manifest'] = str(proposed.relative_to(ROOT))
        cfg['checkpoint_dir'] = str((output / f'{key}_seed42/checkpoints').relative_to(ROOT))
        cfg['source_config_sha256'] = digest(PROJECT / 'configs' / f'{key}.json')
        cfg['campaign_version'] = version
        cfg['data_approval_file'] = str((output / 'data_approval.json').relative_to(ROOT))
        cfg['evidence_status'] = 'not_started_manifest_review_pending'
        if key == 'M5':
            cfg['initialization']['student_checkpoint'] = str((output / 'M3_seed42/checkpoints/best.json').relative_to(ROOT))
            cfg['initialization']['teacher_checkpoint'] = str((output / 'M4_seed42/checkpoints/best.json').relative_to(ROOT))
        (output / 'configs' / f'{key}.json').write_text(json.dumps(cfg, indent=2) + '\n')
        checkpoint_dir = ROOT / cfg['checkpoint_dir']
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / 'README.md').write_text('Reserved; no checkpoint exists. Future epoch checkpoint files must be immutable and uniquely named; best.json may select the best file.\n')
        summary['models'][key] = {'status': 'awaiting_manifest_review', 'epochs_completed': 0,
                                  'duration_seconds': None, 'best_checkpoint': None,
                                  'validation_metrics': None, 'peak_vram_bytes': None,
                                  'configuration': str((output / 'configs' / f'{key}.json').relative_to(ROOT))}
    (output / 'experiment_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    table = ['| Split | NV | MEL | BCC | SCC | Total |', '|---|---:|---:|---:|---:|---:|']
    table.extend('| ' + row['split'] + ' | ' + ' | '.join(f'{row[c]:,}' for c in [*CLASSES, 'total']) + ' |' for row in statistics)
    report = '''# Proposed training exclusions — review required

**No smoke optimization or full training has started. This manifest is not approved yet.**

The user requested preparation of these exclusions for review before training.
Only two training rows are removed in a new manifest; no image, original
manifest, split CSV, source metadata, or validation/test assignment is changed.
The loader must use this manifest explicitly; the excluded image files remain
in their original training folders and must not be rediscovered by folder scans.

| Excluded training image | Class | Retained counterpart | Partition |
|---|---|---|---|
| ISIC_0012887_downsampled | MEL | ISIC_0012989_downsampled | Validation |
| ISIC_0013163_downsampled | NV | ISIC_0013109_downsampled | Test |

Both proposed removals are the sole training image for their source lesion ID;
there are no other same-ID training siblings to retain. The suspicion comes from
the prior recorded visual review, not new test-set model evaluation. No test
pixels or predictions are used in creating this manifest.

''' + '\n'.join(table) + '''

Training changes: **13,686 → 13,684** (NV −1, MEL −1). Known lesion groups:
train 7,004, validation 1,500, test 1,500. Validation/test rows remain identical
and known lesion IDs remain disjoint. Class weights must be recalculated from
the 13,684 retained training rows at run time; all other prepared settings remain.

## Residual concerns

These two exclusions remove the two visually near-identical pairs already
identified, but do not establish a clean split. **1,995 ISIC cross-split candidate
pairs and 187 ISIC–PAD candidate pairs remain** under the original perceptual
hash screen. Candidate pairs are not confirmed duplicate counts. Other exact
duplicates remain within partitions. Patient independence is still unverified.
PAD remains external-only, unchanged. No automatic split regeneration occurs.

## Artifacts in this versioned folder

- `data/manifest.csv`: proposed manifest referencing existing prepared images.
- `data/proposed_exclusions.csv`: excluded rows, reasons and held-out counterparts.
- `data/class_counts.csv`: revised per-class/per-split counts.
- `data_approval.json`: **approved=false**, with original/proposed manifest hashes.
- `configs/M1.json` ... `configs/M5.json`: exactly five config snapshots pointing
  to the proposal and unique checkpoint directories; no hyperparameter changes.
- `experiment_summary.json`: persistent five-model ledger, all waiting for review.

If training is subsequently authorized, immutable epoch checkpoint names and a
`best.json` pointer will avoid overwriting model files. Fresh M3/M4 parents for
M5 must be selected from this same version and recorded by SHA-256.

Prior audit evidence: `dataset/audit/visual_review.json` and
`dataset/audit/duplicate_candidates.csv` (paths relative to repository root).
'''
    (output / 'EXCLUSION_REVIEW.md').write_text(report)
    assert digest(source) == source_hash
    assert not list(output.rglob('*.pt'))
    print(json.dumps({'review_folder': str(output.relative_to(ROOT)), 'manifest_sha256': digest(proposed),
                      'counts': statistics, 'remaining_cross_boundary_candidates': len(remaining),
                      'approved': False, 'training_started': False}, indent=2))


if __name__ == '__main__':
    main()
