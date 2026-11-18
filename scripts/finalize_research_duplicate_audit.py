"""Apply documented identity reviews and conservative training-only exclusions.

Does not load, train, or modify a model. Leaves images, original manifests and
held-out assignments untouched. Approval remains false for unresolved held-out
identity pairs or failed checks. Run against one completed versioned scan.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from audit_datasets import read_csv, write_csv
from audit_final_research_splits import ROOT, AUDIT, BASE_MANIFEST, CLASSES, sha


class Groups:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def identity(row, suffix):
    return row['dataset_' + suffix], row['image_' + suffix]


def run(output):
    if (output / 'final_summary.json').exists():
        raise FileExistsError('Preserving completed audit; create a new version for further adjudication')
    original = read_csv(BASE_MANIFEST)
    raw = read_csv(AUDIT / 'raw_manifest.csv')
    raw_index = {(r['dataset'], r['image']): r for r in raw if r['status'] == 'ok'}
    research = {(r['dataset'], r['image']): r for r in raw if r['status'] == 'ok' and not r['exclusion']}
    pairs = read_csv(output / 'pair_audit.csv')
    orbit_hashes = {(r['dataset'], r['image']): [r[f'phash_d4_{k}'] for k in range(8)]
                    for r in read_csv(output / 'dihedral_hashes.csv')}
    for row in pairs:
        a, b = identity(row, 'a'), identity(row, 'b')
        distances = []
        if a in orbit_hashes:
            distances += [(int(value, 16) ^ int(raw_index[b]['phash'], 16)).bit_count() for value in orbit_hashes[a]]
        if b in orbit_hashes:
            distances += [(int(value, 16) ^ int(raw_index[a]['phash'], 16)).bit_count() for value in orbit_hashes[b]]
        row['phash_distance_best_dihedral'] = min(distances) if distances else ''
    reviews = read_csv(output / 'review_adjudications.csv')
    reviewed = {}
    valid_ids = {r['pair_id'] for r in pairs}
    for review in reviews:
        if (review['pair_id'] not in valid_ids or review['pair_id'] in reviewed or
                review['decision'] not in ['confirmed_near_duplicate', 'distinct_image_pair', 'unresolved'] or
                not review['evidence'].strip() or not review['reviewer'].strip()):
            raise ValueError('Invalid, duplicate or unsupported visual adjudication')
        reviewed[review['pair_id']] = review
    # Include excluded raw records when building identity components: an
    # excluded intermediary must not hide a confirmed train/held-out link.
    groups = Groups(raw_index)
    metadata_groups, exact_groups = {}, {}
    for value, row in raw_index.items():
        if row['lesion_id']:
            lesion_key = (row['dataset'], row['patient_id'] if row['dataset'] == 'PADUFES20' else '', row['lesion_id'])
            if lesion_key in metadata_groups:
                groups.union(value, metadata_groups[lesion_key])
            else:
                metadata_groups[lesion_key] = value
        for attribute in ['sha256', 'pixel_sha256']:
            group_key = attribute, row[attribute]
            if group_key in exact_groups:
                groups.union(value, exact_groups[group_key])
            else:
                exact_groups[group_key] = value
    training_triggers = defaultdict(set)
    for row in pairs:
        a, b = identity(row, 'a'), identity(row, 'b')
        if row['pair_id'] in reviewed:
            review = reviewed[row['pair_id']]
            row['final_decision'] = review['decision']
            row['visual_review_status'] = 'visually_adjudicated'
            row['reviewer'] = review['reviewer']
            row['review_evidence'] += ' Visual review: ' + review['evidence']
        elif row['evidence_category'] in ['confirmed_exact_duplicate_image', 'same_declared_lesion']:
            row['final_decision'] = row['evidence_category']
            row['reviewer'] = 'exact_hash_or_declared_lesion_check'
        else:
            row['final_decision'] = 'unresolved_identity_after_automated_review'
            row['reviewer'] = 'automated_metrics_and_metadata; visual_identity_not_cleared'
        both = a in research and b in research
        if row['final_decision'] in ['confirmed_near_duplicate', 'confirmed_exact_duplicate_image', 'same_declared_lesion']:
            groups.union(a, b)
        if both and row['final_decision'] != 'distinct_image_pair':
            for value in [a, b]:
                if research[value]['dataset'] == 'ISIC2019' and research[value]['split'] == 'train':
                    training_triggers[value].add(row['pair_id'])
    confirmed_members = defaultdict(list)
    for value in research:
        confirmed_members[groups.find(value)].append(value)
    for members in confirmed_members.values():
        if any(research[value]['split'] != 'train' for value in members):
            if any(research[value]['split'] == 'train' for value in members):
                for value in members:
                    if research[value]['dataset'] == 'ISIC2019' and research[value]['split'] == 'train':
                        training_triggers[value].add('confirmed_identity_component_crosses_research_boundary')
    trigger_by_group = defaultdict(set)
    trigger_images_by_group = defaultdict(set)
    for value, pair_ids in training_triggers.items():
        owner = groups.find(value)
        trigger_by_group[owner].update(pair_ids)
        trigger_images_by_group[owner].add(value[1])
    exclusions = []
    for row in original:
        value = ('ISIC2019', row['image'])
        if row['split'] != 'train' or groups.find(value) not in trigger_by_group:
            continue
        owner = groups.find(value)
        reason = ('implicated_training_image_in_confirmed_or_unresolved_boundary_pair'
                  if value in training_triggers else 'same_known_lesion_or_exact_duplicate_group_as_implicated_training_image')
        exclusions.append(dict(row, exclusion_reason=reason,
                               evidence_pair_ids=';'.join(sorted(trigger_by_group[owner])),
                               trigger_training_images=';'.join(sorted(trigger_images_by_group[owner])),
                               action='exclude_from_final_manifest_only; original_file_preserved',
                               certainty='conservative_exclusion; not all candidates confirmed duplicates'))
    excluded_ids = {row['image'] for row in exclusions}
    final = [row for row in original if row['image'] not in excluded_ids]
    for split in ['validation', 'test']:
        if [r for r in final if r['split'] == split] != [r for r in original if r['split'] == split]:
            raise ValueError('Held-out assignments changed')
    if not {'ISIC_0012887_downsampled', 'ISIC_0013163_downsampled'} <= excluded_ids:
        raise ValueError('Required flagged training exclusions missing')
    if len({r['image'] for r in final}) != len(final):
        raise ValueError('Duplicate IDs in proposed final manifest')
    retained = {('ISIC2019', row['image']) for row in final}
    retained |= {value for value, row in research.items() if row['dataset'] == 'PADUFES20'}
    unresolved, confirmed_crossing = [], []
    for row in pairs:
        a, b = identity(row, 'a'), identity(row, 'b')
        row['a_retained_in_final_research'] = a in retained
        row['b_retained_in_final_research'] = b in retained
        row['final_action'] = ('training_image_excluded_or_group_excluded' if
                               any(value[0] == 'ISIC2019' and value[1] in excluded_ids for value in [a, b]) else
                               'outside_primary_research_scope' if a not in research or b not in research else
                               'retained_distinct_images' if row['final_decision'] == 'distinct_image_pair' else
                               'unresolved_heldout_pair_preserved_requires_adjudication')
        if a in retained and b in retained and row['final_decision'] != 'distinct_image_pair':
            if row['final_decision'] in ['confirmed_near_duplicate', 'confirmed_exact_duplicate_image', 'same_declared_lesion']:
                confirmed_crossing.append(row)
            else:
                unresolved.append(row)
    known = defaultdict(set)
    for row in final:
        known[row['lesion_id']].add(row['split'])
    known_crossing = {key: sorted(value) for key, value in known.items() if len(value) > 1}
    group_members = defaultdict(list)
    for value in retained:
        group_members[groups.find(value)].append(value)
    group_crossings = []
    for values in group_members.values():
        boundaries = {research[value]['dataset'] + '/' + research[value]['split'] for value in values}
        if len(boundaries) > 1:
            group_crossings.append({'members': values, 'partitions': sorted(boundaries)})
    component_ids = {value: f'G{number:06d}' for number, value in enumerate(sorted({groups.find(k) for k in raw_index}), 1)}
    membership = []
    for value, row in raw_index.items():
        membership.append({'identity_group': component_ids[groups.find(value)], 'dataset': value[0], 'image': value[1],
                           'original_split': row['split'], 'class': row['label'], 'lesion_id': row['lesion_id'],
                           'patient_id': row['patient_id'], 'retained_in_final_research': value in retained,
                           'basis': 'connected_declared_lesion_or_exact_hash_or_visually_confirmed_near_duplicate'})
    write_csv(output / 'identity_group_membership.csv', membership, list(membership[0]))
    manifests = output / 'manifests'
    manifests.mkdir(exist_ok=False)
    manifest = manifests / 'ISIC2019_final_research_manifest.csv'
    write_csv(manifest, final, list(original[0]))
    write_csv(output / 'training_exclusions.csv', exclusions, list(exclusions[0]))
    fields = list(pairs[0])
    write_csv(output / 'pair_audit.csv', pairs, fields)
    write_csv(output / 'unresolved_retained_pairs.csv', unresolved, fields)
    write_csv(output / 'confirmed_cross_boundary_pairs.csv', confirmed_crossing, fields)
    write_csv(output / 'confirmed_duplicate_pairs_all_scopes.csv', [r for r in pairs if r['final_decision'] in ['confirmed_near_duplicate', 'confirmed_exact_duplicate_image']], fields)
    write_csv(output / 'visually_distinct_pairs.csv', [r for r in pairs if r['final_decision'] == 'distinct_image_pair'], fields)
    statistics = []
    for split in ['train', 'validation', 'test']:
        rows = [row for row in final if row['split'] == split]
        counts = Counter(row['label'] for row in rows)
        statistics.append({'split': split, 'total': len(rows), **{c: counts[c] for c in CLASSES},
                           'known_lesions': len({row['lesion_id'] for row in rows})})
    write_csv(output / 'class_counts.csv', statistics, list(statistics[0]))
    train = statistics[0]
    if any(train[c] <= 0 for c in CLASSES):
        raise ValueError('Conservative exclusions emptied a training class')
    weights = {c: train['total'] / (4 * train[c]) for c in CLASSES}
    (output / 'class_weights.json').write_text(json.dumps({'classes': CLASSES, 'formula': 'N_train/(4*n_train_class)',
                'train_counts': {c: train[c] for c in CLASSES}, 'train_total': train['total'],
                'cross_entropy_weights': weights, 'm5_class_weight_power_from_predefined_protocol': .5,
                'm5_class_factors': {c: weights[c]**.5 for c in CLASSES},
                'statistics_source': str(manifest.relative_to(ROOT)), 'manifest_sha256': sha(manifest)}, indent=2) + '\n')
    snapshots = json.loads((output / 'preservation_snapshot.json').read_text())
    snapshots.update(json.loads((output / 'model_preservation_snapshot.json').read_text()))
    errors = []
    for path, expected in snapshots.items():
        actual = sha(ROOT / path)
        if actual != expected:
            errors.append({'path': path, 'reason': 'preservation_hash_mismatch', 'expected': expected, 'actual': actual})
    raw_checks = [(Path(r['source']), r['sha256']) for r in raw]
    prepared = read_csv(output / 'prepared_image_preservation.csv')
    prepared_checks = [(ROOT / row['path'], row['sha256']) for row in prepared]
    def verify(item):
        path, expected = item
        try:
            actual = sha(path)
            return None if actual == expected else {'path': str(path), 'reason': 'preservation_hash_mismatch', 'expected': expected, 'actual': actual}
        except Exception as exc:
            return {'path': str(path), 'reason': repr(exc), 'expected': expected, 'actual': ''}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, error in enumerate(pool.map(verify, raw_checks + prepared_checks, buffersize=8), 1):
            if error:
                errors.append(error)
            if number % 10000 == 0:
                print(f'Preservation checks: {number}/{len(raw_checks) + len(prepared_checks)}', flush=True)
    write_csv(output / 'preservation_errors.csv', errors, ['path', 'reason', 'expected', 'actual'])
    blockers = []
    if unresolved:
        blockers.append(f'{len(unresolved)} retained held-out/external candidate pairs have unresolved image/lesion identity after automated review.')
    if confirmed_crossing or group_crossings:
        blockers.append('Confirmed duplicate or known-lesion groups still cross retained partition boundaries; held-out assignments must remain unchanged.')
    if known_crossing:
        blockers.append('Known ISIC lesion IDs still cross retained primary partitions.')
    if errors:
        blockers.append('Source/model preservation checks failed.')
    coverage = len([r for r in pairs if 'previous_original_phash_le6' in r['candidate_sources']])
    if coverage != 4318:
        blockers.append('Prior candidate-list coverage is incomplete.')
    checks = {'all_4318_original_cross_boundary_candidates_reviewed_quantitatively': coverage == 4318,
              'validation_test_manifest_rows_unchanged': True,
              'known_lesion_crossings_primary_splits': known_crossing,
              'confirmed_group_crossings_retained_research': group_crossings,
              'unresolved_train_to_heldout_pairs_after_exclusion': sum(r['train_involved'] == 'True' for r in unresolved),
              'original_raw_image_hash_checks': len(raw_checks), 'prepared_image_hash_checks': len(prepared_checks),
              'preservation_errors': len(errors), 'models_and_model_configs_unchanged': not any('final_training' in e['path'] for e in errors),
              'no_model_training_invoked': True}
    approval = {'approved': not blockers, 'status': 'blocked' if blockers else 'passed_documented_audit_scope',
                'reason': ' '.join(blockers) if blockers else 'All required checks passed within documented screening limitations.',
                'review_document': str((output / 'AUDIT_REPORT.md').relative_to(ROOT)),
                'manifest_path': str(manifest.relative_to(ROOT)), 'manifest_sha256': sha(manifest),
                'unresolved_blockers': blockers, 'remaining_identity_review_pairs': len(unresolved),
                'known_patient_independence': 'unverified; ISIC challenge patient identifiers unavailable'}
    summary = {'audit_version': output.name, 'completed_utc': datetime.now(timezone.utc).isoformat(),
               'candidate_pairs_reviewed': len(pairs), 'prior_pairs_quantitatively_reviewed': coverage,
               'additional_candidate_pairs': len(pairs) - coverage,
               'automated_evidence_categories': dict(Counter(row['evidence_category'] for row in pairs)),
               'final_pair_decisions': dict(Counter(row['final_decision'] for row in pairs)),
               'visual_adjudications': len(reviews), 'training_exclusions': len(exclusions),
               'directly_implicated_training_images': len(training_triggers),
               'known_lesion_or_exact_group_additional_training_exclusions': len(exclusions) - len(training_triggers),
               'excluded_class_counts': dict(Counter(row['label'] for row in exclusions)),
               'final_sample_counts': statistics, 'train_class_weights': weights,
               'unresolved_retained_pairs': len(unresolved),
               'unresolved_retained_partition_pairs': dict(Counter('/'.join(sorted([r['dataset_a'] + ':' + r['split_a'], r['dataset_b'] + ':' + r['split_b']])) for r in unresolved)),
               'confirmed_cross_boundary_pairs_after_exclusion': len(confirmed_crossing),
               'checks': checks, 'data_approval': approval,
               'limitations': ['All pairs received automated image/metadata review; only rows in review_adjudications.csv received visual adjudication.',
                               'Low SSIM or differing source lesion IDs is not proof of genuinely different lesions.',
                               'Conservative training exclusions also remove known-lesion and exact-duplicate training siblings; they can alter class/domain composition.',
                               'Screening can miss crop, scale, viewpoint, reflection or temporal recapture duplicates beyond tested thresholds.',
                               'SSIM is computed on directly resized 256px images, not on original-resolution matching coordinates; background can inflate similarity.',
                               'PAD identifiers are dataset-local, with (patient_id, lesion_id) required for within-PAD grouping.',
                               'No validation/test/PAD samples are moved or deleted; unresolved held-out identity blocks remain instead.']}
    (output / 'data_approval.json').write_text(json.dumps(approval, indent=2) + '\n')
    (output / 'final_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Cross-split image-identity audit', '',
             '**Data approval: ' + str(approval['approved']).lower() + '. No models were trained or modified.**', '',
             f'All {coverage:,} prior cross-boundary candidates plus {len(pairs)-coverage:,} additional transformed/exact candidates received pHash, SSIM and metadata review. {len(reviews)} pairs received recorded visual adjudication; automatic review must not be described as full manual identity clearance.', '',
             '## Proposed final research manifest', '',
             f'`{manifest.relative_to(ROOT)}`', '',
             '| Split | NV | MEL | BCC | SCC | Total |', '|---|---:|---:|---:|---:|---:|']
    lines += ['| ' + r['split'] + ' | ' + ' | '.join(f'{r[c]:,}' for c in [*CLASSES, 'total']) + ' |' for r in statistics]
    lines += ['', f'{len(exclusions):,} training images excluded from this manifest only: {len(training_triggers):,} directly implicated, and {len(exclusions)-len(training_triggers):,} known-lesion/exact-group training companions. Raw and prepared files remain in place. Folder scanning must not be used to reconstruct the retained training set.', '',
              '## Required flagged images', '']
    for row in pairs:
        if {row['image_a'], row['image_b']} in [set(['ISIC_0012887_downsampled', 'ISIC_0012989_downsampled']), set(['ISIC_0013163_downsampled', 'ISIC_0013109_downsampled'])]:
            lines += [f'- {row["pair_id"]}: {row["image_a"]} / {row["image_b"]}; pHash distance {row["phash_distance_original"]}, original gray SSIM {float(row["gray_ssim_original_256"]):.6f}, RGB SSIM {float(row["rgb_ssim_best_dihedral_256"]):.6f}; different source lesion IDs {row["lesion_id_a"]} / {row["lesion_id_b"]}. Decision: {row["final_decision"]}; training member excluded.']
    lines += ['', '## Checks and blockers', '',
              f'- Raw image hashes checked: {len(raw_checks):,}; prepared image hashes checked: {len(prepared_checks):,}; preservation mismatches: {len(errors)}.',
              '- Original split CSVs, manifests and held-out assignments are preserved; model/config/official-weight hashes checked.',
              f'- Known lesion overlaps in retained ISIC partitions: {len(known_crossing)}; confirmed-group boundary crossings: {len(group_crossings)}.',
              f'- Unresolved candidate pairs retained after training exclusions: {len(unresolved)}; direct training-boundary unresolved pairs: {checks["unresolved_train_to_heldout_pairs_after_exclusion"]}.',
              *['- ' + b for b in blockers], '',
              'Training-only CE weights: `' + json.dumps(weights) + '`.', '',
              '## Review artifacts', '',
              '- [Pair audit with final decisions](pair_audit.csv)', '- [Recorded visual adjudications](review_adjudications.csv)',
              '- [All contact sheets](CONTACT_SHEETS.md)', '- [Unresolved retained pairs](unresolved_retained_pairs.csv)',
              '- [Training exclusions and justifications](training_exclusions.csv)', '- [Class counts](class_counts.csv)',
              '- [Class weights](class_weights.json)', '- [Approval](data_approval.json)', '- [Machine-readable summary](final_summary.json)',
              '- [Screening policy and software versions](policy.json)', '', '## Limitations', '',
              *['- ' + text for text in summary['limitations']], '']
    (output / 'AUDIT_REPORT.md').write_text('\n'.join(lines))
    active = ROOT / 'final_training/data_approval.json'
    if active.exists():
        (output / 'previous_active_data_approval.json').write_bytes(active.read_bytes())
    active.write_text(json.dumps(approval, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit_directory', type=Path)
    args = parser.parse_args()
    run(args.audit_directory.resolve())
