"""Versioned image-identity audit. Never imports models or changes source data.

Automated review is not clinical lesion-identity adjudication. Unadjudicated
training/held-out pairs cause conservative training exclusions; unresolved
held-out pairs keep approval false. All thresholds are screening heuristics.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.fft import dctn
from skimage.metrics import structural_similarity
import skimage

from audit_datasets import BKTree, read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'dataset/audit'
ISIC = ROOT / 'dataset/ISIC2019'
BASE_MANIFEST = ISIC / 'preprocessed/rgb224_letterbox_gray114_v1/full/manifest.csv'
CLASSES = ['NV', 'MEL', 'BCC', 'SCC']
RAW = {}
METADATA = {}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def key(row):
    return row['dataset'], row['image']


def included(row):
    return row['status'] == 'ok' and not row['exclusion']


def boundary(a, b):
    return a['dataset'] != b['dataset'] or a['split'] != b['split']


def orbit(array):
    return [np.rot90(array, k).copy() for k in range(4)] + [np.rot90(np.fliplr(array), k).copy() for k in range(4)]


def phash(gray):
    low = dctn(gray.astype(np.float32), type=2, norm='ortho')[:8, :8].flatten()
    bits = low > np.median(low[1:])
    bits[0] = False
    return f'{int.from_bytes(np.packbits(bits).tobytes()):016x}'


def fingerprints(row):
    with Image.open(row['source']) as image:
        image.load()
        image = ImageOps.exif_transpose(image).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
        values = [phash(a) for a in orbit(np.asarray(image))]
    if values[0] != row['phash']:
        raise ValueError(f'Prior fingerprint mismatch: {key(row)}')
    return key(row), values


@lru_cache(maxsize=512)
def arrays(identity):
    with Image.open(RAW[identity]['source']) as image:
        image.load()
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB').resize((256, 256), Image.Resampling.LANCZOS))
    return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def ssim(a, b):
    return float(structural_similarity(a, b, data_range=255, gaussian_weights=True,
                                      sigma=1.5, use_sample_covariance=False,
                                      channel_axis=-1 if a.ndim == 3 else None))


@lru_cache(maxsize=512)
def features(identity):
    gray = arrays(identity)[1]
    points, descriptors = cv2.SIFT_create(nfeatures=500).detectAndCompute(gray, None)
    return np.asarray([point.pt for point in points], dtype=np.float32), descriptors


def geometric_evidence(a, b):
    pa, da = features(a)
    pb, db = features(b)
    result = {'sift_good_matches': 0, 'sift_inliers': 0, 'sift_inlier_ratio': 0,
              'sift_spatial_coverage': 0, 'affine_valid_coverage': 0,
              'affine_gray_ssim': '', 'geometric_review': 'insufficient_correspondences'}
    if da is None or db is None or min(len(da), len(db)) < 2:
        return result
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for neighbors in matches if len(neighbors) == 2
            for m, n in [neighbors] if m.distance < .7 * n.distance]
    result['sift_good_matches'] = len(good)
    if len(good) < 6:
        return result
    source = np.asarray([pa[m.queryIdx] for m in good], dtype=np.float32)
    target = np.asarray([pb[m.trainIdx] for m in good], dtype=np.float32)
    affine, mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                             ransacReprojThreshold=2., maxIters=2000, confidence=.99)
    if affine is None or mask is None:
        return result
    selection = mask.ravel().astype(bool)
    result['sift_inliers'] = int(selection.sum())
    result['sift_inlier_ratio'] = float(selection.mean())
    if selection.sum() >= 3:
        result['sift_spatial_coverage'] = float(cv2.contourArea(cv2.convexHull(source[selection])) / 256**2)
    ga, gb = arrays(a)[1], arrays(b)[1]
    aligned = cv2.warpAffine(ga, affine, (256, 256))
    valid = cv2.warpAffine(np.ones_like(ga), affine, (256, 256))
    valid = cv2.erode(valid, np.ones((11, 11), np.uint8)).astype(bool)
    result['affine_valid_coverage'] = float(valid.mean())
    if valid.mean() >= .5:
        _, similarity_map = structural_similarity(aligned, gb, data_range=255, gaussian_weights=True,
                                                  sigma=1.5, use_sample_covariance=False, full=True)
        result['affine_gray_ssim'] = float(similarity_map[valid].mean())
    result['geometric_review'] = 'measured_partial_affine_ransac'
    return result


def score(item):
    identity_a, identity_b, sources = item
    a, b = RAW[identity_a], RAW[identity_b]
    rgb_a, gray_a = arrays(identity_a)
    rgb_b, gray_b = arrays(identity_b)
    small_a = cv2.resize(gray_a, (96, 96), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(gray_b, (96, 96), interpolation=cv2.INTER_AREA)
    orientations = [ssim(small_a, candidate) for candidate in orbit(small_b)]
    best = int(np.argmax(orientations))
    best_gray = orbit(gray_b)[best]
    best_rgb = orbit(rgb_b)[best]
    gray_score, best_score = ssim(gray_a, gray_b), ssim(gray_a, best_gray)
    rgb_score = ssim(rgb_a, best_rgb)
    center = ssim(gray_a[32:224, 32:224], best_gray[32:224, 32:224])
    delta = np.abs(rgb_a.astype(np.float32) - best_rgb.astype(np.float32))
    hamming = (int(a['phash'], 16) ^ int(b['phash'], 16)).bit_count()
    output = {'pair_key': f'{identity_a[0]}:{identity_a[1]}|{identity_b[0]}:{identity_b[1]}',
              'candidate_sources': ';'.join(sorted(sources)),
              'phash_distance_original': hamming,
              'dhash_distance_original': (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count(),
              'gray_ssim_original_256': gray_score, 'gray_ssim_best_dihedral_256': best_score,
              'rgb_ssim_best_dihedral_256': rgb_score, 'central_gray_ssim_192': center,
              'best_dihedral_index': best, 'rgb_mae_0_255': float(delta.mean()),
              'pixel_fraction_abs_difference_le2': float((delta.max(-1) <= 2).mean()),
              'same_file_sha256': a['sha256'] == b['sha256'],
              'same_oriented_pixel_sha256': a['pixel_sha256'] == b['pixel_sha256'],
              'same_known_isic_lesion': a['dataset'] == b['dataset'] == 'ISIC2019' and bool(a['lesion_id']) and a['lesion_id'] == b['lesion_id'],
              'both_originally_included': included(a) and included(b),
              'train_involved': a['split'] == 'train' or b['split'] == 'train',
              'label_conflict': a['label'] != b['label'],
              'review_method': 'automated_phash_ssim_and_metadata_review',
              'visual_review_status': 'not_yet_visually_adjudicated',
              'review_evidence': 'All SSIM values are image-similarity evidence, not proof of different lesion identity.'}
    if max(gray_score, best_score) >= .7 or hamming <= 2:
        output.update(geometric_evidence(identity_a, identity_b))
    else:
        output.update({'sift_good_matches': '', 'sift_inliers': '', 'sift_inlier_ratio': '',
                       'sift_spatial_coverage': '', 'affine_valid_coverage': '', 'affine_gray_ssim': '',
                       'geometric_review': 'not_run_low_ssim_and_phash_distance_gt2'})
    if output['same_file_sha256'] or output['same_oriented_pixel_sha256']:
        output['evidence_category'] = 'confirmed_exact_duplicate_image'
        output['review_evidence'] = 'Matching SHA-256 of original file or fully decoded oriented RGB pixels.'
    elif output['same_known_isic_lesion']:
        output['evidence_category'] = 'same_declared_lesion'
    elif ((best_score >= .98 and rgb_score >= .97 and delta.mean() <= 3) or
          (output.get('sift_inliers', 0) != '' and output.get('sift_inliers', 0) >= 12 and
           output.get('sift_spatial_coverage', 0) >= .05 and output.get('affine_gray_ssim') != '' and
           output['affine_gray_ssim'] >= .94)):
        output['evidence_category'] = 'strong_same_scene_candidate_requires_visual_review'
    elif max(gray_score, best_score) < .7 and center < .7:
        output['evidence_category'] = 'low_structural_similarity_identity_unadjudicated'
    else:
        output['evidence_category'] = 'ambiguous_similarity_identity_unadjudicated'
    metadata_differences = []
    for suffix, row in [('a', a), ('b', b)]:
        meta = METADATA.get(row['image'], {}) if row['dataset'] == 'ISIC2019' else {}
        for field, value in [('dataset', row['dataset']), ('image', row['image']), ('split', row['split']),
                             ('class', row['label']), ('lesion_id', row['lesion_id']), ('patient_id', row['patient_id']),
                             ('source', row['source']), ('age_approx', meta.get('age_approx', '')),
                             ('sex', meta.get('sex', '')), ('anatomic_site', meta.get('anatom_site_general', ''))]:
            output[f'{field}_{suffix}'] = value
    for field in ['age_approx', 'sex', 'anatomic_site']:
        if output[f'{field}_a'] and output[f'{field}_b'] and output[f'{field}_a'] != output[f'{field}_b']:
            metadata_differences.append(field)
    output['metadata_differences'] = ';'.join(metadata_differences)
    output['metadata_caveat'] = 'Different lesion IDs/metadata support different records, not proof of biological independence.'
    return output


def contact_sheets(output, scored):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
    headings = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    groups = defaultdict(list)
    for row in scored:
        category = ('outside_primary_research_scope' if not row['both_originally_included'] else
                    'heldout_only' if not row['train_involved'] else 'training_boundary')
        groups[category].append(row)
    index = ['# All cross-boundary candidate contact sheets', '',
             'Each pair has a stable ID matching pair_audit.csv. Sheets support visual review; generating a sheet does not mean visual adjudication occurred.', '']
    for category, records in groups.items():
        records.sort(key=lambda r: (-float(r['gray_ssim_best_dihedral_256']), r['pair_id']))
        folder = output / 'contact_sheets' / category
        folder.mkdir(parents=True)
        for start in range(0, len(records), 24):
            page = records[start:start + 24]
            canvas = Image.new('RGB', (1320, 6 * 220 + 40), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 8), f'{category} — page {start // 24 + 1}; ORIGINAL orientation; SSIM uses best dihedral orientation', fill='black', font=headings)
            for number, row in enumerate(page):
                x, y = (number % 4) * 330, (number // 4) * 220 + 40
                text = f'{row["pair_id"]}  pH={row["phash_distance_original"]}  SSIM={row["gray_ssim_best_dihedral_256"]:.3f}'
                draw.text((x + 3, y), text, fill='black', font=font)
                for j, suffix in enumerate(['a', 'b']):
                    rgb = arrays((row[f'dataset_{suffix}'], row[f'image_{suffix}']))[0]
                    thumb = ImageOps.contain(Image.fromarray(rgb), (158, 148))
                    canvas.paste(thumb, (x + j * 164 + 3, y + 18))
                    draw.text((x + j * 164 + 3, y + 170), row[f'image_{suffix}'], fill='black', font=font)
                    draw.text((x + j * 164 + 3, y + 183), row[f'split_{suffix}'] + ' / ' + row[f'class_{suffix}'], fill='black', font=font)
                    draw.text((x + j * 164 + 3, y + 196), row[f'lesion_id_{suffix}'], fill='black', font=font)
                row['contact_sheet'] = str((folder / f'page_{start // 24 + 1:03d}.jpg').relative_to(output))
                row['contact_sheet_position'] = number + 1
            path = folder / f'page_{start // 24 + 1:03d}.jpg'
            canvas.save(path, quality=90)
            index.append(f'- [{category}, page {start // 24 + 1}]({path.relative_to(output).as_posix()}): ' + ', '.join(row['pair_id'] for row in page))
        print(f'Contact sheets: {category}: {len(records)} pairs', flush=True)
    (output / 'CONTACT_SHEETS.md').write_text('\n'.join(index) + '\n')


def scan(output):
    global RAW, METADATA
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    original = read_csv(AUDIT / 'raw_manifest.csv')
    RAW = {key(row): row for row in original}
    if len(RAW) != len(original):
        raise ValueError('Duplicate raw image identifiers')
    METADATA = {row['image']: row for row in read_csv(ISIC / 'ISIC_2019_Training_Metadata.csv')}
    sources = [BASE_MANIFEST, AUDIT / 'raw_manifest.csv', AUDIT / 'duplicate_candidates.csv',
               AUDIT / 'visual_review.json', ISIC / 'ISIC_2019_Training_GroundTruth.csv',
               ISIC / 'ISIC_2019_Training_Metadata.csv', ROOT / 'dataset/PADUFES20/raw/metadata.csv']
    sources += list((ISIC / 'splits/lesion_stratified_seed42').glob('*'))
    snapshot = {str(path.relative_to(ROOT)): sha(path) for path in sources if path.is_file()}
    (output / 'preservation_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (output / 'policy.json').write_text(json.dumps({
        'original_phash_threshold': 6, 'additional_dihedral_phash_threshold': 2,
        'ssim': {'size': [256, 256], 'data_range': 255, 'gaussian_weights': True,
                 'sigma': 1.5, 'use_sample_covariance': False, 'central_crop': [32, 224],
                 'orientation_selection_size': [96, 96]},
        'normalization': 'Original oriented RGB is resized directly, without letterbox padding, for comparable image coordinates; this is not training preprocessing.',
        'geometric_check': 'SIFT 500 features; Lowe ratio 0.7; partial-affine RANSAC reprojection 2px',
        'decision': 'Conservatively remove training members and known-lesion/exact-duplicate training siblings for all unadjudicated included boundary pairs. Never change held-out assignments.',
        'approval': 'False while held-out identity candidates or required adjudications/checks remain unresolved.',
        'versions': {'opencv': cv2.__version__, 'scikit_image': skimage.__version__, 'numpy': np.__version__},
    }, indent=2) + '\n')
    pending = defaultdict(set)
    for row in read_csv(AUDIT / 'duplicate_candidates.csv'):
        if row['cross_boundary'] == 'True':
            a, b = (row['dataset_a'], row['image_a']), (row['dataset_b'], row['image_b'])
            pending[tuple(sorted([a, b]))].add('previous_original_phash_le6')
    groups = defaultdict(list)
    for row in original:
        for attribute in ['sha256', 'pixel_sha256']:
            groups[(attribute, row[attribute])].append(key(row))
    for (attribute, value), identities in groups.items():
        if not value or len(identities) < 2:
            continue
        for a, b in itertools.combinations(identities, 2):
            if boundary(RAW[a], RAW[b]):
                pending[tuple(sorted([a, b]))].add('independent_exact_' + attribute)
    tree = BKTree()
    indexed = list(RAW)
    for number, identity in enumerate(indexed):
        tree.add(int(RAW[identity]['phash'], 16), number)
    selected = [row for row in original if included(row)]
    orbit_rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, (identity, hashes) in enumerate(pool.map(fingerprints, selected, buffersize=8), 1):
            orbit_rows.append({'dataset': identity[0], 'image': identity[1], **{f'phash_d4_{k}': value for k, value in enumerate(hashes)}})
            for value in set(hashes[1:]):
                for other_index, distance in tree.query(int(value, 16), 2):
                    other = indexed[other_index]
                    if identity != other and boundary(RAW[identity], RAW[other]):
                        pending[tuple(sorted([identity, other]))].add('additional_dihedral_phash_le2')
            if number % 1000 == 0:
                print(f'Dihedral hash screening: {number}/{len(selected)}', flush=True)
    write_csv(output / 'dihedral_hashes.csv', orbit_rows, list(orbit_rows[0]))
    items = [(a, b, sources) for (a, b), sources in sorted(pending.items())]
    print(f'Scoring every cross-boundary pair: {len(items)} (previous 4318 plus added transformed/exact pairs)', flush=True)
    scored = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, row in enumerate(pool.map(score, items, buffersize=8), 1):
            row['pair_id'] = f'P{number:06d}'
            scored.append(row)
            if number % 200 == 0:
                print(f'SSIM/metadata/geometric evidence: {number}/{len(items)}', flush=True)
    fields = list(scored[0])
    write_csv(output / 'pair_metrics.csv', scored, fields)
    contact_sheets(output, scored)
    fields += ['contact_sheet', 'contact_sheet_position']
    write_csv(output / 'pair_audit.csv', scored, fields)
    strong = [row for row in scored if row['evidence_category'] in ['confirmed_exact_duplicate_image', 'same_declared_lesion', 'strong_same_scene_candidate_requires_visual_review']]
    write_csv(output / 'priority_visual_review.csv', strong, fields)
    (output / 'scan_summary.json').write_text(json.dumps({
        'all_pairs': len(scored), 'original_cross_boundary_pairs_reviewed': 4318,
        'included_pairs': sum(row['both_originally_included'] for row in scored),
        'evidence_categories': dict(Counter(row['evidence_category'] for row in scored)),
        'priority_visual_review_pairs': len(strong), 'status': 'metrics_and_contact_sheets_complete_awaiting_adjudication',
    }, indent=2) + '\n')
    for path, checksum in snapshot.items():
        if sha(ROOT / path) != checksum:
            raise ValueError(f'Original input changed during audit: {path}')
    (output / 'review_adjudications.csv').write_text('pair_id,decision,evidence,reviewer\n')
    print('SCAN COMPLETE:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
    else:
        version = datetime.now(timezone.utc).strftime('v%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        output = ROOT / 'research/thesis_assets/phase4_duplicate_audit' / version
    output.mkdir(parents=True, exist_ok=False)
    print('AUDIT OUTPUT:', output, flush=True)
    scan(output)


if __name__ == '__main__':
    main()
