"""Refine a prior duplicate audit without models or source-data modifications.

Advanced image measurements and narrowly defined screen-clearance are not proof
of biological patient/lesion independence. Every rule is recorded before scoring.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

from audit_datasets import read_csv, write_csv
from audit_final_research_splits import ROOT, AUDIT, BASE_MANIFEST, CLASSES, sha, orbit, ssim

SOURCE = ROOT / 'research/thesis_assets/phase4_duplicate_audit/v20261116T230847Z_34d5abf2'
RAW = {}
PAD = {}
ISIC_META = {}
POLICY = {
    'scope': 'Image-duplicate screening; no claim of independently verified biological patient identity',
    'images': 'Original oriented RGB resized directly to 512x512; central 192x192 crop; no training preprocessing modification',
    'sift': '1500 features, central 60% image mask, Lowe ratio 0.70, partial affine RANSAC 2px at 512px',
    'strict_negative_evidence_rule': {
        'best_dihedral_phash_distance_min': 4,
        'prior_full_gray_ssim_max': .90,
        'max_core_gray_ssim_all_dihedral_max': .45,
        'max_core_rgb_ssim_all_dihedral_max': .45,
        'core_gray_std_min_both': 8.,
        'central_sift_good_matches_max': 3,
        'central_sift_inliers_max': 3,
        'requires_different_nonempty_source_lesion_keys': True,
        'meaning': 'Screen-cleared for duplicate photographs under this rule, not external clinical identity certification',
    },
    'retention': 'Restore a previously excluded known-identity training component only when EVERY included boundary candidate touching the component is visually distinct or strict-negative screen-cleared. Never restore a confirmed train/held-out component.',
    'heldout_policy': 'Original validation/test assignments remain unchanged. Evaluation eligibility is a separate, explicit proposal, not applied silently.',
    'approval': 'False while any likely-same-lesion/unresolved primary or external boundary case or unchecked protocol decision remains.',
}


@lru_cache(maxsize=192)
def image(identity):
    with Image.open(RAW[identity]['source']) as opened:
        opened.load()
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


@lru_cache(maxsize=384)
def feature(identity):
    gray = image(identity)[1]
    mask = np.zeros_like(gray)
    mask[102:410, 102:410] = 255
    points, descriptors = cv2.SIFT_create(nfeatures=1500).detectAndCompute(gray, mask)
    return np.array([p.pt for p in points], dtype=np.float32), descriptors


def geometry(a, b):
    pa, da = feature(a)
    pb, db = feature(b)
    result = {'central_sift_good_matches': 0, 'central_sift_inliers': 0,
              'central_sift_inlier_ratio': 0., 'central_sift_coverage': 0.,
              'aligned_core_ssim': '', 'aligned_valid_coverage': ''}
    if da is None or db is None or min(len(da), len(db)) < 2:
        return result
    neighbors = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for item in neighbors if len(item) == 2 for m, n in [item] if m.distance < .7 * n.distance]
    result['central_sift_good_matches'] = len(good)
    if len(good) < 6:
        return result
    src = np.array([pa[m.queryIdx] for m in good], dtype=np.float32)
    dst = np.array([pb[m.trainIdx] for m in good], dtype=np.float32)
    affine, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.)
    if affine is None or mask is None:
        return result
    selected = mask.ravel().astype(bool)
    result['central_sift_inliers'] = int(selected.sum())
    result['central_sift_inlier_ratio'] = float(selected.mean())
    if selected.sum() >= 3:
        result['central_sift_coverage'] = float(cv2.contourArea(cv2.convexHull(src[selected])) / 512**2)
    ga, gb = image(a)[1], image(b)[1]
    warped = cv2.warpAffine(ga, affine, (512, 512))
    valid = cv2.warpAffine(np.ones_like(ga), affine, (512, 512))
    core = valid[160:352, 160:352]
    result['aligned_valid_coverage'] = float(core.mean())
    if core.mean() > .95:
        result['aligned_core_ssim'] = ssim(warped[160:352, 160:352], gb[160:352, 160:352])
    return result


def coarse_region(value):
    value = value.upper().strip()
    mapping = {'FACE': 'head/neck', 'SCALP': 'head/neck', 'LIP': 'head/neck', 'NOSE': 'head/neck',
               'EAR': 'head/neck', 'NECK': 'head/neck', 'HEAD/NECK': 'head/neck',
               'ARM': 'upper extremity', 'FOREARM': 'upper extremity', 'HAND': 'upper extremity',
               'UPPER EXTREMITY': 'upper extremity', 'THIGH': 'lower extremity', 'FOOT': 'lower extremity',
               'LOWER EXTREMITY': 'lower extremity', 'CHEST': 'torso', 'BACK': 'torso', 'ABDOMEN': 'torso',
               'ANTERIOR TORSO': 'torso', 'POSTERIOR TORSO': 'torso', 'LATERAL TORSO': 'torso'}
    return mapping.get(value, '')


def meta(identity):
    if identity[0] == 'ISIC2019':
        row = ISIC_META[identity[1]]
        return {'age': row.get('age_approx', ''), 'sex': row.get('sex', '').lower(),
                'site': row.get('anatom_site_general', ''), 'patient': '',
                'lesion_key': row.get('lesion_id', ''),
                'provenance': 'Official frozen ISIC 2019 challenge training release; lesion prefix ' + row.get('lesion_id', '').split('_')[0]}
    row = PAD[identity[1]]
    return {'age': row.get('age', ''), 'sex': row.get('gender', '').lower(), 'site': row.get('region', ''),
            'patient': row['patient_id'], 'lesion_key': row['patient_id'] + '/' + row['lesion_id'],
            'provenance': 'Official Mendeley PAD-UFES-20 v1; publisher SHA256-verified metadata and three image archives'}


def measurements(row):
    a, b = (row['dataset_a'], row['image_a']), (row['dataset_b'], row['image_b'])
    ra, ga = image(a)
    rb, gb = image(b)
    ca, cb = ra[160:352, 160:352], rb[160:352, 160:352]
    gca, gcb = ga[160:352, 160:352], gb[160:352, 160:352]
    gray_scores = [ssim(gca, candidate) for candidate in orbit(gcb)]
    rgb_scores = [ssim(ca, candidate) for candidate in orbit(cb)]
    output = {'pair_id': row['pair_id'], 'core_gray_ssim_max_d4': max(gray_scores),
              'core_rgb_ssim_max_d4': max(rgb_scores), 'core_gray_std_a': float(gca.std()),
              'core_gray_std_b': float(gcb.std()), 'best_core_gray_dihedral': int(np.argmax(gray_scores)),
              **geometry(a, b)}
    ma, mb = meta(a), meta(b)
    for label, value in [('a', ma), ('b', mb)]:
        for name, data in value.items():
            output[f'metadata_{name}_{label}'] = data
    try:
        age_difference = abs(float(ma['age']) - float(mb['age'])) if ma['age'] and mb['age'] else ''
    except ValueError:
        age_difference = ''
    output['age_difference_years'] = age_difference
    output['sex_matches_nonmissing'] = bool(ma['sex'] in ['male', 'female'] and ma['sex'] == mb['sex'])
    output['coarse_site_matches_nonmissing'] = bool(coarse_region(ma['site']) and coarse_region(ma['site']) == coarse_region(mb['site']))
    output['different_nonempty_lesion_keys'] = bool(ma['lesion_key'] and mb['lesion_key'] and (a[0], ma['lesion_key']) != (b[0], mb['lesion_key']))
    output['patient_linkage'] = 'No shared global patient identifier; namespace-local IDs do not establish cross-dataset independence'
    negative = (row['phash_distance_best_dihedral'] != '' and float(row['phash_distance_best_dihedral']) >= 4
                and float(row['gray_ssim_best_dihedral_256']) < .90 and max(gray_scores) <= .45 and max(rgb_scores) <= .45
                and min(gca.std(), gcb.std()) >= 8 and output['central_sift_good_matches'] <= 3
                and output['central_sift_inliers'] <= 3 and output['different_nonempty_lesion_keys'])
    output['strict_negative_screen_clearance'] = bool(negative)
    risk = .45 * float(row['gray_ssim_best_dihedral_256']) + .35 * max(gray_scores)
    risk += .08 * output['sex_matches_nonmissing'] + .08 * output['coarse_site_matches_nonmissing']
    risk += .04 * bool(age_difference != '' and age_difference <= 5)
    output['review_priority_score_not_identity_probability'] = risk
    return output


def sheets(output, old, metrics, unresolved_ids):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
    heading = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    index = ['# Revised held-out identity review contact sheets', '',
             'Priority combines structural similarity and matching available demographics/site; it is not an identity probability.',
             'Each panel shows A full / A central crop / B full / B central crop in original orientation. See pair IDs in CSV.', '']
    selected = [r for r in old if r['pair_id'] in unresolved_ids]
    categories = {'validation_test': [r for r in selected if r['dataset_a'] == r['dataset_b'] == 'ISIC2019' and 'train' not in [r['split_a'], r['split_b']]],
                  'isic_training_boundary': [r for r in selected if r['dataset_a'] == r['dataset_b'] == 'ISIC2019' and 'train' in [r['split_a'], r['split_b']]],
                  'isic_pad': [r for r in selected if r['dataset_a'] != r['dataset_b']]}
    for category, rows in categories.items():
        rows.sort(key=lambda r: metrics[r['pair_id']]['review_priority_score_not_identity_probability'], reverse=True)
        folder = output / 'contact_sheets' / category
        folder.mkdir(parents=True)
        for start in range(0, len(rows), 24):
            page = rows[start:start+24]
            canvas = Image.new('RGB', (1440, 8*225+36), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f'{category} priority page {start//24+1} — full image and central crop; not clinical identity verification', fill='black', font=heading)
            for i, row in enumerate(page):
                m = metrics[row['pair_id']]
                x, y = (i%3)*480, (i//3)*225+36
                draw.text((x+3, y), f'{row["pair_id"]} fullSSIM {float(row["gray_ssim_best_dihedral_256"]):.3f} core {m["core_gray_ssim_max_d4"]:.3f} inliers {m["central_sift_inliers"]}', fill='black', font=font)
                for j, suffix in enumerate(['a', 'b']):
                    rgb = image((row['dataset_'+suffix], row['image_'+suffix]))[0]
                    for k, values in enumerate([rgb, rgb[160:352, 160:352]]):
                        tile = ImageOps.contain(Image.fromarray(values), (116, 133))
                        canvas.paste(tile, (x+j*240+k*118+3, y+19))
                    draw.text((x+j*240+3,y+155),row['image_'+suffix],font=font,fill='black')
                    draw.text((x+j*240+3,y+168),row['split_'+suffix]+' / '+row['class_'+suffix]+' / '+row['lesion_id_'+suffix],font=font,fill='black')
                    draw.text((x+j*240+3,y+181),f'age {m["metadata_age_"+suffix]} sex {m["metadata_sex_"+suffix]}',font=font,fill='black')
                    draw.text((x+j*240+3,y+194),str(m['metadata_site_'+suffix]),font=font,fill='black')
                draw.text((x+3,y+208),f'Sex match {m["sex_matches_nonmissing"]}; site match {m["coarse_site_matches_nonmissing"]}; priority {m["review_priority_score_not_identity_probability"]:.3f}',font=font,fill='black')
                m['contact_sheet'] = str((folder/f'page_{start//24+1:03d}.jpg').relative_to(output))
                m['sheet_position'] = i+1
            path = folder/f'page_{start//24+1:03d}.jpg'
            canvas.save(path, quality=92)
            index.append(f'- [{category} page {start//24+1}]({path.relative_to(output).as_posix()}): ' + ', '.join(r['pair_id'] for r in page))
    (output/'CONTACT_SHEETS.md').write_text('\n'.join(index)+'\n')


def prepare():
    global RAW, PAD, ISIC_META
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    version = datetime.now(timezone.utc).strftime('v%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output = ROOT/'research/thesis_assets/phase4_duplicate_audit_revision'/version
    output.mkdir(parents=True, exist_ok=False)
    print('REVISION OUTPUT:',output,flush=True)
    (output/'policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
    RAW = {(r['dataset'],r['image']):r for r in read_csv(AUDIT/'raw_manifest.csv')}
    PAD = {r['img_id']:r for r in read_csv(ROOT/'dataset/PADUFES20/raw/metadata.csv')}
    ISIC_META = {r['image']:r for r in read_csv(ROOT/'dataset/ISIC2019/ISIC_2019_Training_Metadata.csv')}
    old = read_csv(SOURCE/'pair_audit.csv')
    selected = [r for r in old if r['both_originally_included']=='True']
    snapshots = {str(path.relative_to(ROOT)):sha(path) for path in SOURCE.rglob('*') if path.is_file()}
    snapshots.update({str(path.relative_to(ROOT)):sha(path) for path in [BASE_MANIFEST, ROOT/'dataset/PADUFES20/raw/metadata.csv', ROOT/'dataset/ISIC2019/ISIC_2019_Training_Metadata.csv']})
    (output/'input_snapshot.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number,row in enumerate(pool.map(measurements,selected,buffersize=8),1):
            results.append(row)
            if number%200==0:print(f'Lesion-region/central-SIFT review: {number}/{len(selected)}',flush=True)
    write_csv(output/'advanced_metrics.csv',results,list(results[0]))
    metrics={r['pair_id']:r for r in results}
    unresolved_ids={r['pair_id'] for r in read_csv(SOURCE/'unresolved_retained_pairs.csv')}
    sheets(output,old,metrics,unresolved_ids)
    fields=list(results[0])+['contact_sheet','sheet_position']
    write_csv(output/'advanced_metrics.csv',results,fields)
    write_csv(output/'heldout_priority_review.csv',sorted([r for r in results if r['pair_id'] in unresolved_ids],key=lambda r:r['review_priority_score_not_identity_probability'],reverse=True),fields)
    (output/'visual_review.csv').write_text('pair_id,classification,evidence,reviewer\n')
    (output/'stage_summary.json').write_text(json.dumps({'source':str(SOURCE.relative_to(ROOT)),'included_pairs_refined':len(results),'heldout_pairs':len(unresolved_ids),'strict_negative_screen_clearances':sum(r['strict_negative_screen_clearance'] for r in results),'status':'advanced_measurements_complete; visual_adjudication_and_finalization_pending'},indent=2)+'\n')
    print('ADVANCED REVIEW COMPLETE',output,flush=True)


if __name__=='__main__':
    prepare()
