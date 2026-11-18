"""Prepare the four common PAD classes as external-only evaluation images."""
from collections import Counter
import json
from pathlib import Path

from audit_datasets import ROOT, REPORT, read_csv, write_csv
from preprocess_isic2019 import CLASS_TO_INDEX, prepare_image


def main():
    records = [row for row in read_csv(REPORT / 'raw_manifest.csv') if row['dataset'] == 'PADUFES20']
    if not records:
        raise ValueError('Run the raw dataset audit first')
    output = ROOT / 'PADUFES20/preprocessed/rgb224_letterbox_gray114_v1'
    if output.exists():
        raise FileExistsError(f'Preserving existing preparation: {output}')
    output.mkdir(parents=True)
    manifest, exclusions = [], []
    for row in records:
        if row['exclusion']:
            exclusions.append(row)
            continue
        relative = Path('external') / row['label'] / (Path(row['image']).stem + '.jpg')
        width, height = prepare_image(Path(row['source']), output / relative, 224)
        manifest.append(dict(row, label_index=CLASS_TO_INDEX[row['label']],
                             processed_relative_path=relative.as_posix(),
                             original_width=width, original_height=height))
    fields = ['dataset', 'image', 'label', 'original_label', 'label_index', 'lesion_id', 'patient_id',
              'split', 'source', 'processed_relative_path', 'original_width', 'original_height', 'sha256']
    write_csv(output / 'manifest.csv', manifest, fields)
    write_csv(output / 'exclusions.csv', exclusions,
              ['dataset', 'image', 'original_label', 'label', 'patient_id', 'lesion_id', 'exclusion', 'source'])
    summary = {'status': 'complete', 'role': 'external evaluation only; no train/validation split or tuning',
               'class_to_index': CLASS_TO_INDEX, 'label_mapping': {'NEV': 'NV', 'MEL': 'MEL', 'BCC': 'BCC', 'SCC': 'SCC'},
               'label_note': 'PAD source SCC includes Bowen disease/SCC in situ; not necessarily identical to ISIC SCC definitions',
               'images': len(manifest), 'class_counts': dict(Counter(row['label'] for row in manifest)),
               'excluded': len(exclusions), 'excluded_classes': dict(Counter(row['original_label'] for row in exclusions)),
               'patients': len({row['patient_id'] for row in manifest}),
               'lesions': len({row['lesion_id'] for row in manifest}),
               'patient_lesion_pairs': len({(row['patient_id'], row['lesion_id']) for row in manifest}),
               'image_size': [224, 224], 'color_mode': 'RGB',
               'transform': 'EXIF orientation, RGB conversion, LANCZOS aspect-preserving resize and centered gray114 letterboxing',
               'output_format': 'JPEG quality95 subsampling0', 'augmentation': 'none', 'normalization': 'none',
               'source_preserved': True}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
