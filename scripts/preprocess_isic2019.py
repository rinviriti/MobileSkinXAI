"""MobileSkinXAI Phase 5.2: deterministic ISIC 2019 image preparation.

Consumes the frozen four-class lesion-disjoint split CSVs from Phase 5.1.
Does not edit original images, split CSVs, or labels.
No augmentation, class balancing, or model-specific normalization happens here.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps, __version__ as pillow_version

CLASSES = ("NV", "MEL", "BCC", "SCC")
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
SPLITS = ("train", "validation", "test")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_splits(data_dir, split_dir):
    """Check manifests against original labels, metadata, and image paths."""
    ground_truth = {
        row["image"]: row
        for row in read_csv(data_dir / "ISIC_2019_Training_GroundTruth.csv")
    }
    metadata = {
        row["image"]: row
        for row in read_csv(data_dir / "ISIC_2019_Training_Metadata.csv")
    }
    seen_images = set()
    lesion_owners = {}
    by_split = {}

    for split in SPLITS:
        records = read_csv(split_dir / f"{split}.csv")
        if not records:
            raise ValueError(f"Empty split: {split}")

        for row in records:
            image_id = row["image"].strip()
            label = row["label"].strip()
            lesion_id = row["lesion_id"].strip()

            if label not in CLASS_TO_INDEX:
                raise ValueError(f"Unexpected class: {label}")
            if image_id in seen_images:
                raise ValueError(f"Repeated image across splits: {image_id}")
            if not lesion_id:
                raise ValueError(f"Missing lesion ID in {split}: {image_id}")
            if image_id not in ground_truth or image_id not in metadata:
                raise ValueError(f"Unknown image ID: {image_id}")
            if float(ground_truth[image_id][label]) != 1.0:
                raise ValueError(f"Label mismatch: {image_id}")
            if metadata[image_id]["lesion_id"].strip() != lesion_id:
                raise ValueError(f"Lesion ID mismatch: {image_id}")

            owner = lesion_owners.setdefault(lesion_id, split)
            if owner != split:
                raise ValueError(f"Lesion shared by {owner} and {split}: {lesion_id}")

            source = data_dir / "images" / "ISIC_2019_Training_Input" / f"{image_id}.jpg"
            if not source.is_file():
                raise FileNotFoundError(source)
            seen_images.add(image_id)

        by_split[split] = records
        print(f"AUDIT {split}: {len(records)} images, {len({r['lesion_id'] for r in records})} lesions")

    excluded = read_csv(split_dir / "excluded_missing_lesion_id.csv")
    excluded_ids = {row["image"] for row in excluded}
    if len(excluded_ids) != len(excluded) or seen_images.intersection(excluded_ids):
        raise ValueError("Excluded images overlap or contain duplicate IDs")
    if any(row["lesion_id"].strip() for row in excluded):
        raise ValueError("An excluded image unexpectedly has a lesion ID")

    print(f"PASS: {len(seen_images)} images, {len(lesion_owners)} distinct known lesions")
    print(f"PASS: {len(excluded_ids)} excluded images kept outside the splits")
    return by_split


def prepare_image(source, destination, size):
    """Apply EXIF orientation, RGB conversion, aspect-preserving gray padding."""
    with Image.open(source) as opened:
        opened.load()  # Decodes the entire file; raises on unreadable images.
        oriented = ImageOps.exif_transpose(opened)
        original_width, original_height = oriented.size
        if original_width <= 0 or original_height <= 0:
            raise ValueError(f"Invalid dimensions: {source}")
        rgb = oriented.convert("RGB")
        prepared = ImageOps.pad(
            rgb,
            (size, size),
            method=Image.Resampling.LANCZOS,
            color=(114, 114, 114),
            centering=(0.5, 0.5),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        prepared.save(destination, format="JPEG", quality=95, subsampling=0)

    return original_width, original_height


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Desktop" / "MobileSkinXAI-Datasets" / "ISIC2019",
        help="Location of original ISIC 2019 images, labels, and metadata",
    )
    parser.add_argument("--size", type=int, default=224, help="Square image side in pixels")
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=0,
        help="Smoke test: process this many images per class per split (0 means all)",
    )
    parser.add_argument("--audit-only", action="store_true", help="Check split CSVs without processing")
    args = parser.parse_args()
    if args.size < 32 or args.sample_per_class < 0:
        parser.error("--size must be >=32 and --sample-per-class must be >=0")

    data_dir = args.data_dir.expanduser().resolve()
    split_dir = data_dir / "splits" / "lesion_stratified_seed42"
    records_by_split = audit_splits(data_dir, split_dir)
    if args.audit_only:
        print("Audit complete: no files written.")
        return

    version = f"rgb{args.size}_letterbox_gray114_v1"
    run_type = "sample" if args.sample_per_class else "full"
    output_dir = data_dir / "preprocessed" / version / run_type
    if (output_dir / "summary.json").exists():
        raise FileExistsError(
            f"Completed output already exists: {output_dir}. "
            "Do not overwrite a recorded experiment."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    warnings = []
    counts = {}
    for split, records in records_by_split.items():
        if args.sample_per_class:
            subset = []
            for label in CLASSES:
                subset.extend(
                    [r for r in records if r["label"] == label][: args.sample_per_class]
                )
            records = subset

        counts[split] = Counter()
        for number, row in enumerate(records, start=1):
            image_id, label = row["image"], row["label"]
            source = data_dir / "images" / "ISIC_2019_Training_Input" / f"{image_id}.jpg"
            relative_output = Path(split) / label / f"{image_id}.jpg"
            destination = output_dir / relative_output
            try:
                width, height = prepare_image(source, destination, args.size)
            except Exception as error:
                raise RuntimeError(f"Image processing failed: {source}: {error}") from error

            manifest.append({
                "image": image_id,
                "label": label,
                "label_index": CLASS_TO_INDEX[label],
                "lesion_id": row["lesion_id"],
                "split": split,
                "processed_relative_path": relative_output.as_posix(),
                "original_width": width,
                "original_height": height,
            })
            counts[split][label] += 1

            if min(width, height) < args.size or max(width, height) / min(width, height) > 2.5:
                warnings.append({
                    "image": image_id,
                    "split": split,
                    "original_width": width,
                    "original_height": height,
                    "reason": "small_dimension_or_extreme_aspect_ratio",
                })

            if number % 500 == 0 or number == len(records):
                print(f"{split}: {number}/{len(records)} images prepared")

    write_csv(
        output_dir / "manifest.csv",
        manifest,
        ["image", "label", "label_index", "lesion_id", "split",
         "processed_relative_path", "original_width", "original_height"],
    )
    write_csv(
        output_dir / "review_flags.csv",
        warnings,
        ["image", "split", "original_width", "original_height", "reason"],
    )
    summary = {
        "status": "complete",
        "input_split_directory": str(split_dir),
        "output_directory": str(output_dir),
        "run_type": run_type,
        "sample_per_class": args.sample_per_class,
        "image_size": [args.size, args.size],
        "color_mode": "RGB",
        "transform": "EXIF orientation, RGB conversion, aspect-preserving resize + centered gray(114) padding",
        "resampling": "LANCZOS",
        "output_format": "JPEG quality=95 subsampling=0",
        "normalization": "not applied; defer to each model's input preprocessing",
        "augmentation": "not applied; train-only at model training stage",
        "class_to_index": CLASS_TO_INDEX,
        "split_class_counts": {name: dict(counts[name]) for name in SPLITS},
        "total_written": len(manifest),
        "review_flags": len(warnings),
        "pillow_version": pillow_version,
        "limitations": ["Known lesion IDs are disjoint; patient-level independence unverified",
                        "No near-duplicate image search performed"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nCOMPLETE:", len(manifest), "images")
    print("Review flags:", len(warnings))
    print("Output:", output_dir)
    print("Raw images and split CSVs were not modified.")


if __name__ == "__main__":
    main()
