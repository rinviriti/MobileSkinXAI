
"""
MobileSkinXAI — ISIC 2019 four-class split
Version: 1
Method: Class-stratified lesion-group split
Target: 70% train, 15% validation, 15% test

Images without lesion IDs are excluded from
the primary split and recorded separately.

No images or original metadata are modified.
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
CLASSES = ("NV", "MEL", "BCC", "SCC")

DATA_DIR = (
    Path.home()
    / "Desktop"
    / "MobileSkinXAI-Datasets"
    / "ISIC2019"
)

OUTPUT_DIR = (
    DATA_DIR / "splits" / "lesion_stratified_seed42"
)

LABEL_FILE = (
    DATA_DIR / "ISIC_2019_Training_GroundTruth.csv"
)

METADATA_FILE = (
    DATA_DIR / "ISIC_2019_Training_Metadata.csv"
)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "label", "lesion_id"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_distribution(name, rows):
    counts = Counter(row["label"] for row in rows)

    print(f"\n=== {name.upper()} ===")
    print("Total images:", len(rows))

    for label in CLASSES:
        print(f"{label}: {counts[label]}")


def main():
    if not LABEL_FILE.is_file():
        raise FileNotFoundError(LABEL_FILE)

    if not METADATA_FILE.is_file():
        raise FileNotFoundError(METADATA_FILE)

    labels = read_csv(LABEL_FILE)
    metadata_rows = read_csv(METADATA_FILE)

    metadata = {
        row["image"]: row
        for row in metadata_rows
    }

    groups = defaultdict(list)
    group_labels = {}
    excluded = []

    for row in labels:
        selected = [
            label
            for label in CLASSES
            if float(row[label]) == 1.0
        ]

        if not selected:
            continue

        if len(selected) != 1:
            raise ValueError(
                f"Ambiguous diagnosis for {row['image']}"
            )

        label = selected[0]
        image_id = row["image"]

        if image_id not in metadata:
            raise ValueError(
                f"Missing metadata for {image_id}"
            )

        lesion_id = (
            metadata[image_id].get("lesion_id") or ""
        ).strip()

        record = {
            "image": image_id,
            "label": label,
            "lesion_id": lesion_id,
        }

        if not lesion_id:
            excluded.append(record)
            continue

        if (
            lesion_id in group_labels
            and group_labels[lesion_id] != label
        ):
            raise ValueError(
                f"Conflicting labels for lesion {lesion_id}"
            )

        group_labels[lesion_id] = label
        groups[lesion_id].append(record)

    # Organize complete lesion groups by diagnosis.
    class_groups = {
        label: [] for label in CLASSES
    }

    for lesion_id, label in group_labels.items():
        class_groups[label].append(lesion_id)

    rng = random.Random(SEED)

    split_ids = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    # Stratify at lesion-group level, not image level.
    for label in CLASSES:
        lesion_ids = sorted(class_groups[label])
        rng.shuffle(lesion_ids)

        n = len(lesion_ids)

        n_test = round(n * 0.15)
        n_validation = round(n * 0.15)

        test_ids = lesion_ids[:n_test]

        validation_ids = lesion_ids[
            n_test:n_test + n_validation
        ]

        train_ids = lesion_ids[
            n_test + n_validation:
        ]

        split_ids["train"].update(train_ids)
        split_ids["validation"].update(
            validation_ids
        )
        split_ids["test"].update(test_ids)

    # Verify there is no shared known lesion ID.
    assert split_ids["train"].isdisjoint(
        split_ids["validation"]
    )

    assert split_ids["train"].isdisjoint(
        split_ids["test"]
    )

    assert split_ids["validation"].isdisjoint(
        split_ids["test"]
    )

    assert (
        set(group_labels)
        == set().union(*split_ids.values())
    )

    split_rows = {}

    for name, lesion_ids in split_ids.items():
        records = []

        for lesion_id in sorted(lesion_ids):
            records.extend(groups[lesion_id])

        split_rows[name] = sorted(
            records,
            key=lambda row: row["image"],
        )

    excluded.sort(key=lambda row: row["image"])

    all_known_ids = [
        row["image"]
        for records in split_rows.values()
        for row in records
    ]

    assert len(all_known_ids) == len(set(all_known_ids))

    assert (
        len(all_known_ids) + len(excluded)
        == sum(
            1
            for row in labels
            if any(float(row[c]) == 1.0 for c in CLASSES)
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, records in split_rows.items():
        save_csv(
            OUTPUT_DIR / f"{name}.csv",
            records,
        )

    save_csv(
        OUTPUT_DIR / "excluded_missing_lesion_id.csv",
        excluded,
    )

    summary = {
        "random_seed": SEED,
        "classes": list(CLASSES),
        "method": "class-stratified lesion-group split",
        "target_group_proportions": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "splits": {},
        "excluded_missing_lesion_id": len(excluded),
        "limitations": [
            "Patient identifiers unavailable",
            "Missing lesion IDs excluded",
            "Image-level percentages may differ",
            "Cross-patient visual duplicates not ruled out",
        ],
    }

    print("\n=== GROUP SPLIT AUDIT ===")

    for name, records in split_rows.items():
        summary["splits"][name] = {
            "images": len(records),
            "lesion_groups": len(split_ids[name]),
            "class_images": dict(
                Counter(
                    row["label"]
                    for row in records
                )
            ),
        }

        print_distribution(name, records)
        print(
            "Unique lesion groups:",
            len(split_ids[name]),
        )

    print_distribution(
        "excluded: missing lesion IDs",
        excluded,
    )

    with (
        OUTPUT_DIR / "split_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== VERIFICATION ===")
    print("Known lesion groups:", len(groups))
    print("Known-ID images:", len(all_known_ids))
    print("Excluded images:", len(excluded))
    print("Lesion overlap between splits: 0")
    print("\nFiles saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
