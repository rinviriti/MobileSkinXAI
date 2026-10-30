#!/usr/bin/env python3
"""MobileSkinXAI Experiment 7: high-resolution MobileNetV4-Conv-Small.

Hypothesis: resampling source ISIC2019 JPEGs to 320x320 (rather than 224x224)
may improve class-balanced validation performance under the Experiment 7
training protocol. The two stages are frozen ImageNet backbone (up to 8 epochs)
and final-two-block partial fine-tuning (up to 12 epochs).

PREREQUISITE: run scripts/preprocess_isic2019.py --size 320 WITHOUT
reusing or enlarging previously resized 224x224 images. Both 224 and 320
preprocessing outputs must exist. The script checks manifests, labels,
lesion IDs, class counts, and the training/validation image dimensions.

Existing 224x224 Experiment 5 results are READ ONLY. New 320x320 results use new
folders and cannot overwrite past experiments. Best weights are selected only
by validation macro-F1. The independent test set is never used for training,
validation, model selection, or tuning. Image IDs/metadata for all splits are
checked for provenance; no held-out test image pixels are opened by this script.

Caution: this is a single-seed resolution comparison, not a clinically
validated model. Training stochasticity and batch-size changes may confound
resolution comparisons; evaluate multiple seeds subsequently.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    import timm
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: timm. Install with:\n"
        "  ~/Desktop/MobileSkinXAI-train-venv/bin/python -m pip install 'timm>=1.0,<2'"
    ) from exc

from train_baseline import (
    CLASS_NAMES, CLASS_TO_INDEX, SkinLesionDataset,
    confusion_metrics, get_device, make_class_weights,
    seed_everything,
)

MODEL_ID = "mobilenetv4_conv_small.e2400_r224_in1k"
INPUT_SIZE = 320
PRETRAIN_INPUT_SIZE = 224
BASE_PREPROCESSING = Path.home() / "Desktop" / "MobileSkinXAI-Datasets" / "ISIC2019" / "preprocessed"
DEFAULT_DATA_DIR = BASE_PREPROCESSING / "rgb320_letterbox_gray114_v1" / "full"
DEFAULT_REFERENCE_DATA_DIR = BASE_PREPROCESSING / "rgb224_letterbox_gray114_v1" / "full"
DEFAULT_EXPERIMENTS_DIR = (
    Path.home() / "Desktop" / "MobileSkinXAI-Datasets" / "ISIC2019" / "experiments"
)
DEFAULT_MOBILENET_REFERENCE = DEFAULT_EXPERIMENTS_DIR / "mobilenetv3_small_partial_finetune_seed42"
DEFAULT_MOBILENETV4_REFERENCE = DEFAULT_EXPERIMENTS_DIR / "mobilenetv4_conv_small_partial_finetune_seed42"
STAGES = ("frozen", "finetune")


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reference-data-dir", type=Path, default=DEFAULT_REFERENCE_DATA_DIR,
                        help="Frozen 224x224 preprocessing directory; used for label/split audit ONLY")
    parser.add_argument("--audit-only", action="store_true", help="Audit 320x320 output against 224x224 manifests; no training")
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR)
    parser.add_argument("--stage", choices=("all", "frozen", "finetune"), default="all")
    parser.add_argument("--smoke-test", action="store_true", help="One 4-image train/val batch per stage, no files")
    parser.add_argument("--compare-only", action="store_true", help="Print comparison from saved validation metrics without training")
    parser.add_argument("--frozen-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-lr-frozen", type=float, default=0.001)
    parser.add_argument("--head-lr-finetune", type=float, default=0.0001)
    parser.add_argument("--backbone-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mobilenetv4-reference", type=Path, default=DEFAULT_MOBILENETV4_REFERENCE,
                        help="Existing 224x224 MobileNetV4 fine-tuning results (read-only)")
    return parser.parse_args()


def stage_path(args, stage):
    suffix = "frozen" if stage == "frozen" else "partial_finetune"
    return args.experiments_dir / f"mobilenetv4_conv_small_rgb320_{suffix}_seed{args.seed}"


def require_empty_dir(path):
    if path.exists():
        raise FileExistsError(
            f"Output directory already exists: {path}\n"
            "Choose a different --experiments-dir, or run only a missing stage. "
            "Existing research artifacts must not be overwritten."
        )


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SkinLesion320Dataset(SkinLesionDataset):
    """Reuse Experiment 7 ordering and labels, but require true 320x320 inputs."""

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            if source.size != (INPUT_SIZE, INPUT_SIZE):
                raise ValueError(f"Expected {INPUT_SIZE}x{INPUT_SIZE} image: {path}; got {source.size}")
            image = source.convert("RGB")
            tensor = self.transform(image)
        return tensor, label


def read_manifest(directory):
    path = directory / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing preprocessing manifest: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        entries = list(csv.DictReader(handle))
    expected_columns = {"image", "label", "lesion_id", "split", "processed_relative_path", "label_index"}
    if not entries or not expected_columns.issubset(entries[0]):
        raise ValueError(f"Incomplete preprocessing manifest: {path}")
    image_index = {}
    for row in entries:
        image_id = row["image"].strip()
        split = row["split"].strip()
        label = row["label"].strip()
        lesion = row["lesion_id"].strip()
        if not image_id or image_id in image_index:
            raise ValueError(f"Invalid or duplicate image ID in {path}: {image_id!r}")
        if label not in CLASS_TO_INDEX or split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split/label in {path}: {split}/{label}")
        if not lesion:
            raise ValueError(f"Missing lesion ID for {image_id}")
        if int(row["label_index"]) != CLASS_TO_INDEX[label]:
            raise ValueError(f"Class mapping mismatch for {image_id}")
        expected_relative_path = f"{split}/{label}/{image_id}.jpg"
        if row["processed_relative_path"] != expected_relative_path:
            raise ValueError(f"Unexpected relative path for {image_id}")
        image_index[image_id] = row
    return image_index


def audit_resolution_data(high_res, reference):
    """Check all split assignments and lesion grouping; never decode test images."""
    high_res = Path(high_res)
    reference = Path(reference)
    if high_res.resolve() == reference.resolve():
        raise ValueError("320 and 224 preprocessing directories must be different")
    summaries = []
    for folder, expected_side in ((high_res, INPUT_SIZE), (reference, PRETRAIN_INPUT_SIZE)):
        summary = read_json(folder / "summary.json")
        if (summary.get("status") != "complete" or
                summary.get("image_size") != [expected_side, expected_side] or
                summary.get("color_mode") != "RGB" or
                summary.get("class_to_index") != CLASS_TO_INDEX or
                summary.get("run_type") != "full"):
            raise ValueError(f"Unexpected preprocessing summary: {folder}")
        summaries.append(summary)
    current = read_manifest(high_res)
    baseline = read_manifest(reference)
    if current.keys() != baseline.keys():
        missing = sorted(baseline.keys() - current.keys())[:5]
        extra = sorted(current.keys() - baseline.keys())[:5]
        raise ValueError(f"Image IDs differ between 224 and 320: missing={missing}, extra={extra}")
    expected_counts = {"train": 13686, "validation": 2835, "test": 2941}
    by_split = {part: {c: 0 for c in CLASS_NAMES} for part in expected_counts}
    lesion_owner = {}
    for image_id, row in current.items():
        old = baseline[image_id]
        for key in ("split", "label", "label_index", "lesion_id", "original_width", "original_height"):
            if row.get(key) != old.get(key):
                raise ValueError(f"{image_id}: preprocessing metadata changed field {key}")
        split, label, lesion = row["split"], row["label"], row["lesion_id"]
        previous_split = lesion_owner.setdefault(lesion, split)
        if previous_split != split:
            raise ValueError(f"Lesion {lesion} overlaps between {previous_split} and {split}")
        by_split[split][label] += 1
        destination = high_res / row["processed_relative_path"]
        if not destination.is_file():
            raise FileNotFoundError(destination)
        # Only training + validation image pixels will be read during training.
        # For audit, inspect the JPEG header of development images; leave test closed.
        if split != "test":
            with Image.open(destination) as opened:
                if opened.size != (INPUT_SIZE, INPUT_SIZE) or opened.mode != "RGB":
                    raise ValueError(f"Incorrect image size/mode: {destination}")
    for split, classes in by_split.items():
        observed = sum(classes.values())
        if observed != expected_counts[split]:
            raise ValueError(f"{split}: {observed} images, expected {expected_counts[split]}")
        for label, count in classes.items():
            directory = high_res / split / label
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            # Protect against extra image files that are absent from the manifest.
            image_files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
            if len(image_files) != count:
                raise ValueError(f"{split}/{label}: {len(image_files)} files but {count} manifest entries")
        print(f"{split}: {observed:,} images; by class: {classes}", flush=True)
    print(f"PASS: same {len(current):,} image IDs, labels, lesion groups and split assignments at 224/320", flush=True)
    print("PASS: development images have RGB 320x320 headers; independent test pixels were not opened", flush=True)
    print("Note: this is not a perceptual duplicate / clinical image-quality audit.", flush=True)
    return current


def make_model(pretrained):
    # Passing num_classes=4 retains the ImageNet backbone and creates a new head.
    # Checkpoint reloads use pretrained=False and never redownload weights.
    model = timm.create_model(MODEL_ID, pretrained=pretrained, num_classes=len(CLASS_NAMES))
    if not hasattr(model, "blocks") or not hasattr(model, "classifier"):
        raise TypeError("Unexpected timm MobileNetV4 model: missing blocks/classifier")
    if len(model.blocks) < 2:
        raise TypeError("MobileNetV4 Conv Small does not have two block groups")
    if not isinstance(model.classifier, nn.Linear):
        raise TypeError("Expected an nn.Linear MobileNetV4 classifier")
    if model.classifier.out_features != len(CLASS_NAMES):
        raise TypeError("Model head does not produce four logits")
    return model


def get_model_norm(model):
    cfg = timm.data.resolve_model_data_config(model)
    if tuple(cfg.get("input_size", ())) != (3, PRETRAIN_INPUT_SIZE, PRETRAIN_INPUT_SIZE):
        raise ValueError(f"Expected 224x224 pretrained configuration, got {cfg.get('input_size')}")
    # Input tensors at 320x320 are tested in --smoke-test. This convolutional
    # architecture must successfully apply its own adaptive spatial pooling.

    mean = tuple(float(v) for v in cfg["mean"])
    std = tuple(float(v) for v in cfg["std"])
    if len(mean) != 3 or len(std) != 3 or not all(v > 0 for v in std):
        raise ValueError("Unexpected pretrained normalization parameters")
    return mean, std


def make_transforms(mean, std):
    # Same geometric/photometric augmentation as train_baseline.py.
    train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR, fill=(114, 114, 114)),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    validation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train, validation


def configure_trainable(model, stage):
    # Explicitly freeze all backbone parameters and normalization layers.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    if stage == "finetune":
        for block_group in model.blocks[-2:]:
            for p in block_group.parameters():
                p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_batchnorm_running_stats(model):
    # Preserve pretrained BN running means/variances across both stages.
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()


def make_optimizer(model, args, stage):
    head = [p for p in model.classifier.parameters() if p.requires_grad]
    if not head:
        raise RuntimeError("Classifier has no trainable parameters")
    if stage == "frozen":
        return torch.optim.AdamW(head, lr=args.head_lr_frozen, weight_decay=args.weight_decay)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    if not backbone:
        raise RuntimeError("Fine-tuning stage has no unfrozen backbone parameters")
    return torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr},
        {"params": head, "lr": args.head_lr_finetune},
    ], weight_decay=args.weight_decay)


def epoch_pass(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        freeze_batchnorm_running_stats(model)
    confusion = [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]
    total_loss = 0.0
    total_samples = 0
    for images, target in loader:
        images = images.to(device)
        target = target.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
                raise RuntimeError(f"Invalid classification head output: {tuple(logits.shape)}")
            loss = criterion(logits, target)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.detach().item() * target.size(0)
        total_samples += target.size(0)
        for truth, prediction in zip(
            target.detach().cpu().tolist(), logits.detach().argmax(1).cpu().tolist()
        ):
            confusion[truth][prediction] += 1
    if total_samples == 0:
        raise ValueError("Empty DataLoader")
    result = confusion_metrics(confusion)
    result["loss"] = total_loss / total_samples
    return result


def subset_one_per_class(dataset):
    selected = {}
    for index, (_, label) in enumerate(dataset.samples):
        selected.setdefault(label, index)
        if len(selected) == len(CLASS_NAMES):
            break
    if set(selected) != set(range(len(CLASS_NAMES))):
        raise ValueError("One example per class is required for the smoke test")
    return torch.utils.data.Subset(dataset, [selected[i] for i in range(len(CLASS_NAMES))])


def build_loaders(train_data, val_data, args, device, smoke=False):
    if smoke:
        return (
            DataLoader(subset_one_per_class(train_data), batch_size=4, num_workers=0),
            DataLoader(subset_one_per_class(val_data), batch_size=4, num_workers=0),
        )
    return (
        DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                   num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                   generator=torch.Generator().manual_seed(args.seed)),
        DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
    )


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "epoch", "train_weighted_loss", "train_accuracy", "val_loss", "val_accuracy",
            "val_macro_f1", "val_balanced_accuracy", "val_mel_recall", "val_scc_recall",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Frozen backbone checkpoint does not exist: {path}")
    # This is a checkpoint written by our own script. Never load untrusted pickle files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != MODEL_ID:
        raise ValueError("Checkpoint architecture mismatch")
    if checkpoint.get("class_to_index") != CLASS_TO_INDEX:
        raise ValueError("Checkpoint class mapping mismatch")
    if checkpoint.get("input_size") != [3, INPUT_SIZE, INPUT_SIZE]:
        raise ValueError("Frozen checkpoint input resolution mismatch")
    model = make_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def run_stage(args, stage, device, train_data, val_data, mean, std, model=None, initial_checkpoint=None):
    seed_everything(args.seed)
    if stage == "frozen":
        if model is None:
            model = make_model(pretrained=True)
        parent_checkpoint = None
    else:
        if model is None:
            model, initial_checkpoint = load_frozen_checkpoint(stage_path(args, "frozen") / "best_model.pt")
        parent_checkpoint = str(stage_path(args, "frozen") / "best_model.pt")
    trainable = configure_trainable(model, stage)
    model = model.to(device)
    train_weights = make_class_weights(train_data.counts, device)
    train_loss = nn.CrossEntropyLoss(weight=train_weights)
    val_loss = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args, stage)
    smoke = args.smoke_test
    train_loader, val_loader = build_loaders(train_data, val_data, args, device, smoke=smoke)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n=== {stage.upper()} ===", flush=True)
    print(f"Model: {MODEL_ID} | parameters: {param_count:,} | trainable: {trainable:,}", flush=True)
    print("Training weights:", {k: round(float(train_weights[i].detach().cpu()), 4) for i, k in enumerate(CLASS_NAMES)})
    print(f"Pretrained input mean: {mean} | std: {std}")
    if smoke:
        tr = epoch_pass(model, train_loader, device, train_loss, optimizer)
        va = epoch_pass(model, val_loader, device, val_loss)
        print(f"PASS {stage}: 4-image train/validation batches, forward/backward/update complete")
        print(f"  smoke train loss {tr['loss']:.4f}; smoke validation loss {va['loss']:.4f}")
        print("  Smoke tests write no experiment results.")
        return None

    target = stage_path(args, stage)
    require_empty_dir(target)
    target.mkdir(parents=True, exist_ok=False)
    epochs = args.frozen_epochs if stage == "frozen" else args.finetune_epochs
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "MobileSkinXAI Experiment 7: MobileNetV4-Conv-Small 320px resolution study",
        "stage": stage,
        "model_id": MODEL_ID,
        "pretrained": "ImageNet-1K pretrained via timm (third-party timm weights)" if stage == "frozen" else "Experiment 7 frozen-stage best checkpoint",
        "parent_checkpoint": parent_checkpoint,
        "parent_checkpoint_sha256": file_sha256(Path(parent_checkpoint)) if parent_checkpoint else None,
        "architecture": MODEL_ID,
        "classes": list(CLASS_NAMES), "class_to_index": CLASS_TO_INDEX,
        "input_size": [3, INPUT_SIZE, INPUT_SIZE], "mean": list(mean), "std": list(std),
        "pretrained_input_size": [3, PRETRAIN_INPUT_SIZE, PRETRAIN_INPUT_SIZE],
        "input_preprocessing": "RGB 320x320 from original ISIC2019 JPEG, aspect-preserving letterbox gray114; then model-specific normalize",
        "augmentation": "training only: horizontal/vertical flips p=.5, +/-15 deg rotation, brightness/contrast .08, saturation .05",
        "loss": "inverse-frequency weighted cross-entropy (training only)",
        "loss_weights": [float(w) for w in train_weights.cpu()],
        "optimizer": "AdamW", "head_lr": args.head_lr_frozen if stage == "frozen" else args.head_lr_finetune,
        "backbone_lr": None if stage == "frozen" else args.backbone_lr,
        "weight_decay": args.weight_decay, "patience": args.patience,
        "requested_epochs": epochs, "batch_size": args.batch_size,
        "training_images": len(train_data), "validation_images": len(val_data),
        "trainable_parameters": trainable, "total_parameters": param_count,
        "seed": args.seed, "device": str(device), "num_workers": args.num_workers,
        "timm_model_id": MODEL_ID, "pretrained_weights_source": "timm/mobilenetv4_conv_small.e2400_r224_in1k", "data_license_note": "Check ISIC and ImageNet pretrained weight terms before use",
        "torch_version": str(torch.__version__), "timm_version": str(timm.__version__),
        "data_dir": str(args.data_dir),
        "reference_224_data_dir": str(args.reference_data_dir),
        "source_320_manifest_sha256": file_sha256(args.data_dir / "manifest.csv"),
        "source_224_manifest_sha256": file_sha256(args.reference_data_dir / "manifest.csv"),
        "model_id_note": "224px ImageNet pretrained config accepts 320px tensor sizes after verified forward pass",
        "comparison_reference_224": str(args.mobilenetv4_reference),
        "test_set_used": False,
        "limitations": [
            "Known lesion IDs disjoint; patient-level independence not verified",
            "Near-duplicate image search not performed",
            "Architecture may require separate mobile inference conversion and benchmarking",
            "224 vs 320 training stochasticity and potential batch-size changes may confound resolution comparisons",
            "No patient IDs; patient-level independence unverified",
            "Validation repeatedly consulted across model experiments; independent test held out",
            "Single-seed results; GPU kernels may not be fully deterministic",
            "Research only; not clinically validated",
        ],
    }
    (target / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    best_f1 = -1.0
    best_loss = math.inf
    patience_counter = 0
    best_epoch = None
    for epoch in range(1, epochs + 1):
        tr = epoch_pass(model, train_loader, device, train_loss, optimizer)
        va = epoch_pass(model, val_loader, device, val_loss)
        history.append({
            "epoch": epoch,
            "train_weighted_loss": tr["loss"], "train_accuracy": tr["accuracy"],
            "val_loss": va["loss"], "val_accuracy": va["accuracy"],
            "val_macro_f1": va["macro_f1"], "val_balanced_accuracy": va["balanced_accuracy"],
            "val_mel_recall": va["recall_per_class"]["MEL"],
            "val_scc_recall": va["recall_per_class"]["SCC"],
        })
        save_csv(target / "history.csv", history)
        print(
            f"{stage} Epoch {epoch:02d}/{epochs}: train weighted loss {tr['loss']:.4f} "
            f"| val loss {va['loss']:.4f} | val acc {va['accuracy']:.3f} "
            f"| macro-F1 {va['macro_f1']:.4f} "
            f"| MEL recall {va['recall_per_class']['MEL']:.3f} "
            f"| SCC recall {va['recall_per_class']['SCC']:.3f}", flush=True,
        )
        is_better = (va["macro_f1"] > best_f1 + 1e-8 or
                     (abs(va["macro_f1"] - best_f1) <= 1e-8 and va["loss"] < best_loss))
        if is_better:
            best_f1, best_loss, best_epoch = va["macro_f1"], va["loss"], epoch
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "architecture": MODEL_ID, "class_to_index": CLASS_TO_INDEX,
                "epoch": epoch, "stage": stage,
                "input_size": [3, INPUT_SIZE, INPUT_SIZE],
                "normalization_mean": list(mean), "normalization_std": list(std),
                "validation_metrics": va,
            }
            torch.save(checkpoint, target / "best_model.pt")
            (target / "best_validation_metrics.json").write_text(
                json.dumps({"best_epoch": epoch, **va}, indent=2), encoding="utf-8"
            )
            print("  New best validation macro-F1; checkpoint saved.", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping (patience={args.patience}).", flush=True)
                break
    print(f"Finished {stage}. Best val macro-F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"Checkpoint: {target / 'best_model.pt'}")
    return target


def print_comparison(args):
    """Read saved VALIDATION metrics only, never run inference on the test split."""
    references = (
        ("224px MobileNetV4 partial fine-tune", args.mobilenetv4_reference),
        ("320px MobileNetV4 frozen", stage_path(args, "frozen")),
        ("320px MobileNetV4 partial fine-tune", stage_path(args, "finetune")),
    )
    results = []
    for label, folder in references:
        metric_path = folder / "best_validation_metrics.json"
        if metric_path.is_file():
            results.append((label, folder, read_json(metric_path)))
    if not results:
        raise FileNotFoundError("No validation results available for comparison")
    print("\n=== EXPERIMENT 7: VALIDATION-ONLY RESOLUTION COMPARISON ===")
    print("The independent test set has NOT been evaluated.")
    print("NOTE: Training stochasticity and any differing batch sizes affect interpretation.")
    for label, folder, m in results:
        print(f"{label:43} acc={m['accuracy']:.4f} "
              f"macro-F1={m['macro_f1']:.4f} balanced-acc={m['balanced_accuracy']:.4f} "
              f"MEL-recall={m['recall_per_class']['MEL']:.3f} "
              f"SCC-recall={m['recall_per_class']['SCC']:.3f} | {folder}")
    print("Highest single-run validation macro-F1:",
          max(results, key=lambda x: x[2]["macro_f1"])[0])


def main():
    args = args_parser()
    if min(args.frozen_epochs, args.finetune_epochs, args.batch_size, args.patience) < 1:
        raise ValueError("Epoch, batch size and patience values must be positive")
    if args.num_workers < 0 or args.weight_decay < 0:
        raise ValueError("Workers and weight decay must be nonnegative")
    if any(not (0 < value < 1) for value in (
        args.head_lr_frozen, args.head_lr_finetune, args.backbone_lr
    )):
        raise ValueError("All learning rates must be positive and below 1")
    if sum(bool(x) for x in (args.smoke_test, args.compare_only, args.audit_only)) > 1:
        raise ValueError("Choose only one of --smoke-test, --audit-only, --compare-only")
    if args.compare_only:
        print_comparison(args)
        return
    if args.audit_only:
        audit_resolution_data(args.data_dir, args.reference_data_dir)
        print("Audit complete. No training files written.")
        return
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        if not args.smoke_test:
            require_empty_dir(stage_path(args, stage))
    if "finetune" in stages and "frozen" not in stages and not args.smoke_test:
        if not (stage_path(args, "frozen") / "best_model.pt").is_file():
            raise FileNotFoundError("Cannot run --stage finetune without frozen checkpoint")

    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | timm {timm.__version__}")
    print(f"Device: {device} | stages: {', '.join(stages)}")
    audit_resolution_data(args.data_dir, args.reference_data_dir)
    # For stage-only fine-tune, inspect the stored pretrained-data normalization.
    if stages == ("finetune",) and not args.smoke_test:
        model, previous = load_frozen_checkpoint(stage_path(args, "frozen") / "best_model.pt")
        prior_config = read_json(stage_path(args, "frozen") / "run_config.json")
        if prior_config["class_to_index"] != CLASS_TO_INDEX or prior_config["seed"] != args.seed:
            raise ValueError("Frozen-stage configuration is incompatible")
        if prior_config["data_dir"] != str(args.data_dir):
            raise ValueError("Frozen stage used different data directory")
        if prior_config.get("input_size") != [3, INPUT_SIZE, INPUT_SIZE]:
            raise ValueError("Frozen stage used a different resolution")
        if prior_config["batch_size"] != args.batch_size:
            raise ValueError("Frozen stage used a different batch size")
        mean, std = tuple(prior_config["mean"]), tuple(prior_config["std"])
    else:
        print("Loading MobileNetV4-Conv-Small pretrained weights (ImageNet configuration trained at 224px)...", flush=True)
        model = make_model(pretrained=True)
        mean, std = get_model_norm(model)
    train_transform, val_transform = make_transforms(mean, std)
    train_data = SkinLesion320Dataset(args.data_dir / "train", train_transform)
    val_data = SkinLesion320Dataset(args.data_dir / "validation", val_transform)
    for stage in stages:
        if stage == "frozen":
            run_stage(args, stage, device, train_data, val_data, mean, std, model=model)
        else:
            if args.smoke_test:
                # smoke test can use existing in-memory pretrained model, no disk checkpoint
                run_stage(args, stage, device, train_data, val_data, mean, std, model=model)
            else:
                run_stage(args, stage, device, train_data, val_data, mean, std)
    if not args.smoke_test:
        print_comparison(args)
    print("\nNo independent test images loaded or evaluated.")
    print("Research use only; NOT a medical diagnosis system.")


if __name__ == "__main__":
    main()
