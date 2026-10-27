#!/usr/bin/env python3
"""MobileSkinXAI Phase 5.3: four-class MobileNetV3-Small baseline.

Requires the verified ISIC 2019 224x224 RGB preprocessing output:
    full/{train,validation,test}/{NV,MEL,BCC,SCC}/*.jpg

Training only reads train and validation image pixels. The test set is never
loaded or evaluated here. The best model is selected by validation macro-F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import __version__ as torchvision_version
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import InterpolationMode

CLASS_NAMES = ("NV", "MEL", "BCC", "SCC")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1

DEFAULT_DATA_DIR = (
    Path.home()
    / "Desktop/MobileSkinXAI-Datasets/ISIC2019/preprocessed"
    / "rgb224_letterbox_gray114_v1/full"
)
DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "Desktop/MobileSkinXAI-Datasets/ISIC2019/experiments"
    / "mobilenetv3_small_seed42"
)


class SkinLesionDataset(Dataset):
    def __init__(self, directory: Path, transform):
        self.directory = directory
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.counts: dict[str, int] = {}

        if not directory.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {directory}")

        unexpected = sorted(
            path.name for path in directory.iterdir()
            if path.is_dir() and path.name not in CLASS_TO_INDEX
        )
        if unexpected:
            raise ValueError(f"Unexpected class folders in {directory}: {unexpected}")

        for class_name in CLASS_NAMES:
            class_dir = directory / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            paths = sorted(
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
            )
            if not paths:
                raise ValueError(f"No JPEG images in {class_dir}")
            self.counts[class_name] = len(paths)
            self.samples.extend((path, CLASS_TO_INDEX[class_name]) for path in paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            if source.size != (224, 224):
                raise ValueError(f"Expected 224x224 image: {path} (got {source.size})")
            image = source.convert("RGB")
            tensor = self.transform(image)
        return tensor, label


def make_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=15,
            interpolation=InterpolationMode.BILINEAR,
            fill=(114, 114, 114),
        ),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transform, eval_transform


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # MPS and parallel GPU kernels may still have nondeterministic operations.


def get_device(requested: str):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; try --device cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; try --device cpu")
    return torch.device(requested)


def audit_folders(data_dir: Path):
    """Check filenames and expected split counts; do not read test image pixels."""
    expected = {"train": 13686, "validation": 2835, "test": 2941}
    all_ids: set[str] = set()
    for split, expected_count in expected.items():
        split_ids = set()
        class_counts = {}
        for class_name in CLASS_NAMES:
            folder = data_dir / split / class_name
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
            class_counts[class_name] = len(paths)
            for path in paths:
                if path.stem in split_ids:
                    raise ValueError(f"Duplicate image ID in {split}: {path.stem}")
                split_ids.add(path.stem)
        if len(split_ids) != expected_count:
            raise ValueError(f"{split}: expected {expected_count} images, found {len(split_ids)}")
        if not all_ids.isdisjoint(split_ids):
            raise ValueError(f"Image ID overlap detected between {split} and an earlier split")
        all_ids.update(split_ids)
        print(f"{split}: {len(split_ids):,} images; by class: {class_counts}")
    summary_path = data_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("class_to_index") != CLASS_TO_INDEX:
            raise ValueError("Class-to-index mapping differs from preprocessing summary")
        if summary.get("image_size") != [224, 224] or summary.get("color_mode") != "RGB":
            raise ValueError("Unexpected preprocessing configuration")
    print("PASS: Image IDs are disjoint and class indices match the preprocessing specification")


def make_model(fine_tune: bool):
    print("Loading pretrained MobileNetV3-Small ImageNet-1K weights (first run may download them)...")
    model = mobilenet_v3_small(weights=WEIGHTS)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(CLASS_NAMES))
    if not fine_tune:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def make_class_weights(counts: dict[str, int], device: torch.device):
    # Inverse-frequency weighted cross entropy, based on train only.
    total = sum(counts.values())
    weights = [total / (len(CLASS_NAMES) * counts[name]) for name in CLASS_NAMES]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def confusion_metrics(matrix: list[list[int]]):
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(CLASS_NAMES)))
    precision = {}
    recall = {}
    f1 = {}
    for i, label in enumerate(CLASS_NAMES):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(matrix[j][i] for j in range(len(CLASS_NAMES)))
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        score = 2 * p * r / (p + r) if p + r else 0.0
        precision[label] = round(p, 6)
        recall[label] = round(r, 6)
        f1[label] = round(score, 6)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1.values()) / len(CLASS_NAMES),
        "balanced_accuracy": sum(recall.values()) / len(CLASS_NAMES),
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": {name: sum(matrix[i]) for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix_true_rows_pred_columns": matrix,
    }


def run_epoch(model, loader, device, criterion, optimizer=None, freeze_features=False):
    is_train = optimizer is not None
    model.train(is_train)
    if is_train and freeze_features:
        # Freeze BatchNorm running statistics in the pretrained backbone.
        model.features.eval()

    total_loss = 0.0
    total_items = 0
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
        n = labels.size(0)
        total_loss += float(loss.detach().cpu()) * n
        total_items += n
        truth = labels.detach().cpu().tolist()
        predictions = logits.argmax(dim=1).detach().cpu().tolist()
        for truth_label, predicted_label in zip(truth, predictions):
            confusion[truth_label][predicted_label] += 1

    metrics = confusion_metrics(confusion)
    metrics["loss"] = total_loss / total_items
    return metrics


def save_history(path: Path, history: list[dict]):
    columns = ["epoch", "train_loss_weighted", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1", "val_balanced_accuracy", "val_scc_recall"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def smoke_test(model, train_dataset, validation_dataset, device, train_loss, val_loss, lr):
    # Include one example of each class in each smoke-test batch.
    def representative_indices(dataset):
        found = {}
        for index, (_, label) in enumerate(dataset.samples):
            if label not in found:
                found[label] = index
            if len(found) == len(CLASS_NAMES):
                break
        if len(found) != len(CLASS_NAMES):
            raise ValueError("Smoke-test set does not contain every class")
        return [found[index] for index in range(len(CLASS_NAMES))]

    train_loader = DataLoader(Subset(train_dataset, representative_indices(train_dataset)), batch_size=4, shuffle=False, num_workers=0)
    val_loader = DataLoader(Subset(validation_dataset, representative_indices(validation_dataset)), batch_size=4, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    print("SMOKE TEST: one training batch and one validation batch; no checkpoint will be saved")
    train_metrics = run_epoch(model, train_loader, device, train_loss, optimizer, freeze_features=not any(p.requires_grad for p in model.features.parameters()))
    val_metrics = run_epoch(model, val_loader, device, val_loss)
    print("PASS: batch shape [4, 3, 224, 224], four class labels, forward/backward/update completed")
    print(f"Training batch loss: {train_metrics['loss']:.4f}")
    print(f"Validation batch loss: {val_metrics['loss']:.4f}")
    print("This smoke test is NOT a training result or a model evaluation.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke-test", action="store_true", help="Run one train and one validation batch, without saving")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze pretrained backbone (later experiment)")
    parser.add_argument("--overwrite-run", action="store_true", help="Permit overwriting prior training artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.num_workers < 0:
        raise ValueError("Epochs, batch size, patience must be positive and workers >= 0")
    lr = args.learning_rate if args.learning_rate is not None else (0.0001 if args.fine_tune else 0.001)
    if not 0 < lr < 1:
        raise ValueError("Learning rate must be positive and below 1")
    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"Python: {sys.version.split()[0]} | torch: {torch.__version__} | torchvision: {torchvision_version}")
    print(f"Device: {device} | Mode: {'full fine-tuning' if args.fine_tune else 'frozen backbone, trainable classifier'}")
    print(f"Data root: {args.data_dir}")
    audit_folders(args.data_dir)

    train_transform, eval_transform = make_transforms()
    train_data = SkinLesionDataset(args.data_dir / "train", train_transform)
    val_data = SkinLesionDataset(args.data_dir / "validation", eval_transform)
    weights = make_class_weights(train_data.counts, device)
    print(f"Train-only class weights: {dict(zip(CLASS_NAMES, [round(float(w), 4) for w in weights.cpu()]))}")
    model = make_model(args.fine_tune).to(device)
    train_criterion = nn.CrossEntropyLoss(weight=weights)
    val_criterion = nn.CrossEntropyLoss()

    if args.smoke_test:
        smoke_test(model, train_data, val_data, device, train_criterion, val_criterion, lr)
        return

    checkpoint_path = args.output_dir / "best_model.pt"
    if checkpoint_path.exists() and not args.overwrite_run:
        raise FileExistsError(f"Existing checkpoint: {checkpoint_path}. Choose another --output-dir or explicitly pass --overwrite-run")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "mobilenet_v3_small",
        "pretraining": "IMAGENET1K_V1",
        "mode": "fine_tune_all" if args.fine_tune else "classifier_only",
        "classes": list(CLASS_NAMES),
        "class_to_index": CLASS_TO_INDEX,
        "mean": list(MEAN),
        "std": list(STD),
        "preprocessed_image_size": [224, 224],
        "preprocessing": "RGB224 letterbox gray114 (external Phase 5.2)",
        "training_augmentation": "horizontal/vertical flips p=.5; rotation +/-15; mild color jitter",
        "validation_augmentation": "none",
        "class_weights_train_only": [float(w) for w in weights.cpu()],
        "seed": args.seed,
        "device": str(device),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": lr,
        "patience": args.patience,
        "train_image_count": len(train_data),
        "validation_image_count": len(val_data),
        "data_directory": str(args.data_dir),
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version,
        "test_set_used_for_model_selection": False,
        "limitations": ["Not patient-independent", "No independent visual near-duplicate audit", "Not clinically validated"],
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Test set will not be evaluated in this training script.")
    print(f"Artifacts: {args.output_dir}")

    best_f1 = -1.0
    best_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, train_criterion, optimizer, freeze_features=not args.fine_tune)
        val_metrics = run_epoch(model, val_loader, device, val_criterion)
        row = {
            "epoch": epoch,
            "train_loss_weighted": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_scc_recall": val_metrics["recall_per_class"]["SCC"],
        }
        history.append(row)
        save_history(args.output_dir / "history.csv", history)
        print(f"Epoch {epoch:02d}/{args.epochs}: train weighted loss {row['train_loss_weighted']:.4f} | val loss {row['val_loss']:.4f} | val acc {row['val_accuracy']:.3f} | val macro-F1 {row['val_macro_f1']:.3f} | SCC recall {row['val_scc_recall']:.3f}", flush=True)
        improved = (
            val_metrics["macro_f1"] > best_f1 + 1e-8
            or (
                abs(val_metrics["macro_f1"] - best_f1) <= 1e-8
                and val_metrics["loss"] < best_loss
            )
        )
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_index": CLASS_TO_INDEX,
                    "architecture": "mobilenet_v3_small",
                    "weights_source": "IMAGENET1K_V1",
                    "train_mode": config["mode"],
                    "normalization_mean": list(MEAN),
                    "normalization_std": list(STD),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                },
                checkpoint_path,
            )
            (args.output_dir / "best_validation_metrics.json").write_text(
                json.dumps({"best_epoch": epoch, **val_metrics}, indent=2), encoding="utf-8"
            )
            print("  New best validation macro-F1; checkpoint saved.", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    print(f"Finished. Best validation macro-F1: {best_f1:.4f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print("No test-set images were used for training, validation, or model selection.")
    print("Research prototype only; NOT a clinical diagnosis system.")


if __name__ == "__main__":
    main()
