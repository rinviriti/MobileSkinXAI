#!/usr/bin/env python3
"""MobileSkinXAI Experiment 2: controlled partial fine-tuning.

Requires scripts/train_baseline.py beside this file, the verified four-class
ISIC 2019 dataset, and Experiment 1's best_model.pt checkpoint.

Starts from Experiment 1 weights, unfreezes the final TWO feature blocks,
retains frozen feature BatchNorm statistics, and tunes the classifier with a
higher learning rate than the unfrozen feature layers.

Only training and validation images are read; test filenames are audited
for disjointness but test image pixels are NOT used for model selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import __version__ as torchvision_version
from torchvision.models import mobilenet_v3_small

from train_baseline import (
    CLASS_NAMES, CLASS_TO_INDEX, MEAN, STD,
    DEFAULT_DATA_DIR, SkinLesionDataset,
    audit_folders, confusion_metrics, get_device,
    make_class_weights, make_transforms, seed_everything,
)

EXPERIMENT_ROOT = Path.home() / "Desktop/MobileSkinXAI-Datasets/ISIC2019/experiments"
DEFAULT_INITIAL_CHECKPOINT = EXPERIMENT_ROOT / "mobilenetv3_small_seed42/best_model.pt"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "mobilenetv3_small_partial_finetune_seed42"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--smoke-test", action="store_true", help="One train batch and one validation batch; writes nothing")
    return parser.parse_args()


def create_model_from_baseline(checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")
    # Load only a checkpoint YOU generated/trust. weights_only restricts pickled objects.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError("Checkpoint architecture is not MobileNetV3-Small")
    if checkpoint.get("class_to_index") != CLASS_TO_INDEX:
        raise ValueError("Checkpoint class mapping differs from ISIC four-class specification")
    if checkpoint.get("train_mode") != "classifier_only":
        raise ValueError("Expected the frozen-backbone baseline checkpoint")
    if checkpoint.get("normalization_mean") != list(MEAN) or checkpoint.get("normalization_std") != list(STD):
        raise ValueError("Checkpoint normalization differs from current data loader")

    # No ImageNet network download is necessary; load the exact baseline state.
    model = mobilenet_v3_small(weights=None)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # Freeze entire feature extractor, then unfreeze the final two blocks.
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True
    # Keep ALL feature BatchNorm affine parameters and running statistics fixed.
    for module in model.features.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for parameter in module.parameters():
                parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model, checkpoint


def keep_feature_batchnorm_frozen(model):
    for module in model.features.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def make_optimizer(model, backbone_lr, head_lr, weight_decay):
    backbone_parameters = [p for p in model.features.parameters() if p.requires_grad]
    head_parameters = [p for p in model.classifier.parameters() if p.requires_grad]
    if not backbone_parameters or not head_parameters:
        raise RuntimeError("No trainable parameters in one of the intended parameter groups")
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def epoch_pass(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        keep_feature_batchnorm_frozen(model)

    confusion = [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]
    weighted_sum = 0.0
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        size = labels.size(0)
        weighted_sum += loss.detach().item() * size
        count += size
        for truth, prediction in zip(
            labels.detach().cpu().tolist(), logits.detach().argmax(dim=1).cpu().tolist()
        ):
            confusion[truth][prediction] += 1
    if count == 0:
        raise ValueError("Empty training or validation DataLoader")
    metrics = confusion_metrics(confusion)
    metrics["loss"] = weighted_sum / count
    return metrics


def one_each_class(dataset):
    first_indices = {}
    for index, (_, class_idx) in enumerate(dataset.samples):
        first_indices.setdefault(class_idx, index)
        if len(first_indices) == len(CLASS_NAMES):
            break
    if len(first_indices) != len(CLASS_NAMES):
        raise ValueError("Smoke test requires at least one image of every class")
    return Subset(dataset, [first_indices[index] for index in range(len(CLASS_NAMES))])


def write_history(path, history):
    columns = [
        "epoch", "train_weighted_loss", "train_accuracy", "val_loss", "val_accuracy",
        "val_macro_f1", "val_balanced_accuracy", "val_mel_recall", "val_scc_recall",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.num_workers < 0:
        raise ValueError("Epochs/batch-size/patience must be positive; workers >= 0")
    if not (0 < args.backbone_lr < 1 and 0 < args.head_lr < 1):
        raise ValueError("Learning rates must be positive and < 1")
    if args.weight_decay < 0:
        raise ValueError("Weight decay cannot be negative")
    if not args.smoke_test and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {args.output_dir}. "
            "Select a NEW --output-dir to preserve all existing experiments."
        )

    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Torchvision {torchvision_version}")
    print(f"Device: {device}")
    audit_folders(args.data_dir)  # Names/counts only for test; NEVER loads test pixels.
    train_transform, eval_transform = make_transforms()
    train_data = SkinLesionDataset(args.data_dir / "train", train_transform)
    val_data = SkinLesionDataset(args.data_dir / "validation", eval_transform)
    model, initial_checkpoint = create_model_from_baseline(args.initial_checkpoint)
    model = model.to(device)
    criterion_train = nn.CrossEntropyLoss(weight=make_class_weights(train_data.counts, device))
    criterion_val = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded baseline checkpoint from epoch {initial_checkpoint.get('epoch')}")
    print(f"Baseline validation macro-F1: {initial_checkpoint.get('validation_metrics', {}).get('macro_f1')}")
    print(f"Unfrozen feature blocks: last 2 | trainable parameters: {num_trainable:,}")
    print("Feature BatchNorm parameters/statistics remain frozen.")

    if args.smoke_test:
        train_loader = DataLoader(one_each_class(train_data), batch_size=4, num_workers=0)
        val_loader = DataLoader(one_each_class(val_data), batch_size=4, num_workers=0)
        train_stats = epoch_pass(model, train_loader, device, criterion_train, optimizer)
        val_stats = epoch_pass(model, val_loader, device, criterion_val)
        print("PASS: fine-tuning forward/backward, classifier and final feature blocks updated")
        print(f"Smoke train loss: {train_stats['loss']:.4f} | smoke val loss: {val_stats['loss']:.4f}")
        print("SMOKE TEST ONLY — no checkpoint or genuine model results saved.")
        return

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "MobileNetV3-Small partial fine-tuning from frozen-backbone checkpoint",
        "initial_checkpoint": str(args.initial_checkpoint),
        "initial_checkpoint_epoch": initial_checkpoint.get("epoch"),
        "initial_val_macro_f1": initial_checkpoint.get("validation_metrics", {}).get("macro_f1"),
        "classes": list(CLASS_NAMES), "class_to_index": CLASS_TO_INDEX,
        "preprocessing": "224x224 RGB letterbox gray114; ImageNet normalize",
        "mean": list(MEAN), "std": list(STD),
        "augmentation": "Training only; inherited from train_baseline.py",
        "fine_tuned_features": "last two feature blocks; all other feature layers frozen",
        "batchnorm_policy": "all feature BatchNorm affine and running statistics frozen",
        "optimizer": "AdamW", "head_lr": args.head_lr, "backbone_lr": args.backbone_lr,
        "weight_decay": args.weight_decay, "epochs_requested": args.epochs,
        "batch_size": args.batch_size, "patience": args.patience, "seed": args.seed,
        "device": str(device), "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version,
        "test_set_evaluated": False,
        "limitations": ["Not patient independent", "No near-duplicate audit", "Research use only"],
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    best_f1, best_loss = -1.0, float("inf")
    patience_count = 0
    for epoch in range(1, args.epochs + 1):
        train_result = epoch_pass(model, train_loader, device, criterion_train, optimizer)
        val_result = epoch_pass(model, val_loader, device, criterion_val)
        row = {
            "epoch": epoch,
            "train_weighted_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "val_macro_f1": val_result["macro_f1"],
            "val_balanced_accuracy": val_result["balanced_accuracy"],
            "val_mel_recall": val_result["recall_per_class"]["MEL"],
            "val_scc_recall": val_result["recall_per_class"]["SCC"],
        }
        history.append(row)
        write_history(args.output_dir / "history.csv", history)
        print(
            f"Epoch {epoch:02}/{args.epochs}: train weighted loss {row['train_weighted_loss']:.4f} "
            f"| val loss {row['val_loss']:.4f} | val acc {row['val_accuracy']:.3f} "
            f"| macro-F1 {row['val_macro_f1']:.3f} | MEL recall {row['val_mel_recall']:.3f} "
            f"| SCC recall {row['val_scc_recall']:.3f}", flush=True,
        )
        improved = (
            val_result["macro_f1"] > best_f1 + 1e-8
            or (abs(val_result["macro_f1"] - best_f1) <= 1e-8 and val_result["loss"] < best_loss)
        )
        if improved:
            best_f1, best_loss = val_result["macro_f1"], val_result["loss"]
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_index": CLASS_TO_INDEX,
                    "architecture": "mobilenet_v3_small",
                    "weights_source": str(args.initial_checkpoint),
                    "train_mode": "partial_fine_tune_last_two_feature_blocks",
                    "normalization_mean": list(MEAN),
                    "normalization_std": list(STD),
                    "epoch": epoch,
                    "validation_metrics": val_result,
                },
                args.output_dir / "best_model.pt",
            )
            (args.output_dir / "best_validation_metrics.json").write_text(
                json.dumps({"best_epoch": epoch, **val_result}, indent=2), encoding="utf-8"
            )
            print("  New best validation macro-F1; checkpoint saved.", flush=True)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping (patience={args.patience}).")
                break
    print(f"Finished. Best validation macro-F1: {best_f1:.4f}")
    print(f"Best checkpoint: {args.output_dir / 'best_model.pt'}")
    print("Independent test images were not loaded or used for model selection.")
    print("Research prototype — NOT clinically validated.")


if __name__ == "__main__":
    main()
