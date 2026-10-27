#!/usr/bin/env python3
"""MobileSkinXAI Experiment 3: controlled MobileNetV3-Small loss-weight ablation.

Requires train_baseline.py and finetune_mobilenetv3.py in the same scripts/
folder, plus the completed Experiment 1 and 2 artifacts.

Compares TWO new fine-tuning runs against the existing Experiment 2:
  none: unweighted cross entropy, weights [1, 1, 1, 1]
  sqrt: square root of train-only inverse-frequency weights
  inverse: existing Experiment 2 (read-only reference; NEVER retrained here)

All variants start from Experiment 1's SAME epoch-7 baseline checkpoint.
Only the TRAINING loss weighting changes; model, optimizer, augmentation,
random seed and validation protocol are inherited from Experiment 2.

No test images are opened: audit_folders reads test filenames only.
No previously saved checkpoints or results are overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import __version__ as torchvision_version

from train_baseline import (
    CLASS_NAMES, CLASS_TO_INDEX, DEFAULT_DATA_DIR, MEAN, STD,
    SkinLesionDataset, audit_folders, get_device, make_class_weights,
    make_transforms, seed_everything,
)
from finetune_mobilenetv3 import (
    DEFAULT_INITIAL_CHECKPOINT, DEFAULT_OUTPUT_DIR as INVERSE_REFERENCE,
    create_model_from_baseline, epoch_pass, make_optimizer,
    one_each_class, write_history,
)

EXPERIMENTS_DIR = Path.home() / "Desktop/MobileSkinXAI-Datasets/ISIC2019/experiments"
VARIANT_NAMES = ("none", "sqrt")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--inverse-reference", type=Path, default=INVERSE_REFERENCE,
                        help="Completed Experiment 2 directory; read only")
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--weighting", choices=("all", "none", "sqrt"), default="all")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--smoke-test", action="store_true",
                        help="One training and validation batch per variant; write NOTHING")
    parser.add_argument("--compare-only", action="store_true",
                        help="Compare saved results; do not train or rewrite checkpoints")
    return parser.parse_args()


def directory_for(args, variant):
    return args.experiments_dir / f"mobilenetv3_small_weight_{variant}_seed{args.seed}"


def comparison_dir(args):
    return args.experiments_dir / f"mobilenetv3_small_weight_ablation_seed{args.seed}"


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_reference(args):
    """Ensure the established inverse-weight run is genuinely comparable."""
    config = read_json(args.inverse_reference / "run_config.json")
    metrics = read_json(args.inverse_reference / "best_validation_metrics.json")
    checks = {
        "initial_checkpoint": str(args.initial_checkpoint),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "class_to_index": CLASS_TO_INDEX,
        "mean": list(MEAN),
        "std": list(STD),
    }
    mismatches = [
        (name, config.get(name), expected)
        for name, expected in checks.items()
        if config.get(name) != expected
    ]
    if mismatches:
        details = "\n".join(
            f"  {name}: Experiment 2={actual!r}, requested={expected!r}"
            for name, actual, expected in mismatches
        )
        raise ValueError(
            "Experiment 2 configuration differs from this ablation. "
            "For a controlled comparison, match its settings:\n" + details
        )
    if set(config.get("classes", [])) != set(CLASS_NAMES):
        raise ValueError("Experiment 2 class set differs from expected four classes")
    if "macro_f1" not in metrics:
        raise ValueError("Experiment 2 metrics do not include macro-F1")
    return config, metrics


def class_weights(counts, device, variant):
    inverse = make_class_weights(counts, device)
    if variant == "none":
        return torch.ones_like(inverse)
    if variant == "sqrt":
        return torch.sqrt(inverse)
    raise ValueError(f"Invalid weight strategy: {variant}")


def preflight_outputs(args, variants):
    # Fail BEFORE training anything if any destination could be overwritten.
    for variant in variants:
        dest = directory_for(args, variant)
        if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
            raise FileExistsError(
                f"Output already exists: {dest}\n"
                f"Use --weighting to run only a missing variant, or a new "
                f"--experiments-dir. Never overwrite research results."
            )
    target = comparison_dir(args)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(
            f"Comparison directory already exists: {target}; "
            f"existing results will not be overwritten."
        )


def make_loaders(train_data, val_data, args, device):
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader


def run_variant(args, variant, device, train_data, val_data):
    # Independent initialization: each run starts at identical baseline weights,
    # and uses identical seeded data ordering and stochastic augmentations.
    seed_everything(args.seed)
    model, checkpoint = create_model_from_baseline(args.initial_checkpoint)
    model.to(device)
    optimizer = make_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    weights = class_weights(train_data.counts, device, variant)
    train_criterion = nn.CrossEntropyLoss(weight=weights)
    val_criterion = nn.CrossEntropyLoss()  # Comparable unweighted val loss.

    print(f"\n=== VARIANT: {variant.upper()} ===")
    print("Starting checkpoint epoch:", checkpoint.get("epoch"))
    print("Training class weights:", {
        cls: round(float(weights[i].detach().cpu()), 6)
        for i, cls in enumerate(CLASS_NAMES)
    })
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.smoke_test:
        train_loader = DataLoader(one_each_class(train_data), batch_size=4, num_workers=0)
        val_loader = DataLoader(one_each_class(val_data), batch_size=4, num_workers=0)
        train_stats = epoch_pass(model, train_loader, device, train_criterion, optimizer)
        val_stats = epoch_pass(model, val_loader, device, val_criterion)
        print(f"PASS {variant}: forward/backward/update and validation batch ")
        print(f"  train batch loss: {train_stats['loss']:.4f}; validation batch loss: {val_stats['loss']:.4f}")
        print("  SMOKE TEST ONLY; no files written or scientific metrics produced.")
        return

    dest = directory_for(args, variant)
    train_loader, val_loader = make_loaders(train_data, val_data, args, device)
    dest.mkdir(parents=True, exist_ok=True)  # preflight confirmed destination is empty
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "MobileNetV3-Small loss weighting ablation, partial fine-tune",
        "weighting": variant,
        "class_weights_train_only": [float(w) for w in weights.detach().cpu()],
        "initial_checkpoint": str(args.initial_checkpoint),
        "initial_checkpoint_epoch": checkpoint.get("epoch"),
        "initial_val_macro_f1": checkpoint.get("validation_metrics", {}).get("macro_f1"),
        "classes": list(CLASS_NAMES), "class_to_index": CLASS_TO_INDEX,
        "data_directory": str(args.data_dir),
        "preprocessing": "224x224 RGB letterbox gray114; ImageNet normalize",
        "mean": list(MEAN), "std": list(STD),
        "augmentation": "Training only; inherited from train_baseline.py",
        "fine_tuned_features": "last two feature blocks; all other feature layers frozen",
        "batchnorm_policy": "all feature BatchNorm affine and running statistics frozen",
        "optimizer": "AdamW", "head_lr": args.head_lr,
        "backbone_lr": args.backbone_lr, "weight_decay": args.weight_decay,
        "epochs_requested": args.epochs, "batch_size": args.batch_size,
        "patience": args.patience, "seed": args.seed,
        "device": str(device), "num_workers": args.num_workers,
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version,
        "test_set_evaluated": False,
        "limitations": [
            "Known lesion separation only; patient-level independence unverified",
            "Near-duplicate image check not performed",
            "Single seed; GPU kernels may be nondeterministic",
            "Research use only, not clinical diagnosis",
        ],
    }
    (dest / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    best_f1, best_loss = -1.0, math.inf
    patience_count = 0
    for epoch in range(1, args.epochs + 1):
        train_result = epoch_pass(model, train_loader, device, train_criterion, optimizer)
        val_result = epoch_pass(model, val_loader, device, val_criterion)
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
        write_history(dest / "history.csv", history)
        print(
            f"{variant} Epoch {epoch:02}/{args.epochs}: train loss {train_result['loss']:.4f} "
            f"| val loss {val_result['loss']:.4f} "
            f"| val acc {val_result['accuracy']:.3f} "
            f"| macro-F1 {val_result['macro_f1']:.3f} "
            f"| MEL recall {row['val_mel_recall']:.3f} "
            f"| SCC recall {row['val_scc_recall']:.3f}",
            flush=True,
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
                    "class_weighting": variant,
                    "class_weights_train_only": [float(w) for w in weights.detach().cpu()],
                    "normalization_mean": list(MEAN),
                    "normalization_std": list(STD),
                    "epoch": epoch,
                    "validation_metrics": val_result,
                }, dest / "best_model.pt",
            )
            (dest / "best_validation_metrics.json").write_text(
                json.dumps({"best_epoch": epoch, **val_result}, indent=2), encoding="utf-8"
            )
            print("  New best validation macro-F1; checkpoint saved.", flush=True)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping (patience={args.patience}).", flush=True)
                break
    print(f"Finished {variant}: best validation macro-F1 {best_f1:.4f}")
    print(f"Saved: {dest}")


def compare(args):
    sources = {
        "none": directory_for(args, "none"),
        "sqrt": directory_for(args, "sqrt"),
        "inverse_existing_exp2": args.inverse_reference,
    }
    if not all((path / "best_validation_metrics.json").is_file() for path in sources.values()):
        print("Comparison not yet available: complete both new variants first.")
        return False
    metrics = {name: read_json(path / "best_validation_metrics.json") for name, path in sources.items()}
    inverse_f1 = float(metrics["inverse_existing_exp2"]["macro_f1"])
    rows = []
    for name, result in metrics.items():
        row = {
            "variant": name,
            "best_epoch": result["best_epoch"],
            "val_accuracy": result["accuracy"],
            "val_macro_f1": result["macro_f1"],
            "delta_macro_f1_vs_inverse": result["macro_f1"] - inverse_f1,
            "val_balanced_accuracy": result["balanced_accuracy"],
            "val_mel_precision": result["precision_per_class"]["MEL"],
            "val_mel_recall": result["recall_per_class"]["MEL"],
            "val_mel_f1": result["f1_per_class"]["MEL"],
            "val_scc_precision": result["precision_per_class"]["SCC"],
            "val_scc_recall": result["recall_per_class"]["SCC"],
            "val_scc_f1": result["f1_per_class"]["SCC"],
            "source_directory": str(sources[name]),
        }
        rows.append(row)
    dest = comparison_dir(args)
    # Only create comparison artifacts if no prior comparison exists.
    if dest.exists() and any(dest.iterdir()):
        print(f"Comparison already exists; leaving it unchanged: {dest}")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (dest / "comparison.json").write_text(
            json.dumps({"selection_metric": "validation_macro_f1", "rows": rows}, indent=2), encoding="utf-8"
        )
    print("\n=== VALIDATION-ONLY COMPARISON (not independent test results) ===")
    for row in sorted(rows, key=lambda item: item["val_macro_f1"], reverse=True):
        print(f"{row['variant']:<22} macro-F1={row['val_macro_f1']:.4f} "
              f"balanced-acc={row['val_balanced_accuracy']:.4f} "
              f"MEL-recall={row['val_mel_recall']:.3f} "
              f"SCC-recall={row['val_scc_recall']:.3f}")
    print(f"Comparison artifacts: {dest}")
    return True


def main():
    args = parse_args()
    if args.smoke_test and args.compare_only:
        raise ValueError("--smoke-test and --compare-only cannot be combined")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.num_workers < 0:
        raise ValueError("Epochs, batch size and patience must be positive; workers >= 0")
    if not (0 < args.backbone_lr < 1 and 0 < args.head_lr < 1) or args.weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    if args.seed != 42:
        raise ValueError("The completed Experiment 2 reference used seed=42; use 42 for this ablation")
    reference_config, reference_metrics = validate_reference(args)
    print("Experiment 2 reference validated; inverse-frequency macro-F1:", reference_metrics["macro_f1"])

    if args.compare_only:
        if not compare(args):
            raise FileNotFoundError("Both new variants must have saved results before comparison")
        return

    variants = VARIANT_NAMES if args.weighting == "all" else (args.weighting,)
    if not args.smoke_test:
        preflight_outputs(args, variants)
    device = get_device(args.device)
    print(f"Python {sys.version.split()[0]} | Torch {torch.__version__} | Torchvision {torchvision_version}")
    print(f"Device: {device} | Variants: {', '.join(variants)}")
    audit_folders(args.data_dir)  # TEST split: filenames checked only, no test image contents opened.
    train_transform, val_transform = make_transforms()
    train_data = SkinLesionDataset(args.data_dir / "train", train_transform)
    val_data = SkinLesionDataset(args.data_dir / "validation", val_transform)
    for variant in variants:
        run_variant(args, variant, device, train_data, val_data)
    if args.smoke_test:
        print("\nPASS: All requested smoke tests finished. No files were written.")
    else:
        compare(args)
        print("Independent test set was NOT used for training, validation or model selection.")
        print("Research use only; not a clinical diagnosis system.")


if __name__ == "__main__":
    main()
