#!/usr/bin/env python3
"""MobileSkinXAI Experiment 6: ConvNeXt-Tiny teacher-capacity baseline.

Requires scripts/train_baseline.py in the same directory and:
  ~/Desktop/MobileSkinXAI-Datasets/ISIC2019/preprocessed/
      rgb224_letterbox_gray114_v1/full/{train,validation,test}/{NV,MEL,BCC,SCC}/

Two stages, matching the prior MobileNetV3-Small / EfficientNet-Lite0 protocol:
  1. Freeze ImageNet-1K pretrained ConvNeXt backbone; train the four-class head.
  2. Reload best stage-1 checkpoint; fine-tune last N ConvNeXt blocks
     in the final stage plus head (default N=1 for memory safety).

Both stages use train-only inverse-frequency cross-entropy, the established
augmentation protocol, AdamW, macro-F1 validation selection, and patience=4.
Pretrained normalization is read from timm. 224x224 input is used for fairness.
This is a capacity reference: it is NOT intended for Flutter deployment.
If it fails to outperform the compact model, distillation is not justified
without further investigation.

No test image pixels are opened. Only test filenames are audited for split overlap.
Outputs do not overwrite earlier experiments.
Research prototype only; NOT a medical diagnosis tool.
"""
from __future__ import annotations

import argparse
import gc
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    CLASS_NAMES, CLASS_TO_INDEX, DEFAULT_DATA_DIR, SkinLesionDataset,
    audit_folders, confusion_metrics, get_device, make_class_weights,
    seed_everything,
)

MODEL_ID = "convnext_tiny.fb_in1k"
DEFAULT_EXPERIMENTS_DIR = (
    Path.home() / "Desktop" / "MobileSkinXAI-Datasets" / "ISIC2019" / "experiments"
)
DEFAULT_MOBILENET_REFERENCE = DEFAULT_EXPERIMENTS_DIR / "mobilenetv3_small_partial_finetune_seed42"
DEFAULT_EFFICIENTNET_REFERENCE = DEFAULT_EXPERIMENTS_DIR / "efficientnet_lite0_partial_finetune_seed42"
DEFAULT_MOBILENETV4_REFERENCE = DEFAULT_EXPERIMENTS_DIR / "mobilenetv4_conv_small_partial_finetune_seed42"
STAGES = ("frozen", "finetune")


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR)
    parser.add_argument("--stage", choices=("all", "frozen", "finetune"), default="all")
    parser.add_argument("--smoke-test", action="store_true", help="One 4-image train/val batch per stage, no files")
    parser.add_argument("--compare-only", action="store_true", help="Print comparison from saved validation metrics without training")
    parser.add_argument("--frozen-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="ConvNeXt-Tiny is larger; use 4 if MPS runs out of memory")
    parser.add_argument("--finetune-last-blocks", type=int, default=1,
                        help="Number of final-stage ConvNeXt blocks to unfreeze (default 1)")
    parser.add_argument("--head-lr-frozen", type=float, default=0.001)
    parser.add_argument("--head-lr-finetune", type=float, default=0.0001)
    parser.add_argument("--backbone-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mobilenet-reference", type=Path, default=DEFAULT_MOBILENET_REFERENCE)
    parser.add_argument("--efficientnet-reference", type=Path, default=DEFAULT_EFFICIENTNET_REFERENCE)
    parser.add_argument("--mobilenetv4-reference", type=Path, default=DEFAULT_MOBILENETV4_REFERENCE)
    return parser.parse_args()


def stage_path(args, stage):
    suffix = "frozen" if stage == "frozen" else "partial_finetune"
    return args.experiments_dir / f"convnext_tiny_{suffix}_seed{args.seed}"


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


def make_model(pretrained):
    # num_classes=4 replaces the ImageNet classifier while preserving backbone weights.
    model = timm.create_model(MODEL_ID, pretrained=pretrained, num_classes=len(CLASS_NAMES))
    if not hasattr(model, "stages") or not hasattr(model, "head"):
        raise TypeError("Unexpected timm ConvNeXt model structure: missing stages/head")
    head = getattr(model.head, "fc", None)
    if not isinstance(head, nn.Linear) or head.out_features != len(CLASS_NAMES):
        raise TypeError("Expected ConvNeXt head.fc to be a four-class nn.Linear")
    if len(model.stages) < 1 or not hasattr(model.stages[-1], "blocks"):
        raise TypeError("Expected ConvNeXt final stage with blocks")
    if len(model.stages[-1].blocks) < 1:
        raise TypeError("Final ConvNeXt stage is empty")
    return model

def get_model_norm(model):
    cfg = timm.data.resolve_model_data_config(model)
    if tuple(cfg.get("input_size", ())) != (3, 224, 224):
        raise ValueError(f"Expected 224x224 pretrained input, got {cfg.get('input_size')}")
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


def configure_trainable(model, stage, fine_tune_last_blocks=1):
    # Training only the classifier in stage A and final ConvNeXt blocks in B.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.fc.parameters():
        parameter.requires_grad_(True)
    if stage == "finetune":
        last_blocks = model.stages[-1].blocks
        if not 1 <= fine_tune_last_blocks <= len(last_blocks):
            raise ValueError(
                f"--finetune-last-blocks must be between 1 and {len(last_blocks)}"
            )
        for block in last_blocks[-fine_tune_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_batchnorm_running_stats(model):
    # Preserve pretrained BN running means/variances across both stages.
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()


def make_optimizer(model, args, stage):
    head = [p for p in model.head.fc.parameters() if p.requires_grad]
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
    trainable = configure_trainable(model, stage, args.finetune_last_blocks)
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
        "experiment": "MobileSkinXAI Experiment 6: ConvNeXt-Tiny teacher-capacity baseline",
        "finetune_last_blocks": args.finetune_last_blocks,
        "role": "capacity reference and potential teacher (only if validated)",
        "stage": stage,
        "model_id": MODEL_ID,
        "pretrained": "ImageNet-1K pretrained via timm (third-party timm weights)" if stage == "frozen" else "Experiment 6 frozen-stage best checkpoint",
        "parent_checkpoint": parent_checkpoint,
        "parent_checkpoint_sha256": file_sha256(Path(parent_checkpoint)) if parent_checkpoint else None,
        "architecture": MODEL_ID,
        "classes": list(CLASS_NAMES), "class_to_index": CLASS_TO_INDEX,
        "input_size": [3, 224, 224], "mean": list(mean), "std": list(std),
        "input_preprocessing": "RGB 224x224 aspect-preserving letterbox gray114; then model-specific normalize",
        "augmentation": "training only: horizontal/vertical flips p=.5, +/-15 deg rotation, brightness/contrast .08, saturation .05",
        "loss": "inverse-frequency weighted cross-entropy (training only)",
        "loss_weights": [float(w) for w in train_weights.cpu()],
        "optimizer": "AdamW", "head_lr": args.head_lr_frozen if stage == "frozen" else args.head_lr_finetune,
        "backbone_lr": None if stage == "frozen" else args.backbone_lr,
        "weight_decay": args.weight_decay, "patience": args.patience,
        "requested_epochs": epochs, "batch_size": args.batch_size,
        "training_images": len(train_data), "validation_images": len(val_data),
        "train_class_counts": dict(train_data.counts),
        "validation_class_counts": dict(val_data.counts),
        "preprocessing_manifest_sha256": (
            file_sha256(args.data_dir / "manifest.csv")
            if (args.data_dir / "manifest.csv").is_file() else None
        ),
        "trainable_parameters": trainable, "total_parameters": param_count,
        "seed": args.seed, "device": str(device), "num_workers": args.num_workers,
        "timm_model_id": MODEL_ID, "pretrained_weights_source": "timm/convnext_tiny.fb_in1k", "data_license_note": "Check ISIC and ImageNet pretrained weight terms before use",
        "torch_version": str(torch.__version__), "timm_version": str(timm.__version__),
        "data_dir": str(args.data_dir),
        "test_set_used": False,
        "limitations": [
            "Known lesion IDs disjoint; patient-level independence not verified",
            "Near-duplicate image search not performed",
            "Architecture may require separate mobile inference conversion and benchmarking",
            "Models may differ in ImageNet pretraining and effective training budgets",
            "Only final ConvNeXt blocks are unfrozen; this is not end-to-end fine-tuning",
            "Teacher designation conditional on evaluation: may not outperform student",
            "Validation selected architectures; independent test held out",
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
    """Display saved VALIDATION results only; never load a test image."""
    references = (
        ("MobileNetV3-Small partial fine-tune", args.mobilenet_reference),
        ("EfficientNet-Lite0 partial fine-tune", args.efficientnet_reference),
        ("MobileNetV4-Conv-Small partial fine-tune", args.mobilenetv4_reference),
        ("ConvNeXt-Tiny frozen", stage_path(args, "frozen")),
        ("ConvNeXt-Tiny partial fine-tune", stage_path(args, "finetune")),
    )
    results = []
    for label, folder in references:
        metric_path = folder / "best_validation_metrics.json"
        if metric_path.is_file():
            metrics = read_json(metric_path)
            results.append((label, folder, metrics))
    if not results:
        raise FileNotFoundError("No validation results available for comparison")
    print("\n=== VALIDATION-ONLY ARCHITECTURE COMPARISON ===")
    print("The independent test set has not been evaluated.")
    print("NOTE: Existing models may differ in effective training budgets and pretraining configurations.")
    for label, folder, m in results:
        print(
            f"{label:41} macro-F1={m['macro_f1']:.4f} "
            f"balanced-acc={m['balanced_accuracy']:.4f} "
            f"MEL-recall={m['recall_per_class']['MEL']:.3f} "
            f"SCC-recall={m['recall_per_class']['SCC']:.3f} | {folder}"
        )
    print("\nHighest single-run validation macro-F1:",
          max(results, key=lambda x: x[2]["macro_f1"])[0])


def main():
    args = args_parser()
    if min(args.frozen_epochs, args.finetune_epochs, args.batch_size, args.patience, args.finetune_last_blocks) < 1:
        raise ValueError("Epoch, batch size and patience values must be positive")
    if args.num_workers < 0 or args.weight_decay < 0:
        raise ValueError("Workers and weight decay must be nonnegative")
    if any(not (0 < value < 1) for value in (
        args.head_lr_frozen, args.head_lr_finetune, args.backbone_lr
    )):
        raise ValueError("All learning rates must be positive and below 1")
    if args.smoke_test and args.compare_only:
        raise ValueError("Use --smoke-test or --compare-only, not both")
    if args.compare_only:
        print_comparison(args)
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
    audit_folders(args.data_dir)  # test filenames only, never loads test image pixels
    # For stage-only fine-tune, inspect the stored pretrained-data normalization.
    if stages == ("finetune",) and not args.smoke_test:
        model, previous = load_frozen_checkpoint(stage_path(args, "frozen") / "best_model.pt")
        prior_config = read_json(stage_path(args, "frozen") / "run_config.json")
        if prior_config["class_to_index"] != CLASS_TO_INDEX or prior_config["seed"] != args.seed:
            raise ValueError("Frozen-stage configuration is incompatible")
        if prior_config["data_dir"] != str(args.data_dir):
            raise ValueError("Frozen stage used different data directory")
        mean, std = tuple(prior_config["mean"]), tuple(prior_config["std"])
    else:
        print("Loading ConvNeXt-Tiny ImageNet-1K pretrained weights (first run downloads weights)...", flush=True)
        model = make_model(pretrained=True)
        mean, std = get_model_norm(model)
    train_transform, val_transform = make_transforms(mean, std)
    train_data = SkinLesionDataset(args.data_dir / "train", train_transform)
    val_data = SkinLesionDataset(args.data_dir / "validation", val_transform)
    for stage in stages:
        if stage == "frozen":
            run_stage(args, stage, device, train_data, val_data, mean, std, model=model)
            if not args.smoke_test:
                # Release the ImageNet model before loading the best checkpoint for
                # the heavier fine-tuning stage. Important on memory-limited MPS.
                del model
                model = None
                gc.collect()
                if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
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
