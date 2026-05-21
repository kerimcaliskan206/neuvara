#!/usr/bin/env python3
"""
Phase 21 — Stage C: Bilateral/ARDS Pattern Specialization.

Fine-tunes the Stage B calibrated EfficientNet-B0 checkpoint for stronger:
  - Bilateral diffuse opacity detection
  - Lower-lung ARDS-compatible pattern sensitivity
  - Healthy-vs-severe separation (reduced healthy false positives)
  - Spatial activation concentration in lung fields

Strategy
--------
  Base:      Stage B calibrated checkpoint (stage_b_calibrated.pt)
  Unfreeze:  Top 2 EfficientNet feature blocks (features[-2:])
             Stage B unfroze features[-1] only; Stage C goes one block deeper.
  Loss:      FocalLoss(gamma=2.5) — penalizes easy healthy examples,
             concentrates gradient on hard boundary cases.
  Weights:   pneumonia_xray ×2.5, hard_negative ×1.5, fake_medical ×1.5,
             healthy_xray ×1.0
  Augment:   pulmonary_bilateral preset (strong contrast, no vertical flip,
             no random erasing, high grayscale probability)
  LR:        backbone=5e-6 (conservative), head=2.5e-5 (5× differential)
  Epochs:    15

Safety
------
  - V5 production model is NEVER loaded or modified.
  - All outputs are isolated under models/vision/v6_medical/stage_c_bilateral/
  - Checkpoint format is v6_meta compatible (load_v6_calibrated() works directly).

Usage
-----
    python scripts/run_stage_c.py \\
        --calibrated-checkpoint models/vision/v6_medical/calibration/stage_b_calibrated.pt \\
        --dataset-dir data/medical_v6_splits \\
        --device auto

    # Skip training — evaluate an existing Stage C checkpoint:
    python scripts/run_stage_c.py \\
        --checkpoint models/vision/v6_medical/stage_c_bilateral/best.pt \\
        --eval-only \\
        --dataset-dir data/medical_v6_splits

Exit codes: 0 success, 1 error.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("run_stage_c")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

# ── Constants ─────────────────────────────────────────────────────────────────

_V6_CLASSES = ["healthy_xray", "pneumonia_xray", "hard_negative", "fake_medical"]
_ARCHITECTURE = "efficientnet_b0"

# Class weights: emphasize pneumonia (bilateral patterns) over healthy
_CLASS_WEIGHTS = {
    "healthy_xray":   1.0,
    "pneumonia_xray": 2.5,
    "hard_negative":  1.5,
    "fake_medical":   1.5,
}

_STAGE_C_EPOCHS       = 15
_BACKBONE_LR          = 5e-6
_HEAD_LR              = 2.5e-5
_WEIGHT_DECAY         = 1e-4
_GRAD_CLIP            = 1.0
_FOCAL_GAMMA          = 2.5
_LABEL_SMOOTHING      = 0.05
_BATCH_SIZE           = 24
_NUM_WORKERS          = 4

_OUTPUT_DIR = _PROJECT_ROOT / "models" / "vision" / "v6_medical" / "stage_c_bilateral"
_REPORT_DIR = _PROJECT_ROOT / "reports" / "v6_medical" / "stage_c"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _load_base_checkpoint(path: Path, device: torch.device):
    """Load Stage B calibrated checkpoint, return (model, v6_meta, temperature)."""
    from app.modules.vision.models.registry import VisionModelRegistry

    logger.info("Loading Stage B calibrated checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in ckpt or "v6_meta" not in ckpt:
        raise ValueError(
            f"{path} is not a v6 calibrated checkpoint "
            "(expected keys: model_state_dict, v6_meta)"
        )

    meta     = ckpt["v6_meta"]
    classes  = meta.get("classes", _V6_CLASSES)
    temperature = float(meta.get("calibration_temperature", 1.0))

    model = VisionModelRegistry.build(
        architecture=_ARCHITECTURE,
        num_classes=len(classes),
        pretrained=False,
        freeze=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    logger.info(
        "Loaded: classes=%s, T*=%.4f", classes, temperature
    )
    return model, meta, temperature


def _unfreeze_top2_blocks(model: nn.Module) -> None:
    """
    Freeze entire backbone, then selectively unfreeze the top 2 feature blocks
    (features[-2:]) plus the classifier head.

    Stage B unfroze features[-1] only. Stage C goes one block deeper.
    """
    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze top-2 feature blocks
    backbone = model._backbone
    features = list(backbone.features.children())
    for block in features[-2:]:
        for p in block.parameters():
            p.requires_grad_(True)

    # Unfreeze classifier head
    for p in backbone.classifier.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "Unfreeze: top-2 blocks + head | trainable=%s / total=%s (%.1f%%)",
        f"{n_trainable:,}", f"{n_total:,}", 100.0 * n_trainable / n_total,
    )


def _build_dataloaders(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders with pulmonary_bilateral augmentation."""
    from app.modules.vision.datasets.dataset import ImageFolderDataset
    from app.modules.vision.preprocessing.transforms import (
        get_val_transforms,
        get_train_transforms,
    )
    from app.modules.vision.config import VisionConfig, AugmentationConfig

    # Build pulmonary_bilateral augmentation config
    aug_params = {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation_degrees": 8,
        "color_jitter": True,
        "color_jitter_brightness": 0.50,
        "color_jitter_contrast": 0.50,
        "color_jitter_saturation": 0.10,
        "color_jitter_hue": 0.02,
        "random_erasing": False,
        "random_resized_crop": True,
        "random_resized_crop_scale_min": 0.78,
        "grayscale_prob": 0.35,
    }
    aug_cfg = AugmentationConfig(**aug_params)
    train_cfg = VisionConfig(augmentation=aug_cfg)
    val_cfg   = VisionConfig()

    train_tf = get_train_transforms(train_cfg)
    val_tf   = get_val_transforms(val_cfg)

    if not (dataset_dir / "train").exists():
        raise FileNotFoundError(f"Train split not found: {dataset_dir / 'train'}")
    if not (dataset_dir / "val").exists():
        raise FileNotFoundError(f"Val split not found: {dataset_dir / 'val'}")

    train_ds = ImageFolderDataset(root_dir=dataset_dir, split="train", transform=train_tf, classes=_V6_CLASSES)
    val_ds   = ImageFolderDataset(root_dir=dataset_dir, split="val",   transform=val_tf,   classes=_V6_CLASSES)

    # Weighted sampler — oversample pneumonia
    class_weight_tensor = torch.tensor(
        [_CLASS_WEIGHTS.get(c, 1.0) for c in train_ds.classes], dtype=torch.float
    )
    sample_labels = torch.tensor([label for _, label in train_ds.samples])
    sample_weights = class_weight_tensor[sample_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "Dataloaders ready | train=%d  val=%d  classes=%s",
        len(train_ds), len(val_ds), train_ds.classes,
    )
    return train_loader, val_loader


def _build_optimizer_and_scheduler(
    model: nn.Module,
    epochs: int,
) -> tuple:
    from app.modules.vision.training.focal_loss import FocalLoss

    class_weight_tensor = torch.tensor(
        [_CLASS_WEIGHTS.get(c, 1.0) for c in _V6_CLASSES], dtype=torch.float
    )

    criterion = FocalLoss(
        gamma=_FOCAL_GAMMA,
        weight=class_weight_tensor,
        label_smoothing=_LABEL_SMOOTHING,
        reduction="mean",
    )

    # Differential learning rates: backbone conservative, head faster
    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "classifier" not in n
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "classifier" in n
    ]

    optimizer = AdamW([
        {"params": backbone_params, "lr": _BACKBONE_LR},
        {"params": head_params,     "lr": _HEAD_LR},
    ], weight_decay=_WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    return criterion, optimizer, scheduler


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    is_train: bool,
) -> dict:
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss":     total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1,
    }


def _save_checkpoint(
    model: nn.Module,
    out_path: Path,
    meta: dict,
    epoch: int,
    val_f1: float,
    temperature: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v6_meta = {
        **meta,
        "v6_phase":              "21",
        "sub_stage":             "stage_c_bilateral",
        "epoch":                 epoch,
        "best_val_f1":           val_f1,
        "calibration_temperature": temperature,
        "bilateral_specialization": True,
        "focal_gamma":           _FOCAL_GAMMA,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Stage C: bilateral/ARDS specialization. "
            f"Top-2 blocks unfrozen, FocalLoss gamma={_FOCAL_GAMMA}, "
            f"pneumonia weight=2.5, pulmonary_bilateral augmentation."
        ),
    }

    torch.save(
        {"model_state_dict": model.state_dict(), "v6_meta": v6_meta},
        out_path,
    )
    logger.info("Checkpoint saved: %s (val_f1=%.4f)", out_path, val_f1)


# ── Training loop ─────────────────────────────────────────────────────────────


def run_training(
    calibrated_checkpoint: Path,
    dataset_dir: Path,
    device: torch.device,
    epochs: int = _STAGE_C_EPOCHS,
) -> dict:
    model, meta, temperature = _load_base_checkpoint(calibrated_checkpoint, device)
    _unfreeze_top2_blocks(model)

    train_loader, val_loader = _build_dataloaders(
        dataset_dir, batch_size=_BATCH_SIZE, num_workers=_NUM_WORKERS
    )

    criterion, optimizer, scheduler = _build_optimizer_and_scheduler(model, epochs)
    criterion.weight = criterion.weight.to(device) if criterion.weight is not None else None

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_f1  = 0.0
    best_epoch   = 0
    history      = []

    logger.info("═" * 60)
    logger.info("Stage C training: %d epochs, FocalLoss γ=%.1f", epochs, _FOCAL_GAMMA)
    logger.info("backbone_lr=%.2e, head_lr=%.2e", _BACKBONE_LR, _HEAD_LR)
    logger.info("═" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        train_m = _run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_m   = _run_epoch(model, val_loader,   criterion, optimizer, device, is_train=False)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %2d/%d | train_f1=%.4f  val_f1=%.4f  val_loss=%.4f  [%.1fs]",
            epoch, epochs, train_m["macro_f1"], val_m["macro_f1"], val_m["loss"], elapsed,
        )

        row = {"epoch": epoch, "train": train_m, "val": val_m}
        history.append(row)

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_epoch  = epoch
            _save_checkpoint(
                model,
                _OUTPUT_DIR / "best.pt",
                meta=meta,
                epoch=epoch,
                val_f1=best_val_f1,
                temperature=temperature,
            )

    # Also save the final epoch checkpoint
    _save_checkpoint(
        model,
        _OUTPUT_DIR / "last.pt",
        meta=meta,
        epoch=epochs,
        val_f1=val_m["macro_f1"],
        temperature=temperature,
    )

    report = {
        "stage": "stage_c_bilateral",
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "total_epochs": epochs,
        "history": history,
        "checkpoint": str(_OUTPUT_DIR / "best.pt"),
    }

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _REPORT_DIR / "stage_c_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Report saved: %s", report_path)

    logger.info("═" * 60)
    logger.info("Stage C complete | best_val_f1=%.4f @ epoch %d", best_val_f1, best_epoch)
    logger.info("Checkpoint: %s", _OUTPUT_DIR / "best.pt")
    logger.info("═" * 60)

    return report


# ── Eval-only path ────────────────────────────────────────────────────────────


def run_eval_only(checkpoint: Path, dataset_dir: Path, device: torch.device) -> None:
    model, meta, temperature = _load_base_checkpoint(checkpoint, device)
    model.eval()

    _, val_loader = _build_dataloaders(
        dataset_dir, batch_size=_BATCH_SIZE, num_workers=_NUM_WORKERS
    )

    from sklearn.metrics import classification_report

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            cal    = logits / temperature
            preds  = F.softmax(cal, dim=1).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\n" + "═" * 60)
    print("Stage C Evaluation Report")
    print("═" * 60)
    print(classification_report(all_labels, all_preds, target_names=_V6_CLASSES))


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage C: Bilateral/ARDS specialization")
    parser.add_argument(
        "--calibrated-checkpoint",
        type=Path,
        default=_PROJECT_ROOT / "models/vision/v6_medical/calibration/stage_b_calibrated.pt",
        help="Path to Stage B calibrated checkpoint (v6 format).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Existing Stage C checkpoint for eval-only mode.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_PROJECT_ROOT / "data/medical_v6_splits",
        help="Dataset directory with train/ and val/ subdirectories.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_STAGE_C_EPOCHS,
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training — evaluate an existing checkpoint.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    logger.info("Device: %s", device)

    try:
        if args.eval_only:
            ckpt = args.checkpoint or (_OUTPUT_DIR / "best.pt")
            if not ckpt.exists():
                logger.error("Checkpoint not found: %s", ckpt)
                return 1
            run_eval_only(ckpt, args.dataset_dir, device)
        else:
            base = args.calibrated_checkpoint
            if not base.exists():
                logger.error("Calibrated checkpoint not found: %s", base)
                return 1
            run_training(base, args.dataset_dir, device, epochs=args.epochs)
    except Exception:
        logger.exception("Stage C failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
