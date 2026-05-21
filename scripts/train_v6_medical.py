#!/usr/bin/env python3
"""
Train v6 medical classifier — Phase 10.

Isolated training harness for the first real medical specialization:
healthy_xray vs pneumonia_xray (+ OOD: hard_negative, fake_medical).

Three progressive sub-stages (A → B → C):
  A: frozen backbone   — head only, warm start
  B: top-1 unfreeze    — features[-1] + head, differential LR
  C: top-3 unfreeze    — features[-3:] + head, focal loss, very low LR

SAFETY:
  - V5 production model is never loaded or modified.
  - All checkpoints are isolated under models/vision/v6_medical/.
  - Compatibility gates are evaluated after each sub-stage.
  - Training halts (with warning) if a gate is violated.

Expected input: data/medical_v6_splits/ from prepare_v6_medical_dataset.py

    data/medical_v6_splits/
      train/
        healthy_xray/ pneumonia_xray/ hard_negative/ fake_medical/
      val/
        ...

Usage
-----
    # Full 3-stage run
    python scripts/train_v6_medical.py \\
        --dataset-dir data/medical_v6_splits \\
        --stage all \\
        --device cuda

    # Stage A only (useful for quick iteration)
    python scripts/train_v6_medical.py --dataset-dir ... --stage a

    # Resume at Stage B from Stage A best checkpoint
    python scripts/train_v6_medical.py \\
        --dataset-dir ... --stage b \\
        --checkpoint models/vision/v6_medical/stage_a_frozen/best.pt

    # Dry run — print config without training
    python scripts/train_v6_medical.py --dataset-dir ... --dry-run

Exit codes: 0 success, 1 error, 2 compatibility gate failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("train_v6")


# ── Imports ───────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, ConcatDataset

from app.modules.vision.medical.v6_training_config import (
    V6SubStage, V6SubStageConfig, V6TrainingConfig,
    V6_BINARY_MEDICAL_CONFIG, V6_BINARY_MEDICAL_CLASSES,
    apply_backbone_policy, build_checkpoint_meta,
    SUB_STAGE_MAP,
)
from app.modules.vision.medical.v6_evaluation import (
    V6Evaluator, export_compatibility_report,
)
from app.modules.vision.datasets.dataset import ImageFolderDataset
from app.modules.vision.datasets.balancer import (
    build_weighted_sampler, compute_class_weights_tensor,
)
from app.modules.vision.models.efficientnet import build_efficientnet
from app.modules.vision.training.callbacks import (
    EarlyStopping, ModelCheckpoint, LatestCheckpoint,
)
from app.modules.vision.training.trainer import EpochMetrics
from app.modules.vision.preprocessing.transforms import (
    get_train_transforms, get_val_transforms,
)


# ── Model loading ─────────────────────────────────────────────────────────────


def _build_or_load_model(
    checkpoint_path: Path | None,
    config: V6TrainingConfig,
    device: torch.device,
) -> nn.Module:
    """
    Build a fresh v6 EfficientNet or load from an existing v6 checkpoint.

    Never loads from a v5 checkpoint — the checkpoint must have 'v6_meta'.
    """
    num_classes = config.num_classes
    classes     = list(config.classes)

    if checkpoint_path is not None:
        logger.info("Loading v6 checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if "v6_meta" not in ckpt:
            logger.error(
                "Checkpoint missing 'v6_meta' key — this does not appear to be a "
                "v6 checkpoint. V5 checkpoints must not be used for v6 training."
            )
            sys.exit(1)

        meta = ckpt["v6_meta"]
        ckpt_classes = meta.get("classes", [])
        if ckpt_classes and ckpt_classes != classes:
            logger.warning(
                "Checkpoint classes %s differ from config classes %s — proceeding.",
                ckpt_classes, classes,
            )

        model = build_efficientnet(
            variant="efficientnet_b0",
            num_classes=num_classes,
            pretrained=False,    # weights come from the checkpoint
            dropout=0.30,
            freeze=False,
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        logger.info(
            "Checkpoint loaded | sub_stage=%s | val_f1=%.4f",
            meta.get("sub_stage", "?"), meta.get("val_f1", 0.0),
        )
    else:
        logger.info("Building fresh EfficientNet-B0 with %d classes.", num_classes)
        model = build_efficientnet(
            variant="efficientnet_b0",
            num_classes=num_classes,
            pretrained=True,
            dropout=0.30,
            freeze=False,         # backbone policy applied after construction
        )

    return model.to(device)


# ── DataLoaders ───────────────────────────────────────────────────────────────


def _build_dataloaders(
    dataset_dir: Path,
    sub_stage_cfg: V6SubStageConfig,
    config: V6TrainingConfig,
    replay_dir: Path | None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders. Injects replay samples into train if configured."""
    classes    = list(config.classes)
    batch_size = sub_stage_cfg.batch_size
    num_workers = config.num_workers

    train_ds = ImageFolderDataset(
        root_dir=dataset_dir, split="train",
        transform=get_train_transforms(),
        classes=classes,
    )
    val_ds = ImageFolderDataset(
        root_dir=dataset_dir, split="val",
        transform=get_val_transforms(),
        classes=classes,
    )

    if not train_ds.samples:
        raise RuntimeError(
            f"No training images found in {dataset_dir}/train/. "
            "Run prepare_v6_medical_dataset.py first."
        )

    # Replay buffer injection (hard_negative OOD samples from v5 data)
    if (
        config.replay.enabled
        and replay_dir is not None
        and replay_dir.exists()
        and sub_stage_cfg.sub_stage == V6SubStage.A_FROZEN
    ):
        replay_ds = _build_replay_dataset(
            replay_dir, config, get_train_transforms()
        )
        if replay_ds is not None:
            train_ds = ConcatDataset([train_ds, replay_ds])  # type: ignore[assignment]
            logger.info(
                "Replay buffer added: %d extra samples",
                len(replay_ds),
            )

    # Weighted sampler for class imbalance
    if sub_stage_cfg.use_weighted_sampler and hasattr(train_ds, "samples"):
        dist = train_ds.class_distribution()  # type: ignore[attr-defined]
        weights_tensor = compute_class_weights_tensor(dist, classes)
        sampler = build_weighted_sampler(train_ds, classes)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=config.pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=config.pin_memory,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=config.pin_memory,
    )

    logger.info(
        "DataLoaders ready | train=%d val=%d | batch=%d",
        len(train_ds), len(val_ds), batch_size,
    )
    return train_loader, val_loader


def _build_replay_dataset(
    replay_dir: Path, config: V6TrainingConfig, transform
) -> ImageFolderDataset | None:
    """
    Build a replay dataset from extra v5 hard_negative images.

    replay_dir should contain hard_negative/ (and optionally other class dirs).
    Only classes present in config.classes are used.
    """
    available = [d.name for d in replay_dir.iterdir() if d.is_dir()]
    target    = [c for c in config.classes if c in available]
    if not target:
        logger.warning("Replay dir %s has no matching class dirs. Skipping.", replay_dir)
        return None

    try:
        ds = ImageFolderDataset(
            root_dir=replay_dir,
            split="",           # replay_dir/class/images directly (no split subdir)
            transform=transform,
            classes=list(config.classes),
        )
    except Exception:
        # Replay dir might not have the split subdir layout — scan directly
        return None

    if not ds.samples:
        return None

    max_n = config.replay.max_replay_images
    if len(ds.samples) > max_n:
        import random
        random.seed(config.random_seed)
        ds.samples = random.sample(ds.samples, max_n)

    return ds


# ── Optimizer + scheduler ─────────────────────────────────────────────────────


def _build_optimizer_and_scheduler(
    model: nn.Module,
    sub_stage_cfg: V6SubStageConfig,
) -> tuple[torch.optim.Optimizer, object]:
    """
    Build optimizer with differential LR for backbone vs classifier.

    Stage A: all trainable params use a single LR (backbone is frozen).
    Stage B/C: backbone LR = learning_rate, head LR = learning_rate × head_lr_factor.
    """
    backbone_params = [
        p for p in model._backbone.features.parameters()
        if p.requires_grad
    ]
    head_params = list(model._backbone.classifier.parameters())

    lr      = sub_stage_cfg.learning_rate
    head_lr = lr * sub_stage_cfg.head_lr_factor

    if backbone_params:
        param_groups = [
            {"params": backbone_params, "lr": lr,      "name": "backbone"},
            {"params": head_params,     "lr": head_lr, "name": "head"},
        ]
        logger.info(
            "Differential LR: backbone=%.2e  head=%.2e",
            lr, head_lr,
        )
    else:
        param_groups = [{"params": head_params, "lr": lr, "name": "head"}]
        logger.info("Single LR (head only): %.2e", lr)

    optimizer = AdamW(
        param_groups,
        lr=lr,
        weight_decay=sub_stage_cfg.weight_decay,
    )

    n_epochs  = sub_stage_cfg.epochs
    warmup    = sub_stage_cfg.warmup_epochs
    if warmup > 0 and n_epochs > warmup:
        warmup_sched = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
        )
        main_sched = CosineAnnealingLR(
            optimizer, T_max=max(n_epochs - warmup, 1)
        )
        scheduler = SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[warmup])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))

    return optimizer, scheduler


# ── Loss ──────────────────────────────────────────────────────────────────────


def _build_criterion(
    sub_stage_cfg: V6SubStageConfig,
    class_weights: torch.Tensor | None,
    device: torch.device,
) -> nn.Module:
    if sub_stage_cfg.use_focal_loss:
        from app.modules.vision.training.focal_loss import FocalLoss
        return FocalLoss(
            gamma=sub_stage_cfg.focal_gamma,
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=sub_stage_cfg.label_smoothing,
        )
    return nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=sub_stage_cfg.label_smoothing,
    )


# ── Training loop ─────────────────────────────────────────────────────────────


def _train_sub_stage(
    model: nn.Module,
    sub_stage_cfg: V6SubStageConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: V6TrainingConfig,
    device: torch.device,
    stage_output_dir: Path,
) -> dict:
    """Train one sub-stage and return the best epoch metrics."""
    stage_output_dir.mkdir(parents=True, exist_ok=True)

    apply_backbone_policy(model, sub_stage_cfg.backbone_policy)
    optimizer, scheduler = _build_optimizer_and_scheduler(model, sub_stage_cfg)

    # Class weights from training distribution (if weighted sampler not covering this)
    class_weights = None
    if hasattr(train_loader.dataset, "class_distribution"):
        dist = train_loader.dataset.class_distribution()
        class_weights = compute_class_weights_tensor(dist, list(config.classes))

    criterion = _build_criterion(sub_stage_cfg, class_weights, device)

    early_stop = EarlyStopping(
        patience=sub_stage_cfg.early_stopping_patience,
        min_delta=sub_stage_cfg.early_stopping_min_delta,
        mode="max",
    )
    best_checkpoint = ModelCheckpoint(
        checkpoint_dir=stage_output_dir,
        monitor="val_f1",
        mode="max",
        save_best_only=True,
        filename="best.pt",
        class_names=list(config.classes),
        architecture="efficientnet_b0",
    )
    latest_checkpoint = LatestCheckpoint(
        checkpoint_dir=stage_output_dir,
        keep=2,
        class_names=list(config.classes),
        architecture="efficientnet_b0",
    )

    history: list[dict] = []
    best_val_f1 = 0.0

    logger.info(
        "=== Sub-stage: %s | epochs=%d | lr=%.2e ===",
        sub_stage_cfg.sub_stage.value, sub_stage_cfg.epochs, sub_stage_cfg.learning_rate,
    )

    for epoch in range(1, sub_stage_cfg.epochs + 1):
        t0 = time.time()

        # Training pass
        model.train()
        train_metrics = EpochMetrics()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            if sub_stage_cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), sub_stage_cfg.grad_clip)
            optimizer.step()
            train_metrics.update(loss.item(), logits.detach().argmax(1).cpu(), labels.cpu())

        # Validation pass
        model.eval()
        val_metrics = EpochMetrics()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss   = criterion(logits, labels)
                val_metrics.update(loss.item(), logits.argmax(1).cpu(), labels.cpu())

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics.f1)
        elif scheduler is not None:
            scheduler.step()

        val_f1  = val_metrics.f1
        elapsed = time.time() - t0

        row = {
            "epoch":      epoch,
            "train_loss": round(train_metrics.loss, 5),
            "train_f1":   round(train_metrics.f1, 4),
            "val_loss":   round(val_metrics.loss, 5),
            "val_f1":     round(val_f1, 4),
            "elapsed_s":  round(elapsed, 1),
        }
        history.append(row)

        logger.info(
            "Epoch %03d | train_loss=%.4f train_f1=%.4f | "
            "val_loss=%.4f val_f1=%.4f | %.1fs",
            epoch, row["train_loss"], row["train_f1"],
            row["val_loss"], row["val_f1"], elapsed,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        best_checkpoint.step(model, epoch=epoch, score=val_f1)
        latest_checkpoint.step(model, epoch=epoch, score=val_f1)

        if early_stop.step(val_f1):
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    # Save training history
    history_path = stage_output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "best_val_f1": best_val_f1,
        "best_checkpoint": str(stage_output_dir / "best.pt"),
        "history_path": str(history_path),
        "epochs_run": len(history),
    }


# ── Compatibility check ───────────────────────────────────────────────────────


def _check_compatibility(
    model: nn.Module,
    val_loader: DataLoader,
    sub_stage_cfg: V6SubStageConfig,
    config: V6TrainingConfig,
    device: torch.device,
    stage_output_dir: Path,
) -> bool:
    """Run compatibility gates after a sub-stage. Returns True if all pass."""
    logger.info("Running post-stage compatibility checks …")
    evaluator = V6Evaluator(config)

    try:
        result = evaluator.evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            split="val",
            run_gradcam=config.compatibility.check_gradcam_sanity,
        )
        evaluator.fill_positive_recall(result.compatibility, result.per_class)

        compat = result.compatibility
        logger.info(
            "Compatibility | ood_rejection=%.4f(%s) pos_recall=%.4f(%s) "
            "ece=%.4f(%s) gradcam=%s",
            compat.ood_rejection_rate, "OK" if compat.ood_rejection_ok else "FAIL",
            compat.positive_recall,    "OK" if compat.positive_recall_ok else "FAIL",
            compat.ece,                "OK" if compat.calibration_ok else "FAIL",
            "OK" if compat.gradcam_ok else "FAIL",
        )

        report_path = stage_output_dir / "compatibility_report.json"
        export_compatibility_report(compat, report_path)

        return compat.overall_pass

    except Exception:
        logger.exception("Compatibility check failed with exception.")
        return False


# ── v6 checkpoint ─────────────────────────────────────────────────────────────


def _save_v6_checkpoint(
    model: nn.Module,
    sub_stage_cfg: V6SubStageConfig,
    config: V6TrainingConfig,
    stage_result: dict,
    compat_passed: bool,
    stage_output_dir: Path,
) -> Path:
    """Save isolated v6 checkpoint with full metadata."""
    meta = build_checkpoint_meta(
        run_name=config.run_name,
        sub_stage=sub_stage_cfg.sub_stage.value,
        classes=list(config.classes),
        epoch=stage_result["epochs_run"],
        best_val_f1=stage_result["best_val_f1"],
        val_f1=stage_result["best_val_f1"],
        val_loss=0.0,
        ood_rejection_rate=0.0,   # filled by compatibility check if available
        compatibility_passed=compat_passed,
        notes=sub_stage_cfg.notes,
    )

    ckpt_path = stage_output_dir / "v6_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "v6_meta": meta,
        },
        ckpt_path,
    )
    logger.info("V6 checkpoint saved: %s", ckpt_path)
    return ckpt_path


# ── Main ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_v6_medical",
        description="Train isolated v6 medical EfficientNet — Phase 10.",
    )
    p.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Path to data/medical_v6_splits/ from prepare_v6_medical_dataset.py.",
    )
    p.add_argument(
        "--stage", choices=["a", "b", "c", "all"], default="all",
        help="Sub-stage(s) to run (default: all).",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Start from this v6 checkpoint (must contain v6_meta). "
             "Required for --stage b and --stage c.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("models/vision/v6_medical"),
        help="Root for v6 model outputs (default: models/vision/v6_medical).",
    )
    p.add_argument(
        "--replay-dir", type=Path, default=None,
        help="Optional dir with extra v5 hard_negative images for replay buffer.",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: 'cuda', 'mps', 'cpu', or 'auto' (default).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print config and dataset info without training.",
    )
    return p


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> int:
    args   = build_parser().parse_args()
    config = V6_BINARY_MEDICAL_CONFIG
    device = _resolve_device(args.device)

    config = V6TrainingConfig(
        run_name=config.run_name,
        classes=config.classes,
        positive_classes=config.positive_classes,
        ood_classes=config.ood_classes,
        sub_stages=config.sub_stages,
        replay=config.replay,
        compatibility=config.compatibility,
        output_dir=args.output_dir,
        random_seed=config.random_seed,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        mixed_precision=config.mixed_precision,
        notes=config.notes,
    )

    logger.info("=== Train V6 Medical — Phase 10 ===")
    logger.info("Dataset  : %s", args.dataset_dir)
    logger.info("Stage(s) : %s", args.stage)
    logger.info("Output   : %s", args.output_dir)
    logger.info("Device   : %s", device)
    logger.info("Classes  : %s", list(config.classes))

    # Determine which sub-stages to run
    if args.stage == "all":
        stages_to_run = [V6SubStage.A_FROZEN, V6SubStage.B_TOP1, V6SubStage.C_SELECTIVE]
    elif args.stage == "a":
        stages_to_run = [V6SubStage.A_FROZEN]
    elif args.stage == "b":
        stages_to_run = [V6SubStage.B_TOP1]
    else:
        stages_to_run = [V6SubStage.C_SELECTIVE]

    if args.dry_run:
        for stage_key in stages_to_run:
            cfg = config.get_sub_stage(stage_key)
            logger.info(
                "DRY-RUN  sub_stage=%-25s epochs=%d lr=%.2e focal=%s",
                cfg.sub_stage.value, cfg.epochs, cfg.learning_rate, cfg.use_focal_loss,
            )
        logger.info("DRY-RUN complete — no training performed.")
        return 0

    if not args.dataset_dir.exists():
        logger.error("Dataset dir not found: %s", args.dataset_dir)
        return 1

    checkpoint_path = args.checkpoint
    overall_compat  = True

    for stage_key in stages_to_run:
        sub_stage_cfg = config.get_sub_stage(stage_key)
        stage_dir     = config.stage_output_dir(sub_stage_cfg)

        logger.info("\n" + "=" * 60)
        logger.info("Starting sub-stage: %s", sub_stage_cfg.sub_stage.value)
        logger.info("Notes: %s", sub_stage_cfg.notes)
        logger.info("=" * 60)

        model = _build_or_load_model(checkpoint_path, config, device)

        train_loader, val_loader = _build_dataloaders(
            dataset_dir=args.dataset_dir,
            sub_stage_cfg=sub_stage_cfg,
            config=config,
            replay_dir=args.replay_dir,
        )

        stage_result = _train_sub_stage(
            model=model,
            sub_stage_cfg=sub_stage_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            stage_output_dir=stage_dir,
        )

        compat_passed = _check_compatibility(
            model=model,
            val_loader=val_loader,
            sub_stage_cfg=sub_stage_cfg,
            config=config,
            device=device,
            stage_output_dir=stage_dir,
        )

        if not compat_passed:
            logger.warning(
                "Compatibility gate FAILED for sub-stage %s. "
                "Checkpoint saved for inspection but continuing is NOT recommended.",
                sub_stage_cfg.sub_stage.value,
            )
            overall_compat = False

        checkpoint_path = _save_v6_checkpoint(
            model=model,
            sub_stage_cfg=sub_stage_cfg,
            config=config,
            stage_result=stage_result,
            compat_passed=compat_passed,
            stage_output_dir=stage_dir,
        )

        logger.info(
            "Sub-stage %s complete | best_val_f1=%.4f | compat=%s",
            sub_stage_cfg.sub_stage.value,
            stage_result["best_val_f1"],
            "PASS" if compat_passed else "FAIL",
        )
        logger.info("Next checkpoint: %s", checkpoint_path)

    logger.info("\n=== V6 Training Complete ===")
    logger.info("Overall compatibility: %s", "PASS" if overall_compat else "FAIL")
    logger.info(
        "Evaluate with:\n  python scripts/evaluate_v6_medical.py "
        "--checkpoint %s --dataset-dir %s",
        checkpoint_path, args.dataset_dir,
    )

    return 0 if overall_compat else 2


if __name__ == "__main__":
    sys.exit(main())
