"""
HantaProject — Professional Vision Training Script
====================================================

Trains a CNN image classifier with two-phase transfer learning.

Features in this version
------------------------
  - WeightedRandomSampler for class-imbalance handling
  - Class-weighted CrossEntropyLoss or Focal Loss
  - Macro-F1 tracking (correct for 3-class and binary)
  - Differential LR in Phase B (backbone ≪ head)
  - Rollback-safe LatestCheckpoint (last 2 epochs)
  - Best-metric ModelCheckpoint (with embedded class_names + architecture)
  - Post-training temperature calibration (ECE before/after)
  - Calibration temperature saved to VisionModelStore metadata
  - Test-set evaluation with per-class breakdown
  - Training statistics summary

Expected dataset layout (produced by prepare_dataset.py):

    data/vision/datasets/<version>/splits/
    ├── train/<class_a>/, train/<class_b>/, ...
    ├── val/<class_a>/,   val/<class_b>/,   ...
    └── test/<class_a>/,  test/<class_b>/,  ...

Usage
-----
    # 3-class baseline (related / unrelated / hard_negative)
    python scripts/train_vision.py \\
        --dataset-dir data/vision/datasets/v1/splits

    # Full fine-tune with focal loss
    python scripts/train_vision.py \\
        --dataset-dir data/vision/datasets/v1/splits \\
        --use-focal-loss --epochs 40

    # Binary gate (related vs unrelated), no calibration
    python scripts/train_vision.py \\
        --dataset-dir data/vision/datasets/v1/splits \\
        --classes unrelated related \\
        --no-calibration
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from torch.utils.data import DataLoader  # noqa: E402

from app.core.logging import setup_logging  # noqa: E402
from app.modules.vision.config import (  # noqa: E402
    AugmentationConfig,
    ImageSizeConfig,
    VisionConfig,
    VisionModelConfig,
    VisionStorageConfig,
)
from app.modules.vision.datasets.balancer import (  # noqa: E402
    build_weighted_sampler,
    compute_class_weights_tensor,
    imbalance_report,
)
from app.modules.vision.datasets.dataset import ImageFolderDataset  # noqa: E402
from app.modules.vision.evaluation.metrics import VisionEvaluator  # noqa: E402
from app.modules.vision.models.registry import VisionModelRegistry  # noqa: E402
from app.modules.vision.persistence.model_store import VisionModelStore  # noqa: E402
from app.modules.vision.preprocessing.augmentation import PRESETS, get_preset  # noqa: E402
from app.modules.vision.preprocessing.transforms import (  # noqa: E402
    get_train_transforms,
    get_val_transforms,
)
from app.modules.vision.training.calibration import calibrate_model  # noqa: E402
from app.modules.vision.training.config import VisionTrainingConfig  # noqa: E402
from app.modules.vision.training.trainer import VisionTrainer  # noqa: E402

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HantaProject — Train a vision classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--dataset-dir", required=True, type=Path,
        help="Root splits directory (contains train/, val/, test/).",
    )
    parser.add_argument(
        "--dataset-version", default=None,
        help="Dataset version string for audit trail (e.g. 'v1').",
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Class names in label-index order. Auto-detected from disk if omitted.",
    )

    # Model
    parser.add_argument(
        "--architecture", default="efficientnet_b0",
        choices=VisionModelRegistry.list_available(),
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Disable ImageNet pretrained weights (rarely useful).",
    )

    # Image
    parser.add_argument("--image-size", type=int, default=224)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--unfreeze-lr", type=float, default=1e-5)
    parser.add_argument("--differential-lr-factor", type=float, default=5.0)
    parser.add_argument(
        "--freeze-epochs", type=int, default=5,
        help="Warm-up epochs with backbone frozen (Phase A).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "plateau", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--early-stopping-patience", type=int, default=7,
    )

    # Imbalance handling
    parser.add_argument(
        "--no-weighted-sampler", action="store_true",
        help="Disable WeightedRandomSampler (use plain shuffle instead).",
    )
    parser.add_argument(
        "--use-focal-loss", action="store_true",
        help="Replace CrossEntropyLoss with Focal Loss (gamma=2.0 default).",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # Augmentation
    parser.add_argument(
        "--augmentation", default="standard", choices=sorted(PRESETS.keys()),
    )

    # Calibration
    parser.add_argument(
        "--no-calibration", action="store_true",
        help="Skip post-training temperature calibration.",
    )

    # Runtime
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-test", action="store_true")

    # Persistence
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--version", default=None,
                        help="Override the auto-generated model version string.")
    parser.add_argument("--save-all-checkpoints", action="store_true",
                        help="Keep all checkpoints (default: best + last 2).")

    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_vision_config(args: argparse.Namespace, num_classes: int) -> VisionConfig:
    preset_params = get_preset(args.augmentation).params
    aug = AugmentationConfig(**{
        k: v for k, v in preset_params.items()
        if k in AugmentationConfig.model_fields
    })
    storage = VisionStorageConfig()
    if args.models_dir is not None:
        storage = storage.model_copy(update={"models_dir": args.models_dir})
    return VisionConfig(
        storage=storage,
        model=VisionModelConfig(
            architecture=args.architecture,
            pretrained=not args.no_pretrained,
            num_classes=num_classes,
            dropout=args.dropout,
            freeze_backbone=args.freeze_epochs > 0,
        ),
        image_size=ImageSizeConfig(width=args.image_size, height=args.image_size),
        augmentation=aug,
        device=args.device,
    )


def _build_dataloader(
    dataset: ImageFolderDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler=None,
) -> DataLoader:
    import torch
    # pin_memory is CUDA-only; MPS and CPU do not benefit and produce a warning
    pin = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _summary_block(title: str, rows: list[tuple[str, str]]) -> str:
    width = 62
    sep = "─" * width
    lines = [sep, f"  {title}", sep]
    for label, value in rows:
        lines.append(f"  {label:<28} {value}")
    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    setup_logging(debug=True, environment="development")

    logger.info("=" * 62)
    logger.info("  HantaProject — Vision Training")
    logger.info("  Dataset     : %s", args.dataset_dir)
    logger.info("  Architecture: %s", args.architecture)
    logger.info("  Augmentation: %s", args.augmentation)
    logger.info("  Epochs      : %d (freeze=%d)", args.epochs, args.freeze_epochs)
    logger.info("  Focal Loss  : %s (gamma=%.1f)", args.use_focal_loss, args.focal_gamma)
    logger.info("  WRS         : %s", not args.no_weighted_sampler)
    logger.info("=" * 62)

    if not args.dataset_dir.exists():
        logger.error("Dataset directory not found: %s", args.dataset_dir)
        sys.exit(1)

    # ── Datasets ──────────────────────────────────────────────────────────────

    train_ds = ImageFolderDataset(
        root_dir=args.dataset_dir,
        split="train",
        classes=args.classes,
    )
    val_ds = ImageFolderDataset(
        root_dir=args.dataset_dir,
        split="val",
        classes=args.classes,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error(
            "Empty train or val split at %s.\n"
            "Run: python scripts/prepare_dataset.py --version v1",
            args.dataset_dir,
        )
        sys.exit(1)

    class_names = train_ds.classes
    num_classes = len(class_names)
    logger.info("Classes (%d): %s", num_classes, class_names)

    # ── Class imbalance report ─────────────────────────────────────────────────

    class_dist = train_ds.class_distribution()
    report = imbalance_report(class_dist)
    logger.info(
        "Imbalance: ratio=%.1f:1 | severity=%s",
        report["imbalance_ratio"], report["severity"],
    )
    if report["severity"] not in ("balanced", "mild"):
        logger.info("Recommendation: %s", report["recommendation"])

    # ── Transforms ────────────────────────────────────────────────────────────

    vision_cfg = _build_vision_config(args, num_classes)
    train_tf = get_train_transforms(vision_cfg)
    val_tf = get_val_transforms(vision_cfg)

    train_ds.transform = train_tf
    val_ds.transform = val_tf

    # ── WeightedRandomSampler ─────────────────────────────────────────────────

    if not args.no_weighted_sampler:
        logger.info("Building WeightedRandomSampler…")
        sampler = build_weighted_sampler(train_ds, class_names)
        train_loader = _build_dataloader(
            train_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, sampler=sampler,
        )
    else:
        train_loader = _build_dataloader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
        )

    val_loader = _build_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ── Class weights for loss ─────────────────────────────────────────────────

    class_weights = compute_class_weights_tensor(class_dist, class_names)
    logger.info(
        "Class weights for loss: %s",
        {name: round(float(w), 4) for name, w in zip(class_names, class_weights)},
    )

    # ── Model ─────────────────────────────────────────────────────────────────

    model = VisionModelRegistry.build(
        architecture=args.architecture,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        freeze=args.freeze_epochs > 0,
    )
    model.log_parameter_summary()

    # ── Trainer ───────────────────────────────────────────────────────────────

    training_cfg = VisionTrainingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        mixed_precision=args.mixed_precision,
        label_smoothing=args.label_smoothing,
        freeze_epochs=args.freeze_epochs,
        unfreeze_lr=args.unfreeze_lr,
        differential_lr_factor=args.differential_lr_factor,
        early_stopping_patience=args.early_stopping_patience,
        save_best_only=not args.save_all_checkpoints,
        random_state=args.seed,
        use_weighted_sampler=not args.no_weighted_sampler,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
    )

    run_dir = vision_cfg.storage.checkpoints_dir / f"run_{args.architecture}"
    device = vision_cfg.resolve_device()

    trainer = VisionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_cfg,
        output_dir=run_dir,
        device=device,
        class_names=class_names,
        class_weights=class_weights,
    )

    history = trainer.train()

    # ── Test evaluation ───────────────────────────────────────────────────────

    test_metrics: dict | None = None
    if not args.skip_test:
        test_root = args.dataset_dir / "test"
        if test_root.exists():
            test_ds = ImageFolderDataset(
                root_dir=args.dataset_dir,
                split="test",
                transform=val_tf,
                classes=args.classes,
            )
            if len(test_ds) > 0:
                test_loader = _build_dataloader(
                    test_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                )
                evaluator = VisionEvaluator(class_names=class_names)
                result = evaluator.evaluate(
                    model=trainer.model,
                    dataloader=test_loader,
                    device=trainer.device,
                )
                result.log(prefix="test")
                test_metrics = result.as_dict()
            else:
                logger.warning("Test split is empty — skipping.")
        else:
            logger.info("No test/ split at %s — skipping.", test_root)

    # ── Temperature calibration ───────────────────────────────────────────────

    calibration_temperature = 1.0
    calibration_data: dict | None = None

    if not args.no_calibration:
        logger.info("Running post-training temperature calibration on val set…")
        try:
            cal_scaler = calibrate_model(
                model=trainer.model,
                val_loader=val_loader,
                device=trainer.device,
            )
            calibration_temperature = cal_scaler.temperature
            calibration_data = cal_scaler.as_dict()
            logger.info(
                "Calibration: T=%.4f | ECE %.4f → %.4f",
                cal_scaler.temperature,
                cal_scaler.ece_before, cal_scaler.ece_after,
            )
        except Exception:
            logger.exception("Calibration failed — proceeding with T=1.0")
            calibration_temperature = 1.0
    else:
        logger.info("Calibration skipped (--no-calibration).")

    # ── Save model ────────────────────────────────────────────────────────────

    best_val = trainer.checkpoint.best_score
    metrics: dict = {
        "best_val_score": round(float(best_val), 4),
        "monitor_metric": training_cfg.monitor_metric,
        "epochs_trained": len(history),
    }
    if test_metrics:
        metrics["test"] = test_metrics
        metrics["f1"] = test_metrics.get("f1", best_val)
    else:
        metrics["f1"] = float(best_val)

    if calibration_data:
        metrics["calibration"] = calibration_data

    store = VisionModelStore(base_dir=vision_cfg.storage.models_dir)
    version = store.save(
        model=trainer.model,
        architecture=args.architecture,
        class_names=class_names,
        image_size=(vision_cfg.image_size.width, vision_cfg.image_size.height),
        metrics=metrics,
        training_config=training_cfg.model_dump(),
        version=args.version,
        calibration_temperature=calibration_temperature,
        confidence_threshold=0.6,
        dataset_version=args.dataset_version,
    )

    # Persist calibration separately
    if calibration_data:
        store.save_calibration(
            version=version,
            temperature=calibration_temperature,
            ece_before=calibration_data.get("ece_before"),
            ece_after=calibration_data.get("ece_after"),
            nll_before=calibration_data.get("nll_before"),
            nll_after=calibration_data.get("nll_after"),
        )

    # Persist epoch-by-epoch history
    version_dir = store.version_dir(version)
    (version_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # ── Summary ───────────────────────────────────────────────────────────────

    rows = [
        ("Version", version),
        ("Architecture", args.architecture),
        ("Classes", ", ".join(class_names)),
        ("Epochs trained", str(metrics["epochs_trained"])),
        ("Best val score", f"{metrics['best_val_score']:.4f}  ({training_cfg.monitor_metric})"),
        ("Imbalance severity", report["severity"]),
        ("WeightedRandomSampler", str(not args.no_weighted_sampler)),
        ("Focal Loss", f"{args.use_focal_loss}  (gamma={args.focal_gamma:.1f})"),
        ("Calibration T", f"{calibration_temperature:.4f}"),
    ]
    if calibration_data:
        rows.append(("ECE (before → after)", f"{calibration_data.get('ece_before', 0):.4f} → {calibration_data.get('ece_after', 0):.4f}"))
    if test_metrics:
        rows.append(("Test F1 (macro)", f"{test_metrics['f1']:.4f}"))
        rows.append(("Test accuracy", f"{test_metrics['accuracy']:.4f}"))
    rows.append(("Saved to", str(version_dir)))

    print("\n" + _summary_block("Training Complete", rows) + "\n")


if __name__ == "__main__":
    main()
