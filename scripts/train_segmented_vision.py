"""
HantaProject — ROI-Only Vision Training Script
===============================================

Trains a CNN classifier on pre-segmented lung ROI crops produced by
scripts/regenerate_segmented_dataset.py.

Key differences from train_vision.py
--------------------------------------
  - Loads SegmentedROIDataset (no fallback to full images).
  - Uses lung-specific augmentation (lung_roi preset + GaussianNoise +
    small affine translation).
  - Logs segmentation telemetry stats before training starts.
  - Defaults to augmentation preset "lung_roi".

Expected dataset layout (produced by regenerate_segmented_dataset.py):
    <dataset_dir>/
    ├── train/healthy_xray/img001.jpg
    ├── train/healthy_xray/img001_telemetry.json
    ├── val/...
    └── test/...

Usage
-----
    # Basic
    python scripts/train_segmented_vision.py \\
        --dataset-dir data/segmented_dataset

    # Full fine-tune with focal loss
    python scripts/train_segmented_vision.py \\
        --dataset-dir data/segmented_dataset \\
        --architecture efficientnet_b4 \\
        --use-focal-loss --epochs 40 \\
        --freeze-epochs 8

    # Skip test + skip calibration for a quick sweep
    python scripts/train_segmented_vision.py \\
        --dataset-dir data/segmented_dataset \\
        --epochs 5 --skip-test --no-calibration
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
from app.modules.vision.datasets.segmented_dataset import SegmentedROIDataset  # noqa: E402
from app.modules.vision.evaluation.metrics import VisionEvaluator  # noqa: E402
from app.modules.vision.models.registry import VisionModelRegistry  # noqa: E402
from app.modules.vision.persistence.model_store import VisionModelStore  # noqa: E402
from app.modules.vision.preprocessing.lung_augmentation import (  # noqa: E402
    get_lung_roi_train_transforms,
    get_lung_roi_val_transforms,
)
from app.modules.vision.training.calibration import calibrate_model  # noqa: E402
from app.modules.vision.training.config import VisionTrainingConfig  # noqa: E402
from app.modules.vision.training.trainer import VisionTrainer  # noqa: E402

logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HantaProject — Train on segmented lung ROI crops",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--dataset-dir", required=True, type=Path,
        help="Root segmented dataset directory (train/ val/ test/ layout).",
    )
    parser.add_argument(
        "--dataset-version", default=None,
        help="Dataset version string for audit trail (e.g. 'seg_v1').",
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
    parser.add_argument("--no-pretrained", action="store_true")

    # Image
    parser.add_argument("--image-size", type=int, default=224)

    # Augmentation
    parser.add_argument(
        "--noise-std", type=float, default=0.03,
        help="Gaussian noise max std for lung_roi augmentation.",
    )
    parser.add_argument(
        "--blur-prob", type=float, default=0.25,
        help="Probability of Gaussian blur per sample.",
    )
    parser.add_argument(
        "--affine-translate", type=float, default=0.05,
        help="Affine translation fraction (per side).",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--unfreeze-lr", type=float, default=1e-5)
    parser.add_argument("--differential-lr-factor", type=float, default=5.0)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", default="cosine",
                        choices=["cosine", "step", "plateau", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=7)

    # Imbalance
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # Calibration
    parser.add_argument("--no-calibration", action="store_true")

    # Runtime
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-test", action="store_true")

    # Persistence
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--version", default=None,
                        help="Override the auto-generated version string.")
    parser.add_argument("--version-prefix", default="seg",
                        help="Prefix for auto-generated version string.")

    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_dataloader(
    dataset: SegmentedROIDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler=None,
) -> DataLoader:
    import torch
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
    width = 66
    sep = "─" * width
    lines = [sep, f"  {title}", sep]
    for label, value in rows:
        lines.append(f"  {label:<32} {value}")
    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    setup_logging(debug=True, environment="development")

    logger.info("=" * 66)
    logger.info("  HantaProject — Segmented Lung ROI Training")
    logger.info("  Dataset     : %s", args.dataset_dir)
    logger.info("  Architecture: %s", args.architecture)
    logger.info("  Epochs      : %d (freeze=%d)", args.epochs, args.freeze_epochs)
    logger.info("  Focal Loss  : %s (gamma=%.1f)", args.use_focal_loss, args.focal_gamma)
    logger.info("=" * 66)

    if not args.dataset_dir.exists():
        logger.error(
            "Dataset directory not found: %s\n"
            "Run: python scripts/regenerate_segmented_dataset.py first.",
            args.dataset_dir,
        )
        sys.exit(1)

    # ── Vision config (for transforms / storage) ──────────────────────────────

    storage = VisionStorageConfig()
    if args.models_dir is not None:
        storage = storage.model_copy(update={"models_dir": args.models_dir})

    vision_cfg = VisionConfig(
        storage=storage,
        model=VisionModelConfig(
            architecture=args.architecture,
            pretrained=not args.no_pretrained,
            num_classes=2,          # overridden below after class detection
            dropout=args.dropout,
            freeze_backbone=args.freeze_epochs > 0,
        ),
        image_size=ImageSizeConfig(width=args.image_size, height=args.image_size),
        device=args.device,
    )

    # ── Datasets ──────────────────────────────────────────────────────────────

    train_ds = SegmentedROIDataset(
        root_dir=args.dataset_dir,
        split="train",
        classes=args.classes,
    )
    val_ds = SegmentedROIDataset(
        root_dir=args.dataset_dir,
        split="val",
        classes=args.classes,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error(
            "Empty train or val split at %s.\n"
            "Run: python scripts/regenerate_segmented_dataset.py "
            "--source-dir <src> --output-dir %s",
            args.dataset_dir, args.dataset_dir,
        )
        sys.exit(1)

    class_names = train_ds.classes
    num_classes = len(class_names)
    logger.info("Classes (%d): %s", num_classes, class_names)

    # ── Telemetry stats ───────────────────────────────────────────────────────

    train_tel = train_ds.telemetry_summary()
    val_tel = val_ds.telemetry_summary()
    train_tel.log(split="train")
    val_tel.log(split="val")

    # ── Class imbalance ───────────────────────────────────────────────────────

    class_dist = train_ds.class_distribution()
    report = imbalance_report(class_dist)
    logger.info(
        "Imbalance: ratio=%.1f:1 severity=%s",
        report["imbalance_ratio"], report["severity"],
    )

    # ── Transforms (lung ROI specific) ────────────────────────────────────────

    train_tf = get_lung_roi_train_transforms(
        vision_cfg,
        noise_std=args.noise_std,
        blur_prob=args.blur_prob,
        affine_translate_pct=args.affine_translate,
    )
    val_tf = get_lung_roi_val_transforms(vision_cfg)

    train_ds.transform = train_tf
    val_ds.transform = val_tf

    # ── Sampler ───────────────────────────────────────────────────────────────

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

    # ── Class weights ─────────────────────────────────────────────────────────

    class_weights = compute_class_weights_tensor(class_dist, class_names)
    logger.info(
        "Class weights: %s",
        {n: round(float(w), 4) for n, w in zip(class_names, class_weights)},
    )

    # ── Model ─────────────────────────────────────────────────────────────────

    vision_cfg.model = vision_cfg.model.model_copy(update={"num_classes": num_classes})
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
        random_state=args.seed,
        use_weighted_sampler=not args.no_weighted_sampler,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
    )

    device = vision_cfg.resolve_device()
    run_dir = vision_cfg.storage.checkpoints_dir / f"seg_{args.architecture}"

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
        test_split = args.dataset_dir / "test"
        if test_split.exists():
            test_ds = SegmentedROIDataset(
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

    # ── Calibration ───────────────────────────────────────────────────────────

    calibration_temperature = 1.0
    calibration_data: dict | None = None

    if not args.no_calibration:
        logger.info("Running post-training temperature calibration…")
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
                cal_scaler.temperature, cal_scaler.ece_before, cal_scaler.ece_after,
            )
        except Exception:
            logger.exception("Calibration failed — proceeding with T=1.0")

    # ── Save model ────────────────────────────────────────────────────────────

    best_val = trainer.checkpoint.best_score
    metrics: dict = {
        "best_val_score": round(float(best_val), 4),
        "monitor_metric": training_cfg.monitor_metric,
        "epochs_trained": len(history),
        "training_mode": "segmented_roi",
        "augmentation": "lung_roi",
        "telemetry": {
            "train": train_tel.as_dict(),
            "val": val_tel.as_dict(),
        },
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

    if calibration_data:
        store.save_calibration(
            version=version,
            temperature=calibration_temperature,
            ece_before=calibration_data.get("ece_before"),
            ece_after=calibration_data.get("ece_after"),
            nll_before=calibration_data.get("nll_before"),
            nll_after=calibration_data.get("nll_after"),
        )

    version_dir = store.version_dir(version)
    (version_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # ── Summary ───────────────────────────────────────────────────────────────

    rows = [
        ("Version", version),
        ("Architecture", args.architecture),
        ("Training mode", "Segmented lung ROI"),
        ("Classes", ", ".join(class_names)),
        ("Epochs trained", str(metrics["epochs_trained"])),
        ("Best val score", f"{metrics['best_val_score']:.4f}  ({training_cfg.monitor_metric})"),
        ("Imbalance severity", report["severity"]),
        ("WeightedRandomSampler", str(not args.no_weighted_sampler)),
        ("Focal Loss", f"{args.use_focal_loss}  (gamma={args.focal_gamma:.1f})"),
        ("Calibration T", f"{calibration_temperature:.4f}"),
        ("Train lung_area mean", f"{train_tel.lung_area_pct_mean:.3f}"),
        ("Val lung_area mean", f"{val_tel.lung_area_pct_mean:.3f}"),
    ]
    if calibration_data:
        rows.append((
            "ECE (before → after)",
            f"{calibration_data.get('ece_before', 0):.4f} → {calibration_data.get('ece_after', 0):.4f}",
        ))
    if test_metrics:
        rows.append(("Test F1 (macro)", f"{test_metrics['f1']:.4f}"))
    rows.append(("Saved to", str(version_dir)))

    print("\n" + _summary_block("Segmented ROI Training Complete", rows) + "\n")


if __name__ == "__main__":
    main()
