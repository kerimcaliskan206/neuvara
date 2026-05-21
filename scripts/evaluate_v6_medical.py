#!/usr/bin/env python3
"""
Evaluate v6 medical classifier — Phase 10.

Loads an isolated v6 checkpoint and runs full evaluation on val or test split.

Metrics computed:
  - macro F1, per-class precision/recall/F1
  - OOD rejection recall + leakage rate (per class)
  - Expected Calibration Error (ECE) + Maximum CE
  - Confusion matrix
  - GradCAM activation sanity check
  - Compatibility report (pass/fail per gate)

Outputs (written to --output-dir):
  evaluation_<split>_<timestamp>.json
  compatibility_<split>_<timestamp>.json

SAFETY:
  - Only loads v6 checkpoints (verified by 'v6_meta' key).
  - V5 production model is never touched.

Usage
-----
    # Evaluate on validation split (default)
    python scripts/evaluate_v6_medical.py \\
        --checkpoint models/vision/v6_medical/stage_c_selective/v6_checkpoint.pt \\
        --dataset-dir data/medical_v6_splits

    # Evaluate on test split
    python scripts/evaluate_v6_medical.py \\
        --checkpoint models/vision/v6_medical/stage_c_selective/v6_checkpoint.pt \\
        --dataset-dir data/medical_v6_splits \\
        --split test

    # Skip GradCAM (faster, useful for quick checks)
    python scripts/evaluate_v6_medical.py ... --no-gradcam

    # Custom output directory
    python scripts/evaluate_v6_medical.py ... --output-dir reports/v6_eval/

Exit codes: 0 all gates pass, 1 error, 2 compatibility gate failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("evaluate_v6")

import torch
from torch.utils.data import DataLoader

from app.modules.vision.medical.v6_training_config import (
    V6_BINARY_MEDICAL_CONFIG, V6TrainingConfig,
)
from app.modules.vision.medical.v6_evaluation import (
    V6Evaluator,
    export_evaluation_report,
    export_compatibility_report,
)
from app.modules.vision.datasets.dataset import ImageFolderDataset
from app.modules.vision.models.efficientnet import build_efficientnet
from app.modules.vision.preprocessing.transforms import get_val_transforms


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_v6_model(
    checkpoint_path: Path, config: V6TrainingConfig, device: torch.device
) -> torch.nn.Module:
    """Load a v6 checkpoint and verify it is not a v5 checkpoint."""
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "v6_meta" not in ckpt:
        logger.error(
            "Checkpoint %s does not have a 'v6_meta' key. "
            "This is not a v6 checkpoint — V5 checkpoints must not be used.",
            checkpoint_path,
        )
        sys.exit(1)

    meta = ckpt["v6_meta"]
    logger.info(
        "V6 checkpoint | run=%s sub_stage=%s val_f1=%.4f compat=%s",
        meta.get("run_name", "?"),
        meta.get("sub_stage", "?"),
        meta.get("val_f1", 0.0),
        meta.get("compatibility_passed", "?"),
    )

    ckpt_classes  = meta.get("classes", list(config.classes))
    ckpt_n_classes = len(ckpt_classes)

    model = build_efficientnet(
        variant="efficientnet_b0",
        num_classes=ckpt_n_classes,
        pretrained=False,
        dropout=0.30,
        freeze=False,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model.to(device)


# ── DataLoader ────────────────────────────────────────────────────────────────


def _build_eval_loader(
    dataset_dir: Path, split: str, config: V6TrainingConfig, batch_size: int
) -> DataLoader:
    classes = list(config.classes)
    ds = ImageFolderDataset(
        root_dir=dataset_dir, split=split,
        transform=get_val_transforms(),
        classes=classes,
    )
    if not ds.samples:
        raise RuntimeError(
            f"No images found in {dataset_dir}/{split}/. "
            f"Run prepare_v6_medical_dataset.py first."
        )
    logger.info("Eval dataset [%s]: %d samples | %s", split, len(ds), ds._distribution_str())
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )


# ── Summary printer ───────────────────────────────────────────────────────────


def _print_summary(result, compat_passed: bool) -> None:
    print("\n=== V6 Evaluation Summary ===")
    print(f"  Split        : {result.split}  ({result.num_samples} samples)")
    print(f"  Macro F1     : {result.macro_f1:.4f}")
    print(f"  Accuracy     : {result.accuracy:.4f}")
    print(f"  ECE          : {result.calibration.ece:.4f}")
    print()
    print("  Per-class metrics:")
    for m in result.per_class:
        flag = ""
        if m.class_name in {"hard_negative", "fake_medical"}:
            flag = "  [OOD]"
        print(
            f"    {m.class_name:<22}  recall={m.recall:.4f}  "
            f"prec={m.precision:.4f}  f1={m.f1:.4f}  n={m.support}{flag}"
        )
    print()
    print(f"  OOD rejection recall : {result.ood.rejection_recall:.4f}")
    print(f"  OOD leakage rate     : {result.ood.leakage_rate:.4f}")
    print()
    print("  Compatibility gates:")
    compat = result.compatibility
    gates = compat.gate_results
    for gate_name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {gate_name:<24} {status}")
    print()
    print(f"  Overall compatibility: {'PASS' if compat_passed else 'FAIL'}")
    if result.gradcam is not None:
        print(
            f"  GradCAM sanity: entropy={result.gradcam.mean_entropy:.4f}  "
            f"degenerate={result.gradcam.degenerate_fraction:.4f}  "
            f"passed={result.gradcam.passed}"
        )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_v6_medical",
        description="Evaluate isolated v6 medical classifier — Phase 10.",
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a v6 checkpoint (.pt file with v6_meta key).",
    )
    p.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Path to data/medical_v6_splits/ from prepare_v6_medical_dataset.py.",
    )
    p.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Dataset split to evaluate on (default: val).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for evaluation reports. "
             "Defaults to <checkpoint_parent>/evaluation/.",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: 'cuda', 'mps', 'cpu', or 'auto' (default).",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for evaluation (default: 64).",
    )
    p.add_argument(
        "--gradcam", dest="gradcam", action="store_true",  default=True,
        help="Run GradCAM sanity check (default: enabled).",
    )
    p.add_argument(
        "--no-gradcam", dest="gradcam", action="store_false",
        help="Skip GradCAM sanity check.",
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.checkpoint.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Evaluate V6 Medical — Phase 10 ===")
    logger.info("Checkpoint  : %s", args.checkpoint)
    logger.info("Dataset     : %s", args.dataset_dir)
    logger.info("Split       : %s", args.split)
    logger.info("Device      : %s", device)
    logger.info("GradCAM     : %s", args.gradcam)
    logger.info("Output      : %s", output_dir)

    try:
        model       = _load_v6_model(args.checkpoint, config, device)
        eval_loader = _build_eval_loader(
            args.dataset_dir, args.split, config, args.batch_size
        )

        evaluator = V6Evaluator(config)
        result    = evaluator.evaluate(
            model=model,
            dataloader=eval_loader,
            device=device,
            split=args.split,
            run_gradcam=args.gradcam,
            checkpoint_path=str(args.checkpoint),
        )
        evaluator.fill_positive_recall(result.compatibility, result.per_class)

        compat_passed = result.compatibility.overall_pass

        # Export reports
        eval_path   = output_dir / f"evaluation_{args.split}_{timestamp}.json"
        compat_path = output_dir / f"compatibility_{args.split}_{timestamp}.json"
        export_evaluation_report(result, eval_path)
        export_compatibility_report(result.compatibility, compat_path)

        _print_summary(result, compat_passed)
        logger.info("Reports written to: %s", output_dir)

        return 0 if compat_passed else 2

    except Exception:
        logger.exception("Evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
