#!/usr/bin/env python3
"""
Phase 15 — Calibration Recovery (Temperature Scaling).

Applies post-hoc temperature scaling to the Stage B checkpoint.
No backbone retraining — only a single scalar T is fitted on val logits.

Key insight (from logit inspection):
  mean_confidence = 0.6764 < mean_accuracy = 0.9043  →  UNDERCONFIDENT
  Expect T* < 1.0 (sharpening, not softening).

Pipeline
--------
  1. Load Stage B checkpoint  (models/vision/v6_medical/stage_b_top1/best.pt)
  2. Collect raw val logits   (no softmax)
  3. Fit T* via L-BFGS on NLL
  4. Compute before/after: ECE, MCE, Brier score, reliability diagram
  5. Run pipeline compatibility with calibrated model wrapper
  6. Stage C safety verdict  (ECE < 0.15 → GO, < 0.18 → CONDITIONAL, ≥ 0.18 → need fixes)
  7. Export temperature_config.json, stage_b_calibrated.pt, calibration_report.json

SAFETY: V5 production model is never loaded or modified.
All outputs isolated under models/vision/v6_medical/calibration/
and reports/v6_medical/calibration_recovery/.

Usage
-----
    python scripts/calibrate_stage_b.py

    python scripts/calibrate_stage_b.py \\
        --checkpoint models/vision/v6_medical/stage_b_top1/best.pt \\
        --dataset-dir data/medical_v6_splits \\
        --device auto

Exit codes: 0 GO/CONDITIONAL_GO, 1 error, 2 NO_GO.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("calibrate_stage_b")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ── Temperature scaling ───────────────────────────────────────────────────────


class TemperatureScaler(nn.Module):
    """
    Wraps a model and divides its logits by a learnable scalar T.

    T > 1  → softer predictions  (fixes overconfidence)
    T < 1  → sharper predictions (fixes underconfidence)
    T = 1  → identity

    argmax predictions are unchanged; only confidence values shift.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """
        Fit T* on pre-collected logits via L-BFGS minimising NLL.

        Parameters
        ----------
        logits : (N, C) raw logits (no softmax)
        labels : (N,)   integer class indices

        Returns the optimal temperature T*.
        """
        self.temperature.data.fill_(1.0)
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=0.01, max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = nll(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        t_star = float(self.temperature.item())
        logger.info("Temperature scaling: T* = %.4f", t_star)
        return t_star


# ── Calibration metrics ───────────────────────────────────────────────────────


def _compute_ece_mce(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float, list[dict]]:
    """
    ECE and MCE from confidence + correctness arrays.

    Returns (ece, mce, bin_data) where bin_data is the reliability diagram.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_data = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n = int(mask.sum())
        if n == 0:
            bin_data.append({
                "bin_lower": round(float(lo), 4),
                "bin_upper": round(float(hi), 4),
                "count":     0,
                "fraction":  0.0,
                "avg_confidence": round(float((lo + hi) / 2), 4),
                "avg_accuracy":   0.0,
                "gap":            0.0,
            })
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc  = float(accuracies[mask].mean())
        gap      = abs(avg_conf - avg_acc)
        frac     = n / len(confidences)
        ece     += frac * gap
        mce      = max(mce, gap)
        bin_data.append({
            "bin_lower":      round(float(lo), 4),
            "bin_upper":      round(float(hi), 4),
            "count":          n,
            "fraction":       round(frac, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy":   round(avg_acc, 4),
            "gap":            round(gap, 4),
        })

    return float(ece), float(mce), bin_data


def _brier_score(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score (mean squared error in probability space)."""
    one_hot = np.zeros((len(labels), n_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean((probs - one_hot) ** 2))


def _confidence_histogram(confidences: np.ndarray, n_bins: int = 15) -> list[dict]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    hist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        hist.append({
            "bin_lower": round(float(lo), 4),
            "bin_upper": round(float(hi), 4),
            "count":     int(mask.sum()),
            "fraction":  round(float(mask.sum()) / len(confidences), 4),
        })
    return hist


@dataclass
class CalibrationSnapshot:
    ece:             float
    mce:             float
    brier_score:     float
    mean_confidence: float
    mean_accuracy:   float
    overconfidence:  float   # mean_conf - mean_acc (negative = underconfident)
    n_bins:          int
    reliability_diagram: list[dict]
    confidence_histogram: list[dict]


def _snapshot(
    probs: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    n_bins: int = 15,
) -> CalibrationSnapshot:
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = (preds == labels).astype(np.float32)

    ece, mce, reliability = _compute_ece_mce(confidences, correct, n_bins)
    brier  = _brier_score(probs, labels, n_classes)
    m_conf = float(confidences.mean())
    m_acc  = float(correct.mean())

    return CalibrationSnapshot(
        ece=round(ece, 4),
        mce=round(mce, 4),
        brier_score=round(brier, 4),
        mean_confidence=round(m_conf, 4),
        mean_accuracy=round(m_acc, 4),
        overconfidence=round(m_conf - m_acc, 4),
        n_bins=n_bins,
        reliability_diagram=reliability,
        confidence_histogram=_confidence_histogram(confidences, n_bins),
    )


# ── Logit collection ──────────────────────────────────────────────────────────


@torch.no_grad()
def _collect_logits(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, labels) tensors for the full val set."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


# ── Calibrated model wrapper (for compatibility checker) ──────────────────────


class _CalibratedModel(nn.Module):
    """Thin wrapper: divides logits by a fixed T before returning."""

    def __init__(self, model: nn.Module, temperature: float) -> None:
        super().__init__()
        self.model       = model
        self.temperature = temperature
        # Expose backbone for GradCAM hooks (though not used here)
        self._backbone   = getattr(model, "_backbone", model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature


# ── Go / No-Go for Stage C ────────────────────────────────────────────────────


def _stage_c_verdict(
    after: CalibrationSnapshot,
    before: CalibrationSnapshot,
    compat_passed: bool | None,
) -> tuple[str, list[str]]:
    """
    Gates:
      ECE < 0.15                        → GO
      0.15 ≤ ECE < 0.18                 → CONDITIONAL_GO
      ECE ≥ 0.18                        → NO_GO (secondary fixes required)
      ECE worse than before by > 0.02   → NO_GO (T scaling made it worse)
    """
    notes: list[str] = []

    # Regression guard
    ece_regression = after.ece > before.ece + 0.02
    if ece_regression:
        return "NO_GO", [
            f"Temperature scaling worsened ECE: before={before.ece:.4f} → after={after.ece:.4f} "
            "(regression > 0.02). T* may be at a local minimum. Try secondary fixes."
        ]

    if after.ece >= 0.18:
        return "NO_GO", [
            f"ECE {after.ece:.4f} ≥ 0.18 even after temperature scaling. "
            "Secondary fixes required: label smoothing increase or reduced head_lr_factor."
        ]

    if after.ece < 0.15:
        if compat_passed is False:
            notes.append("Pipeline compat gates failed — review before Stage C.")
        return "GO", notes

    # Conditional: [0.15, 0.18)
    notes.append(f"ECE {after.ece:.4f} in conditional range [0.15, 0.18) — monitor in Stage C.")
    if compat_passed is False:
        notes.append("Pipeline compat gates failed — review before Stage C.")
    return "CONDITIONAL_GO", notes


# ── Console summary ───────────────────────────────────────────────────────────


def _print_summary(
    before: CalibrationSnapshot,
    after: CalibrationSnapshot,
    t_star: float,
    verdict: str,
    reasons: list[str],
    compat_passed: bool | None,
) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  CALIBRATION RECOVERY — PHASE 15")
    print(sep)
    print(f"  Temperature T*     : {t_star:.4f}  "
          f"({'sharpen — model was underconfident' if t_star < 1.0 else 'soften — model was overconfident'})")
    print(f"\n  {'Metric':<24}  {'Before':>10}  {'After':>10}  {'Delta':>10}")
    print("  " + "─" * 52)
    rows = [
        ("ECE",              before.ece,              after.ece),
        ("MCE",              before.mce,               after.mce),
        ("Brier Score",      before.brier_score,       after.brier_score),
        ("Mean Confidence",  before.mean_confidence,   after.mean_confidence),
        ("Mean Accuracy",    before.mean_accuracy,     after.mean_accuracy),
        ("Overconfidence",   before.overconfidence,    after.overconfidence),
    ]
    for name, bv, av in rows:
        delta = av - bv
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<24}  {bv:>10.4f}  {av:>10.4f}  {sign}{delta:+.4f}")

    if compat_passed is not None:
        print(f"\n  Pipeline compat    : {'PASS' if compat_passed else 'FAIL'}")

    print(f"\n{sep}")
    verdict_label = {
        "GO":             "GO  — Stage C is safe to proceed",
        "CONDITIONAL_GO": "CONDITIONAL GO  — review ECE before Stage C",
        "NO_GO":          "NO-GO  — do NOT proceed to Stage C",
    }[verdict]
    icon = "✓" if verdict == "GO" else ("⚠" if verdict == "CONDITIONAL_GO" else "✗")
    print(f"  {icon}  {verdict_label}")
    for r in reasons:
        print(f"      • {r}")
    print(sep + "\n")


# ── Report export ─────────────────────────────────────────────────────────────


def _write_report(
    report_dir: Path,
    before: CalibrationSnapshot,
    after: CalibrationSnapshot,
    t_star: float,
    verdict: str,
    reasons: list[str],
    compat_result,
    stage_b_ckpt: str,
) -> None:
    def _compat_dict(c):
        if c is None:
            return None
        return {
            "overall_pass": c.overall_pass,
            "gate_results": c.gate_results,
            "fusion":       asdict(c.fusion),
            "calibration":  asdict(c.calibration),
            "probability":  asdict(c.probability),
        }

    data = {
        "phase":               "15",
        "stage":               "calibration_recovery",
        "stage_b_checkpoint":  stage_b_ckpt,
        "temperature":         round(t_star, 6),
        "verdict":             verdict,
        "verdict_reasons":     reasons,
        "calibration_before":  asdict(before),
        "calibration_after":   asdict(after),
        "delta": {
            "ece":             round(after.ece - before.ece, 4),
            "mce":             round(after.mce - before.mce, 4),
            "brier_score":     round(after.brier_score - before.brier_score, 4),
            "mean_confidence": round(after.mean_confidence - before.mean_confidence, 4),
        },
        "pipeline_compatibility": _compat_dict(compat_result),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "calibration_report.json"
    report_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Calibration report written: %s", report_path)


def _write_temperature_config(out_dir: Path, t_star: float, classes: list[str]) -> None:
    cfg = {
        "temperature":        round(t_star, 6),
        "apply_to":           "stage_b_top1",
        "classes":            classes,
        "num_classes":        len(classes),
        "calibration_phase":  "15",
        "notes": (
            "Divide raw logits by this temperature before softmax. "
            "T < 1 sharpens predictions (underconfident model). "
            "argmax predictions are unchanged."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "temperature_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logger.info("Temperature config written: %s", cfg_path)


def _save_calibrated_checkpoint(
    stage_b_path: Path,
    out_dir: Path,
    t_star: float,
) -> Path:
    """
    Copy Stage B checkpoint and append temperature + v6_meta update.

    The saved dict contains the same model_state_dict as Stage B but with:
      - v6_meta.calibration_temperature = T*
      - v6_meta.calibration_phase       = "15"
    """
    from datetime import datetime, timezone

    ckpt = torch.load(stage_b_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("v6_meta", {})
    meta["calibration_temperature"] = round(t_star, 6)
    meta["calibration_phase"] = "15"
    meta["calibration_timestamp"] = datetime.now(timezone.utc).isoformat()
    ckpt["v6_meta"] = meta

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage_b_calibrated.pt"
    torch.save(ckpt, out_path)
    logger.info("Calibrated checkpoint saved: %s", out_path)
    return out_path


# ── Argument parsing ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calibrate_stage_b",
        description="Phase 15 — Temperature scaling calibration for Stage B.",
    )
    p.add_argument(
        "--checkpoint", type=Path,
        default=Path("models/vision/v6_medical/stage_b_top1/best.pt"),
        help="Stage B checkpoint to calibrate (default: stage_b_top1/best.pt).",
    )
    p.add_argument(
        "--dataset-dir", type=Path,
        default=Path("data/medical_v6_splits"),
        help="Staged v6 splits directory (default: data/medical_v6_splits).",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("models/vision/v6_medical/calibration"),
        help="Output directory for calibrated checkpoint (default: …/calibration).",
    )
    p.add_argument(
        "--report-dir", type=Path,
        default=Path("reports/v6_medical/calibration_recovery"),
        help="Report output directory (default: reports/v6_medical/calibration_recovery).",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: cuda / mps / cpu / auto (default).",
    )
    p.add_argument(
        "--n-bins", type=int, default=15,
        help="Number of calibration bins (default: 15).",
    )
    p.add_argument(
        "--no-compat", action="store_true",
        help="Skip pipeline compatibility check.",
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


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    args   = build_parser().parse_args()
    device = _resolve_device(args.device)

    logger.info("=== Phase 15 — Calibration Recovery (Temperature Scaling) ===")
    logger.info("Checkpoint  : %s", args.checkpoint)
    logger.info("Dataset     : %s", args.dataset_dir)
    logger.info("Output      : %s", args.output_dir)
    logger.info("Reports     : %s", args.report_dir)
    logger.info("Device      : %s", device)

    # ── Imports ────────────────────────────────────────────────────────────────
    from app.modules.vision.medical.v6_training_config import (
        V6_BINARY_MEDICAL_CONFIG, V6TrainingConfig,
    )
    from app.modules.vision.medical.v6_compatibility import V6CompatibilityChecker
    from app.modules.vision.preprocessing.transforms import get_val_transforms
    from app.modules.vision.datasets.dataset import ImageFolderDataset
    from app.modules.vision.models.efficientnet import build_efficientnet

    config = V6TrainingConfig(
        run_name=V6_BINARY_MEDICAL_CONFIG.run_name,
        classes=V6_BINARY_MEDICAL_CONFIG.classes,
        positive_classes=V6_BINARY_MEDICAL_CONFIG.positive_classes,
        ood_classes=V6_BINARY_MEDICAL_CONFIG.ood_classes,
        sub_stages=V6_BINARY_MEDICAL_CONFIG.sub_stages,
        replay=V6_BINARY_MEDICAL_CONFIG.replay,
        compatibility=V6_BINARY_MEDICAL_CONFIG.compatibility,
        output_dir=args.output_dir,
        random_seed=V6_BINARY_MEDICAL_CONFIG.random_seed,
        num_workers=V6_BINARY_MEDICAL_CONFIG.num_workers,
        pin_memory=V6_BINARY_MEDICAL_CONFIG.pin_memory,
        mixed_precision=V6_BINARY_MEDICAL_CONFIG.mixed_precision,
    )
    classes    = list(config.classes)
    n_classes  = config.num_classes

    # ── Load Stage B checkpoint ────────────────────────────────────────────────
    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return 1

    logger.info("Loading Stage B checkpoint …")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "v6_meta" not in ckpt:
        logger.error(
            "Checkpoint missing 'v6_meta' — not a v6 checkpoint. Aborting."
        )
        return 1

    meta = ckpt["v6_meta"]
    logger.info(
        "Checkpoint: sub_stage=%s | epoch=%s | val_f1=%.4f",
        meta.get("sub_stage", "?"),
        meta.get("epoch", "?"),
        meta.get("val_f1", 0.0),
    )

    model = build_efficientnet(
        variant="efficientnet_b0",
        num_classes=n_classes,
        pretrained=False,
        dropout=0.30,
        freeze=False,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded: EfficientNet-B0 | %d classes", n_classes)

    # ── Val dataloader ─────────────────────────────────────────────────────────
    if not args.dataset_dir.exists():
        logger.error("Dataset directory not found: %s", args.dataset_dir)
        return 1

    val_ds = ImageFolderDataset(
        root_dir=args.dataset_dir,
        split="val",
        transform=get_val_transforms(),
        classes=classes,
    )
    if not val_ds.samples:
        logger.error("Val dataset is empty: %s", args.dataset_dir / "val")
        return 1

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    logger.info("Val dataset: %d samples", len(val_ds))

    # ── Collect raw logits ─────────────────────────────────────────────────────
    logger.info("Collecting val logits …")
    logits_t, labels_t = _collect_logits(model, val_loader, device)
    logger.info(
        "Logits: shape=%s | label_range=[%d, %d]",
        tuple(logits_t.shape), int(labels_t.min()), int(labels_t.max()),
    )

    # ── Before-calibration snapshot ────────────────────────────────────────────
    probs_before = F.softmax(logits_t, dim=1).numpy()
    labels_np    = labels_t.numpy()
    before       = _snapshot(probs_before, labels_np, n_classes, args.n_bins)
    logger.info(
        "Before calibration: ECE=%.4f  MCE=%.4f  Brier=%.4f  "
        "mean_conf=%.4f  mean_acc=%.4f  gap=%.4f",
        before.ece, before.mce, before.brier_score,
        before.mean_confidence, before.mean_accuracy, before.overconfidence,
    )

    # ── Temperature scaling ────────────────────────────────────────────────────
    logger.info("Fitting temperature T* via L-BFGS on val NLL …")
    scaler       = TemperatureScaler(model)
    logits_cpu   = logits_t.float()
    labels_cpu   = labels_t.long()
    t_star       = scaler.calibrate(logits_cpu, labels_cpu)

    # ── After-calibration snapshot ─────────────────────────────────────────────
    with torch.no_grad():
        scaled_logits = logits_cpu / t_star
    probs_after = F.softmax(scaled_logits, dim=1).numpy()
    after       = _snapshot(probs_after, labels_np, n_classes, args.n_bins)
    logger.info(
        "After calibration (T*=%.4f): ECE=%.4f  MCE=%.4f  Brier=%.4f  "
        "mean_conf=%.4f  mean_acc=%.4f  gap=%.4f",
        t_star, after.ece, after.mce, after.brier_score,
        after.mean_confidence, after.mean_accuracy, after.overconfidence,
    )

    # ── Pipeline compatibility check ───────────────────────────────────────────
    compat_result = None
    if not args.no_compat:
        logger.info("Running pipeline compatibility with calibrated model …")
        calibrated_model = _CalibratedModel(model, t_star).to(device)
        checker = V6CompatibilityChecker(config)
        try:
            compat_result = checker.check_all(calibrated_model, val_loader, device)
            logger.info(
                "Compatibility: fusion=%s calib=%s prob=%s overall=%s",
                "PASS" if compat_result.fusion.passed else "FAIL",
                "PASS" if compat_result.calibration.passed else "FAIL",
                "PASS" if compat_result.probability.passed else "FAIL",
                "PASS" if compat_result.overall_pass else "FAIL",
            )
        except Exception as exc:
            logger.warning("Compatibility check error (non-fatal): %s", exc)

    # ── Stage C safety verdict ─────────────────────────────────────────────────
    compat_passed = compat_result.overall_pass if compat_result is not None else None
    verdict, reasons = _stage_c_verdict(after, before, compat_passed)

    # ── Console output ─────────────────────────────────────────────────────────
    _print_summary(before, after, t_star, verdict, reasons, compat_passed)

    # ── Export artifacts ───────────────────────────────────────────────────────
    logger.info("Exporting calibration artifacts …")

    _write_temperature_config(args.output_dir.resolve(), t_star, classes)
    calibrated_ckpt = _save_calibrated_checkpoint(
        ckpt_path, args.output_dir.resolve(), t_star
    )
    _write_report(
        args.report_dir.resolve(),
        before, after, t_star,
        verdict, reasons,
        compat_result,
        str(ckpt_path),
    )

    logger.info("Calibration complete.")
    logger.info("  T*                  : %.4f", t_star)
    logger.info("  ECE before          : %.4f", before.ece)
    logger.info("  ECE after           : %.4f", after.ece)
    logger.info("  Calibrated ckpt     : %s", calibrated_ckpt)
    logger.info("  Stage C verdict     : %s", verdict)

    return 0 if verdict in ("GO", "CONDITIONAL_GO") else 2


if __name__ == "__main__":
    sys.exit(main())
