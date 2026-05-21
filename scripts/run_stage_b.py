#!/usr/bin/env python3
"""
Phase 14 — Stage B orchestrator.

Progressive backbone adaptation: unfreezes the top-1 EfficientNet feature
block (features[-1]) on top of the Stage A frozen-backbone checkpoint.

Goal: improve pneumonia sensitivity while preserving OOD robustness,
semantic compatibility, GradCAM quality, and calibration stability.

Pipeline
--------
  1. Load Stage A checkpoint (v6_meta verified)
  2. Train Stage B (top-1 unfreeze, differential LR, 20 epochs)
  3. Full evaluation (per-class, OOD, ECE, GradCAM)
  4. Delta comparison vs Stage A baseline
  5. GradCAM PNG overlays (4 per class)
  6. Pipeline compatibility gates (fusion / calibration / probability)
  7. Go / No-Go decision — OOD regression is an automatic NO-GO

SAFETY: V5 production model is never loaded or modified.
All outputs isolated under models/vision/v6_medical/stage_b_top1/
and reports/v6_medical/stage_b/.

Usage
-----
    python scripts/run_stage_b.py \\
        --stage-a-checkpoint models/vision/v6_medical/stage_a_frozen/best.pt \\
        --dataset-dir data/medical_v6_splits \\
        --device auto

    # Skip training (evaluate existing Stage B checkpoint):
    python scripts/run_stage_b.py \\
        --checkpoint models/vision/v6_medical/stage_b_top1/best.pt \\
        --eval-only \\
        --dataset-dir data/medical_v6_splits

Exit codes: 0 GO, 1 error, 2 NO-GO.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("run_stage_b")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ── Import train_v6_medical internals ─────────────────────────────────────────

def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_v6_medical",
        _PROJECT_ROOT / "scripts" / "train_v6_medical.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_v6_medical"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── GradCAM PNG export (shared with run_stage_a) ──────────────────────────────

def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0.0, 1.0)


def _cam_to_heatmap(cam_np: np.ndarray) -> np.ndarray:
    cam = cam_np.astype(np.float32)
    mn, mx = cam.min(), cam.max()
    cam = (cam - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(cam)
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _save_gradcam_overlays(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    classes: list[str],
    n_per_class: int = 4,
) -> None:
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed — skipping GradCAM PNG export.")
        return

    target_layer = model._backbone.features[-1]
    _activations: dict[str, torch.Tensor] = {}
    _gradients:   dict[str, torch.Tensor] = {}

    fwd_h = target_layer.register_forward_hook(
        lambda _, __, out: _activations.__setitem__("feat", out.detach().clone())
    )
    bwd_h = target_layer.register_full_backward_hook(
        lambda _, __, g: _gradients.__setitem__("feat", g[0].detach().clone())
    )

    saved: dict[int, int] = {i: 0 for i in range(len(classes))}
    total_needed = n_per_class * len(classes)
    total_saved  = 0

    try:
        for images, labels in val_loader:
            if total_saved >= total_needed:
                break
            for img_t, lbl in zip(images, labels):
                cls_idx = int(lbl.item())
                if saved[cls_idx] >= n_per_class:
                    continue

                img = img_t.unsqueeze(0).to(device).clone().requires_grad_(True)
                model.zero_grad()
                logits = model(img)
                pred = int(logits.argmax(1).item())
                logits[0, pred].backward()

                if "feat" not in _activations or "feat" not in _gradients:
                    continue

                alpha  = _gradients["feat"].mean(dim=[2, 3], keepdim=True)
                cam    = F.relu((alpha * _activations["feat"]).sum(1, keepdim=True))
                cam_np = cam.squeeze().cpu().detach().numpy()

                cam_pil   = Image.fromarray(
                    (cam_np / (cam_np.max() + 1e-8) * 255).astype(np.uint8)
                ).resize((224, 224), Image.BILINEAR)
                cam_hw    = np.array(cam_pil) / 255.0
                heatmap   = _cam_to_heatmap(cam_hw)

                orig_np   = (_denormalize(img_t).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                orig_pil  = Image.fromarray(orig_np).resize((224, 224), Image.BILINEAR)
                overlay   = Image.blend(orig_pil, Image.fromarray(heatmap), alpha=0.40)

                cls_name  = classes[cls_idx]
                save_dir  = out_dir / "gradcam" / cls_name
                save_dir.mkdir(parents=True, exist_ok=True)
                overlay.save(save_dir / f"{cls_name}_{saved[cls_idx]:02d}_pred{classes[pred]}.png")

                saved[cls_idx] += 1
                total_saved    += 1
                _activations.clear()
                _gradients.clear()

                if total_saved >= total_needed:
                    break
    finally:
        fwd_h.remove()
        bwd_h.remove()

    logger.info("GradCAM overlays saved: %s  (%d images)", out_dir / "gradcam", total_saved)


# ── v6_meta checkpoint wrapper ───────────────────────────────────────────────

def _ensure_v6_meta(
    checkpoint_path: Path,
    config,
    sub_stage_cfg,
    train_result: dict,
) -> Path:
    """
    If the checkpoint lacks 'v6_meta' (saved by ModelCheckpoint), wrap it.

    ModelCheckpoint only stores {epoch, score, model_state_dict, ...}.
    The v6 safety guard requires v6_meta.  We read the raw state and re-save
    with v6_meta embedded, then return the (same) path.
    """
    from datetime import datetime, timezone

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "v6_meta" in ckpt:
        return checkpoint_path

    meta = {
        "v6_phase":             "14",
        "run_name":             config.run_name,
        "sub_stage":            sub_stage_cfg.sub_stage.value,
        "classes":              list(config.classes),
        "class_to_idx":         {c: i for i, c in enumerate(config.classes)},
        "num_classes":          config.num_classes,
        "architecture":         "efficientnet_b0",
        "epoch":                ckpt.get("epoch", train_result.get("epochs_run", 0)),
        "best_val_f1":          train_result.get("best_val_f1", ckpt.get("score", 0.0)),
        "val_f1":               ckpt.get("score", train_result.get("best_val_f1", 0.0)),
        "val_loss":             0.0,
        "ood_rejection_rate":   0.0,
        "compatibility_passed": None,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "notes":                "v6_meta injected post-training (ModelCheckpoint output)",
    }
    torch.save({"model_state_dict": ckpt["model_state_dict"], "v6_meta": meta}, checkpoint_path)
    logger.info("v6_meta injected into %s", checkpoint_path)
    return checkpoint_path


# ── Delta comparison ──────────────────────────────────────────────────────────

def _load_stage_a_baseline(report_path: Path) -> dict | None:
    if not report_path.exists():
        logger.warning("Stage A report not found: %s — delta comparison skipped.", report_path)
        return None
    try:
        return json.loads(report_path.read_text())["evaluation"]
    except Exception as exc:
        logger.warning("Could not parse Stage A report: %s", exc)
        return None


def _compute_delta(b_val, a_val) -> str:
    delta = b_val - a_val
    sign  = "+" if delta >= 0 else ""
    return f"{sign}{delta:+.4f}"


def _print_delta_table(eval_b, baseline_a: dict | None) -> None:
    if baseline_a is None:
        return

    sep = "─" * 60
    print(f"\n{sep}")
    print("  STAGE B vs STAGE A — DELTA COMPARISON")
    print(sep)

    metrics = [
        ("Macro F1",      eval_b.macro_f1,                      baseline_a["macro_f1"]),
        ("Accuracy",      eval_b.accuracy,                       baseline_a["accuracy"]),
        ("Pos. Recall",   eval_b.compatibility.positive_recall,  baseline_a["compatibility"]["positive_recall"]),
        ("OOD Rejection", eval_b.ood.rejection_recall,           baseline_a["ood"]["rejection_recall"]),
        ("OOD Leakage",   eval_b.ood.leakage_rate,               baseline_a["ood"]["leakage_rate"]),
        ("ECE",           eval_b.calibration.ece,                baseline_a["calibration"]["ece"]),
    ]
    print(f"  {'Metric':<20}  {'Stage A':>10}  {'Stage B':>10}  {'Delta':>10}  Status")
    print("  " + "─" * 58)
    for name, b_val, a_val in metrics:
        delta     = b_val - a_val
        sign      = "+" if delta >= 0 else ""
        direction = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        status    = "OK" if _metric_improved_or_neutral(name, delta) else "⚠"
        print(f"  {name:<20}  {a_val:>10.4f}  {b_val:>10.4f}  {sign}{delta:+.4f} {direction}  {status}")

    # Per-class recall delta
    a_per = {m["class_name"]: m for m in baseline_a["per_class"]}
    print(f"\n  Per-class recall:")
    for m in eval_b.per_class:
        a_m    = a_per.get(m.class_name)
        a_rec  = a_m["recall"] if a_m else 0.0
        delta  = m.recall - a_rec
        sign   = "+" if delta >= 0 else ""
        tag    = "OOD" if m.class_name in ("hard_negative", "fake_medical") else "POS"
        print(f"    [{tag}] {m.class_name:<20}  A={a_rec:.4f}  B={m.recall:.4f}  {sign}{delta:+.4f}")

    print(sep)


def _metric_improved_or_neutral(name: str, delta: float) -> bool:
    """True if the direction of change is beneficial or neutral."""
    lower_is_better = {"ECE", "OOD Leakage"}
    if name in lower_is_better:
        return delta <= 0.01   # allow tiny regressions
    return delta >= -0.02      # allow tiny regressions for other metrics


# ── Go / No-Go decision ───────────────────────────────────────────────────────

def _make_decision(
    eval_result,
    compat_result,
    baseline_a: dict | None,
) -> tuple[str, list[str]]:
    ood_ok       = eval_result.ood.rejection_recall >= 0.90
    f1_go        = eval_result.macro_f1 >= 0.78
    f1_cond      = eval_result.macro_f1 >= 0.70
    pos_ok       = eval_result.compatibility.positive_recall >= 0.80
    ece_ok       = eval_result.calibration.ece <= 0.12
    ece_hard     = eval_result.calibration.ece <= 0.18
    gradcam_ok   = (
        eval_result.gradcam is None
        or eval_result.gradcam.passed
        or eval_result.gradcam.degenerate_fraction < 0.30
    )
    zero_recall  = [
        m.class_name for m in eval_result.per_class
        if m.recall == 0.0 and m.support > 0
    ]

    # OOD regression vs Stage A (automatic NO-GO)
    ood_regression = False
    if baseline_a is not None:
        a_ood = baseline_a["ood"]["rejection_recall"]
        ood_regression = eval_result.ood.rejection_recall < a_ood - 0.05

    # GradCAM degeneration vs Stage A
    gradcam_degen = (
        eval_result.gradcam is not None
        and eval_result.gradcam.degenerate_fraction >= 0.30
    )

    # Hard blockers → NO-GO
    hard_reasons: list[str] = []
    if ood_regression:
        hard_reasons.append(
            f"OOD regression: Stage A={baseline_a['ood']['rejection_recall']:.4f} → "
            f"Stage B={eval_result.ood.rejection_recall:.4f} (Δ > 0.05 — STOP)"
        )
    if not ood_ok:
        hard_reasons.append(f"OOD rejection {eval_result.ood.rejection_recall:.4f} < 0.90 absolute floor")
    if zero_recall:
        hard_reasons.append(f"Zero recall for class(es): {', '.join(zero_recall)}")
    if gradcam_degen:
        hard_reasons.append(
            f"GradCAM degeneration: {eval_result.gradcam.degenerate_fraction:.2f} ≥ 0.30 "
            "(backbone fine-tuning degraded spatial attention)"
        )
    if not ece_hard:
        hard_reasons.append(f"ECE {eval_result.calibration.ece:.4f} > 0.18 — calibration collapsed")
    if hard_reasons:
        return "NO_GO", hard_reasons

    # Full GO
    if ood_ok and f1_go and pos_ok and ece_ok and gradcam_ok:
        notes = []
        if compat_result is not None and not compat_result.overall_pass:
            notes.append("pipeline compat gates failed — review before Stage C")
        return "GO", notes

    # Conditional GO
    notes = []
    if not f1_go:
        notes.append(f"macro F1 {eval_result.macro_f1:.4f} in conditional range [0.70, 0.78)")
    if not pos_ok:
        notes.append(f"positive recall {eval_result.compatibility.positive_recall:.4f} < 0.80")
    if not ece_ok:
        notes.append(f"ECE {eval_result.calibration.ece:.4f} in [0.12, 0.18]")
    if eval_result.gradcam and not eval_result.gradcam.passed:
        notes.append(f"GradCAM degraded: entropy={eval_result.gradcam.mean_entropy:.4f}")
    if compat_result is not None and not compat_result.overall_pass:
        notes.append("pipeline compat gates failed — review before Stage C")

    if f1_cond:
        return "CONDITIONAL_GO", notes

    return "NO_GO", [f"macro F1 {eval_result.macro_f1:.4f} < 0.70 minimum"]


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(eval_result, compat_v6, verdict: str, reasons: list[str]) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  STAGE B — EVALUATION SUMMARY")
    print(sep)
    print(f"  Samples        : {eval_result.num_samples}")
    print(f"  Accuracy       : {eval_result.accuracy:.4f}")
    print(f"  Macro F1       : {eval_result.macro_f1:.4f}")
    print(f"  Macro Recall   : {eval_result.macro_recall:.4f}")
    print(f"  Macro Precision: {eval_result.macro_precision:.4f}")
    print(f"\n  OOD Rejection  : {eval_result.ood.rejection_recall:.4f}  "
          f"(leakage={eval_result.ood.leakage_rate:.4f})")
    print(f"  Pos. Recall    : {eval_result.compatibility.positive_recall:.4f}")
    print(f"  ECE            : {eval_result.calibration.ece:.4f}  "
          f"(MCE={eval_result.calibration.mce:.4f})")

    if eval_result.gradcam:
        gc = eval_result.gradcam
        print(f"  GradCAM        : entropy={gc.mean_entropy:.4f}  "
              f"degenerate={gc.degenerate_fraction:.4f}  "
              f"{'PASS' if gc.passed else 'FAIL'}")

    print(f"\n  Per-class metrics:")
    for m in eval_result.per_class:
        tag = "OOD" if m.class_name in ("hard_negative", "fake_medical") else "POS"
        print(f"    [{tag}] {m.class_name:<20} "
              f"recall={m.recall:.4f}  prec={m.precision:.4f}  "
              f"F1={m.f1:.4f}  n={m.support}")

    print(f"\n  Confusion matrix ({' / '.join(eval_result.confusion_matrix_labels)}):")
    for row in eval_result.confusion_matrix:
        print(f"    {row}")

    print(f"\n  Pipeline compat (fusion/calib/prob):")
    if compat_v6 is not None:
        for gate, ok in compat_v6.gate_results.items():
            print(f"    {'✓' if ok else '✗'} {gate}")

    print(f"\n{sep}")
    verdict_label = {
        "GO":             "GO  — proceed to Stage C if desired",
        "CONDITIONAL_GO": "CONDITIONAL GO  — review notes before Stage C",
        "NO_GO":          "NO-GO  — do NOT proceed to Stage C",
    }[verdict]
    icon = "✓" if verdict == "GO" else ("⚠" if verdict == "CONDITIONAL_GO" else "✗")
    print(f"  {icon}  {verdict_label}")
    if reasons:
        for r in reasons:
            print(f"      • {r}")
    print(sep + "\n")


# ── Report writer ─────────────────────────────────────────────────────────────

def _write_report(
    report_path: Path,
    eval_result,
    compat_v6,
    verdict: str,
    reasons: list[str],
    train_result: dict | None,
    baseline_a: dict | None,
) -> None:
    from app.modules.vision.medical.v6_evaluation import _to_serializable

    def _compat_v6_dict(c):
        if c is None:
            return None
        return {
            "overall_pass": c.overall_pass,
            "gate_results": c.gate_results,
            "fusion":      asdict(c.fusion),
            "calibration": asdict(c.calibration),
            "probability": asdict(c.probability),
        }

    # Per-class delta vs Stage A
    delta_table = None
    if baseline_a is not None:
        a_per = {m["class_name"]: m for m in baseline_a["per_class"]}
        delta_table = {}
        for m in eval_result.per_class:
            a_m = a_per.get(m.class_name, {})
            delta_table[m.class_name] = {
                "recall_a":    a_m.get("recall", 0.0),
                "recall_b":    m.recall,
                "recall_delta": round(m.recall - a_m.get("recall", 0.0), 4),
                "f1_a":        a_m.get("f1", 0.0),
                "f1_b":        m.f1,
                "f1_delta":    round(m.f1 - a_m.get("f1", 0.0), 4),
            }

    data = {
        "phase":    "14",
        "stage":    "stage_b_top1",
        "verdict":  verdict,
        "verdict_reasons":       reasons,
        "evaluation":            _to_serializable(eval_result),
        "pipeline_compatibility": _compat_v6_dict(compat_v6),
        "training":              train_result,
        "stage_a_baseline":      baseline_a,
        "per_class_delta":       delta_table,
        "summary_delta": {
            "macro_f1_delta":    round(eval_result.macro_f1     - (baseline_a or {}).get("macro_f1", 0.0), 4),
            "ood_delta":         round(eval_result.ood.rejection_recall - (baseline_a or {}).get("ood", {}).get("rejection_recall", 0.0), 4),
            "ece_delta":         round(eval_result.calibration.ece      - (baseline_a or {}).get("calibration", {}).get("ece", 0.0), 4),
            "pos_recall_delta":  round(eval_result.compatibility.positive_recall - (baseline_a or {}).get("compatibility", {}).get("positive_recall", 0.0), 4),
        } if baseline_a else {},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Stage B report written: %s", report_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_stage_b",
        description="Phase 14 — Stage B training + evaluation + Go/No-Go.",
    )
    p.add_argument(
        "--stage-a-checkpoint", type=Path,
        default=Path("models/vision/v6_medical/stage_a_frozen/best.pt"),
        help="Stage A v6 checkpoint to warm-start from (default: stage_a_frozen/best.pt).",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Existing Stage B checkpoint to skip training (implies --eval-only).",
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=Path("data/medical_v6_splits"),
        help="Path to staged v6 splits (default: data/medical_v6_splits).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("models/vision/v6_medical"),
        help="Root for v6 model outputs (default: models/vision/v6_medical).",
    )
    p.add_argument(
        "--report-dir", type=Path, default=Path("reports/v6_medical/stage_b"),
        help="Directory for evaluation reports (default: reports/v6_medical/stage_b).",
    )
    p.add_argument(
        "--stage-a-report", type=Path,
        default=Path("reports/v6_medical/stage_a/stage_a_report.json"),
        help="Stage A report for delta comparison.",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: cuda / mps / cpu / auto (default).",
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; requires --checkpoint.",
    )
    p.add_argument(
        "--no-gradcam", action="store_true",
        help="Skip GradCAM export (faster).",
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
    device = _resolve_device(args.device)

    if args.checkpoint is not None:
        args.eval_only = True

    logger.info("=== Phase 14 — Stage B Orchestrator ===")
    logger.info("Dataset    : %s", args.dataset_dir)
    logger.info("Stage A ck : %s", args.stage_a_checkpoint)
    logger.info("Output     : %s", args.output_dir)
    logger.info("Reports    : %s", args.report_dir)
    logger.info("Device     : %s", device)
    logger.info("Eval-only  : %s", args.eval_only)

    # ── Imports ────────────────────────────────────────────────────────────────
    from app.modules.vision.medical.v6_training_config import (
        V6SubStage, V6_BINARY_MEDICAL_CONFIG, V6TrainingConfig,
    )
    from app.modules.vision.medical.v6_evaluation import (
        V6Evaluator,
        export_evaluation_report,
        export_compatibility_report,
    )
    from app.modules.vision.medical.v6_compatibility import V6CompatibilityChecker
    from app.modules.vision.preprocessing.transforms import get_val_transforms
    from app.modules.vision.datasets.dataset import ImageFolderDataset

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

    sub_stage_cfg = config.get_sub_stage(V6SubStage.B_TOP1)
    stage_dir     = config.stage_output_dir(sub_stage_cfg)
    report_dir    = args.report_dir.resolve()

    train_mod = _load_train_module()

    # ── Val loader ─────────────────────────────────────────────────────────────
    if not args.dataset_dir.exists():
        logger.error("Dataset directory not found: %s", args.dataset_dir)
        return 1

    val_ds = ImageFolderDataset(
        root_dir=args.dataset_dir,
        split="val",
        transform=get_val_transforms(),
        classes=list(config.classes),
    )
    if not val_ds.samples:
        logger.error("No val samples found in %s/val/", args.dataset_dir)
        return 1

    val_loader = DataLoader(
        val_ds,
        batch_size=sub_stage_cfg.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    logger.info("Val dataset: %d samples", len(val_ds))

    # ── Stage A baseline ───────────────────────────────────────────────────────
    baseline_a = _load_stage_a_baseline(args.stage_a_report)

    # ── Step 1: Training ───────────────────────────────────────────────────────
    train_result: dict | None = None

    if args.eval_only:
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            logger.error("--eval-only requires --checkpoint.")
            return 1
        logger.info("Eval-only mode — checkpoint: %s", checkpoint_path)
    else:
        if not args.stage_a_checkpoint.exists():
            logger.error("Stage A checkpoint not found: %s", args.stage_a_checkpoint)
            return 1

        logger.info("\n%s", "=" * 60)
        logger.info("STEP 1: Training Stage B (top-1 backbone unfreeze)")
        logger.info("  LR backbone=%.2e  head=%.2e  epochs=%d",
                    sub_stage_cfg.learning_rate,
                    sub_stage_cfg.learning_rate * sub_stage_cfg.head_lr_factor,
                    sub_stage_cfg.epochs)
        logger.info("=" * 60)

        model = train_mod._build_or_load_model(args.stage_a_checkpoint, config, device)
        train_loader, _ = train_mod._build_dataloaders(
            dataset_dir=args.dataset_dir,
            sub_stage_cfg=sub_stage_cfg,
            config=config,
            replay_dir=None,
        )
        train_result = train_mod._train_sub_stage(
            model=model,
            sub_stage_cfg=sub_stage_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            stage_output_dir=stage_dir,
        )
        checkpoint_path = Path(train_result["best_checkpoint"])
        _ensure_v6_meta(checkpoint_path, config, sub_stage_cfg, train_result)
        logger.info(
            "Training complete | best_val_f1=%.4f | checkpoint=%s",
            train_result["best_val_f1"], checkpoint_path,
        )

    # ── Step 2: Load checkpoint ────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 2: Loading checkpoint for evaluation")
    logger.info("=" * 60)

    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return 1

    model = train_mod._build_or_load_model(checkpoint_path, config, device)
    model.eval()

    # ── Step 3: Full evaluation ────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 3: Full evaluation")
    logger.info("=" * 60)

    evaluator   = V6Evaluator(config)
    eval_result = evaluator.evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        split="val",
        run_gradcam=not args.no_gradcam,
        checkpoint_path=str(checkpoint_path),
    )
    evaluator.fill_positive_recall(eval_result.compatibility, eval_result.per_class)

    export_evaluation_report(eval_result, report_dir / "evaluation_val.json")
    export_compatibility_report(eval_result.compatibility, report_dir / "compatibility_eval.json")

    # ── Step 4: GradCAM overlays ───────────────────────────────────────────────
    if not args.no_gradcam:
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 4: GradCAM PNG overlays")
        logger.info("=" * 60)
        _save_gradcam_overlays(
            model=model,
            val_loader=val_loader,
            device=device,
            out_dir=report_dir,
            classes=list(config.classes),
            n_per_class=4,
        )

    # ── Step 5: Pipeline compatibility ────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 5: Pipeline compatibility (fusion / calibration / probability)")
    logger.info("=" * 60)

    compat_v6 = None
    try:
        checker   = V6CompatibilityChecker(config)
        compat_v6 = checker.check_all(
            model=model,
            val_loader=val_loader,
            device=device,
            max_samples=300,
        )
        compat_v6_path = report_dir / "pipeline_compat.json"
        compat_v6_path.parent.mkdir(parents=True, exist_ok=True)
        compat_v6_path.write_text(
            json.dumps({
                "overall_pass": compat_v6.overall_pass,
                "gate_results": compat_v6.gate_results,
                "fusion": {
                    "passed":                 compat_v6.fusion.passed,
                    "exception_rate":         compat_v6.fusion.exception_rate,
                    "mean_delta":             compat_v6.fusion.mean_delta,
                    "max_delta":              compat_v6.fusion.max_delta,
                    "alignment_distribution": compat_v6.fusion.alignment_distribution,
                    "notes":                  compat_v6.fusion.notes,
                },
                "calibration": {
                    "passed":                         compat_v6.calibration.passed,
                    "trust_tier_distribution":        compat_v6.calibration.trust_tier_distribution,
                    "mean_trust_score":               compat_v6.calibration.mean_trust_score,
                    "calibration_state_distribution": compat_v6.calibration.calibration_state_distribution,
                    "fraction_suspicious":            compat_v6.calibration.fraction_suspicious,
                    "notes":                          compat_v6.calibration.notes,
                },
                "probability": {
                    "passed":                   compat_v6.probability.passed,
                    "mean_positive_class_prob": compat_v6.probability.mean_positive_class_prob,
                    "mean_ood_rejection_prob":  compat_v6.probability.mean_ood_rejection_prob,
                    "fraction_correct":         compat_v6.probability.fraction_correct,
                    "catastrophic_class_recall":compat_v6.probability.catastrophic_class_recall,
                    "notes":                    compat_v6.probability.notes,
                },
            }, indent=2),
            encoding="utf-8",
        )
        logger.info("Pipeline compat report: %s", compat_v6_path)
    except Exception:
        logger.exception("Pipeline compatibility check raised an exception.")

    # ── Step 6: Go / No-Go ────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 6: Go / No-Go decision")
    logger.info("=" * 60)

    verdict, reasons = _make_decision(eval_result, compat_v6, baseline_a)

    # ── Step 7: Write reports ──────────────────────────────────────────────────
    _write_report(
        report_path=report_dir / "stage_b_report.json",
        eval_result=eval_result,
        compat_v6=compat_v6,
        verdict=verdict,
        reasons=reasons,
        train_result=train_result,
        baseline_a=baseline_a,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(eval_result, compat_v6, verdict, reasons)
    _print_delta_table(eval_result, baseline_a)

    return 2 if verdict == "NO_GO" else 0


if __name__ == "__main__":
    sys.exit(main())
