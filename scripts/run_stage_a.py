#!/usr/bin/env python3
"""
Phase 12 — Stage A orchestrator.

Runs the complete Stage A (frozen backbone) training and evaluation pipeline:
  1. Train Stage A  — frozen backbone, head-only, warm-start
  2. Full evaluation — confusion matrix, macro F1, per-class recall, OOD recall, ECE
  3. GradCAM PNG overlays — 4 samples per class saved to reports/
  4. V6 compatibility gates — fusion / calibration / probability sanity
  5. Go / No-Go decision — explicit gate with printed verdict

SAFETY: V5 production model is never loaded or modified.
All outputs are isolated under models/vision/v6_medical/stage_a_frozen/
and reports/v6_medical/stage_a/.

Usage
-----
    python scripts/run_stage_a.py \\
        --dataset-dir data/medical_v6_splits \\
        --device auto

    # Skip training (evaluate an existing checkpoint):
    python scripts/run_stage_a.py \\
        --dataset-dir data/medical_v6_splits \\
        --checkpoint models/vision/v6_medical/stage_a_frozen/best.pt \\
        --eval-only

Exit codes: 0 GO, 1 error, 2 NO-GO (stability gate failed).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("run_stage_a")


# ── Lazy imports (heavy deps deferred until needed) ───────────────────────────

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ── Load train_v6_medical internals via importlib ─────────────────────────────

def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_v6_medical",
        _PROJECT_ROOT / "scripts" / "train_v6_medical.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── GradCAM PNG export ────────────────────────────────────────────────────────

def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0.0, 1.0)


def _cam_to_heatmap(cam_np: np.ndarray) -> np.ndarray:
    """Jet-ish colormap (pure numpy, no matplotlib dependency)."""
    cam = cam_np.astype(np.float32)
    mn, mx = cam.min(), cam.max()
    if mx - mn > 1e-8:
        cam = (cam - mn) / (mx - mn)
    else:
        cam = np.zeros_like(cam)
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
    """Save GradCAM overlay PNGs per class to out_dir/gradcam/<class>/."""
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

                alpha = _gradients["feat"].mean(dim=[2, 3], keepdim=True)
                cam   = F.relu((alpha * _activations["feat"]).sum(1, keepdim=True))
                cam_np = cam.squeeze().cpu().detach().numpy()

                # Resize cam to image spatial size (224×224)
                cam_h, cam_w = cam_np.shape
                img_np_hw = (
                    (np.array(
                        Image.fromarray(
                            (cam_np / (cam_np.max() + 1e-8) * 255).astype(np.uint8)
                        ).resize((224, 224), Image.BILINEAR)
                    ) / 255.0)
                )
                heatmap = _cam_to_heatmap(img_np_hw)  # (224, 224, 3)

                # Denorm original image → (H, W, 3) uint8
                orig_np = (_denormalize(img_t).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                orig_pil = Image.fromarray(orig_np).resize((224, 224), Image.BILINEAR)
                heat_pil = Image.fromarray(heatmap)

                # Overlay: 60% original + 40% heatmap
                overlay = Image.blend(orig_pil, heat_pil, alpha=0.40)

                # Save
                cls_name = classes[cls_idx]
                save_dir = out_dir / "gradcam" / cls_name
                save_dir.mkdir(parents=True, exist_ok=True)
                n = saved[cls_idx]
                fname = f"{cls_name}_{n:02d}_pred{classes[pred]}.png"
                overlay.save(save_dir / fname)

                saved[cls_idx] += 1
                total_saved += 1
                _activations.clear()
                _gradients.clear()

                if total_saved >= total_needed:
                    break
    finally:
        fwd_h.remove()
        bwd_h.remove()

    logger.info("GradCAM overlays saved to %s  (%d images)", out_dir / "gradcam", total_saved)


# ── v6_meta checkpoint wrapper ───────────────────────────────────────────────

def _ensure_v6_meta(
    checkpoint_path: Path,
    config,
    sub_stage_cfg,
    train_result: dict,
) -> Path:
    """Inject v6_meta into a ModelCheckpoint output if it's missing."""
    from datetime import datetime, timezone

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "v6_meta" in ckpt:
        return checkpoint_path

    meta = {
        "v6_phase":             "12",
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


# ── Go / No-Go decision ───────────────────────────────────────────────────────

def _make_decision(
    eval_result,
    compat_result,
) -> tuple[str, list[str]]:
    """
    Returns (verdict, reasons).
    verdict: "GO" | "CONDITIONAL_GO" | "NO_GO"
    """
    from app.modules.vision.medical.v6_evaluation import V6EvaluationResult

    ood_ok      = eval_result.ood.rejection_recall >= 0.90
    f1_go       = eval_result.macro_f1 >= 0.70
    f1_cond     = eval_result.macro_f1 >= 0.60
    pos_ok      = eval_result.compatibility.positive_recall >= 0.75
    ece_ok      = eval_result.calibration.ece <= 0.15
    ece_hard    = eval_result.calibration.ece <= 0.20
    gradcam_ok  = (
        eval_result.gradcam is None
        or eval_result.gradcam.passed
        or eval_result.gradcam.degenerate_fraction < 0.90
    )
    # Only flag zero recall for classes that have val samples (support > 0).
    # Classes absent from the staging run (e.g. fake_medical not yet collected)
    # have support=0 and cannot be blamed for zero recall.
    zero_recall = [
        m.class_name for m in eval_result.per_class
        if m.recall == 0.0 and m.support > 0
    ]
    compat_ok   = compat_result.overall_pass

    reasons: list[str] = []

    # Hard blockers → NO-GO
    if zero_recall:
        return "NO_GO", [f"Zero recall for class(es): {', '.join(zero_recall)}"]
    if not ood_ok:
        reasons.append(f"OOD rejection recall too low: {eval_result.ood.rejection_recall:.3f} < 0.90")
    if not ece_hard:
        reasons.append(f"ECE too high: {eval_result.calibration.ece:.4f} > 0.20")
    if eval_result.gradcam and eval_result.gradcam.degenerate_fraction >= 0.90:
        reasons.append(
            f"GradCAM fully degenerate: {eval_result.gradcam.degenerate_fraction:.2f}"
        )
    if reasons:
        return "NO_GO", reasons

    # Full GO
    if ood_ok and f1_go and pos_ok and ece_ok and gradcam_ok:
        notes = []
        if not compat_ok:
            notes.append("pipeline compat gates failed — review before Stage B")
        return "GO", notes

    # Conditional GO (weakened thresholds)
    notes = []
    if not f1_go:
        notes.append(f"macro F1 {eval_result.macro_f1:.3f} in conditional range [0.60, 0.70)")
    if not pos_ok:
        notes.append(f"positive recall {eval_result.compatibility.positive_recall:.3f} < 0.75")
    if not ece_ok:
        notes.append(f"ECE {eval_result.calibration.ece:.4f} in [0.15, 0.20]")
    if not gradcam_ok and eval_result.gradcam:
        notes.append(f"GradCAM degraded: entropy={eval_result.gradcam.mean_entropy:.3f}")
    if not compat_ok:
        notes.append("pipeline compat gates failed — review before Stage B")

    if f1_cond:
        return "CONDITIONAL_GO", notes

    return "NO_GO", [
        f"macro F1 {eval_result.macro_f1:.3f} < 0.60 minimum"
    ]


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(eval_result, compat_v6, verdict: str, reasons: list[str]) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("  STAGE A — EVALUATION SUMMARY")
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
        "GO":             "GO  — proceed to Stage B",
        "CONDITIONAL_GO": "CONDITIONAL GO  — review notes before Stage B",
        "NO_GO":          "NO-GO  — do NOT proceed to Stage B",
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
) -> None:
    from app.modules.vision.medical.v6_evaluation import _to_serializable

    def _compat_v6_dict(c):
        if c is None:
            return None
        return {
            "overall_pass": c.overall_pass,
            "gate_results": c.gate_results,
            "fusion": asdict(c.fusion),
            "calibration": asdict(c.calibration),
            "probability": asdict(c.probability),
        }

    data = {
        "phase": "12",
        "stage": "stage_a_frozen",
        "verdict": verdict,
        "verdict_reasons": reasons,
        "evaluation": _to_serializable(eval_result),
        "pipeline_compatibility": _compat_v6_dict(compat_v6),
        "training": train_result,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Stage A report written: %s", report_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_stage_a",
        description="Phase 12 — Stage A training + evaluation + Go/No-Go.",
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=Path("data/medical_v6_splits"),
        help="Path to data/medical_v6_splits/ (default: data/medical_v6_splits).",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Existing v6 Stage A checkpoint to skip training.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("models/vision/v6_medical"),
        help="Root for v6 model outputs (default: models/vision/v6_medical).",
    )
    p.add_argument(
        "--report-dir", type=Path, default=Path("reports/v6_medical/stage_a"),
        help="Directory for evaluation reports (default: reports/v6_medical/stage_a).",
    )
    p.add_argument(
        "--replay-dir", type=Path, default=None,
        help="Optional dir with extra v5 hard_negative images for replay buffer.",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: cuda / mps / cpu / auto (default).",
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; evaluate an existing checkpoint (--checkpoint required).",
    )
    p.add_argument(
        "--no-gradcam", action="store_true",
        help="Skip GradCAM sanity + PNG export (faster, useful for testing).",
    )
    p.add_argument(
        "--max-eval-samples", type=int, default=None,
        help="Limit val samples for compatibility checker (default: all).",
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

    logger.info("=== Phase 12 — Stage A Orchestrator ===")
    logger.info("Dataset  : %s", args.dataset_dir)
    logger.info("Output   : %s", args.output_dir)
    logger.info("Reports  : %s", args.report_dir)
    logger.info("Device   : %s", device)
    logger.info("Eval-only: %s", args.eval_only)

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

    sub_stage_cfg = config.get_sub_stage(V6SubStage.A_FROZEN)
    stage_dir     = config.stage_output_dir(sub_stage_cfg)   # models/vision/v6_medical/stage_a_frozen/
    report_dir    = args.report_dir.resolve()

    # ── Load training internals ────────────────────────────────────────────────
    train_mod = _load_train_module()

    # ── Val dataloader (always needed) ────────────────────────────────────────
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

    # ── Step 1: Training ───────────────────────────────────────────────────────
    train_result: dict | None = None

    if args.eval_only:
        if args.checkpoint is None:
            logger.error("--eval-only requires --checkpoint.")
            return 1
        checkpoint_path = args.checkpoint
        logger.info("Eval-only mode — loading checkpoint: %s", checkpoint_path)
    else:
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 1: Training Stage A (frozen backbone)")
        logger.info("=" * 60)
        model = train_mod._build_or_load_model(None, config, device)
        train_loader, _ = train_mod._build_dataloaders(
            dataset_dir=args.dataset_dir,
            sub_stage_cfg=sub_stage_cfg,
            config=config,
            replay_dir=args.replay_dir,
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

    # ── Step 2: Load best checkpoint for evaluation ────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 2: Loading best checkpoint for evaluation")
    logger.info("=" * 60)

    if not checkpoint_path.exists():
        logger.error("Best checkpoint not found: %s", checkpoint_path)
        return 1

    model = train_mod._build_or_load_model(checkpoint_path, config, device)
    model.eval()

    # ── Step 3: Full evaluation ────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 3: Full evaluation")
    logger.info("=" * 60)

    evaluator = V6Evaluator(config)
    eval_result = evaluator.evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        split="val",
        run_gradcam=not args.no_gradcam,
        checkpoint_path=str(checkpoint_path),
    )
    evaluator.fill_positive_recall(eval_result.compatibility, eval_result.per_class)

    eval_report_path = report_dir / "evaluation_val.json"
    export_evaluation_report(eval_result, eval_report_path)

    compat_report_path = report_dir / "compatibility_eval.json"
    export_compatibility_report(eval_result.compatibility, compat_report_path)

    # ── Step 4: GradCAM PNG overlays ──────────────────────────────────────────
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

    # ── Step 5: Pipeline compatibility check ──────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 5: Pipeline compatibility (fusion / calibration / probability)")
    logger.info("=" * 60)

    compat_v6 = None
    try:
        checker = V6CompatibilityChecker(config)
        max_s = args.max_eval_samples or len(val_ds)
        compat_v6 = checker.check_all(
            model=model,
            val_loader=val_loader,
            device=device,
            max_samples=min(max_s, 300),
        )
        compat_v6_path = report_dir / "pipeline_compat.json"
        compat_v6_path.parent.mkdir(parents=True, exist_ok=True)
        compat_v6_path.write_text(
            json.dumps(
                {
                    "overall_pass": compat_v6.overall_pass,
                    "gate_results": compat_v6.gate_results,
                    "fusion": {
                        "passed": compat_v6.fusion.passed,
                        "exception_rate": compat_v6.fusion.exception_rate,
                        "mean_delta": compat_v6.fusion.mean_delta,
                        "max_delta": compat_v6.fusion.max_delta,
                        "alignment_distribution": compat_v6.fusion.alignment_distribution,
                        "notes": compat_v6.fusion.notes,
                    },
                    "calibration": {
                        "passed": compat_v6.calibration.passed,
                        "trust_tier_distribution": compat_v6.calibration.trust_tier_distribution,
                        "mean_trust_score": compat_v6.calibration.mean_trust_score,
                        "calibration_state_distribution": compat_v6.calibration.calibration_state_distribution,
                        "fraction_suspicious": compat_v6.calibration.fraction_suspicious,
                        "notes": compat_v6.calibration.notes,
                    },
                    "probability": {
                        "passed": compat_v6.probability.passed,
                        "mean_positive_class_prob": compat_v6.probability.mean_positive_class_prob,
                        "mean_ood_rejection_prob": compat_v6.probability.mean_ood_rejection_prob,
                        "fraction_correct": compat_v6.probability.fraction_correct,
                        "catastrophic_class_recall": compat_v6.probability.catastrophic_class_recall,
                        "notes": compat_v6.probability.notes,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Pipeline compat report: %s", compat_v6_path)
    except Exception:
        logger.exception("Pipeline compatibility check raised an exception — results excluded.")

    # ── Step 6: Go / No-Go decision ───────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 6: Go / No-Go decision")
    logger.info("=" * 60)

    verdict, reasons = _make_decision(eval_result, compat_v6)

    # ── Step 7: Write master report ───────────────────────────────────────────
    _write_report(
        report_path=report_dir / "stage_a_report.json",
        eval_result=eval_result,
        compat_v6=compat_v6,
        verdict=verdict,
        reasons=reasons,
        train_result=train_result,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(eval_result, compat_v6, verdict, reasons)

    # Exit code
    if verdict == "NO_GO":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
