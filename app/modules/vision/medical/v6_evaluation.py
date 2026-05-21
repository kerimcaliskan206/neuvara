"""
V6 medical evaluation engine — Phase 10.

Evaluation metrics:
  - macro F1, per-class precision/recall/F1
  - OOD rejection recall + leakage rate
  - Expected Calibration Error (ECE) + MCE
  - confusion matrix
  - GradCAM activation sanity check
  - compatibility report (pass/fail per gate)

All evaluation is isolated: reads only the v6 checkpoint.
V5 production model is not loaded or modified.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from app.modules.vision.medical.v6_training_config import (
    V6CompatibilityConfig,
    V6TrainingConfig,
    V6_OOD_CLASSES,
    V6_POSITIVE_CLASSES,
)

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class OODMetrics:
    rejection_recall: float       # fraction of OOD samples correctly rejected
    rejection_precision: float    # of all rejected predictions, fraction truly OOD
    leakage_rate: float           # fraction of OOD samples predicted as positive
    per_ood_class: dict[str, float] = field(default_factory=dict)  # per-class recall


@dataclass
class CalibrationMetrics:
    ece: float                    # Expected Calibration Error
    mce: float                    # Maximum Calibration Error (worst bin)
    mean_confidence: float
    mean_accuracy: float
    overconfidence: float         # mean_confidence - mean_accuracy (>0 = overconfident)
    n_bins: int = 15


@dataclass
class PerClassMetrics:
    class_name: str
    recall: float
    precision: float
    f1: float
    support: int


@dataclass
class GradCAMSanity:
    passed: bool
    mean_entropy: float           # normalized spatial entropy (higher = better distributed)
    degenerate_fraction: float    # fraction of samples with near-zero activation entropy
    n_samples_checked: int
    notes: str


@dataclass
class CompatibilityReport:
    ood_rejection_rate: float
    ood_rejection_ok: bool

    positive_recall: float
    positive_recall_ok: bool

    ece: float
    calibration_ok: bool

    gradcam_passed: bool | None
    gradcam_ok: bool

    semantic_conflict_rate: float | None = None
    semantic_conflict_ok: bool | None = None
    semantic_conflict_checked: bool = False

    overall_pass: bool = False
    gate_results: dict[str, bool] = field(default_factory=dict)

    def compute_overall(self) -> None:
        gates = [
            self.ood_rejection_ok,
            self.positive_recall_ok,
            self.calibration_ok,
            self.gradcam_ok,
        ]
        if self.semantic_conflict_ok is not None:
            gates.append(self.semantic_conflict_ok)
        self.overall_pass = all(gates)
        self.gate_results = {
            "ood_rejection":   self.ood_rejection_ok,
            "positive_recall": self.positive_recall_ok,
            "calibration_ece": self.calibration_ok,
            "gradcam_sanity":  self.gradcam_ok,
        }
        if self.semantic_conflict_ok is not None:
            self.gate_results["semantic_conflict"] = self.semantic_conflict_ok


@dataclass
class V6EvaluationResult:
    split: str
    num_samples: int
    classes: list[str]
    class_to_idx: dict[str, int]

    macro_f1: float
    macro_recall: float
    macro_precision: float
    accuracy: float

    per_class: list[PerClassMetrics]
    ood: OODMetrics
    calibration: CalibrationMetrics
    gradcam: GradCAMSanity | None
    compatibility: CompatibilityReport

    confusion_matrix: list[list[int]]
    confusion_matrix_labels: list[str]

    elapsed_s: float = 0.0
    checkpoint_path: str = ""
    notes: str = ""


# ── Evaluator ─────────────────────────────────────────────────────────────────


class V6Evaluator:
    """
    Runs all v6 medical evaluation metrics on a model + dataloader pair.

    The evaluator never modifies the model; all passes are under torch.no_grad()
    except the GradCAM sanity check (which requires a backward pass on a small sample).
    """

    def __init__(
        self,
        config: V6TrainingConfig,
        compat_config: V6CompatibilityConfig | None = None,
    ) -> None:
        self.config = config
        self.compat = compat_config or config.compatibility
        self.class_to_idx = config.class_to_idx()
        self.idx_to_class = config.idx_to_class()
        self.classes = list(config.classes)
        self.ood_indices = frozenset(
            self.class_to_idx[c]
            for c in config.ood_classes
            if c in self.class_to_idx
        )
        self.pos_indices = frozenset(
            self.class_to_idx[c]
            for c in config.positive_classes
            if c in self.class_to_idx
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def evaluate(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        split: str = "val",
        run_gradcam: bool = True,
        checkpoint_path: str = "",
    ) -> V6EvaluationResult:
        t0 = time.time()
        logger.info("V6Evaluator: evaluating split='%s' on %s", split, device)

        all_preds, all_labels, all_probs = self._collect_predictions(
            model, dataloader, device
        )

        per_class = self._per_class_metrics(all_labels, all_preds)
        ood       = self._ood_metrics(all_labels, all_preds)
        calib     = self._calibration_metrics(all_labels, all_probs)
        cm, cm_labels = self._confusion_matrix(all_labels, all_preds)

        gradcam: GradCAMSanity | None = None
        if run_gradcam and self.compat.check_gradcam_sanity:
            try:
                gradcam = self._gradcam_sanity(model, dataloader, device)
            except Exception as exc:
                logger.warning("GradCAM sanity check failed: %s", exc)
                gradcam = GradCAMSanity(
                    passed=False, mean_entropy=0.0, degenerate_fraction=1.0,
                    n_samples_checked=0, notes=str(exc),
                )

        n = len(all_labels)
        accuracy = float((all_preds == all_labels).sum() / n) if n > 0 else 0.0

        # Only average over classes that actually appear in the eval split.
        # Classes with zero support (e.g. fake_medical when no data was staged)
        # would otherwise collapse macro_f1 to ~0 via the zero_division default.
        present_labels = sorted(np.unique(all_labels).tolist())
        macro_f1 = float(f1_score(
            all_labels, all_preds,
            labels=present_labels, average="macro", zero_division=0,
        ))
        p_arr, r_arr, _, _ = precision_recall_fscore_support(
            all_labels, all_preds,
            labels=present_labels, average="macro", zero_division=0,
        )
        macro_recall    = float(r_arr)
        macro_precision = float(p_arr)

        compat = self._compatibility_report(ood, calib, gradcam)

        elapsed = time.time() - t0
        logger.info(
            "V6Evaluator: macro_f1=%.4f ood_rejection=%.4f ece=%.4f compat=%s (%.1fs)",
            macro_f1, ood.rejection_recall, calib.ece, compat.overall_pass, elapsed,
        )

        return V6EvaluationResult(
            split=split,
            num_samples=n,
            classes=self.classes,
            class_to_idx=self.class_to_idx,
            macro_f1=macro_f1,
            macro_recall=macro_recall,
            macro_precision=macro_precision,
            accuracy=accuracy,
            per_class=per_class,
            ood=ood,
            calibration=calib,
            gradcam=gradcam,
            compatibility=compat,
            confusion_matrix=cm,
            confusion_matrix_labels=cm_labels,
            elapsed_s=round(elapsed, 2),
            checkpoint_path=checkpoint_path,
        )

    # ── Prediction collection ─────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_predictions(
        self, model: nn.Module, dataloader, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = F.softmax(logits, dim=1).cpu()
            preds  = probs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.numpy())

        return (
            np.array(all_preds, dtype=np.int64),
            np.array(all_labels, dtype=np.int64),
            np.vstack(all_probs) if all_probs else np.empty((0, len(self.classes))),
        )

    # ── Per-class metrics ─────────────────────────────────────────────────────

    def _per_class_metrics(
        self, labels: np.ndarray, preds: np.ndarray
    ) -> list[PerClassMetrics]:
        p_arr, r_arr, f_arr, s_arr = precision_recall_fscore_support(
            labels, preds,
            labels=list(range(len(self.classes))),
            average=None,
            zero_division=0,
        )
        return [
            PerClassMetrics(
                class_name=self.idx_to_class[i],
                recall=round(float(r_arr[i]), 4),
                precision=round(float(p_arr[i]), 4),
                f1=round(float(f_arr[i]), 4),
                support=int(s_arr[i]),
            )
            for i in range(len(self.classes))
        ]

    # ── OOD metrics ───────────────────────────────────────────────────────────

    def _ood_metrics(
        self, labels: np.ndarray, preds: np.ndarray
    ) -> OODMetrics:
        ood_mask = np.isin(labels, list(self.ood_indices))
        n_ood = int(ood_mask.sum())

        if n_ood == 0:
            return OODMetrics(
                rejection_recall=0.0, rejection_precision=0.0,
                leakage_rate=0.0, per_ood_class={},
            )

        ood_preds = preds[ood_mask]
        ood_labels = labels[ood_mask]

        # Rejected = predicted as any OOD class
        correctly_rejected = np.isin(ood_preds, list(self.ood_indices)).sum()
        leaked_to_positive = np.isin(ood_preds, list(self.pos_indices)).sum()

        # Precision: of all OOD predictions, how many were truly OOD
        all_ood_predictions_mask = np.isin(preds, list(self.ood_indices))
        n_pred_ood = int(all_ood_predictions_mask.sum())
        if n_pred_ood > 0:
            true_ood_of_pred_ood = (
                np.isin(labels[all_ood_predictions_mask], list(self.ood_indices)).sum()
            )
            rejection_precision = float(true_ood_of_pred_ood / n_pred_ood)
        else:
            rejection_precision = 0.0

        # Per OOD class recall
        per_ood: dict[str, float] = {}
        for idx in self.ood_indices:
            cls_name = self.idx_to_class[idx]
            mask = labels == idx
            if mask.sum() > 0:
                per_ood[cls_name] = round(
                    float((preds[mask] == idx).sum() / mask.sum()), 4
                )

        return OODMetrics(
            rejection_recall=round(float(correctly_rejected / n_ood), 4),
            rejection_precision=round(rejection_precision, 4),
            leakage_rate=round(float(leaked_to_positive / n_ood), 4),
            per_ood_class=per_ood,
        )

    # ── Calibration ───────────────────────────────────────────────────────────

    def _calibration_metrics(
        self, labels: np.ndarray, probs: np.ndarray, n_bins: int = 15
    ) -> CalibrationMetrics:
        if len(labels) == 0:
            return CalibrationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, n_bins)

        conf = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = (conf >= lo) & (conf < hi)
            n_in = int(in_bin.sum())
            if n_in == 0:
                continue
            bin_acc  = float(correct[in_bin].mean())
            bin_conf = float(conf[in_bin].mean())
            gap = abs(bin_conf - bin_acc)
            ece += (n_in / len(labels)) * gap
            mce = max(mce, gap)

        mean_conf = float(conf.mean())
        mean_acc  = float(correct.mean())

        return CalibrationMetrics(
            ece=round(ece, 4),
            mce=round(mce, 4),
            mean_confidence=round(mean_conf, 4),
            mean_accuracy=round(mean_acc, 4),
            overconfidence=round(mean_conf - mean_acc, 4),
            n_bins=n_bins,
        )

    # ── Confusion matrix ──────────────────────────────────────────────────────

    def _confusion_matrix(
        self, labels: np.ndarray, preds: np.ndarray
    ) -> tuple[list[list[int]], list[str]]:
        label_indices = list(range(len(self.classes)))
        cm = sk_confusion_matrix(labels, preds, labels=label_indices)
        return cm.tolist(), self.classes[:]

    # ── GradCAM sanity check ──────────────────────────────────────────────────

    def _gradcam_sanity(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        n_samples: int = 30,
    ) -> GradCAMSanity:
        """
        Check that GradCAM activations are spatially distributed (not degenerate).

        A degenerate activation map — all zeros or focused on a single pixel —
        indicates the model is not using spatial features for its decisions,
        which is a red flag for medical imaging models.
        """
        target_layer = model._backbone.features[-1]

        _activations: dict[str, torch.Tensor] = {}
        _gradients:   dict[str, torch.Tensor] = {}

        fwd_handle = target_layer.register_forward_hook(
            lambda _, __, out: _activations.__setitem__("feat", out.detach().clone())
        )
        bwd_handle = target_layer.register_full_backward_hook(
            lambda _, __, grad_out: _gradients.__setitem__("feat", grad_out[0].detach().clone())
        )

        entropies: list[float] = []
        n_checked = 0
        n_degenerate = 0
        max_ent = None

        try:
            for images, _ in dataloader:
                if n_checked >= n_samples:
                    break
                batch = images[:min(4, len(images))].to(device)

                for i in range(len(batch)):
                    if n_checked >= n_samples:
                        break

                    img = batch[i:i+1].clone().requires_grad_(True)
                    model.zero_grad()

                    logits = model(img)
                    pred_class = int(logits.argmax(dim=1).item())
                    logits[0, pred_class].backward()

                    if "feat" not in _activations or "feat" not in _gradients:
                        continue

                    # GradCAM: alpha × activations, then ReLU
                    alpha = _gradients["feat"].mean(dim=[2, 3], keepdim=True)
                    cam = F.relu(
                        (alpha * _activations["feat"]).sum(dim=1, keepdim=True)
                    )
                    cam_np = cam.squeeze().cpu().detach().numpy()

                    if max_ent is None:
                        max_ent = float(np.log(cam_np.size))

                    if cam_np.sum() < 1e-8:
                        n_degenerate += 1
                        entropies.append(0.0)
                    else:
                        cam_flat = cam_np.flatten()
                        cam_flat = cam_flat / (cam_flat.sum() + 1e-10)
                        raw_ent = -float(np.sum(cam_flat * np.log(cam_flat + 1e-10)))
                        norm_ent = raw_ent / (max_ent or 1.0)
                        if norm_ent < 0.10:
                            n_degenerate += 1
                        entropies.append(norm_ent)

                    n_checked += 1
                    _activations.clear()
                    _gradients.clear()
        finally:
            fwd_handle.remove()
            bwd_handle.remove()

        if not entropies:
            return GradCAMSanity(
                passed=False, mean_entropy=0.0, degenerate_fraction=1.0,
                n_samples_checked=0, notes="No samples could be evaluated.",
            )

        mean_ent = float(np.mean(entropies))
        deg_frac = n_degenerate / n_checked
        passed = mean_ent > 0.15 and deg_frac < 0.30

        return GradCAMSanity(
            passed=passed,
            mean_entropy=round(mean_ent, 4),
            degenerate_fraction=round(deg_frac, 4),
            n_samples_checked=n_checked,
            notes=(
                f"Evaluated {n_checked} samples. "
                f"Degenerate={n_degenerate} (threshold: entropy>0.15, deg_frac<0.30)."
            ),
        )

    # ── Compatibility report ──────────────────────────────────────────────────

    def _compatibility_report(
        self,
        ood: OODMetrics,
        calib: CalibrationMetrics,
        gradcam: GradCAMSanity | None,
    ) -> CompatibilityReport:
        ood_ok     = ood.rejection_recall >= self.compat.min_ood_rejection_rate
        calib_ok   = calib.ece <= self.compat.max_ece
        gradcam_ok = (not self.compat.check_gradcam_sanity) or (
            gradcam is not None and gradcam.passed
        )

        # positive_recall: mean recall across positive classes only
        report = CompatibilityReport(
            ood_rejection_rate=ood.rejection_recall,
            ood_rejection_ok=ood_ok,
            positive_recall=0.0,          # filled below
            positive_recall_ok=False,     # filled below
            ece=calib.ece,
            calibration_ok=calib_ok,
            gradcam_passed=gradcam.passed if gradcam else None,
            gradcam_ok=gradcam_ok,
        )
        report.compute_overall()
        return report

    def fill_positive_recall(
        self,
        report: CompatibilityReport,
        per_class: list[PerClassMetrics],
    ) -> None:
        """Fill positive_recall after per-class metrics are computed."""
        pos_recalls = [
            m.recall for m in per_class
            if m.class_name in self.config.positive_classes
        ]
        mean_pos_recall = float(np.mean(pos_recalls)) if pos_recalls else 0.0
        report.positive_recall = round(mean_pos_recall, 4)
        report.positive_recall_ok = (
            mean_pos_recall >= self.compat.min_positive_recall
        )
        report.compute_overall()


# ── Export helpers ────────────────────────────────────────────────────────────


def _to_serializable(obj) -> object:
    """Recursively convert dataclasses and numpy types to JSON-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def export_evaluation_report(result: V6EvaluationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_serializable(result)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Evaluation report written: %s", path)


def export_compatibility_report(report: CompatibilityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_serializable(report)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Compatibility report written: %s", path)
