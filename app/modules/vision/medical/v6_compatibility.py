"""
V6 pipeline compatibility checker — Phase 12.

Verifies that a trained v6 model's outputs are compatible with the
existing Phase 5–6 production pipeline components:

  - Fusion Intelligence Layer (Phase 5)  — delta bounded, no crashes
  - Calibration V2 / Explainability (Phase 6) — trust tiers sane
  - Trust score distribution — not degenerate (not all suspicious / all very_high)
  - Probability sanity — model is actually learning the task

Does NOT require loading the v5 EfficientNet model.
Does NOT require CLIP (semantic check is advisory/optional only).
All checks derive from val split predictions of the v6 model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_NEUTRAL_OOD_SCORE         = 0.15
_NEUTRAL_MEDICAL_RELEVANCE = 0.75
_NEUTRAL_PLAUSIBILITY      = 0.70
_NEUTRAL_FAKE_SCORE        = 0.05
_NEUTRAL_REASONING_CONF    = 0.65
_NEUTRAL_UNCERTAINTY       = 0.25

_TRUST_TIER_ORDER = (
    "very_high_trust", "high_trust", "moderate_trust", "uncertain", "suspicious"
)


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class FusionCompatResult:
    passed: bool
    exception_rate: float          # fraction of samples that raised exceptions
    mean_delta: float              # mean |fusion_delta| — should stay ≤ 0.08
    max_delta: float
    alignment_distribution: dict[str, float]  # fraction per alignment label
    notes: str = ""


@dataclass
class CalibrationCompatResult:
    passed: bool
    trust_tier_distribution: dict[str, float]   # fraction per tier
    mean_trust_score: float
    calibration_state_distribution: dict[str, float]
    fraction_suspicious: float
    notes: str = ""


@dataclass
class ProbabilitySanityResult:
    passed: bool
    mean_positive_class_prob: float    # mean prob for the correct class (positive samples)
    mean_ood_rejection_prob: float     # mean prob predicted as OOD (OOD samples)
    fraction_correct: float            # fraction where predicted class == true class
    catastrophic_class_recall: list[str]  # classes with recall = 0.0
    notes: str = ""


@dataclass
class V6CompatibilityResult:
    fusion: FusionCompatResult
    calibration: CalibrationCompatResult
    probability: ProbabilitySanityResult
    overall_pass: bool
    gate_results: dict[str, bool] = field(default_factory=dict)
    notes: str = ""

    def compute_overall(self) -> None:
        self.gate_results = {
            "fusion":         self.fusion.passed,
            "calibration":    self.calibration.passed,
            "probability":    self.probability.passed,
        }
        self.overall_pass = all(self.gate_results.values())


# ── Val output collector ──────────────────────────────────────────────────────


@dataclass
class _ValOutputs:
    probs: np.ndarray       # (N, num_classes) softmax probabilities
    preds: np.ndarray       # (N,) predicted class indices
    labels: np.ndarray      # (N,) true class indices
    confidences: np.ndarray # (N,) max prob
    classes: list[str]
    ood_indices: frozenset[int]
    pos_indices: frozenset[int]


@torch.no_grad()
def _collect_val_outputs(
    model: nn.Module,
    val_loader,
    device: torch.device,
    max_samples: int = 500,
) -> _ValOutputs:
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in val_loader:
        if sum(len(p) for p in all_probs) >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.tolist())

    probs_arr  = np.vstack(all_probs)[:max_samples]
    labels_arr = np.array(all_labels[:max_samples], dtype=np.int64)
    preds_arr  = probs_arr.argmax(axis=1)
    conf_arr   = probs_arr.max(axis=1)

    return _ValOutputs(
        probs=probs_arr,
        preds=preds_arr,
        labels=labels_arr,
        confidences=conf_arr,
        classes=[],           # filled by caller
        ood_indices=frozenset(),
        pos_indices=frozenset(),
    )


# ── Checker ───────────────────────────────────────────────────────────────────


class V6CompatibilityChecker:
    """
    Tests v6 model outputs against the existing Phase 5–6 pipeline components.

    All checks use the val split predictions — no need to reload the training data.
    """

    def __init__(self, config) -> None:
        """config: V6TrainingConfig from v6_training_config.py"""
        self.config = config
        self.class_to_idx = config.class_to_idx()
        self.idx_to_class = config.idx_to_class()
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

    def check_all(
        self,
        model: nn.Module,
        val_loader,
        device: torch.device,
        max_samples: int = 300,
    ) -> V6CompatibilityResult:
        logger.info("V6CompatibilityChecker: collecting val outputs …")
        outputs = _collect_val_outputs(model, val_loader, device, max_samples)
        outputs.classes     = list(self.config.classes)
        outputs.ood_indices = self.ood_indices
        outputs.pos_indices = self.pos_indices

        logger.info("Checking fusion compatibility …")
        fusion_result = self._check_fusion(outputs)

        logger.info("Checking calibration V2 compatibility …")
        calib_result = self._check_calibration(outputs, fusion_result)

        logger.info("Checking probability sanity …")
        prob_result = self._check_probability_sanity(outputs)

        result = V6CompatibilityResult(
            fusion=fusion_result,
            calibration=calib_result,
            probability=prob_result,
            overall_pass=False,
        )
        result.compute_overall()

        logger.info(
            "Compatibility: fusion=%s calib=%s prob=%s overall=%s",
            "PASS" if fusion_result.passed else "FAIL",
            "PASS" if calib_result.passed else "FAIL",
            "PASS" if prob_result.passed else "FAIL",
            "PASS" if result.overall_pass else "FAIL",
        )
        return result

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _check_fusion(self, outputs: _ValOutputs) -> FusionCompatResult:
        from app.modules.vision.fusion.intelligent_fusion import default_fusion

        deltas: list[float] = []
        alignments: list[str] = []
        n_errors = 0
        n = len(outputs.confidences)

        for i in range(min(n, 200)):
            conf     = float(outputs.confidences[i])
            pred_idx = int(outputs.preds[i])
            is_ood   = pred_idx in outputs.ood_indices

            reasoning_decision = "allow" if not is_ood else "reject"

            try:
                result = default_fusion.fuse(
                    classifier_confidence=conf,
                    reasoning_decision=reasoning_decision,
                    reasoning_confidence=_NEUTRAL_REASONING_CONF,
                    semantic_uncertainty=_NEUTRAL_UNCERTAINTY,
                    semantic_consistency=None,
                    medical_plausibility=_NEUTRAL_PLAUSIBILITY if not is_ood else 0.20,
                    fake_medical_score=_NEUTRAL_FAKE_SCORE if not is_ood else 0.35,
                    ood_score=_NEUTRAL_OOD_SCORE if not is_ood else 0.70,
                    medical_relevance_score=_NEUTRAL_MEDICAL_RELEVANCE,
                )
                deltas.append(abs(result.fusion_delta))
                alignments.append(result.semantic_alignment)
            except Exception as exc:
                logger.debug("Fusion error at sample %d: %s", i, exc)
                n_errors += 1

        n_tested = max(min(n, 200), 1)
        exc_rate = n_errors / n_tested
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        max_delta  = float(np.max(deltas))  if deltas else 0.0

        align_counts: dict[str, int] = {"aligned": 0, "misaligned": 0, "uncertain": 0}
        for a in alignments:
            if a in align_counts:
                align_counts[a] += 1
        total_aligned = max(len(alignments), 1)
        align_dist = {k: round(v / total_aligned, 4) for k, v in align_counts.items()}

        # Gate: exception rate < 2%, delta always within bounds, not all uncertain
        passed = (
            exc_rate < 0.02
            and max_delta <= 0.081         # 0.08 + float tolerance
            and align_dist.get("uncertain", 1.0) < 0.90
        )
        notes = (
            f"Tested {n_tested} samples. "
            f"exception_rate={exc_rate:.3f} max_delta={max_delta:.4f}"
        )

        return FusionCompatResult(
            passed=passed,
            exception_rate=round(exc_rate, 4),
            mean_delta=round(mean_delta, 4),
            max_delta=round(max_delta, 4),
            alignment_distribution=align_dist,
            notes=notes,
        )

    # ── Calibration V2 ────────────────────────────────────────────────────────

    def _check_calibration(
        self,
        outputs: _ValOutputs,
        fusion_result: FusionCompatResult,
    ) -> CalibrationCompatResult:
        from app.modules.vision.explainability.calibration_v2 import build_calibration_v2
        from app.modules.vision.fusion.intelligent_fusion import default_fusion

        tier_counts: dict[str, int] = {t: 0 for t in _TRUST_TIER_ORDER}
        state_counts: dict[str, int] = {}
        trust_scores: list[float] = []
        n_errors = 0
        n = len(outputs.confidences)

        for i in range(min(n, 200)):
            conf     = float(outputs.confidences[i])
            pred_idx = int(outputs.preds[i])
            is_ood   = pred_idx in outputs.ood_indices

            reasoning_decision = "allow" if not is_ood else "reject"

            try:
                fir = default_fusion.fuse(
                    classifier_confidence=conf,
                    reasoning_decision=reasoning_decision,
                    reasoning_confidence=_NEUTRAL_REASONING_CONF,
                    semantic_uncertainty=_NEUTRAL_UNCERTAINTY,
                    semantic_consistency=None,
                    medical_plausibility=_NEUTRAL_PLAUSIBILITY if not is_ood else 0.20,
                    fake_medical_score=_NEUTRAL_FAKE_SCORE if not is_ood else 0.35,
                    ood_score=_NEUTRAL_OOD_SCORE if not is_ood else 0.70,
                    medical_relevance_score=_NEUTRAL_MEDICAL_RELEVANCE,
                )
                cal = build_calibration_v2(
                    classifier_confidence=conf,
                    threshold=0.70,
                    fusion_confidence=fir.fusion_confidence,
                    fusion_delta=fir.fusion_delta,
                    agreement_score=fir.agreement_score,
                    uncertainty_score=fir.uncertainty_score,
                    semantic_alignment=fir.semantic_alignment,
                    reasoning_type="chest_xray" if not is_ood else "ood_image",
                    reasoning_decision=reasoning_decision,
                    semantic_uncertainty=_NEUTRAL_UNCERTAINTY,
                    medical_plausibility=_NEUTRAL_PLAUSIBILITY if not is_ood else 0.20,
                    fake_medical_score=_NEUTRAL_FAKE_SCORE if not is_ood else 0.35,
                    ood_score=_NEUTRAL_OOD_SCORE if not is_ood else 0.70,
                )
                tier = cal.trust_tier
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                state = cal.calibration_state
                state_counts[state] = state_counts.get(state, 0) + 1
                trust_scores.append(cal.trust_score)
            except Exception as exc:
                logger.debug("Calibration V2 error at sample %d: %s", i, exc)
                n_errors += 1

        n_tested = min(n, 200)
        total    = max(n_tested - n_errors, 1)

        tier_dist  = {t: round(tier_counts.get(t, 0) / total, 4) for t in _TRUST_TIER_ORDER}
        state_dist = {s: round(v / total, 4) for s, v in state_counts.items()}

        mean_trust = float(np.mean(trust_scores)) if trust_scores else 0.0
        frac_suspicious = tier_dist.get("suspicious", 0.0)

        # Gate: not all suspicious, not all very_high_trust, mean trust > 0.15
        passed = (
            frac_suspicious < 0.50           # model not flagging everything suspicious
            and tier_dist.get("very_high_trust", 0.0) < 0.95  # not unrealistically perfect
            and mean_trust > 0.15             # not all garbage
            and n_errors / max(n_tested, 1) < 0.05
        )
        notes = (
            f"Tested {n_tested} samples. "
            f"suspicious={frac_suspicious:.3f} mean_trust={mean_trust:.3f}"
        )

        return CalibrationCompatResult(
            passed=passed,
            trust_tier_distribution=tier_dist,
            mean_trust_score=round(mean_trust, 4),
            calibration_state_distribution=state_dist,
            fraction_suspicious=round(frac_suspicious, 4),
            notes=notes,
        )

    # ── Probability sanity ────────────────────────────────────────────────────

    def _check_probability_sanity(
        self, outputs: _ValOutputs
    ) -> ProbabilitySanityResult:
        labels = outputs.labels
        probs  = outputs.probs
        preds  = outputs.preds

        # Mean predicted prob for the true class (all samples)
        true_class_probs = probs[np.arange(len(labels)), labels]
        mean_true = float(np.mean(true_class_probs))

        # Positive samples: mean prob for their true class
        pos_mask = np.isin(labels, list(outputs.pos_indices))
        ood_mask = np.isin(labels, list(outputs.ood_indices))

        mean_pos_prob = float(np.mean(true_class_probs[pos_mask])) if pos_mask.any() else 0.0

        # OOD samples: mean prob predicted as any OOD class
        if ood_mask.any():
            ood_preds = preds[ood_mask]
            ood_rejected = np.isin(ood_preds, list(outputs.ood_indices))
            mean_ood_rej = float(np.mean(ood_rejected.astype(float)))
        else:
            mean_ood_rej = 0.0

        # Accuracy
        frac_correct = float(np.mean((preds == labels).astype(float)))

        # Catastrophic failure: any class with recall = 0
        catastrophic: list[str] = []
        n_classes = probs.shape[1]
        for idx in range(n_classes):
            cls_mask = labels == idx
            if not cls_mask.any():
                continue
            cls_recall = float(np.mean((preds[cls_mask] == idx).astype(float)))
            if cls_recall == 0.0:
                cls_name = self.idx_to_class.get(idx, str(idx))
                catastrophic.append(cls_name)
                logger.warning(
                    "Catastrophic failure: class '%s' has recall = 0.0", cls_name
                )

        # Gate: model must be learning something (not random)
        # Random baseline for 4 classes = 0.25 accuracy
        n_classes_actual = max(n_classes, 1)
        random_baseline = 1.0 / n_classes_actual
        passed = (
            frac_correct > random_baseline + 0.10   # meaningfully above random
            and len(catastrophic) == 0              # no class with zero recall
            and mean_pos_prob > 0.35                # positive class probabilities reasonable
        )
        notes = (
            f"Correct={frac_correct:.4f} (random={random_baseline:.2f}) "
            f"pos_prob={mean_pos_prob:.4f} ood_rej={mean_ood_rej:.4f}"
        )

        return ProbabilitySanityResult(
            passed=passed,
            mean_positive_class_prob=round(mean_pos_prob, 4),
            mean_ood_rejection_prob=round(mean_ood_rej, 4),
            fraction_correct=round(frac_correct, 4),
            catastrophic_class_recall=catastrophic,
            notes=notes,
        )
