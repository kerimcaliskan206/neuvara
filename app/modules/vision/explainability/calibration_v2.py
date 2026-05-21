"""
Calibration V2 — Phase 6.

Orchestrates the explainability + confidence-calibration pipeline that
runs after the Fusion Intelligence Layer but before the final response.

Role in the pipeline
--------------------
    EfficientNet → Fusion Intelligence Layer
        → *** Calibration V2 ***       ← THIS MODULE
        → final VisionPredictionResponse

What it produces
----------------
  trust_tier         — 5-level label: very_high_trust | high_trust |
                        moderate_trust | uncertain | suspicious
  trust_score        — [0, 1] holistic trustworthiness score
  calibration_state  — stable | near_threshold | softened | suspicious
  explanation_summary — single Turkish sentence explaining the result
  uncertainty_reason  — Turkish uncertainty source string (None if negligible)
  semantic_warning    — Turkish semantic-conflict warning (None if no conflict)

Design contract
---------------
  - Advisory only. Does not change accepted/rejected state.
  - All inputs are optional-safe (None → neutral defaults).
  - Imports from explanation_builder and uncertainty_formatter are local
    so this module stays unit-testable without the full package.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_NEAR_THRESHOLD_BAND: float = 0.10
_SOFTENED_DELTA: float = 0.06      # |fusion_delta| ≥ this → "softened"
_SUSPICIOUS_FAKE: float = 0.40
_SUSPICIOUS_FAKE_MISALIGN: float = 0.15


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationV2Result:
    """Advisory output of the Calibration V2 layer."""

    trust_tier: str              # very_high_trust | high_trust | moderate_trust | uncertain | suspicious
    trust_score: float           # [0, 1]
    calibration_state: str       # stable | near_threshold | softened | suspicious
    explanation_summary: str     # Turkish
    uncertainty_reason: str | None
    semantic_warning: str | None


# ── Trust score ───────────────────────────────────────────────────────────────


def _trust_score(
    fusion_confidence: float,
    uncertainty_score: float,
    agreement_score: float,
    medical_plausibility: float | None,
    fake_medical_score: float | None,
) -> float:
    """
    Holistic [0, 1] trustworthiness score.

    Combines fusion confidence (primary) with penalties from uncertainty,
    fake-medical suspicion, and bonuses from alignment + plausibility.
    """
    plausibility = medical_plausibility or 0.50
    fake = fake_medical_score or 0.0

    uncertainty_discount = 1.0 - uncertainty_score * 0.35
    alignment_factor     = 0.65 + agreement_score * 0.35
    plausibility_factor  = 0.80 + plausibility * 0.20
    fake_penalty         = max(0.10, 1.0 - fake * 0.60)

    score = (
        fusion_confidence
        * uncertainty_discount
        * alignment_factor
        * plausibility_factor
        * fake_penalty
    )
    return round(min(1.0, max(0.0, score)), 4)


# ── Trust tier ────────────────────────────────────────────────────────────────


def _trust_tier(
    trust_score: float,
    semantic_alignment: str,
    uncertainty_score: float,
    fake_medical_score: float | None,
) -> str:
    fake = fake_medical_score or 0.0

    # Suspicious first — takes precedence over numeric score
    if fake > _SUSPICIOUS_FAKE:
        return "suspicious"
    if semantic_alignment == "misaligned" and fake > _SUSPICIOUS_FAKE_MISALIGN:
        return "suspicious"

    if trust_score >= 0.72 and semantic_alignment == "aligned" and uncertainty_score < 0.30:
        return "very_high_trust"

    if trust_score >= 0.55 and semantic_alignment in ("aligned", "uncertain") and uncertainty_score < 0.55:
        return "high_trust"

    if trust_score >= 0.38 and uncertainty_score < 0.70:
        return "moderate_trust"

    return "uncertain"


# ── Calibration state ─────────────────────────────────────────────────────────


def _calibration_state(
    classifier_confidence: float,
    threshold: float,
    semantic_alignment: str,
    fake_medical_score: float | None,
    fusion_delta: float,
) -> str:
    fake = fake_medical_score or 0.0

    if fake > _SUSPICIOUS_FAKE:
        return "suspicious"

    if semantic_alignment == "misaligned" or abs(fusion_delta) >= _SOFTENED_DELTA:
        return "softened"

    if abs(classifier_confidence - threshold) <= _NEAR_THRESHOLD_BAND:
        return "near_threshold"

    return "stable"


# ── Public entry point ────────────────────────────────────────────────────────


def build_calibration_v2(
    *,
    classifier_confidence: float,
    threshold: float,
    fusion_confidence: float,
    fusion_delta: float,
    agreement_score: float,
    uncertainty_score: float,
    semantic_alignment: str,
    reasoning_type: str | None,
    reasoning_decision: str | None,
    semantic_uncertainty: float | None,
    medical_plausibility: float | None,
    fake_medical_score: float | None,
    ood_score: float,
) -> CalibrationV2Result:
    """
    Build a full CalibrationV2Result from post-fusion signals.

    All numeric inputs derived from SemanticInfo, VisionFusionResult, and
    the raw EfficientNet prediction.  None values receive neutral defaults.
    """
    from app.modules.vision.explainability.explanation_builder import build_explanation_summary
    from app.modules.vision.explainability.uncertainty_formatter import (
        format_uncertainty_reason,
        format_semantic_warning,
    )

    score = _trust_score(
        fusion_confidence,
        uncertainty_score,
        agreement_score,
        medical_plausibility,
        fake_medical_score,
    )

    tier = _trust_tier(score, semantic_alignment, uncertainty_score, fake_medical_score)

    cal_state = _calibration_state(
        classifier_confidence,
        threshold,
        semantic_alignment,
        fake_medical_score,
        fusion_delta,
    )

    explanation = build_explanation_summary(
        trust_tier=tier,
        semantic_alignment=semantic_alignment,
        reasoning_type=reasoning_type,
        classifier_confidence=classifier_confidence,
        fusion_delta=fusion_delta,
        medical_plausibility=medical_plausibility,
        fake_medical_score=fake_medical_score,
    )

    unc_reason = format_uncertainty_reason(
        uncertainty_score=uncertainty_score,
        semantic_alignment=semantic_alignment,
        semantic_uncertainty=semantic_uncertainty,
        ood_score=ood_score,
        fake_medical_score=fake_medical_score,
        reasoning_decision=reasoning_decision,
    )

    sem_warning = format_semantic_warning(
        semantic_alignment=semantic_alignment,
        reasoning_decision=reasoning_decision,
        fake_medical_score=fake_medical_score,
    )

    logger.debug(
        "CalibrationV2: tier=%s score=%.3f state=%s align=%s",
        tier, score, cal_state, semantic_alignment,
    )

    return CalibrationV2Result(
        trust_tier=tier,
        trust_score=score,
        calibration_state=cal_state,
        explanation_summary=explanation,
        uncertainty_reason=unc_reason,
        semantic_warning=sem_warning,
    )
