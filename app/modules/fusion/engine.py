"""
Multimodal Fusion Engine.

Combines:
  - Structured symptom/risk ML score  (PRIMARY signal)
  - Calibrated vision confidence       (SUPPORTING evidence)

into a single multimodal risk score.

Architecture
------------
  final_risk = α × ml_probability + β × vision_contribution

  α >> β always  (ML is primary)
  β = 0          when vision is rejected / unrelated / unavailable / low-confidence

Vision NEVER overrides ML.
Vision CAN moderately increase or decrease the final score when accepted.

Safe Fallbacks
--------------
  - No image provided           → α=1.0, β=0 (pure ML)
  - Image rejected (gate/threshold) → α=1.0, β=0
  - Image unrelated (gate class) → α=1.0, β=0
  - Image low confidence         → α=1.0, β=0
  - Inference error              → α=1.0, β=0 + uncertainty flag
"""
from __future__ import annotations

import logging
from typing import Optional

from app.modules.fusion.schema import (
    ExplanationPayload,
    FusionConfidence,
    FusionResult,
    FusionWeightsUsed,
    MLResult,
    RiskLevel,
    VisionResult,
    VisionStatus,
)
from app.modules.fusion.weights import DEFAULT_WEIGHT_POLICY, FusionWeightPolicy

logger = logging.getLogger(__name__)

# Vision classes that explicitly mean "not relevant" (gate output labels)
_UNRELATED_CLASSES = frozenset({"unrelated", "hard_negative", "irrelevant"})


class MultimodalFusionEngine:
    """
    Symptom-first multimodal fusion for hantavirus risk assessment.

    Parameters
    ----------
    policy : FusionWeightPolicy
        Weight configuration. Defaults to ml=0.75, vision=0.25.
    """

    def __init__(self, policy: Optional[FusionWeightPolicy] = None) -> None:
        self._policy = policy or DEFAULT_WEIGHT_POLICY

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse(
        self,
        ml_result: MLResult,
        vision_result: Optional[VisionResult] = None,
    ) -> FusionResult:
        """
        Perform multimodal fusion.

        Parameters
        ----------
        ml_result : MLResult
            Structured ML prediction (required — ML is always the primary signal).
        vision_result : VisionResult | None
            Vision prediction from VisionInferenceService, or None when no image
            was provided.

        Returns
        -------
        FusionResult with the final composite risk assessment.
        """
        ml_prob = float(ml_result.probability)
        uncertainty_flags: list[str] = []

        # ── Classify vision status ────────────────────────────────────────────
        vision_status, vision_rejection_reason = self._classify_vision_status(
            vision_result, uncertainty_flags
        )

        # ── Determine effective weights ───────────────────────────────────────
        vision_confidence = vision_result.confidence if vision_result else None
        α, β, weight_reason = self._resolve_weights(vision_status, vision_confidence)

        # ── Compute vision contribution ───────────────────────────────────────
        vision_contrib = self._vision_contribution(vision_result, β)

        # ── ML contribution ───────────────────────────────────────────────────
        ml_contrib = α * ml_prob

        # ── Final composite risk ──────────────────────────────────────────────
        final_score = round(ml_contrib + vision_contrib, 4)
        final_score = max(0.0, min(1.0, final_score))

        # ── Risk level + boundary proximity ──────────────────────────────────
        risk_level, risk_proximity = self._risk_level(final_score)
        near_risk_boundary = risk_proximity <= 0.05

        # ── ML uncertainty signals ────────────────────────────────────────────
        self._check_ml_uncertainty(ml_result, uncertainty_flags)

        # ── Fusion confidence ─────────────────────────────────────────────────
        fusion_confidence = self._fusion_confidence(
            ml_result, vision_status, vision_result
        )

        # ── Weights used ──────────────────────────────────────────────────────
        weights_used = FusionWeightsUsed(
            ml_weight=round(α, 4),
            vision_weight=round(β, 4),
            vision_status=vision_status,
            reason=weight_reason,
        )

        # ── Explanation payload ───────────────────────────────────────────────
        explanation = self._build_explanation(
            ml_result=ml_result,
            vision_result=vision_result,
            vision_status=vision_status,
            vision_rejection_reason=vision_rejection_reason,
            risk_level=risk_level,
            final_score=final_score,
            uncertainty_flags=uncertainty_flags,
            alpha=α,
            near_risk_boundary=near_risk_boundary,
            risk_proximity=risk_proximity,
        )

        logger.info(
            "Fusion: ml_prob=%.4f vision_status=%s α=%.2f β=%.2f "
            "final_score=%.4f risk=%s confidence=%s flags=%s",
            ml_prob, vision_status.value, α, β,
            final_score, risk_level.value, fusion_confidence.value,
            uncertainty_flags,
        )

        return FusionResult(
            final_risk_score=final_score,
            risk_level=risk_level,
            fusion_confidence=fusion_confidence,
            ml_risk_score=round(ml_prob, 4),
            ml_contribution=round(ml_contrib, 4),
            vision_contribution=round(vision_contrib, 4),
            vision_status=vision_status,
            vision_rejection_reason=vision_rejection_reason,
            uncertainty_flags=uncertainty_flags,
            weights_used=weights_used,
            explanation_payload=explanation,
            near_risk_boundary=near_risk_boundary,
            risk_proximity=round(risk_proximity, 4),
            ml_model_name=ml_result.model_name,
            ml_model_version=ml_result.model_version,
            vision_model_name=vision_result.model_name if vision_result else None,
            vision_model_version=vision_result.model_version if vision_result else None,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify_vision_status(
        self,
        vision_result: Optional[VisionResult],
        uncertainty_flags: list[str],
    ) -> tuple[VisionStatus, Optional[str]]:
        """Determine how (or whether) the vision result will be used."""
        if vision_result is None:
            return VisionStatus.UNAVAILABLE, None

        if not vision_result.accepted:
            reason = vision_result.rejection_reason or "threshold_rejection"
            # Distinguish content-relevance rejection from low-confidence rejection.
            predicted = (vision_result.predicted_class or "").lower()
            if predicted in _UNRELATED_CLASSES:
                return VisionStatus.UNRELATED, reason
            return VisionStatus.REJECTED, reason

        # Accepted, but confidence may still be too low.
        confidence = vision_result.confidence
        if confidence is None or confidence < self._policy.min_vision_confidence:
            uncertainty_flags.append("vision_low_confidence_ignored")
            return VisionStatus.LOW_CONFIDENCE, "confidence_below_threshold"

        return VisionStatus.USED, None

    def _resolve_weights(
        self,
        vision_status: VisionStatus,
        vision_confidence: Optional[float] = None,
    ) -> tuple[float, float, str]:
        """
        Return (α, β, reason_string) for the given vision status.

        When vision is USED, β is ramped linearly from 0 at min_vision_confidence
        to β_max at confidence=1.0, eliminating the binary jump that caused
        abrupt score changes near the acceptance threshold.
        """
        if vision_status == VisionStatus.USED and vision_confidence is not None:
            _, β_max = self._policy.normalised()
            min_conf = self._policy.min_vision_confidence
            ramp = (vision_confidence - min_conf) / (1.0 - min_conf)
            ramp = max(0.0, min(1.0, ramp))
            β = β_max * ramp
            α = 1.0 - β
            reason = "ml_primary_vision_supporting_ramped"
        else:
            α, β = self._policy.ml_only()
            reason = {
                VisionStatus.UNAVAILABLE: "no_image_provided",
                VisionStatus.REJECTED: "image_rejected",
                VisionStatus.UNRELATED: "image_unrelated",
                VisionStatus.LOW_CONFIDENCE: "vision_low_confidence",
            }.get(vision_status, "vision_ignored")
        return α, β, reason

    def _vision_contribution(
        self,
        vision_result: Optional[VisionResult],
        β: float,
    ) -> float:
        """Compute β × vision_signal. Returns 0.0 when β=0 or vision unavailable."""
        if β == 0.0 or vision_result is None or vision_result.confidence is None:
            return 0.0

        # Vision signal: use confidence of the accepted class as the risk proxy.
        # A "related" (positive-indicator) class contributes positively;
        # any non-related class contributes inversely (lowers the vision signal).
        predicted = (vision_result.predicted_class or "").lower()
        confidence = vision_result.confidence

        if predicted in {"related", "positive"}:
            vision_signal = confidence
        elif predicted in _UNRELATED_CLASSES:
            # Should never reach here (classified as UNRELATED/REJECTED), but defensive:
            vision_signal = 0.0
        else:
            # Unknown class label — use raw confidence as a neutral positive signal
            vision_signal = confidence

        return β * vision_signal

    def _risk_level(self, score: float) -> tuple[RiskLevel, float]:
        """Return (risk_level, proximity) where proximity = distance to nearest threshold."""
        high_t = self._policy.high_risk_threshold
        med_t = self._policy.medium_risk_threshold
        proximity = min(abs(score - high_t), abs(score - med_t))
        if score >= high_t:
            return RiskLevel.HIGH, proximity
        if score >= med_t:
            return RiskLevel.MEDIUM, proximity
        return RiskLevel.LOW, proximity

    def _check_ml_uncertainty(
        self, ml_result: MLResult, flags: list[str]
    ) -> None:
        prob = ml_result.probability
        # Near decision boundary → uncertain ML
        if 0.40 <= prob <= 0.65:
            flags.append("ml_near_decision_boundary")
        if ml_result.confidence == "low":
            flags.append("ml_low_confidence")

    def _fusion_confidence(
        self,
        ml_result: MLResult,
        vision_status: VisionStatus,
        vision_result: Optional[VisionResult],
    ) -> FusionConfidence:
        """
        Assess overall confidence in the fusion result.

        HIGH   → ML high-confidence AND (no vision OR vision high-confidence)
        MEDIUM → ML medium-confidence OR vision contributing but medium-confidence
        LOW    → ML low-confidence OR vision rejected with ML near boundary
        """
        ml_conf = ml_result.confidence

        if ml_conf == "low":
            return FusionConfidence.LOW

        if vision_status == VisionStatus.USED:
            v_conf = vision_result.confidence if vision_result else None
            if ml_conf == "high" and v_conf is not None and v_conf >= 0.75:
                return FusionConfidence.HIGH
            return FusionConfidence.MEDIUM

        # Vision ignored / unavailable
        if ml_conf == "high":
            return FusionConfidence.HIGH
        return FusionConfidence.MEDIUM

    def _build_explanation(
        self,
        *,
        ml_result: MLResult,
        vision_result: Optional[VisionResult],
        vision_status: VisionStatus,
        vision_rejection_reason: Optional[str],
        risk_level: RiskLevel,
        final_score: float,
        uncertainty_flags: list[str],
        alpha: float,
        near_risk_boundary: bool,
        risk_proximity: float,
    ) -> ExplanationPayload:
        dominant = (
            "ml_and_vision" if vision_status == VisionStatus.USED else "ml_only"
        )
        return ExplanationPayload(
            risk_level=risk_level.value,
            final_risk_score=final_score,
            ml_probability=round(ml_result.probability, 4),
            ml_label=ml_result.label,
            ml_confidence=ml_result.confidence,
            vision_used=vision_status == VisionStatus.USED,
            vision_class=vision_result.predicted_class if vision_result else None,
            vision_confidence=round(vision_result.confidence, 4)
            if vision_result and vision_result.confidence is not None
            else None,
            vision_status=vision_status.value,
            vision_rejection_reason=vision_rejection_reason,
            uncertainty_flags=uncertainty_flags,
            dominant_signal=dominant,
            near_risk_boundary=near_risk_boundary,
            risk_proximity=round(risk_proximity, 4),
        )
