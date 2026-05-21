"""
Semantic gate — evaluates whether an image is medically relevant before
forwarding it to the EfficientNet classifier.

Gate signals (applied in order)
--------------------------------

  1. Label gate  — top_semantic_label in REJECT_SEMANTIC_LABELS
                   Reliable: CLIP's relative ranking is robust even when
                   absolute scores overlap.  Catches wildlife, food, vehicles,
                   furniture, and other clearly non-medical scenes.

  2. Score gate  — medical_relevance < threshold  OR  ood_score > threshold
                   Catches edge cases where the label gate passes but the
                   scores signal OOD content (e.g. gorilla face classified
                   as 'human' but with very low medical_relevance).

  3. Reasoning override  — if both threshold gates pass BUT the SemanticReasoner
                   identifies a non-medical scene with confidence ≥ 0.80, the
                   reasoning engine overrides the pass to a reject.  This adds
                   a probabilistic second line of defence for edge cases that
                   fall just inside the threshold boundaries.

The reasoning engine always runs (on pass AND reject paths) and its output
is attached to SemanticGateResult for inclusion in the API response.

Usage
-----
    from app.modules.vision.semantic.semantic_gate import semantic_gate

    result = semantic_gate.evaluate(semantic_result)
    if not result.passed:
        # reject — return result.rejection_code, result.reasoning.explanation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from app.modules.vision.semantic.semantic_config import (
    LABEL_TO_REJECTION_CODE,
    SemanticGateConfig,
    semantic_gate_config,
)
from app.modules.vision.semantic.semantic_types import SemanticResult

if TYPE_CHECKING:
    from app.modules.vision.reasoning.reasoning_types import ReasoningOutput

logger = logging.getLogger(__name__)

TriggerType = Literal["label", "score", "reasoning", "disabled", "none"]

# Reasoning type → rejection code when the reasoning override fires.
# Only non-medical reasoning types can trigger an override.
_REASONING_TYPE_TO_CODE: dict[str, str] = {
    "wildlife_scene":   "wildlife_detected",
    "portrait_scene":   "human_portrait_detected",
    "consumer_object":  "random_object_detected",
    "natural_scene":    "non_medical_scene",
    "clear_non_medical": "non_medical_scene",
    # ambiguous / uncertain types never override (they map to "uncertain" decision)
}


# ── Gate result ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SemanticGateResult:
    """
    Outcome of a single semantic gate evaluation.

    Attributes
    ----------
    passed : bool
        True  → image is medically relevant; forward to EfficientNet.
        False → image should be rejected; do NOT run the medical classifier.
    semantic_result : SemanticResult
        The raw CLIP output that was evaluated.
    rejection_code : str | None
        Machine-readable rejection code.  None when passed=True.
    rejection_reason : str | None
        Human-readable rejection description.  None when passed=True.
    triggered_by : TriggerType
        Which signal fired: "label", "score", "reasoning" (reasoning override),
        "disabled" (gate off → always passes), or "none" (image passed all checks).
    reasoning : ReasoningOutput | None
        Full reasoning engine output.  Present on all paths when the reasoning
        engine ran successfully.  None only when the engine is unavailable.
    """

    passed: bool
    semantic_result: SemanticResult
    rejection_code: str | None = None
    rejection_reason: str | None = None
    triggered_by: TriggerType = "none"
    reasoning: Optional["ReasoningOutput"] = field(default=None, compare=False)

    def as_dict(self) -> dict:
        d: dict = {
            "passed": self.passed,
            "rejection_code": self.rejection_code,
            "rejection_reason": self.rejection_reason,
            "triggered_by": self.triggered_by,
            "semantic": self.semantic_result.as_dict(),
        }
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning.as_dict()
        return d


# ── Human-readable rejection messages ────────────────────────────────────────

_REJECTION_MESSAGES: dict[str, str] = {
    "wildlife_detected":        "Görüntü tıbbi bağlamla ilgisiz yabani hayvan içeriyor.",
    "food_detected":            "Görüntü tıbbi bağlamla ilgisiz yiyecek içeriyor.",
    "non_medical_scene":        "Görüntü tıbbi analiz için uygun değil.",
    "vehicle_detected":         "Görüntü araç veya taşıt içeriyor; tıbbi görüntü değil.",
    "random_object_detected":   "Görüntü tanımlanamayan günlük nesne içeriyor.",
    "human_portrait_detected":  "Görüntü tıbbi bağlamda olmayan bir portre fotoğrafı.",
    "low_medical_relevance":    "CLIP semantik analizi düşük tıbbi uygunluk puanı üretti.",
    "high_ood_score":           "Görüntü aşırı yüksek OOD skoru nedeniyle reddedildi.",
}


def _rejection_message(code: str) -> str:
    return _REJECTION_MESSAGES.get(code, "Görüntü semantik analiz tarafından reddedildi.")


# ── Gate class ────────────────────────────────────────────────────────────────


class SemanticGate:
    """
    Three-layer semantic gate that evaluates images before the medical classifier.

    Layer 1 — Label gate  (fast, reliable):
        Top CLIP label in REJECT_SEMANTIC_LABELS → immediate reject.

    Layer 2 — Score gate  (calibrated thresholds):
        medical_relevance < threshold  OR  ood_score > threshold → reject.

    Layer 3 — Reasoning override  (probabilistic, defence-in-depth):
        SemanticReasoner confidence ≥ 0.80 for a reject type → override pass to reject.
        The reasoning engine also runs on rejected images to enrich API metadata.

    Thread-safe: all mutable state is in __init__ (config only).
    The SemanticReasoner is stateless after construction.
    """

    def __init__(
        self,
        config: SemanticGateConfig = semantic_gate_config,
        reasoner: Optional["SemanticReasoner"] = None,  # type: ignore[name-defined]
    ) -> None:
        self._cfg = config
        # Reasoner is imported lazily to avoid circular imports at module load.
        # It is resolved once on first use and cached.
        self._reasoner = reasoner
        self._reasoner_ready: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, result: SemanticResult) -> SemanticGateResult:
        """
        Evaluate all gate layers for a SemanticResult.

        Returns SemanticGateResult with passed=True (proceed to EfficientNet)
        or passed=False (reject with structured reason + reasoning metadata).
        """
        if not self._cfg.enabled:
            logger.debug("SemanticGate: disabled — passing image through")
            reasoning = self._run_reasoner(result)
            return SemanticGateResult(
                passed=True,
                semantic_result=result,
                triggered_by="disabled",
                reasoning=reasoning,
            )

        # ── Layer 1: Label gate ───────────────────────────────────────────────
        label_code = self._check_label_gate(result)
        if label_code is not None:
            reasoning = self._run_reasoner(result)
            logger.info(
                "SemanticGate [label]: rejected top_label=%s code=%s "
                "medical_rel=%.3f ood=%.3f",
                result.top_semantic_label, label_code,
                result.medical_relevance_score, result.ood_score,
            )
            return self._build_rejection(result, label_code, "label", reasoning)

        # ── Layer 2: Score gate ───────────────────────────────────────────────
        score_code = self._check_score_gate(result)
        if score_code is not None:
            reasoning = self._run_reasoner(result)
            logger.info(
                "SemanticGate [score]: rejected code=%s "
                "medical_rel=%.3f (thr=%.3f) ood=%.3f (thr=%.3f)",
                score_code,
                result.medical_relevance_score, self._cfg.medical_relevance_threshold,
                result.ood_score, self._cfg.ood_rejection_threshold,
            )
            return self._build_rejection(result, score_code, "score", reasoning)

        # ── Layer 3: Reasoning engine (enrich + possible override) ────────────
        reasoning = self._run_reasoner(result)

        if reasoning is not None:
            override_code = self._check_reasoning_override(reasoning)
            if override_code is not None:
                logger.info(
                    "SemanticGate [reasoning override]: rejected type=%s "
                    "confidence=%.3f code=%s",
                    reasoning.reasoning_type,
                    reasoning.reasoning_confidence,
                    override_code,
                )
                return self._build_rejection(result, override_code, "reasoning", reasoning)

        logger.debug(
            "SemanticGate: passed top=%s medical_rel=%.3f ood=%.3f reasoning=%s(%.2f)",
            result.top_semantic_label,
            result.medical_relevance_score,
            result.ood_score,
            reasoning.reasoning_type if reasoning else "n/a",
            reasoning.reasoning_confidence if reasoning else 0.0,
        )
        return SemanticGateResult(
            passed=True,
            semantic_result=result,
            triggered_by="none",
            reasoning=reasoning,
        )

    # ── Helper predicates (public for unit testing) ───────────────────────────

    def is_medically_relevant(self, result: SemanticResult) -> bool:
        """True when medical_relevance_score meets the configured threshold."""
        return result.medical_relevance_score >= self._cfg.medical_relevance_threshold

    def should_reject_as_ood(self, result: SemanticResult) -> bool:
        """True when ood_score exceeds the configured rejection threshold."""
        return result.ood_score > self._cfg.ood_rejection_threshold

    def build_semantic_rejection(
        self,
        result: SemanticResult,
        code: str,
        triggered_by: TriggerType = "score",
        reasoning: Optional["ReasoningOutput"] = None,
    ) -> SemanticGateResult:
        """Construct a rejection result externally (useful for testing)."""
        return self._build_rejection(result, code, triggered_by, reasoning)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_label_gate(self, result: SemanticResult) -> str | None:
        label = result.top_semantic_label
        if label in self._cfg.reject_labels:
            return LABEL_TO_REJECTION_CODE.get(label, "non_medical_scene")
        return None

    def _check_score_gate(self, result: SemanticResult) -> str | None:
        if not self.is_medically_relevant(result):
            return "low_medical_relevance"
        if self.should_reject_as_ood(result):
            return "high_ood_score"
        return None

    def _check_reasoning_override(
        self,
        reasoning: "ReasoningOutput",
    ) -> str | None:
        """
        Return a rejection code when the reasoning engine is very confident
        about a non-medical scene type that the threshold gates missed.

        Fires only when:
          - semantic_decision == "reject"
          - reasoning_confidence >= OVERRIDE_CONFIDENCE_THRESHOLD (0.80)
          - reasoning_type is in _REASONING_TYPE_TO_CODE
        """
        from app.modules.vision.reasoning.semantic_reasoner import (
            OVERRIDE_CONFIDENCE_THRESHOLD,
        )

        if reasoning.semantic_decision != "reject":
            return None
        if reasoning.reasoning_confidence < OVERRIDE_CONFIDENCE_THRESHOLD:
            return None
        return _REASONING_TYPE_TO_CODE.get(reasoning.reasoning_type)

    def _run_reasoner(
        self,
        result: SemanticResult,
    ) -> Optional["ReasoningOutput"]:
        """
        Run the SemanticReasoner and return its output.

        Returns None gracefully if the reasoner is unavailable or raises.
        The gate never fails because the reasoner failed.
        """
        try:
            if not self._reasoner_ready:
                from app.modules.vision.reasoning.semantic_reasoner import default_reasoner
                self._reasoner = default_reasoner
                self._reasoner_ready = True
            return self._reasoner.reason(result)  # type: ignore[union-attr]
        except Exception:
            logger.warning(
                "SemanticGate: reasoner unavailable — skipping reasoning layer",
                exc_info=True,
            )
            return None

    def _build_rejection(
        self,
        result: SemanticResult,
        code: str,
        triggered_by: TriggerType,
        reasoning: Optional["ReasoningOutput"] = None,
    ) -> SemanticGateResult:
        return SemanticGateResult(
            passed=False,
            semantic_result=result,
            rejection_code=code,
            rejection_reason=_rejection_message(code),
            triggered_by=triggered_by,
            reasoning=reasoning,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

semantic_gate = SemanticGate(semantic_gate_config)
