"""
Fusion Intelligence Layer — Phase 5.

Confidence orchestration layer that combines:
  - classifier confidence (EfficientNet softmax)
  - semantic reasoning decision + confidence (CLIP reasoner)
  - medical plausibility (medical refiner)
  - fake_medical_score (medical refiner)
  - semantic uncertainty (reasoning layer — normalised Shannon entropy)
  - OOD tendency (semantic gate — weighted OOD score)
  - medical_relevance_score (semantic analyzer)

Role in the pipeline
--------------------
    upload → validation → semantic gate → semantic reasoner
        → medical refiner → EfficientNet classifier
        → *** Fusion Intelligence Layer ***       ← THIS MODULE
        → final VisionPredictionResponse

Design contract
---------------
  MAX_DELTA ±0.08  — classifier confidence is adjusted by at most 8 pp.
  Advisory only    — fusion_confidence is informational; acceptance decisions
                     still use the raw classifier confidence + threshold.
  No hard overrides — alignment state never flips accepted ↔ rejected.
  Transparent      — every score factor is logged at DEBUG level.
  Graceful         — if fusion is unavailable the response is unchanged.

Alignment states
----------------
  aligned    — semantic and classifier both support a medical interpretation
  misaligned — semantic signals reject while classifier is confident
  uncertain  — semantic reasoning could not reach a confident conclusion
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Maximum advisory delta the fusion layer can apply.  Kept at ±0.08 so the
# fusion_confidence field remains a narrow advisory signal — heavier dampening
# of overconfident predictions is performed by the multiplicative attenuation
# in unified_reasoning.analyze() and the multi-branch CRITICAL safeguard.
_MAX_DELTA: float = 0.08
_ALIGN_THRESHOLD: float = 0.60   # agreement score above which we call it "aligned"
# Asymmetric dampening: when the semantic branch disagrees we are willing to
# pull the fusion advisory further down than we'd ever push it up.
_MAX_NEGATIVE_DELTA: float = 0.20


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FusionIntelligenceResult:
    """Advisory output of the Fusion Intelligence Layer."""

    fusion_confidence: float    # adjusted confidence [0, 1]
    fusion_delta: float         # signed delta applied (bounded ±_MAX_DELTA)
    agreement_score: float      # [0, 1] semantic ↔ classifier alignment
    uncertainty_score: float    # [0, 1] combined uncertainty signal
    semantic_alignment: str     # "aligned" | "misaligned" | "uncertain"
    fusion_reason: str          # human-readable Turkish explanation


# ── Sub-computations ──────────────────────────────────────────────────────────


def _agreement_score(
    clf_conf: float,
    reasoning_decision: str | None,
    reasoning_confidence: float | None,
) -> tuple[float, str]:
    """
    Compute (agreement_score, alignment_label).

    Agreement measures how consistently the semantic reasoning and the
    EfficientNet classifier interpret the image.
    """
    rc = reasoning_confidence or 0.50
    decision = reasoning_decision or "uncertain"

    if decision == "allow":
        if clf_conf >= 0.50:
            # Semantic and classifier both support medical — strong alignment
            score = 0.55 + min(clf_conf, rc) * 0.45
            return min(1.0, round(score, 4)), "aligned"
        # Semantic allows but classifier is hesitant
        score = 0.35 + rc * 0.20
        return min(1.0, round(score, 4)), "uncertain"

    if decision == "reject":
        if clf_conf >= 0.60:
            # Strong disagreement: classifier is confident, semantic rejects
            score = max(0.0, 0.30 - clf_conf * 0.40)
            return round(score, 4), "misaligned"
        # Both leaning non-medical via different paths
        score = max(0.0, 0.42 - clf_conf * 0.25)
        return round(score, 4), "uncertain"

    # decision == "uncertain"
    score = 0.30 + (1.0 - rc) * 0.15
    return round(min(1.0, score), 4), "uncertain"


def _uncertainty_score(
    semantic_uncertainty: float | None,
    ood_score: float,
    clf_conf: float,
    fake_medical_score: float | None,
) -> float:
    """Combined uncertainty signal in [0, 1]."""
    sem_unc = semantic_uncertainty or 0.50
    fake = fake_medical_score or 0.0

    # Low classifier confidence → high uncertainty contribution
    clf_unc = max(0.0, (0.75 - clf_conf) / 0.75)

    # OOD score: typically 0 – 2+; normalize and cap at 1
    ood_unc = min(1.0, ood_score * 0.60)

    # Fake-medical suspicion inflates uncertainty
    fake_unc = min(1.0, fake * 0.80)

    unc = (
        sem_unc  * 0.30
        + clf_unc  * 0.35
        + ood_unc  * 0.20
        + fake_unc * 0.15
    )
    return round(min(1.0, max(0.0, unc)), 4)


def _delta_and_reason(
    alignment: str,
    agreement_score: float,
    uncertainty_score: float,
    clf_conf: float,
    medical_plausibility: float | None,
    fake_medical_score: float | None,
) -> tuple[float, str]:
    """
    Compute a bounded confidence delta (±_MAX_DELTA) and a Turkish reason string.

    Boost triggers  : aligned + high agreement + high plausibility + low uncertainty
    Dampen triggers : misaligned (semantic/classifier disagreement)
    Mild dampen     : uncertain + high uncertainty or high fake-medical score
    """
    plausibility = medical_plausibility or 0.50
    fake = fake_medical_score or 0.0

    if alignment == "aligned":
        # Boost proportional to: agreement excess above threshold, plausibility,
        # inverse uncertainty, and absence of fake-medical signal.
        agreement_excess = max(0.0, agreement_score - _ALIGN_THRESHOLD) / (1.0 - _ALIGN_THRESHOLD)
        uncertainty_discount = 1.0 - uncertainty_score * 0.80
        plausibility_factor = max(0.0, plausibility - 0.30) / 0.70
        fake_brake = max(0.0, 1.0 - fake * 4.0)   # fake > 0.25 kills the boost

        delta = _MAX_DELTA * agreement_excess * uncertainty_discount * plausibility_factor * fake_brake
        delta = round(min(_MAX_DELTA, max(0.0, delta)), 4)

        if delta >= 0.04:
            reason = (
                f"Semantik analiz ve sınıflandırıcı güçlü uyum gösteriyor "
                f"(agreement={agreement_score:.2f}, plausibility={plausibility:.2f}); "
                "güven hafifçe artırıldı."
            )
        elif delta > 0.005:
            reason = (
                "Orta düzeyde semantik-sınıflandırıcı uyumu; "
                "güven ılımlı biçimde desteklendi."
            )
        else:
            reason = (
                "Semantik uyum var ancak yüksek belirsizlik veya düşük tıbbi "
                "plausibility nedeniyle boost uygulanmadı."
            )

    elif alignment == "misaligned":
        # Dampen: degree of disagreement × classifier confidence.  We allow a
        # larger negative delta than positive — false positives (classifier
        # confident, semantic rejects) are far more dangerous than false
        # negatives.
        disagreement = max(0.0, 0.35 - agreement_score) / 0.35
        delta = -_MAX_NEGATIVE_DELTA * disagreement * clf_conf
        delta = round(max(-_MAX_NEGATIVE_DELTA, min(0.0, delta)), 4)

        reason = (
            f"Semantik gate ve sınıflandırıcı arasında anlaşmazlık tespit edildi "
            f"(agreement={agreement_score:.2f}, clf_conf={clf_conf:.2f}); "
            "tahmin güvenilirliği düşürüldü."
        )

    else:  # uncertain
        # Dampening for high uncertainty + fake-medical suspicion.  Allow up to
        # _MAX_NEGATIVE_DELTA downward when both signals stack.
        unc_penalty = max(0.0, uncertainty_score - 0.45) * _MAX_NEGATIVE_DELTA * 0.90
        fake_penalty = fake * _MAX_NEGATIVE_DELTA * 0.80
        delta = -(unc_penalty + fake_penalty)
        delta = round(max(-_MAX_NEGATIVE_DELTA, min(0.0, delta)), 4)

        if abs(delta) >= 0.04:
            reason = (
                f"Yüksek semantik belirsizlik (uncertainty={uncertainty_score:.2f}); "
                "güven koruyucu olarak hafifçe düşürüldü."
            )
        elif fake > 0.30:
            reason = (
                f"Sahte/yapay tıbbi içerik şüphesi (fake_score={fake:.2f}); "
                "güven denetimli biçimde ayarlandı."
            )
        else:
            reason = "Semantik sinyaller belirsiz; güven ayarı minimal tutuldu."

    return delta, reason


# ── Main class ────────────────────────────────────────────────────────────────


class IntelligentFusion:
    """
    Confidence orchestration layer.

    Stateless — all parameters are passed per-call, making this thread-safe
    without locks.  Instantiate once at module level; call fuse() per request.
    """

    def fuse(
        self,
        *,
        classifier_confidence: float,
        reasoning_decision: str | None,
        reasoning_confidence: float | None,
        semantic_uncertainty: float | None,
        semantic_consistency: float | None,   # reserved for future use
        medical_plausibility: float | None,
        fake_medical_score: float | None,
        ood_score: float,
        medical_relevance_score: float,       # reserved for future use
    ) -> FusionIntelligenceResult:
        """
        Produce an advisory FusionIntelligenceResult.

        All inputs are optional-safe: None values are replaced with neutral
        defaults so the layer degrades gracefully when upstream stages are
        unavailable.
        """
        agreement_score, semantic_alignment = _agreement_score(
            classifier_confidence, reasoning_decision, reasoning_confidence
        )

        uncertainty_score = _uncertainty_score(
            semantic_uncertainty, ood_score, classifier_confidence, fake_medical_score
        )

        delta, fusion_reason = _delta_and_reason(
            semantic_alignment,
            agreement_score,
            uncertainty_score,
            classifier_confidence,
            medical_plausibility,
            fake_medical_score,
        )

        fusion_confidence = round(
            min(1.0, max(0.0, classifier_confidence + delta)), 4
        )

        logger.debug(
            "FusionIntel: align=%s agreement=%.3f uncertainty=%.3f "
            "delta=%+.4f clf_conf=%.4f → fusion_conf=%.4f",
            semantic_alignment, agreement_score, uncertainty_score,
            delta, classifier_confidence, fusion_confidence,
        )

        return FusionIntelligenceResult(
            fusion_confidence=fusion_confidence,
            fusion_delta=delta,
            agreement_score=agreement_score,
            uncertainty_score=uncertainty_score,
            semantic_alignment=semantic_alignment,
            fusion_reason=fusion_reason,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

default_fusion = IntelligentFusion()
