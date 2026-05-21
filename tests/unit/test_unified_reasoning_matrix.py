"""
Smoke test for the three-case behavior matrix:

  1. Suspicious image + severe clinical → HIGH
  2. Normal image     + severe clinical → MODERATE  (clinical lift)
  3. Normal image     + normal clinical → LOW

Exercises the deterministic pipeline pieces (clinical modifier → fusion
policy → clinical lift → tier mapping) without spinning up the vision
model, so the test runs in milliseconds.
"""
from __future__ import annotations

import pytest

from app.modules.vision.medical.unified_reasoning import (
    BoundedClinicalModifier,
    ClinicalContext,
    ImagingFirstFusionPolicy,
    MedicalRiskTier,
    _apply_clinical_lift,
    _risk_tier_from_score,
)


def _resolve_tier(
    imaging_score: float,
    ctx: ClinicalContext | None,
    *,
    predicted_class: str = "pneumonia_xray",
    high_conf_healthy: bool = False,
) -> tuple[MedicalRiskTier, float, float]:
    """Mirror the deterministic portion of UnifiedMedicalReasoningEngine.analyze()."""
    modifier = BoundedClinicalModifier()
    fusion = ImagingFirstFusionPolicy()

    clin = modifier.compute(ctx, imaging_score, is_ood=False)
    fused, _ood, _override = fusion.fuse(
        imaging_score=imaging_score,
        clinical_delta=clin.applied_delta,
        is_ood=False,
        predicted_class=predicted_class,
    )
    lifted, _applied = _apply_clinical_lift(
        final_score=fused,
        clinical_alarm=clin.contradiction.clinical_alarm,
        is_ood=False,
        high_conf_healthy_suppression=high_conf_healthy,
    )
    return _risk_tier_from_score(lifted), lifted, clin.contradiction.clinical_alarm


# ── Severe symptom payload — matches a realistic "hantavirus-suspicious" panel.
SEVERE_CTX = ClinicalContext(
    symptoms=["fever", "dyspnea", "cough", "hypoxia"],
    respiratory_severity="severe",
    oxygenation_context="severe_drop",
    fever_severity="high",
    recent_worsening="rapid_48h",
    rodent_exposure_level="possible_contact",
    symptom_duration_tier="3_7_days",
)


def test_suspicious_image_plus_severe_clinical_is_high():
    """Imaging score in the suspicious band + severe symptoms → HIGH (not CRITICAL)."""
    tier, score, alarm = _resolve_tier(imaging_score=0.55, ctx=SEVERE_CTX)
    assert alarm >= 0.60, f"sanity: severe ctx should produce high alarm, got {alarm}"
    assert tier == MedicalRiskTier.HIGH_DIFFERENTIAL_RISK, (
        f"suspicious + severe should be HIGH, got {tier} (score={score:.3f})"
    )


def test_normal_image_plus_severe_clinical_is_moderate():
    """Clean image + severe symptoms → MODERATE via clinical lift (never HIGH)."""
    tier, score, alarm = _resolve_tier(
        imaging_score=0.10,
        ctx=SEVERE_CTX,
        predicted_class="healthy_xray",
        high_conf_healthy=False,  # not high-confidence-healthy → lift is allowed
    )
    assert alarm >= 0.60, f"sanity: severe ctx should produce high alarm, got {alarm}"
    assert tier == MedicalRiskTier.MODERATE, (
        f"clean image + severe symptoms should be MODERATE, got {tier} (score={score:.3f})"
    )
    assert score < 0.60, f"lift must stay below HIGH tier, got {score:.3f}"


def test_normal_image_plus_normal_clinical_is_low():
    """Clean image + no symptoms → LOW."""
    tier, score, _alarm = _resolve_tier(
        imaging_score=0.10,
        ctx=ClinicalContext(),  # empty → neutral
        predicted_class="healthy_xray",
    )
    assert tier == MedicalRiskTier.LOW, (
        f"clean + no symptoms should be LOW, got {tier} (score={score:.3f})"
    )


def test_lift_suppressed_for_high_conf_healthy():
    """High-confidence healthy + near-zero bilateral burden → lift does NOT engage."""
    tier, score, alarm = _resolve_tier(
        imaging_score=0.08,
        ctx=SEVERE_CTX,
        predicted_class="healthy_xray",
        high_conf_healthy=True,
    )
    assert alarm >= 0.60
    assert tier == MedicalRiskTier.LOW, (
        f"high-conf healthy must suppress lift, got {tier} (score={score:.3f})"
    )


def test_lift_never_reaches_high_from_clean_image():
    """Even with maximal clinical alarm, lift caps at CLINICAL_LIFT_CEILING (<0.60)."""
    from app.modules.vision.medical.unified_reasoning import CLINICAL_LIFT_CEILING
    extreme = ClinicalContext(
        symptoms=["fever", "dyspnea", "cough", "hypoxia", "hemoptysis", "chest_pain"],
        respiratory_severity="severe",
        oxygenation_context="severe_drop",
        fever_severity="high",
        recent_worsening="rapid_48h",
        rodent_exposure_level="possible_contact",
        symptom_duration_tier="over_1_week",
        immunocompromised=True,
    )
    tier, score, _ = _resolve_tier(imaging_score=0.05, ctx=extreme)
    assert tier == MedicalRiskTier.MODERATE
    assert score <= CLINICAL_LIFT_CEILING + 1e-6, (
        f"lift must not cross into HIGH, got {score:.3f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
