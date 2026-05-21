"""
Phase 26 — imaging-chain refactor tests.

Covers the three named trust factors (_semantic_trust, _refiner_trust,
_spatial_trust), the symmetric healthy-doubt helper, the low-trust soft
cap, and one end-to-end integration through UnifiedMedicalReasoningEngine.

Runs without conftest because the repo's conftest pulls in app.main, which
needs xgboost; this file imports only unified_reasoning.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.modules.vision.medical.unified_reasoning import (
    BoundedClinicalModifier,
    CLINICAL_LIFT_FLOOR,
    ClinicalContext,
    HEALTHY_DOUBT_PINNED_SCORE,
    HEALTHY_DOUBT_REQUIRED_VOTES,
    ImagingFirstFusionPolicy,
    LOW_TRUST_IMAGING_CAP,
    LOW_TRUST_THRESHOLD,
    MedicalRiskTier,
    UnifiedMedicalReasoningEngine,
    _apply_clinical_lift,
    _doubt_healthy_prediction,
    _refiner_trust,
    _risk_tier_from_score,
    _semantic_trust,
    _spatial_trust,
    _TRUST_FLOOR,
)


# ───────────────────────────── _semantic_trust ─────────────────────────────


def test_semantic_trust_all_clear_returns_one():
    trust, reasons = _semantic_trust(
        semantic_alignment="aligned",
        medical_relevance_score=0.80,
        medical_plausibility=0.85,
        fake_medical_score=0.05,
    )
    assert trust == 1.0
    assert reasons == []


def test_semantic_trust_misaligned_attenuates():
    trust, reasons = _semantic_trust(
        semantic_alignment="misaligned",
        medical_relevance_score=None,
        medical_plausibility=None,
        fake_medical_score=None,
    )
    assert trust == pytest.approx(0.40, abs=1e-4)
    assert any("semantic_misaligned" in r for r in reasons)


def test_semantic_trust_floors_at_TRUST_FLOOR():
    """Worst case (all attenuators fire) is clamped to _TRUST_FLOOR (0.30)."""
    trust, _ = _semantic_trust(
        semantic_alignment="misaligned",
        medical_relevance_score=0.05,
        medical_plausibility=0.10,
        fake_medical_score=0.95,
    )
    assert trust == pytest.approx(_TRUST_FLOOR, abs=1e-4)


# ───────────────────────────── _refiner_trust ─────────────────────────────


def test_refiner_trust_healthy_top_with_pneumonia_class_clamps():
    """Old refiner_healthy_veto behavior preserved at <=0.30."""
    trust, reasons = _refiner_trust(
        refiner_top_type="healthy_xray",
        refiner_group_scores=None,
        semantic_margin=None,
        predicted_class="pneumonia_xray",
    )
    assert trust <= 0.30 + 1e-6
    assert any("refiner_healthy_veto" in r for r in reasons)


def test_refiner_trust_aligned_returns_one():
    trust, reasons = _refiner_trust(
        refiner_top_type="pneumonia_xray",
        refiner_group_scores={"pneumonia_xray": 0.85, "healthy_xray": 0.05},
        semantic_margin=0.40,
        predicted_class="pneumonia_xray",
    )
    assert trust == 1.0
    assert reasons == []


# ───────────────────────────── _spatial_trust ─────────────────────────────


def test_spatial_trust_low_burden_attenuates():
    trust, reasons = _spatial_trust(
        bilateral_burden=0.10,
        uncertainty_score=0.20,
        fusion_delta=0.0,
    )
    assert trust == pytest.approx(0.55, abs=1e-4)
    assert any("low_bilateral_burden" in r for r in reasons)


def test_spatial_trust_high_uncertainty_attenuates():
    trust, _ = _spatial_trust(
        bilateral_burden=0.70,  # high burden → no attenuation from this leg
        uncertainty_score=0.70,
        fusion_delta=0.0,
    )
    assert trust == pytest.approx(0.65, abs=1e-4)


# ───────────────────────────── soft cap behavior ─────────────────────────────


def test_evidence_trust_collapse_caps_imaging_at_LOW_TRUST_IMAGING_CAP():
    """When the product of trust factors collapses below LOW_TRUST_THRESHOLD,
    imaging_score must not exceed LOW_TRUST_IMAGING_CAP. We simulate the
    analyze() math here directly."""
    semantic_trust, _ = _semantic_trust(
        semantic_alignment="misaligned",
        medical_relevance_score=0.05,
        medical_plausibility=0.10,
        fake_medical_score=0.95,
    )
    refiner_trust, _ = _refiner_trust(
        refiner_top_type="healthy_xray",
        refiner_group_scores={"pneumonia_xray": 0.05, "healthy_xray": 0.85},
        semantic_margin=0.01,
        predicted_class="pneumonia_xray",
    )
    spatial_trust, _ = _spatial_trust(
        bilateral_burden=0.10,
        uncertainty_score=0.80,
        fusion_delta=0.0,
    )
    evidence_trust = semantic_trust * refiner_trust * spatial_trust
    assert evidence_trust <= LOW_TRUST_THRESHOLD

    # Simulate analyze(): raw=0.95, prior_corrected≈0.95×0.775≈0.736
    raw_pneumonia_score = 0.95
    prior_corrected = 0.95 * (0.5 / 0.6450)
    imaging_score = min(prior_corrected, raw_pneumonia_score * evidence_trust)
    if evidence_trust <= LOW_TRUST_THRESHOLD:
        imaging_score = min(imaging_score, LOW_TRUST_IMAGING_CAP)
    assert imaging_score <= LOW_TRUST_IMAGING_CAP + 1e-6


# ─────────────────────── _doubt_healthy_prediction ──────────────────────────
# Phase 29: votes are now weighted (refiner=2, misaligned=2, bilateral=1),
# threshold ≥ 4. Bilateral alone or bilateral + ONE strong is no longer
# enough — strong-strong (refiner+misaligned) or all-three required.


def test_doubt_healthy_no_votes_does_not_trigger():
    triggered, votes, weight = _doubt_healthy_prediction(
        predicted_class="healthy_xray",
        is_ood=False,
        refiner_top_type="healthy_xray",
        bilateral_burden=0.10,
        semantic_alignment="aligned",
    )
    assert triggered is False
    assert votes == []
    assert weight == 0


def test_doubt_healthy_single_vote_does_not_trigger():
    triggered, votes, weight = _doubt_healthy_prediction(
        predicted_class="healthy_xray",
        is_ood=False,
        refiner_top_type="healthy_xray",
        bilateral_burden=0.60,   # bilateral weight = 1
        semantic_alignment="aligned",
    )
    assert triggered is False
    assert weight == 1


def test_doubt_healthy_refiner_plus_bilateral_does_NOT_trigger():
    """Phase 29 calmness: refiner (2) + bilateral (1) = 3 < 4 → no trigger.
    Previously this combination DID trigger; now it doesn't because bilateral
    alone with one strong signal is treated as weak dissent."""
    triggered, votes, weight = _doubt_healthy_prediction(
        predicted_class="healthy_xray",
        is_ood=False,
        refiner_top_type="pneumonia_xray",   # weight 2
        bilateral_burden=0.60,                # weight 1
        semantic_alignment="aligned",
    )
    assert triggered is False, f"weight {weight} must be < 4 to not trigger"
    assert weight == 3


def test_doubt_healthy_misaligned_plus_bilateral_does_NOT_trigger():
    """Phase 29 calmness: misaligned (2) + bilateral (1) = 3 < 4 → no trigger."""
    triggered, votes, weight = _doubt_healthy_prediction(
        predicted_class="healthy_xray",
        is_ood=False,
        refiner_top_type="healthy_xray",
        bilateral_burden=0.60,                # weight 1
        semantic_alignment="misaligned",      # weight 2
    )
    assert triggered is False
    assert weight == 3


def test_doubt_healthy_refiner_plus_misaligned_triggers():
    """Phase 29: two strong dissents (refiner + misaligned) = weight 4 → trigger."""
    triggered, votes, weight = _doubt_healthy_prediction(
        predicted_class="healthy_xray",
        is_ood=False,
        refiner_top_type="pneumonia_xray",    # weight 2
        bilateral_burden=0.10,
        semantic_alignment="misaligned",       # weight 2
    )
    assert triggered is True
    assert weight == 4
    assert len(votes) == 2


def test_doubt_healthy_three_votes_triggers():
    triggered, votes, weight = _doubt_healthy_prediction(
        predicted_class="healthy_xray",
        is_ood=False,
        refiner_top_type="pneumonia_xray",
        bilateral_burden=0.70,
        semantic_alignment="misaligned",
    )
    assert triggered is True
    assert weight == 5  # 2 + 1 + 2
    assert len(votes) == 3


def test_doubt_healthy_skipped_for_pneumonia_class():
    """The helper must not trigger when classifier already predicts pneumonia."""
    triggered, votes, weight = _doubt_healthy_prediction(
        predicted_class="pneumonia_xray",
        is_ood=False,
        refiner_top_type="pneumonia_xray",
        bilateral_burden=0.80,
        semantic_alignment="misaligned",
    )
    assert triggered is False
    assert votes == []
    assert weight == 0


def test_doubt_healthy_skipped_when_ood():
    triggered, votes, weight = _doubt_healthy_prediction(
        predicted_class="healthy_xray",
        is_ood=True,
        refiner_top_type="pneumonia_xray",
        bilateral_burden=0.80,
        semantic_alignment="misaligned",
    )
    assert triggered is False
    assert votes == []
    assert weight == 0


def test_doubt_healthy_pinned_score_is_cautious_moderate():
    """Phase 29: pin lowered to 0.40 (LOW/MODERATE boundary) for calmness."""
    assert HEALTHY_DOUBT_PINNED_SCORE == 0.40


# ───────────────────────────── happy path (no drift) ─────────────────────────


def test_pneumonia_full_evidence_no_drift():
    """High raw score, all trust factors ≈ 1.0 → imaging_score lands near
    prior_corrected (~raw × 0.775), not below. Captures regression on the
    'everything looks good' happy path."""
    semantic_trust, _ = _semantic_trust(
        semantic_alignment="aligned",
        medical_relevance_score=0.85,
        medical_plausibility=0.80,
        fake_medical_score=0.05,
    )
    refiner_trust, _ = _refiner_trust(
        refiner_top_type="pneumonia_xray",
        refiner_group_scores={"pneumonia_xray": 0.90, "healthy_xray": 0.05},
        semantic_margin=0.40,
        predicted_class="pneumonia_xray",
    )
    spatial_trust, _ = _spatial_trust(
        bilateral_burden=0.70,
        uncertainty_score=0.25,
        fusion_delta=0.0,
    )
    assert semantic_trust == 1.0
    assert refiner_trust == 1.0
    assert spatial_trust == 1.0
    evidence_trust = semantic_trust * refiner_trust * spatial_trust
    raw = 0.85
    prior_corrected = raw * (0.5 / 0.6450)
    imaging_score = min(prior_corrected, raw * evidence_trust)
    # No drift more than 0.02 from prior_corrected.
    assert abs(imaging_score - prior_corrected) < 0.02


# ───────────────────────────── integration ─────────────────────────────


def test_analyze_healthy_doubt_pins_to_pinned_score_and_lands_in_MODERATE():
    """End-to-end: classifier says healthy, but refiner says pneumonia AND
    semantic_alignment is misaligned (weight 2+2=4, meets Phase 29 threshold).
    Expect imaging_score pinned, MODERATE tier, [1c/HEALTHY_DOUBT] step in
    the chain, and Phase 25 clinical lift does NOT compound on top."""
    engine = UnifiedMedicalReasoningEngine()
    bilateral_stub = SimpleNamespace(bilateral_burden=0.70)

    result = engine.analyze(
        predicted_class="healthy_xray",
        calibrated_confidence=0.65,
        probabilities={"healthy_xray": 0.65, "pneumonia_xray": 0.30},
        is_ood=False,
        ood_class=None,
        trust_tier="medium_trust",
        trust_score=0.55,
        calibration_state="stable",
        uncertainty_reason=None,
        semantic_warning=None,
        semantic_alignment="misaligned",   # weight 2 (strong dissent)
        agreement_score=0.50,
        uncertainty_score=0.30,
        fusion_delta=0.0,
        clinical_context=None,
        bilateral_score=bilateral_stub,
        medical_relevance_score=0.60,
        medical_plausibility=0.65,
        fake_medical_score=0.10,
        semantic_margin=0.20,
        refiner_top_type="pneumonia_xray", # weight 2 (strong dissent)
        refiner_group_scores={"pneumonia_xray": 0.55, "healthy_xray": 0.30},
        source_filename="test.jpg",
    )

    # Imaging score lifted to (or above) the pinned score.
    assert result.imaging_score >= HEALTHY_DOUBT_PINNED_SCORE - 1e-6
    # Risk tier landed in MODERATE.
    assert result.risk_tier == MedicalRiskTier.MODERATE
    # New chain step appears.
    assert any("[1c/HEALTHY_DOUBT]" in step for step in result.reasoning_chain)
    # Phase 25 lift did NOT also fire (no double-elevation).
    assert not any("[4/LIFT]" in step for step in result.reasoning_chain)
    # Healthy-doubt warning recorded.
    assert any(
        "Healthy prediction contradicted by independent evidence" in w
        for w in result.pipeline_warnings
    )


def test_analyze_clean_healthy_no_doubt_no_lift_stays_LOW():
    """Regression: an obviously-healthy image with no dissenting evidence
    still lands in LOW with no extra warnings."""
    engine = UnifiedMedicalReasoningEngine()
    bilateral_stub = SimpleNamespace(bilateral_burden=0.08)

    result = engine.analyze(
        predicted_class="healthy_xray",
        calibrated_confidence=0.92,
        probabilities={"healthy_xray": 0.92, "pneumonia_xray": 0.05},
        is_ood=False,
        ood_class=None,
        trust_tier="high_trust",
        trust_score=0.85,
        calibration_state="stable",
        uncertainty_reason=None,
        semantic_warning=None,
        semantic_alignment="aligned",
        agreement_score=0.85,
        uncertainty_score=0.15,
        fusion_delta=0.0,
        clinical_context=None,
        bilateral_score=bilateral_stub,
        medical_relevance_score=0.80,
        medical_plausibility=0.85,
        fake_medical_score=0.05,
        semantic_margin=0.40,
        refiner_top_type="healthy_xray",
        refiner_group_scores={"pneumonia_xray": 0.05, "healthy_xray": 0.90},
        source_filename="clean.jpg",
    )

    assert result.risk_tier == MedicalRiskTier.LOW
    assert not any("[1c/HEALTHY_DOUBT]" in step for step in result.reasoning_chain)
    assert not any(
        "Healthy prediction contradicted" in w
        for w in result.pipeline_warnings
    )


# ───────────────────────────── Phase 29 calmness ──────────────────────────────


def test_semantic_uncertain_is_softer_than_misaligned():
    """Phase 29: 'uncertain' (×0.65) must score higher than 'misaligned' (×0.40)
    so weak semantic noise no longer feels as aggressive as positive dissent."""
    uncertain_trust, _ = _semantic_trust(
        semantic_alignment="uncertain",
        medical_relevance_score=None,
        medical_plausibility=None,
        fake_medical_score=None,
    )
    misaligned_trust, _ = _semantic_trust(
        semantic_alignment="misaligned",
        medical_relevance_score=None,
        medical_plausibility=None,
        fake_medical_score=None,
    )
    assert uncertain_trust == pytest.approx(0.65, abs=1e-4)
    assert uncertain_trust > misaligned_trust


def test_low_trust_cap_suppresses_clinical_lift():
    """Phase 29: when imaging evidence collapses (evidence_trust ≤ LOW_TRUST_
    THRESHOLD) on a pneumonia prediction, clinical lift must NOT fire on
    top — even with severe clinical alarm. Prevents low-trust + symptoms
    from compounding into MODERATE."""
    from app.modules.vision.medical.unified_reasoning import ClinicalContext

    engine = UnifiedMedicalReasoningEngine()
    severe_ctx = ClinicalContext(
        symptoms=["fever", "dyspnea", "cough", "hypoxia"],
        respiratory_severity="severe",
        oxygenation_context="severe_drop",
        fever_severity="high",
        recent_worsening="rapid_48h",
    )

    # Inputs that collapse all three trust factors to ~floor:
    result = engine.analyze(
        predicted_class="pneumonia_xray",
        calibrated_confidence=0.85,
        probabilities={"healthy_xray": 0.10, "pneumonia_xray": 0.85},
        is_ood=False,
        ood_class=None,
        trust_tier="low_trust",
        trust_score=0.25,
        calibration_state="stable",
        uncertainty_reason="multi_branch_disagreement",
        semantic_warning="low_relevance",
        semantic_alignment="misaligned",
        agreement_score=0.20,
        uncertainty_score=0.75,
        fusion_delta=-0.05,
        clinical_context=severe_ctx,
        bilateral_score=SimpleNamespace(bilateral_burden=0.15),
        medical_relevance_score=0.10,
        medical_plausibility=0.20,
        fake_medical_score=0.50,
        semantic_margin=0.02,
        refiner_top_type="healthy_xray",
        refiner_group_scores={"pneumonia_xray": 0.10, "healthy_xray": 0.80},
        source_filename="low_trust.jpg",
    )
    # Low-trust cap forces the imaging score ≤ 0.30, AND clinical lift
    # must be suppressed → final stays in LOW.
    assert result.imaging_score <= 0.30 + 1e-6
    assert result.risk_tier == MedicalRiskTier.LOW, (
        f"low-trust + severe clinical must stay LOW, got {result.risk_tier} "
        f"(final={result.final_score:.3f})"
    )
    # The escalation telemetry should record both signals.
    assert result.escalation_reason_count >= 1   # low_trust_cap counted
    # clinical_lift must NOT be in the escalations list.
    assert not any("[4/LIFT]" in step for step in result.reasoning_chain)


def test_calmness_telemetry_fields_present_on_clean_healthy():
    """Clean healthy image with no dissents → telemetry zeros across the board."""
    engine = UnifiedMedicalReasoningEngine()
    result = engine.analyze(
        predicted_class="healthy_xray",
        calibrated_confidence=0.92,
        probabilities={"healthy_xray": 0.92, "pneumonia_xray": 0.05},
        is_ood=False,
        ood_class=None,
        trust_tier="high_trust",
        trust_score=0.85,
        calibration_state="stable",
        uncertainty_reason=None,
        semantic_warning=None,
        semantic_alignment="aligned",
        agreement_score=0.85,
        uncertainty_score=0.15,
        fusion_delta=0.0,
        clinical_context=None,
        bilateral_score=SimpleNamespace(bilateral_burden=0.08),
        medical_relevance_score=0.80,
        medical_plausibility=0.85,
        fake_medical_score=0.05,
        semantic_margin=0.40,
        refiner_top_type="healthy_xray",
        refiner_group_scores={"pneumonia_xray": 0.05, "healthy_xray": 0.90},
        source_filename="clean.jpg",
    )
    assert result.escalation_reason_count == 0
    assert result.weak_signal_count == 0
    assert result.disagreement_strength == 0.0
    # No [1d/CALMNESS] step because nothing to surface.
    assert not any("[1d/CALMNESS]" in step for step in result.reasoning_chain)


def test_calmness_telemetry_records_near_miss_healthy_doubt():
    """A healthy image with ONE strong dissent (weight 2 < 4 threshold)
    should NOT trigger doubt-pin but SHOULD surface as a weak signal."""
    engine = UnifiedMedicalReasoningEngine()
    result = engine.analyze(
        predicted_class="healthy_xray",
        calibrated_confidence=0.75,
        probabilities={"healthy_xray": 0.75, "pneumonia_xray": 0.20},
        is_ood=False,
        ood_class=None,
        trust_tier="medium_trust",
        trust_score=0.60,
        calibration_state="stable",
        uncertainty_reason=None,
        semantic_warning=None,
        semantic_alignment="aligned",
        agreement_score=0.65,
        uncertainty_score=0.30,
        fusion_delta=0.0,
        clinical_context=None,
        bilateral_score=SimpleNamespace(bilateral_burden=0.08),
        medical_relevance_score=0.70,
        medical_plausibility=0.75,
        fake_medical_score=0.10,
        semantic_margin=0.20,
        refiner_top_type="pneumonia_xray",  # ONE strong dissent (weight 2)
        refiner_group_scores={"pneumonia_xray": 0.55, "healthy_xray": 0.40},
        source_filename="near_miss.jpg",
    )
    # Doubt did NOT trigger (weight 2 < threshold 4).
    assert result.risk_tier == MedicalRiskTier.LOW
    assert not any("[1c/HEALTHY_DOUBT]" in step for step in result.reasoning_chain)
    # But the near-miss IS recorded as a weak signal.
    assert result.weak_signal_count >= 1
    assert result.disagreement_strength > 0.0
    # And the [1d/CALMNESS] step shows up in the chain.
    assert any("[1d/CALMNESS]" in step for step in result.reasoning_chain)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
