"""
Unified Medical Reasoning API schemas — Phase 17.

Single response contract for the POST /api/v1/medical/analyze endpoint.
Replaces the fragmented VisionPredictionResponse + FusionResult pattern
with one self-describing UnifiedAnalysisSession.

Signal hierarchy encoded in schema:
  imaging     → primary signal, always present
  clinical    → bounded modifier, optional
  fusion      → advisory weight, optional
  trust       → calibration tier, optional
  risk        → final verdict, always present
  explainability → GradCAM + reasoning chain, always present
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Risk tier ─────────────────────────────────────────────────────────────────


class MedicalRiskTierSchema(str, Enum):
    LOW                     = "LOW"
    MODERATE                = "MODERATE"
    HIGH_DIFFERENTIAL_RISK  = "HIGH_DIFFERENTIAL_RISK"
    CRITICAL_PULMONARY_RISK = "CRITICAL_PULMONARY_RISK"


# ── Sub-schemas ───────────────────────────────────────────────────────────────


class ImagingSignal(BaseModel):
    """Primary signal — EfficientNet + temperature calibration."""

    predicted_class:        str
    calibrated_confidence:  float = Field(ge=0.0, le=1.0)
    raw_confidence:         float = Field(ge=0.0, le=1.0)
    temperature_applied:    float = Field(
        description="T* used for confidence calibration (Phase 15). T<1 = model was underconfident."
    )
    class_probabilities:    dict[str, float]
    ood_detected:           bool
    ood_class:              Optional[str] = None
    imaging_score:          float = Field(
        ge=0.0, le=1.0,
        description="Calibrated confidence used as the primary risk signal."
    )
    model_version:          str
    inference_ms:           float


class ClinicalModifierSchema(BaseModel):
    """Bounded clinical influence — never fully overrides imaging."""

    provided:               bool = Field(description="Whether clinical data was submitted.")
    clinical_delta:         float = Field(
        ge=-0.15, le=0.15,
        description="Applied clinical score modifier. Bounded to ±0.15 of imaging score."
    )
    delta_direction:        str  = Field(
        description="'upward' | 'downward' | 'neutral'"
    )
    symptoms_flagged:       list[str]
    exposure_flagged:       Optional[str] = None
    symptom_score:          float = Field(ge=0.0, le=1.0)
    exposure_score:         float = Field(ge=0.0, le=1.0)
    contradiction_detected: bool
    contradiction_severity: Optional[str] = Field(
        None, description="'mild' | 'moderate' | 'severe' — when imaging and clinical disagree."
    )
    contradiction_note:     Optional[str] = None
    weight_applied:         float = Field(
        ge=0.0, le=1.0,
        description="Weight factor from contradiction analysis. 1.0=no contradiction, 0.2=severe."
    )


class FusionReasoning(BaseModel):
    """Advisory fusion intelligence — alignment of semantic and imaging signals."""

    imaging_weight:         float = Field(
        description="Imaging signal dominance (always > 0.5 in imaging-first policy)."
    )
    clinical_weight:        float = Field(
        description="Bounded clinical contribution (<= 0.15 of total score)."
    )
    semantic_alignment:     str  = Field(
        description="'aligned' | 'misaligned' | 'uncertain'"
    )
    agreement_score:        float = Field(ge=0.0, le=1.0)
    uncertainty_score:      float = Field(ge=0.0, le=1.0)
    fusion_delta:           float = Field(
        description="Semantic-classifier alignment delta (advisory, ±0.08 max)."
    )
    ood_guard_applied:      bool


class UnifiedTrustReport(BaseModel):
    """Calibration V2 trust assessment — post-fusion advisory."""

    trust_tier:             str  = Field(
        description="'very_high_trust' | 'high_trust' | 'moderate_trust' | 'uncertain' | 'suspicious'"
    )
    trust_score:            float = Field(ge=0.0, le=1.0)
    calibration_state:      str  = Field(
        description="'stable' | 'near_threshold' | 'softened' | 'suspicious'"
    )
    ece_at_training:        float = Field(
        description="ECE after temperature scaling (Phase 15). Target: < 0.05."
    )
    temperature_used:       float = Field(
        description="Temperature T* applied to logits before softmax."
    )
    uncertainty_reason:     Optional[str] = None
    semantic_warning:       Optional[str] = None


class UnifiedRiskAssessment(BaseModel):
    """Final risk verdict — imaging-first, clinically-assisted."""

    risk_tier:              MedicalRiskTierSchema
    final_score:            float = Field(ge=0.0, le=1.0)
    imaging_score:          float = Field(ge=0.0, le=1.0)
    clinical_modifier:      float = Field(
        ge=-0.15, le=0.15,
        description="Clinical delta actually applied to imaging score."
    )
    near_boundary:          bool  = Field(
        description="True when final_score is within 0.05 of a tier boundary."
    )
    boundary_proximity:     float = Field(
        description="Distance to nearest tier boundary."
    )
    requires_immediate_action: bool = Field(
        description="True only for CRITICAL_PULMONARY_RISK. NOT a diagnosis."
    )
    differential_classes:   list[str] = Field(
        description="Clinically plausible differential diagnoses for this tier."
    )

    # Tier thresholds for UI rendering
    tier_thresholds: dict[str, float] = Field(
        default={
            "LOW_upper":                   0.35,
            "MODERATE_upper":              0.60,
            "HIGH_DIFFERENTIAL_RISK_upper": 0.80,
        },
        description="Score boundaries between tiers.",
    )


class UnifiedExplainability(BaseModel):
    """GradCAM + structured reasoning chain + narrative summaries."""

    summary:                    str  = Field(description="One-sentence final risk summary.")
    imaging_findings:           str  = Field(description="What the imaging signal detected.")
    clinical_context_applied:   Optional[str] = Field(
        None, description="Clinical context summary as applied."
    )
    contradiction_note:         Optional[str] = None
    gradcam_base64:             Optional[str] = Field(
        None, description="Base64 JPEG GradCAM overlay."
    )
    gradcam_target_class:       Optional[str] = None
    reasoning_chain:            list[str] = Field(
        description="Ordered reasoning steps from imaging to final verdict."
    )
    pipeline_warnings:          list[str] = Field(
        description="Advisory warnings from pipeline (boundary proximity, trust, etc.)."
    )


class SemanticSignal(BaseModel):
    """CLIP semantic gate result — included for dashboard transparency."""

    label:                  str
    medical_relevance_score: float = Field(ge=0.0, le=1.0)
    ood_score:              float  = Field(ge=0.0, le=1.0)
    gate_passed:            bool
    rejection_code:         Optional[str] = None
    reasoning_decision:     Optional[str] = None
    reasoning_confidence:   Optional[float] = None
    top_matches:            list[dict[str, Any]] = Field(default_factory=list)


# ── Top-level session schema ──────────────────────────────────────────────────


class UnifiedAnalysisSession(BaseModel):
    """
    Complete unified dashboard response for a single analysis session.

    One session = one uploaded image + optional clinical context.
    All downstream components (imaging, semantic, fusion, trust, risk)
    are included in a single coherent response object.

    Design contract
    ---------------
      imaging.imaging_score  → always the primary signal
      clinical.clinical_delta → bounded ±0.15, never overrides imaging
      risk.risk_tier         → derived from final_score (imaging + clinical)
      ood_guard_applied      → if True, clinical was ignored and score capped

    API usage
    ---------
      POST /api/v1/medical/analyze
      Content-Type: multipart/form-data
        file:              image file
        clinical_context:  optional JSON string (ClinicalContextRequest)
        gradcam:           bool (default: true)
    """

    session_id:         str  = Field(description="Unique session identifier.")
    timestamp:          str  = Field(description="ISO 8601 UTC timestamp.")

    imaging:            ImagingSignal
    semantic:           Optional[SemanticSignal] = None
    clinical:           ClinicalModifierSchema
    fusion:             FusionReasoning
    trust:              UnifiedTrustReport
    risk:               UnifiedRiskAssessment
    explainability:     UnifiedExplainability

    # Session-level flags
    ood_guard_applied:          bool
    clinical_override_attempted: bool = Field(
        description="True if clinical context tried to downgrade a CRITICAL finding — blocked."
    )

    # Model provenance
    model_version:      str
    pipeline_version:   str = "v6_phase17"


# ── Request schemas ───────────────────────────────────────────────────────────


class ClinicalContextRequest(BaseModel):
    """
    Optional clinical context submitted with the image.

    All fields are optional. Missing (None) = neutral contribution — NOT treated as normal.
    Clinical data can only shift the imaging score by ±0.15 (MAX_CLINICAL_DELTA).
    """

    # ── Legacy fields (backward-compatible) ──────────────────────────────────
    symptoms:           list[str] = Field(
        default_factory=list,
        description=(
            "Clinical symptoms. Recognized: fever, cough, dyspnea, "
            "shortness_of_breath, chest_pain, hemoptysis, tachypnea, hypoxia, "
            "fatigue, myalgia, night_sweats, weight_loss, wheezing, productive_cough."
        ),
    )
    exposure_history:   Optional[str] = Field(
        None,
        description="hospital | sick_contact | travel | healthcare_worker | immunocompromised",
    )
    duration_days:      Optional[int] = Field(None, ge=0, le=365)
    severity:           Optional[str] = Field(
        None, description="Legacy severity: 'mild' | 'moderate' | 'severe'"
    )
    immunocompromised:  bool = False
    age_group:          Optional[str] = Field(
        None, description="adolescent | young_adult | adult | older_adult | elderly"
    )
    notes:              Optional[str] = Field(None, max_length=500)

    # ── Phase 22 — Structured clinical signals ────────────────────────────────
    age:                Optional[int] = Field(
        None, ge=0, le=120,
        description="Patient age in years. Internally mapped to age_group.",
    )
    sex:                Optional[str] = Field(
        None, description="'male' | 'female'. Very low-weight contextual modifier."
    )
    respiratory_severity: Optional[str] = Field(
        None,
        description="'normal' | 'mild' | 'severe'. Strongest new clinical modifier.",
    )
    oxygenation_context:  Optional[str] = Field(
        None,
        description="'normal' | 'mild_drop' | 'severe_drop'. High-importance signal.",
    )
    fever_severity:       Optional[str] = Field(
        None, description="'none' | 'mild' | 'moderate' | 'high'."
    )
    recent_worsening:     Optional[str] = Field(
        None, description="'none' | 'some' | 'rapid_48h'. Progression modifier."
    )
    rodent_exposure_level: Optional[str] = Field(
        None,
        description=(
            "'none' | 'unsure' | 'rural_env' | 'possible_contact'. "
            "Differential-shaping signal. Does NOT directly create CRITICAL risk."
        ),
    )
    symptom_duration_tier: Optional[str] = Field(
        None, description="'1_2_days' | '3_7_days' | 'over_1_week'."
    )


class ClinicalPersistRequest(BaseModel):
    """Payload sent by the frontend to persist a clinical-only analysis result."""
    session_id:  str
    risk_tier:   str
    final_score: float
    summary:     Optional[str]   = None
    duration_ms: Optional[float] = None
