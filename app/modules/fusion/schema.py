"""
Fusion input/output schemas.

All public-facing types used by the fusion engine and its API route.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FusionConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VisionStatus(str, Enum):
    USED = "used"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    UNRELATED = "unrelated"
    LOW_CONFIDENCE = "low_confidence"


# ── Input types ───────────────────────────────────────────────────────────────


class MLResult(BaseModel):
    """Structured ML prediction from InferenceService.predict_single()."""

    prediction: int = Field(description="0=negative, 1=positive")
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None


class VisionResult(BaseModel):
    """
    Vision prediction to be fused. Maps to VisionPredictionResponse fields.

    Only ``accepted``, ``predicted_class``, ``confidence``, and
    ``rejection_reason`` affect fusion; the rest is metadata.
    """

    accepted: bool
    predicted_class: Optional[str] = None
    predicted_class_index: Optional[int] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    probabilities: Optional[dict[str, float]] = None
    rejection_reason: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    gradcam_base64: Optional[str] = None


# ── Output types ──────────────────────────────────────────────────────────────


class FusionWeightsUsed(BaseModel):
    """Actual α/β weights applied in this fusion call."""

    ml_weight: float
    vision_weight: float
    vision_status: VisionStatus
    reason: str


class ExplanationPayload(BaseModel):
    """Structured payload for the AI assistant to use in explanation generation."""

    risk_level: str
    final_risk_score: float
    ml_probability: float
    ml_label: str
    ml_confidence: str
    vision_used: bool
    vision_class: Optional[str]
    vision_confidence: Optional[float]
    vision_status: str
    vision_rejection_reason: Optional[str]
    uncertainty_flags: list[str]
    dominant_signal: str
    near_risk_boundary: bool = False
    risk_proximity: float = 0.0


class FusionResult(BaseModel):
    """Complete output of the multimodal fusion engine."""

    # ── Core verdict ─────────────────────────────────────────────────────────
    final_risk_score: float = Field(ge=0.0, le=1.0, description="Composite risk 0–1")
    risk_level: RiskLevel
    fusion_confidence: FusionConfidence

    # ── Per-signal contributions ──────────────────────────────────────────────
    ml_risk_score: float = Field(ge=0.0, le=1.0)
    ml_contribution: float = Field(ge=0.0, le=1.0, description="Weighted ML term")
    vision_contribution: float = Field(
        ge=0.0, le=1.0, description="Weighted vision term (0 if ignored)"
    )

    # ── Vision handling ───────────────────────────────────────────────────────
    vision_status: VisionStatus
    vision_rejection_reason: Optional[str] = None

    # ── Uncertainty signals ───────────────────────────────────────────────────
    uncertainty_flags: list[str] = Field(default_factory=list)

    # ── Weights used ─────────────────────────────────────────────────────────
    weights_used: FusionWeightsUsed

    # ── Explainability ────────────────────────────────────────────────────────
    explanation_payload: ExplanationPayload

    # ── Boundary proximity ────────────────────────────────────────────────────
    near_risk_boundary: bool = False
    risk_proximity: float = 0.0

    # ── Provenance ────────────────────────────────────────────────────────────
    ml_model_name: Optional[str] = None
    ml_model_version: Optional[str] = None
    vision_model_name: Optional[str] = None
    vision_model_version: Optional[str] = None
