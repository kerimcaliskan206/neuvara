"""
API-level request/response schemas for the fusion endpoint.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.predict import PatientInput


# ── Request ───────────────────────────────────────────────────────────────────


class FusionVisionInput(BaseModel):
    """Vision prediction to include in fusion. Mirrors VisionPredictionResponse."""

    model_config = ConfigDict(extra="ignore")

    accepted: bool
    predicted_class: Optional[str] = None
    predicted_class_index: Optional[int] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    probabilities: Optional[dict[str, float]] = None
    rejection_reason: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    gradcam_base64: Optional[str] = None


class FusionRequest(BaseModel):
    """Request body for POST /fusion/predict."""

    model_config = ConfigDict(extra="forbid")

    patient: PatientInput = Field(description="Structured patient features for ML prediction")
    vision: Optional[FusionVisionInput] = Field(
        None,
        description=(
            "Optional vision prediction from /vision/predict. "
            "Omit entirely when no image is available."
        ),
    )
    ml_model_name: Optional[str] = Field(
        None,
        description="Override the ML model to use. Defaults to best available.",
    )


# ── Response ──────────────────────────────────────────────────────────────────


class FusionWeightsResponse(BaseModel):
    ml_weight: float
    vision_weight: float
    vision_status: str
    reason: str


class ExplanationPayloadResponse(BaseModel):
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


class FusionResponse(BaseModel):
    """Response from POST /fusion/predict."""

    # Core verdict
    final_risk_score: float = Field(description="Composite risk 0–1 (ML primary + vision supporting)")
    risk_level: str = Field(description="high | medium | low")
    fusion_confidence: str = Field(description="high | medium | low")

    # Per-signal details
    ml_risk_score: float
    ml_contribution: float
    vision_contribution: float

    # Vision handling
    vision_status: str = Field(description="used | rejected | unrelated | unavailable | low_confidence")
    vision_rejection_reason: Optional[str]

    # Uncertainty
    uncertainty_flags: list[str]

    # Weights
    weights_used: FusionWeightsResponse

    # AI-ready explanation data
    explanation_payload: ExplanationPayloadResponse

    # Model provenance
    ml_model_name: Optional[str]
    ml_model_version: Optional[str]
    vision_model_name: Optional[str]
    vision_model_version: Optional[str]

    # Boundary proximity
    near_risk_boundary: bool = False
    risk_proximity: float = 0.0

    # Inline ML result (convenience — avoids a second API call for callers)
    ml_raw: dict[str, Any] = Field(description="Full ML InferenceService result")
