"""
Schemas for the medical AI assistant endpoint (Phase 26).

MedicalAnalysisContext carries the curated, clinically-relevant subset of
UnifiedAnalysisSession — no raw internal metrics (ECE, T*, fusion_delta, etc.).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MedicalAnalysisContext(BaseModel):
    """
    Curated analysis state injected by the frontend into each assistant request.
    Fields mirror UnifiedAnalysisSession but exclude all engineering internals.
    """
    model_config = ConfigDict(extra="ignore")

    # ── Risk assessment ───────────────────────────────────────────────────────
    risk_tier: str  # LOW | MODERATE | HIGH_DIFFERENTIAL_RISK | CRITICAL_PULMONARY_RISK
    final_score: float = Field(ge=0.0, le=1.0)
    requires_immediate_action: bool = False
    near_boundary: bool = False

    # ── Imaging ───────────────────────────────────────────────────────────────
    has_image: bool
    predicted_class: str | None = None     # healthy_xray | pneumonia_xray | hard_negative | fake_medical
    imaging_score: float | None = Field(default=None, ge=0.0, le=1.0)
    bilateral_burden: float | None = Field(default=None, ge=0.0, le=1.0)

    # ── OOD guard ─────────────────────────────────────────────────────────────
    ood_detected: bool = False
    ood_label: str | None = None           # e.g., "Doğa fotoğrafı"

    # ── Clinical context ──────────────────────────────────────────────────────
    has_clinical: bool = False
    symptoms_flagged: list[str] = Field(default_factory=list)
    respiratory_severity: str | None = None   # normal | mild | severe
    oxygenation_context: str | None = None    # normal | mild_drop | severe_drop
    fever_severity: str | None = None         # none | mild | moderate | high
    recent_worsening: str | None = None       # none | some | rapid_48h
    rodent_exposure_level: str | None = None  # none | unsure | rural_env | possible_contact
    symptom_duration_tier: str | None = None  # 1_2_days | 3_7_days | over_1_week
    exposure_history: str | None = None
    age: int | None = Field(default=None, ge=0, le=130)
    sex: str | None = None                    # male | female

    # ── Reasoning output (Turkish prose from backend) ─────────────────────────
    summary: str
    imaging_findings: str | None = None


class MedicalAssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        min_length=1,
        max_length=2000,
        description="Kullanıcının soru veya mesajı (Türkçe).",
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="Konuşma oturumu kimliği.",
    )
    analysis_context: MedicalAnalysisContext


class MedicalAssistantResponse(BaseModel):
    content: str
    refused: bool = False
    refusal_reason: str | None = None
    model: str
    duration_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    timestamp: str
    session_id: str
