"""
Pydantic schemas for the AI API.

Conventions:
  * All free-text the user sees is Turkish — fields keep English names but
    the assistant's content is always Turkish.
  * Token counts are optional because Ollama only reports them when the
    underlying runner does.
  * Refusals are first-class — clients should branch on ``refused`` rather
    than HTTP status.
"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Requests ─────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000, description="Kullanıcı mesajı.")
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="Konuşma oturumu kimliği. Yoksa 'default' kullanılır.",
    )


class MLInterpretationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    prediction: int = Field(ge=0, le=1)
    label: str
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    feature_summary: str | None = Field(
        default=None,
        description="İsteğe bağlı, hastanın öne çıkan girdilerini özetleyen kısa metin.",
    )


class VisionInterpretationGate(BaseModel):
    enabled: bool = False
    predicted_class: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class VisionInterpretationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    accepted: bool
    predicted_class: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    rejection_reason: str | None = None
    gate: VisionInterpretationGate = Field(default_factory=VisionInterpretationGate)
    model_name: str | None = None
    model_version: str | None = None


class FusionExplanationPayload(BaseModel):
    risk_level: str
    final_risk_score: float = Field(ge=0.0, le=1.0)
    ml_probability: float = Field(ge=0.0, le=1.0)
    ml_label: str
    ml_confidence: str
    vision_used: bool
    vision_class: str | None = None
    vision_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    vision_status: str
    vision_rejection_reason: str | None = None
    uncertainty_flags: list[str] = Field(default_factory=list)
    dominant_signal: str


class FusionInterpretationRequest(BaseModel):
    """Pass the full FusionResponse or just its explanation_payload."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    fusion_confidence: str = Field(default="unknown")
    explanation_payload: FusionExplanationPayload


# ── Responses ────────────────────────────────────────────────────────────────


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    content: str = Field(description="Asistanın Türkçe yanıtı.")
    intent: str = Field(description="Tespit edilen niyet.")
    refused: bool = Field(description="Yanıt güvenlik nedeniyle reddedildi mi?")
    refusal_reason: str | None = Field(
        default=None,
        description="Reddedildiyse sebebi (off_topic | prompt_injection | invalid_input).",
    )
    model: str = Field(description="Kullanılan LLM modeli.")
    duration_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    timestamp: str


class InterpretationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    content: str
    model: str
    duration_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    timestamp: str


class AIHealthResponse(BaseModel):
    ok: bool
    enabled: bool
    base_url: str | None = None
    model: str | None = None
    model_loaded: bool = False
    available_models: list[str] = Field(default_factory=list)
    reason: str | None = None
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)