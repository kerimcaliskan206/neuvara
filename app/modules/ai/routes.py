"""
AI API routes.

Endpoints
---------
POST /ai/chat                 — Free-form chat (Turkish, domain-locked).
POST /ai/explain/ml           — Explain a tabular ML prediction.
POST /ai/explain/vision       — Explain a vision prediction.
GET  /ai/health               — Ollama / model health probe.

All blocking work happens inside the service layer. Routes only parse,
delegate, and serialise.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import (
    get_ai_chat_service,
    get_ai_health_service,
    get_fusion_interpretation_service,
    get_ml_interpretation_service,
    get_vision_interpretation_service,
)
from app.modules.ai.providers.base import ProviderError
from app.modules.ai.services.chat_service import AIChatService
from app.modules.ai.services.health import AIHealthService
from app.modules.ai.services.interpretation import (
    FusionInterpretationService,
    MLInterpretationService,
    VisionInterpretationService,
)
from app.schemas.ai import (
    AIHealthResponse,
    ChatRequest,
    ChatResponse,
    FusionInterpretationRequest,
    InterpretationResponse,
    MLInterpretationRequest,
    VisionInterpretationRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="HantaProject yapay zekâ asistanı ile sohbet",
    responses={
        503: {"description": "AI servisi şu anda kullanılamıyor."},
    },
)
async def chat(
    body: ChatRequest,
    service: AIChatService = Depends(get_ai_chat_service),
) -> ChatResponse:
    """
    Türkçe, alan-kilitli sohbet uç noktası. Konu dışı istekler ve
    prompt-injection denemeleri 200 dönüşüyle birlikte ``refused=true``
    olarak işaretlenir; üst hata yalnızca sağlayıcı erişilemezse atılır.
    """
    try:
        result = await service.chat(
            message=body.message,
            session_id=body.session_id,
        )
    except ProviderError as exc:
        logger.error("AI chat: provider error — %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI sağlayıcı erişilemiyor: {exc}",
        )

    return ChatResponse(
        content=result.content,
        intent=result.intent,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        model=result.model,
        duration_ms=result.duration_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        timestamp=result.timestamp,
    )


@router.post(
    "/explain/ml",
    response_model=InterpretationResponse,
    summary="ML tahmin sonucunu Türkçe açıkla",
)
async def explain_ml(
    body: MLInterpretationRequest,
    service: MLInterpretationService = Depends(get_ml_interpretation_service),
) -> InterpretationResponse:
    try:
        result = await service.explain(body.model_dump())
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI sağlayıcı erişilemiyor: {exc}",
        )

    return InterpretationResponse(
        content=result.content,
        model=result.model,
        duration_ms=result.duration_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        timestamp=result.timestamp,
    )


@router.post(
    "/explain/vision",
    response_model=InterpretationResponse,
    summary="Görüntü tahmin sonucunu Türkçe açıkla",
)
async def explain_vision(
    body: VisionInterpretationRequest,
    service: VisionInterpretationService = Depends(get_vision_interpretation_service),
) -> InterpretationResponse:
    try:
        result = await service.explain(body.model_dump())
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI sağlayıcı erişilemiyor: {exc}",
        )

    return InterpretationResponse(
        content=result.content,
        model=result.model,
        duration_ms=result.duration_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        timestamp=result.timestamp,
    )


@router.post(
    "/explain/fusion",
    response_model=InterpretationResponse,
    summary="Çok modlu füzyon sonucunu Türkçe açıkla",
)
async def explain_fusion(
    body: FusionInterpretationRequest,
    service: FusionInterpretationService = Depends(get_fusion_interpretation_service),
) -> InterpretationResponse:
    """
    Explain a multimodal fusion result in plain Turkish.

    Pass the full response from ``POST /fusion/predict`` (or at minimum
    ``fusion_confidence`` + ``explanation_payload``).  The AI assistant
    describes what the composite risk score means, how each signal contributed,
    and reminds the user that this is not a clinical diagnosis.
    """
    try:
        result = await service.explain(body.model_dump())
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI sağlayıcı erişilemiyor: {exc}",
        )

    return InterpretationResponse(
        content=result.content,
        model=result.model,
        duration_ms=result.duration_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        timestamp=result.timestamp,
    )


@router.get(
    "/health",
    response_model=AIHealthResponse,
    summary="AI sağlığı: Ollama erişilebilirliği ve modelin yüklü olup olmadığı",
)
async def health(
    service: AIHealthService = Depends(get_ai_health_service),
) -> AIHealthResponse:
    data = await service.check()
    return AIHealthResponse(**data)