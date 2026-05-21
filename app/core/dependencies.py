from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.jwt import decode_access_token
from app.models.user import User
from app.repositories.user_repository import UserRepository

# auto_error=False so we surface 401 (no/invalid auth) explicitly instead of
# FastAPI's default 403 for a missing Authorization header.  Frontends and
# spec-compliant REST clients distinguish 401 (please authenticate) from
# 403 (authenticated but not allowed).
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        user_id = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await UserRepository(db).get_by_id(int(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive.")
    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Returns the authenticated User if a valid Bearer token is present, else None."""
    if credentials is None or not credentials.credentials:
        return None
    try:
        user_id = decode_access_token(credentials.credentials)
    except ValueError:
        return None
    return await UserRepository(db).get_by_id(int(user_id))


def get_inference_service(request: Request):
    """
    FastAPI dependency that returns the application-level InferenceService.

    Raises HTTP 503 if no model has been loaded (e.g. no training run yet).
    The service itself is stored in app.state.inference_service during lifespan startup.
    """
    from app.modules.ml.inference.service import InferenceService  # local import avoids circular

    service: InferenceService | None = getattr(
        request.app.state, "inference_service", None
    )
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ML model is not available. "
                "Train a model first: python scripts/train.py"
            ),
        )
    return service


# ── Vision dependencies ──────────────────────────────────────────────────────


def get_vision_inference_service(request: Request):
    """
    FastAPI dependency that returns the main vision inference service.

    The service is created and (optionally) loaded during lifespan startup.
    Raises 503 if no vision model is loaded.
    """
    from app.modules.vision.inference.service import VisionInferenceService  # noqa: WPS433

    service: VisionInferenceService | None = getattr(
        request.app.state, "vision_service", None
    )
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vision model is not available. "
                "Train one first: python scripts/train_vision.py"
            ),
        )
    return service


def get_vision_gate_service(request: Request):
    """
    Returns the related/unrelated gate service if one is loaded, else ``None``.

    Routes that depend on this should treat ``None`` as 'gate disabled'
    rather than an error — the gate is optional infrastructure.
    """
    return getattr(request.app.state, "vision_gate_service", None)


def get_image_upload_handler(request: Request):
    """Returns the shared ImageUploadHandler (created once at startup)."""
    from app.modules.vision.upload.handler import ImageUploadHandler  # noqa: WPS433

    handler: ImageUploadHandler | None = getattr(
        request.app.state, "image_upload_handler", None
    )
    if handler is None:
        # Defensive: shouldn't happen if lifespan ran, but stay graceful.
        handler = ImageUploadHandler()
        request.app.state.image_upload_handler = handler
    return handler


# ── AI dependencies ──────────────────────────────────────────────────────────


def _require_ai_enabled() -> None:
    """Raise 503 when the AI module is switched off via config."""
    from app.modules.ai.config import ai_config  # noqa: WPS433

    if not ai_config.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI servisi devre dışı.",
        )


def get_ai_chat_service(request: Request):
    """Returns the shared AIChatService (created once at startup)."""
    from app.modules.ai.services.chat_service import AIChatService  # noqa: WPS433

    _require_ai_enabled()
    service: AIChatService | None = getattr(request.app.state, "ai_chat_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI sohbet servisi başlatılamadı.",
        )
    return service


def get_ai_health_service(request: Request):
    """Returns the shared AIHealthService (always available — even if AI disabled)."""
    from app.modules.ai.services.health import AIHealthService  # noqa: WPS433

    service: AIHealthService | None = getattr(
        request.app.state, "ai_health_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI sağlık servisi başlatılamadı.",
        )
    return service


def get_ml_interpretation_service(request: Request):
    from app.modules.ai.services.interpretation import (  # noqa: WPS433
        MLInterpretationService,
    )

    _require_ai_enabled()
    service: MLInterpretationService | None = getattr(
        request.app.state, "ai_ml_interpretation_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML yorumlama servisi başlatılamadı.",
        )
    return service


def get_vision_interpretation_service(request: Request):
    from app.modules.ai.services.interpretation import (  # noqa: WPS433
        VisionInterpretationService,
    )

    _require_ai_enabled()
    service: VisionInterpretationService | None = getattr(
        request.app.state, "ai_vision_interpretation_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Görüntü yorumlama servisi başlatılamadı.",
        )
    return service


def get_fusion_interpretation_service(request: Request):
    from app.modules.ai.services.interpretation import (  # noqa: WPS433
        FusionInterpretationService,
    )

    _require_ai_enabled()
    service: FusionInterpretationService | None = getattr(
        request.app.state, "ai_fusion_interpretation_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Füzyon yorumlama servisi başlatılamadı.",
        )
    return service


def get_medical_assistant_service(request: Request):
    """Returns the shared MedicalAssistantService (created once at startup)."""
    from app.modules.ai.services.medical_assistant import MedicalAssistantService  # noqa: WPS433

    _require_ai_enabled()
    service: MedicalAssistantService | None = getattr(
        request.app.state, "ai_medical_assistant_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tıbbi asistan servisi başlatılamadı.",
        )
    return service
