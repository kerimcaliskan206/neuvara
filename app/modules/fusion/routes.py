"""
Fusion API routes.

Endpoints
---------
POST /fusion/predict
    Combine ML risk prediction + optional vision evidence into a single
    multimodal risk assessment. ML is always the primary signal.

GET  /fusion/health
    Reports whether both ML and vision services are loaded and ready.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.dependencies import get_inference_service
from app.modules.fusion.engine import MultimodalFusionEngine
from app.modules.fusion.schema import MLResult, VisionResult
from app.modules.ml.inference.service import InferenceService
from app.schemas.fusion import (
    ExplanationPayloadResponse,
    FusionRequest,
    FusionResponse,
    FusionWeightsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fusion", tags=["fusion"])

_engine = MultimodalFusionEngine()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_fusion(
    body: FusionRequest,
    ml_service: InferenceService,
) -> dict[str, Any]:
    """Synchronous fusion pipeline. Invoked via asyncio.to_thread."""

    # ── 1. ML inference (primary signal) ──────────────────────────────────────
    ml_raw = ml_service.predict_single(body.patient.model_dump())

    ml_probability = ml_raw.get("probability")
    if ml_probability is None:
        # Model has no predict_proba — treat prediction as probability proxy
        ml_probability = float(ml_raw["prediction"])

    ml_result = MLResult(
        prediction=ml_raw["prediction"],
        label=ml_raw["label"],
        probability=float(ml_probability),
        confidence=ml_raw["confidence"],
        model_name=ml_raw.get("model_name"),
        model_version=ml_raw.get("model_version"),
    )

    # ── 2. Vision input (supporting evidence, optional) ────────────────────────
    vision_result: Optional[VisionResult] = None
    if body.vision is not None:
        vision_result = VisionResult(**body.vision.model_dump())

    # ── 3. Fusion ──────────────────────────────────────────────────────────────
    result = _engine.fuse(ml_result=ml_result, vision_result=vision_result)

    # ── 4. Serialise ───────────────────────────────────────────────────────────
    ep = result.explanation_payload
    wu = result.weights_used

    return {
        "final_risk_score": result.final_risk_score,
        "risk_level": result.risk_level.value,
        "fusion_confidence": result.fusion_confidence.value,
        "ml_risk_score": result.ml_risk_score,
        "ml_contribution": result.ml_contribution,
        "vision_contribution": result.vision_contribution,
        "vision_status": result.vision_status.value,
        "vision_rejection_reason": result.vision_rejection_reason,
        "uncertainty_flags": result.uncertainty_flags,
        "weights_used": {
            "ml_weight": wu.ml_weight,
            "vision_weight": wu.vision_weight,
            "vision_status": wu.vision_status.value,
            "reason": wu.reason,
        },
        "explanation_payload": {
            "risk_level": ep.risk_level,
            "final_risk_score": ep.final_risk_score,
            "ml_probability": ep.ml_probability,
            "ml_label": ep.ml_label,
            "ml_confidence": ep.ml_confidence,
            "vision_used": ep.vision_used,
            "vision_class": ep.vision_class,
            "vision_confidence": ep.vision_confidence,
            "vision_status": ep.vision_status,
            "vision_rejection_reason": ep.vision_rejection_reason,
            "uncertainty_flags": ep.uncertainty_flags,
            "dominant_signal": ep.dominant_signal,
            "near_risk_boundary": ep.near_risk_boundary,
            "risk_proximity": ep.risk_proximity,
        },
        "near_risk_boundary": result.near_risk_boundary,
        "risk_proximity": result.risk_proximity,
        "ml_model_name": result.ml_model_name,
        "ml_model_version": result.ml_model_version,
        "vision_model_name": result.vision_model_name,
        "vision_model_version": result.vision_model_version,
        "ml_raw": ml_raw,
    }


@router.post(
    "/predict",
    response_model=FusionResponse,
    summary="Multimodal risk fusion (ML + vision)",
    responses={
        503: {"description": "ML model not loaded."},
    },
)
async def fusion_predict(
    body: FusionRequest,
    ml_service: InferenceService = Depends(get_inference_service),
) -> FusionResponse:
    """
    Combine structured ML risk with optional vision evidence into one risk score.

    **Signal hierarchy (enforced):**
    - Structured ML is always the **primary** signal (weight ≥ 0.75).
    - Vision is **supporting evidence** only (weight ≤ 0.25).
    - Vision is **automatically ignored** when: not provided / rejected by
      threshold / classified as unrelated / confidence below minimum.

    **Safe fallback chain:**
    1. No ``vision`` field → pure ML result (α=1.0, β=0).
    2. ``vision.accepted=false`` → vision ignored, pure ML.
    3. ``vision.confidence`` below threshold → vision ignored, pure ML.
    4. Vision accepted → ML-primary weighted blend.

    **This is a medical-support tool, not a diagnosis system.**
    All results must be reviewed by a qualified clinician.
    """
    try:
        result = await asyncio.to_thread(_run_fusion, body, ml_service)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except Exception:
        logger.exception("Fusion predict failed unexpectedly")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fusion inference error. Check server logs for details.",
        )

    return FusionResponse(**result)


@router.get(
    "/health",
    summary="Fusion service readiness",
)
async def fusion_health(request: Request) -> dict:
    """
    Returns readiness of both ML and vision services.

    Fusion always works if ML is ready. Vision is optional.
    """
    from app.modules.ml.inference.service import InferenceService
    from app.modules.vision.inference.service import VisionInferenceService

    ml_svc: Optional[InferenceService] = getattr(
        request.app.state, "inference_service", None
    )
    vision_svc: Optional[VisionInferenceService] = getattr(
        request.app.state, "vision_service", None
    )

    ml_ready = ml_svc is not None and ml_svc.is_ready
    vision_ready = vision_svc is not None and vision_svc.is_ready

    return {
        "fusion_ready": ml_ready,
        "ml_ready": ml_ready,
        "vision_ready": vision_ready,
        "ml_model": ml_svc.model_name if ml_svc and ml_ready else None,
        "ml_version": ml_svc.model_version if ml_svc and ml_ready else None,
        "vision_model": vision_svc.architecture if vision_svc and vision_ready else None,
        "vision_version": vision_svc.version if vision_svc and vision_ready else None,
        "note": (
            "Vision is optional supporting evidence. "
            "Fusion works with ML alone when no image is provided."
        ),
        "timestamp": _utc_now(),
    }
