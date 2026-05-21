import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.dependencies import get_inference_service
from app.modules.ml.inference.service import InferenceService
from app.schemas.predict import (
    BatchPredictionRequest,
    BatchPredictionResult,
    ModelInfoResponse,
    PredictionRequest,
    SinglePredictionResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])


@router.post(
    "/predict",
    response_model=SinglePredictionResult,
    summary="Single-patient hantavirus risk prediction",
)
async def predict(
    body: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> SinglePredictionResult:
    """
    Predict hantavirus infection risk for a single patient.

    All fields are **optional** — the preprocessing pipeline imputes any missing
    value using the statistics from the training data. Sending an empty `patient`
    object `{}` is valid and will return a prediction based entirely on imputed values.

    **Key risk factors (most predictive):**
    - `rodent_contact` — direct exposure to rodents or their droppings
    - `fever`, `myalgia`, `thrombocytopenia` — clinical HPS markers
    - `rodent_density`, `humidity_pct` — environmental risk indicators
    """
    # If a specific model was requested, swap it for this request
    if body.model_name and body.model_name != service.model_name:
        alt = InferenceService(service.config)
        loaded = alt.load(body.model_name)
        if not loaded:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{body.model_name}' not found. "
                       f"Use GET /ml/models/current to see what is available.",
            )
        active = alt
    else:
        active = service

    try:
        result = await asyncio.to_thread(
            active.predict_single,
            body.patient.model_dump(),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except Exception as exc:
        logger.exception("Prediction failed unexpectedly")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference error. Check server logs for details.",
        )

    return result


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResult,
    summary="Batch hantavirus risk prediction (up to 100 patients)",
)
async def predict_batch(
    body: BatchPredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> BatchPredictionResult:
    """
    Predict hantavirus infection risk for a batch of up to 100 patients.

    Returns predictions in the same order as the input list.
    Total inference time is reported at the batch level.
    """
    if body.model_name and body.model_name != service.model_name:
        alt = InferenceService(service.config)
        loaded = alt.load(body.model_name)
        if not loaded:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{body.model_name}' not found.",
            )
        active = alt
    else:
        active = service

    try:
        result = await asyncio.to_thread(
            active.predict_batch,
            [p.model_dump() for p in body.patients],
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except Exception as exc:
        logger.exception("Batch prediction failed unexpectedly")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch inference error. Check server logs for details.",
        )

    return result


@router.get(
    "/models/current",
    response_model=ModelInfoResponse,
    summary="Currently loaded model info",
)
async def current_model(request: Request) -> ModelInfoResponse:
    """
    Returns metadata about the model currently loaded in the inference service.
    Returns **503** if no model is available (train one first).
    """
    service: InferenceService | None = getattr(
        request.app.state, "inference_service", None
    )
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model is currently loaded. Run: python scripts/train.py",
        )
    return ModelInfoResponse(
        model_name=service.model_name,
        model_version=service.model_version,
        is_ready=service.is_ready,
        metadata=service.metadata,
    )
