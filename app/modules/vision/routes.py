"""
Vision API routes.

Endpoints
---------
POST /vision/predict
    Upload an image (multipart/form-data) and receive a structured
    prediction. Optionally returns a Grad-CAM heatmap.

GET  /vision/models/current
    Returns metadata about the currently loaded vision model.

The router is intentionally thin — all heavy lifting lives in services
(``ImageUploadHandler``, ``ImageValidator``, ``VisionInferenceService``).
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

from app.core.dependencies import (
    get_image_upload_handler,
    get_vision_gate_service,
    get_vision_inference_service,
)
from app.modules.vision.config import vision_config
from app.modules.vision.explainability.gradcam import build_gradcam
from app.modules.vision.explainability.heatmap import HeatmapRenderer
from app.modules.vision.inference.acceptance import (
    AcceptanceLevel,
    decide_acceptance,
)
from app.modules.vision.inference.service import VisionInferenceService
from app.modules.vision.upload.handler import ImageUploadHandler
from app.modules.vision.utils.io import load_image_from_bytes
from app.modules.vision.validation.validator import (
    DEFAULT_RELEVANT_CLASS,
    ImageValidator,
)
from app.schemas.vision import (
    GateInfo,
    ImageInfo,
    SemanticInfo,
    SemanticMatch as SemanticMatchSchema,
    UploadInfo,
    VisionExplainabilityResult as VisionExplainSchema,
    VisionFusionResult as VisionFusionSchema,
    VisionModelInfoResponse,
    VisionPredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["vision"])

# Default minimum confidence for the main classifier.  Independent of the
# gate's own threshold; both must pass for the response to be accepted.
#
# Set to 0.50 as a balanced demo default.  At this value:
#   • genuine microscopy with confidence ≥ 0.70 → ACCEPTED (Grad-CAM on)
#   • microscopy with confidence in [0.50, 0.70) → ACCEPTED_LOW_CONFIDENCE
#   • unrelated / hard_negative predictions      → REJECTED regardless of
#                                                   confidence (class-aware)
#   • predicted=target but non-target mass ≥ predicted mass → REJECTED
#   • confidence < 0.50                          → REJECTED
#
# The full pipeline lives in ``inference/acceptance.py``.
DEFAULT_PREDICTION_THRESHOLD = 0.5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enforce_size_limit(content_length: int | None) -> None:
    """Reject obviously oversized uploads before reading bytes into memory."""
    if content_length is None:
        return
    max_bytes = int(vision_config.upload.max_file_size_mb * 1024 * 1024)
    if content_length > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Upload too large: {content_length} bytes > "
                f"{vision_config.upload.max_file_size_mb} MB limit."
            ),
        )


def _gradcam_to_base64(
    service: VisionInferenceService,
    pil_image,
    class_idx: int,
) -> str | None:
    """Generate a Grad-CAM overlay for the given prediction and base64-encode it."""
    if service.model is None or service.architecture is None:
        return None
    try:
        tensor = service._pipeline.preprocess_for_inference(pil_image).to(service.device)
        with build_gradcam(service.model, service.architecture) as cam:
            heatmap = cam.generate(tensor, class_idx=class_idx)
        overlay = HeatmapRenderer().overlay(pil_image, heatmap)
        buf = io.BytesIO()
        overlay.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        logger.exception("Grad-CAM generation failed")
        return None


def _run_prediction(
    *,
    file_bytes: bytes,
    original_filename: str,
    upload_handler: ImageUploadHandler,
    gate_service: VisionInferenceService | None,
    main_service: VisionInferenceService,
    gradcam: bool,
    threshold: float,
) -> VisionPredictionResponse:
    """Synchronous prediction pipeline. Invoked via asyncio.to_thread."""
    start = time.perf_counter()

    # ── 1. Persist upload (validates size, format, dimensions internally) ──
    upload = upload_handler.handle(file_bytes, original_filename)
    if not upload.success:
        # Validator already logged details — surface the first error.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=upload.error or "Image validation failed.",
        )

    image = load_image_from_bytes(file_bytes)

    image_info = ImageInfo(
        original_filename=upload.original_filename,
        width=image.width,
        height=image.height,
        format=(upload.metadata.format if upload.metadata else "unknown"),
        size_bytes=len(file_bytes),
    )
    upload_info = UploadInfo(
        stored=True,
        safe_filename=upload.safe_filename,
        storage_path=str(upload.upload_path) if upload.upload_path else None,
    )

    # ── 2. Content-relevance gate (optional) ────────────────────────────
    gate_info: GateInfo
    if gate_service is not None and gate_service.is_ready:
        validator = ImageValidator(
            relevance_service=gate_service,
            relevance_threshold=threshold,
        )
        gate_result = validator.validate_content_relevance(image)
        info = gate_result.image_info or {}
        gate_info = GateInfo(
            enabled=True,
            predicted_class=info.get("gate_predicted_class"),
            confidence=info.get("gate_confidence"),
            threshold=info.get("gate_threshold"),
            relevant_class=validator.relevant_class,
        )
        if not gate_result.passed:
            reason = gate_result.errors[0] if gate_result.errors else "Rejected by gate."
            logger.info(
                "Vision predict: gate rejected upload=%s reason=%s",
                upload.safe_filename, reason,
            )
            return VisionPredictionResponse(
                accepted=False,
                acceptance_level=AcceptanceLevel.REJECTED.value,
                low_confidence_reason=None,
                predicted_class=None,
                predicted_class_index=None,
                confidence=info.get("gate_confidence"),
                probabilities=None,
                threshold=threshold,
                rejection_reason=reason,
                gate=gate_info,
                image=image_info,
                upload=upload_info,
                model_name=main_service.architecture or "unknown",
                model_version=main_service.version or "unknown",
                inference_duration_ms=round((time.perf_counter() - start) * 1000, 2),
                gradcam_base64=None,
                timestamp=_utc_now(),
            )
    else:
        gate_info = GateInfo(
            enabled=False,
            predicted_class=None,
            confidence=None,
            threshold=None,
            relevant_class=DEFAULT_RELEVANT_CLASS,
        )

    # ── 3. Semantic gate (CLIP) ────────────────────────────────────────
    semantic_info: SemanticInfo | None = None
    try:
        from app.modules.vision.semantic.semantic_analyzer import get_semantic_analyzer
        from app.modules.vision.semantic.semantic_gate import semantic_gate

        sem_result = get_semantic_analyzer().analyze(image)
        gate_outcome = semantic_gate.evaluate(sem_result)

        rsn = gate_outcome.reasoning  # ReasoningOutput | None
        semantic_info = SemanticInfo(
            label=sem_result.top_semantic_label,
            medical_relevance_score=sem_result.medical_relevance_score,
            ood_score=sem_result.ood_score,
            rejection_code=gate_outcome.rejection_code,
            rejection_reason=gate_outcome.rejection_reason,
            triggered_by=gate_outcome.triggered_by,
            top_matches=[
                SemanticMatchSchema(label=m.label, score=m.score, rank=m.rank)
                for m in sem_result.top_matches[:3]
            ],
            inference_ms=sem_result.inference_ms,
            # Reasoning layer fields (None when reasoner unavailable)
            reasoning_type=rsn.reasoning_type if rsn else None,
            reasoning_confidence=rsn.reasoning_confidence if rsn else None,
            reasoning_decision=rsn.semantic_decision if rsn else None,
            semantic_uncertainty=rsn.semantic_uncertainty if rsn else None,
            semantic_consistency=rsn.semantic_consistency if rsn else None,
            explanation=rsn.explanation if rsn else None,
            group_scores=rsn.evidence.group_scores.as_dict() if rsn else None,
        )

        # ── Medical semantic refinement (advisory, gate-pass only) ───────────
        if gate_outcome.passed:
            try:
                from app.modules.vision.semantic.medical_refiner import get_medical_refiner
                from app.schemas.vision import MedicalRefinement as MedRefSchema

                ref = get_medical_refiner().refine(image)
                semantic_info = semantic_info.model_copy(update={
                    "medical_refinement": MedRefSchema(
                        semantic_medical_type=ref.semantic_medical_type,
                        medical_plausibility=ref.medical_plausibility,
                        fake_medical_score=ref.fake_medical_score,
                        semantic_margin=ref.semantic_margin,
                        refinement_reason=ref.refinement_reason,
                        inference_ms=ref.inference_ms,
                    )
                })
                logger.debug(
                    "MedicalRefiner: type=%s plausibility=%.3f fake=%.3f",
                    ref.semantic_medical_type, ref.medical_plausibility, ref.fake_medical_score,
                )
            except Exception:
                logger.warning(
                    "MedicalRefiner unavailable for upload=%s — skipping",
                    upload.safe_filename, exc_info=True,
                )

        if not gate_outcome.passed:
            logger.info(
                "Vision predict: semantic gate rejected upload=%s code=%s triggered_by=%s",
                upload.safe_filename,
                gate_outcome.rejection_code,
                gate_outcome.triggered_by,
            )
            return VisionPredictionResponse(
                accepted=False,
                acceptance_level=AcceptanceLevel.REJECTED.value,
                low_confidence_reason=None,
                predicted_class=None,
                predicted_class_index=None,
                confidence=None,
                probabilities=None,
                threshold=threshold,
                rejection_reason=gate_outcome.rejection_reason,
                semantic=semantic_info,
                gate=gate_info,
                image=image_info,
                upload=upload_info,
                model_name=main_service.architecture or "unknown",
                model_version=main_service.version or "unknown",
                inference_duration_ms=round((time.perf_counter() - start) * 1000, 2),
                gradcam_base64=None,
                timestamp=_utc_now(),
            )
    except Exception:
        logger.warning(
            "Semantic gate unavailable for upload=%s — skipping gate",
            upload.safe_filename,
            exc_info=True,
        )

    # ── 4. Main classifier + multi-tier acceptance ─────────────────────
    prediction = main_service.predict(image)
    acceptance = decide_acceptance(
        predicted_class=prediction.class_label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        class_names=main_service.class_names,
        threshold=threshold,
    )

    probabilities = {
        label: round(p, 4)
        for label, p in zip(main_service.class_names, prediction.probabilities)
    }

    # ── 5. Fusion Intelligence Layer (advisory) ────────────────────────
    vision_fusion: VisionFusionSchema | None = None
    if semantic_info is not None:
        try:
            from app.modules.vision.fusion.intelligent_fusion import default_fusion

            ref = semantic_info.medical_refinement
            fir = default_fusion.fuse(
                classifier_confidence=prediction.confidence,
                reasoning_decision=semantic_info.reasoning_decision,
                reasoning_confidence=semantic_info.reasoning_confidence,
                semantic_uncertainty=semantic_info.semantic_uncertainty,
                semantic_consistency=semantic_info.semantic_consistency,
                medical_plausibility=ref.medical_plausibility if ref else None,
                fake_medical_score=ref.fake_medical_score if ref else None,
                ood_score=semantic_info.ood_score,
                medical_relevance_score=semantic_info.medical_relevance_score,
            )
            vision_fusion = VisionFusionSchema(
                fusion_confidence=fir.fusion_confidence,
                fusion_delta=fir.fusion_delta,
                agreement_score=fir.agreement_score,
                uncertainty_score=fir.uncertainty_score,
                semantic_alignment=fir.semantic_alignment,
                fusion_reason=fir.fusion_reason,
            )
            logger.debug(
                "Vision fusion: align=%s agreement=%.3f uncertainty=%.3f "
                "delta=%+.4f fusion_conf=%.4f file=%s",
                fir.semantic_alignment, fir.agreement_score, fir.uncertainty_score,
                fir.fusion_delta, fir.fusion_confidence, upload.safe_filename,
            )
        except Exception:
            logger.warning(
                "Fusion intelligence layer unavailable for upload=%s — skipping",
                upload.safe_filename, exc_info=True,
            )

    # ── 6. Calibration V2 + Explainability (advisory) ─────────────────
    vision_explain: VisionExplainSchema | None = None
    if vision_fusion is not None and semantic_info is not None:
        try:
            from app.modules.vision.explainability.calibration_v2 import build_calibration_v2

            ref = semantic_info.medical_refinement
            cal = build_calibration_v2(
                classifier_confidence=prediction.confidence,
                threshold=threshold,
                fusion_confidence=vision_fusion.fusion_confidence,
                fusion_delta=vision_fusion.fusion_delta,
                agreement_score=vision_fusion.agreement_score,
                uncertainty_score=vision_fusion.uncertainty_score,
                semantic_alignment=vision_fusion.semantic_alignment,
                reasoning_type=semantic_info.reasoning_type,
                reasoning_decision=semantic_info.reasoning_decision,
                semantic_uncertainty=semantic_info.semantic_uncertainty,
                medical_plausibility=ref.medical_plausibility if ref else None,
                fake_medical_score=ref.fake_medical_score if ref else None,
                ood_score=semantic_info.ood_score,
            )
            vision_explain = VisionExplainSchema(
                trust_tier=cal.trust_tier,
                trust_score=cal.trust_score,
                calibration_state=cal.calibration_state,
                explanation_summary=cal.explanation_summary,
                uncertainty_reason=cal.uncertainty_reason,
                semantic_warning=cal.semantic_warning,
            )
        except Exception:
            logger.warning(
                "Calibration V2 unavailable for upload=%s — skipping",
                upload.safe_filename, exc_info=True,
            )

    # Grad-CAM is generated for any predicted class when the caller requests
    # it.  Acceptance level does not gate Grad-CAM — shortcut-learning analysis
    # and explainability auditing require heatmaps for rejected classes too
    # (hard_negative, unrelated).
    gradcam_b64: str | None = None
    if gradcam:
        gradcam_b64 = _gradcam_to_base64(main_service, image, prediction.class_index)

    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Vision predict: file=%s class=%s confidence=%.4f level=%s "
        "model=%s@%s duration_ms=%.1f",
        upload.safe_filename,
        prediction.class_label,
        prediction.confidence,
        acceptance.level.value,
        main_service.architecture,
        main_service.version,
        duration_ms,
    )

    return VisionPredictionResponse(
        accepted=acceptance.accepted,
        acceptance_level=acceptance.level.value,
        low_confidence_reason=(
            acceptance.reason if acceptance.is_low_confidence else None
        ),
        predicted_class=prediction.class_label,
        predicted_class_index=prediction.class_index,
        confidence=round(prediction.confidence, 4),
        probabilities=probabilities,
        threshold=threshold,
        rejection_reason=(
            acceptance.reason if acceptance.level is AcceptanceLevel.REJECTED else None
        ),
        semantic=semantic_info,
        fusion=vision_fusion,
        explainability=vision_explain,
        gate=gate_info,
        image=image_info,
        upload=upload_info,
        model_name=main_service.architecture or "unknown",
        model_version=main_service.version or "unknown",
        inference_duration_ms=round(duration_ms, 2),
        gradcam_base64=gradcam_b64,
        timestamp=_utc_now(),
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post(
    "/predict",
    response_model=VisionPredictionResponse,
    summary="Image-based prediction",
    responses={
        413: {"description": "Upload exceeds the configured size limit."},
        415: {"description": "Unsupported image format."},
        422: {"description": "Image failed validation (corrupted, too small, etc.)."},
        503: {"description": "Vision model is not loaded."},
    },
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image file (jpg/png/bmp/tiff)."),
    gradcam: bool = Query(
        False,
        description="When true, include a base64-encoded Grad-CAM overlay in the response.",
    ),
    threshold: float = Query(
        DEFAULT_PREDICTION_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to mark a prediction as accepted.",
    ),
    main_service: VisionInferenceService = Depends(get_vision_inference_service),
    upload_handler: ImageUploadHandler = Depends(get_image_upload_handler),
    gate_service: VisionInferenceService | None = Depends(get_vision_gate_service),
) -> VisionPredictionResponse:
    """
    Run vision inference on an uploaded image.

    The pipeline is:

    1. **Size & MIME guard** — reject obvious oversize uploads before reading.
    2. **Upload + validation** — extension / size / decode-integrity / dimension checks.
    3. **Content-relevance gate** (optional) — drops "unrelated" or low-confidence images.
    4. **Semantic gate** (CLIP ViT-B-32) — rejects wildlife, food, vehicles, and other
       non-medical scenes before the EfficientNet classifier ever runs.
    5. **Main classifier** — softmax prediction + threshold filter.
    6. **Grad-CAM** (optional) — heatmap overlay returned as base64 JPEG.

    Rejected predictions still return **200 OK** with ``accepted=false`` and
    a ``rejection_reason``. Errors (oversize, corrupt, no model) return a
    proper HTTP status code with a structured error body.
    """
    logger.info(
        "Vision upload received: filename=%s content_type=%s declared_size=%s",
        file.filename, file.content_type,
        request.headers.get("content-length", "unknown"),
    )

    # Early size guard from headers (cheap, before reading).
    try:
        cl = int(request.headers.get("content-length", "0") or 0)
    except ValueError:
        cl = 0
    _enforce_size_limit(cl)

    # MIME-type guard
    if file.content_type and file.content_type not in vision_config.upload.allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content-type '{file.content_type}'. "
                f"Allowed: {sorted(vision_config.upload.allowed_mime_types)}."
            ),
        )

    file_bytes = await file.read()
    _enforce_size_limit(len(file_bytes))  # also check actual size

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty upload — 0 bytes received.",
        )

    try:
        response = await asyncio.to_thread(
            _run_prediction,
            file_bytes=file_bytes,
            original_filename=file.filename or "upload",
            upload_handler=upload_handler,
            gate_service=gate_service,
            main_service=main_service,
            gradcam=gradcam,
            threshold=threshold,
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception:
        logger.exception("Vision predict failed unexpectedly for file=%s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vision inference error. Check server logs for details.",
        )

    return response


@router.get(
    "/models/current",
    response_model=VisionModelInfoResponse,
    summary="Currently loaded vision model info",
)
async def current_model(
    request: Request,
    main_service: VisionInferenceService = Depends(get_vision_inference_service),
    gate_service: VisionInferenceService | None = Depends(get_vision_gate_service),
) -> VisionModelInfoResponse:
    """Returns metadata about the vision model loaded into the API."""
    info = main_service.model_info()
    return VisionModelInfoResponse(
        is_ready=info.get("is_ready", False),
        architecture=info.get("architecture"),
        model_version=info.get("model_version"),
        class_names=info.get("class_names", []),
        image_size=info.get("image_size"),
        metrics=info.get("metrics", {}),
        gate_loaded=bool(gate_service is not None and gate_service.is_ready),
    )