"""
Unified Medical Analysis endpoint — Phase 17.

POST /api/v1/medical/analyze

Pipeline (imaging-first, clinically-assisted):
  1. Image upload + validation
  2. Semantic gate (CLIP)            — OOD veto
  3. Medical refiner (advisory)      — plausibility + fake-medical score
  4. EfficientNet classifier         — calibrated T*=0.4585
  5. Fusion Intelligence Layer       — semantic ↔ classifier alignment
  6. Calibration V2                  — trust tier
  7. GradCAM overlay (optional)
  8. Unified Reasoning Engine        — imaging-first + bounded clinical fusion
  → UnifiedAnalysisSession           — single dashboard response

Design rules enforced here:
  - Imaging is always the primary signal.
  - Clinical context is bounded (±0.15 max delta).
  - OOD detection always caps score and ignores clinical.
  - CRITICAL imaging cannot be downgraded to MODERATE by clinical alone.
  - All steps are graceful-degrading: missing components → neutral defaults.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import (
    get_image_upload_handler,
    get_medical_assistant_service,
    get_optional_user,
    get_vision_inference_service,
)
from app.models.user import User
from app.services.analysis_service import AnalysisService
from app.modules.vision.utils.io import load_image_from_bytes
from app.modules.vision.upload.handler import ImageUploadHandler
from app.modules.vision.inference.service import VisionInferenceService
from app.modules.vision.medical.unified_reasoning import (
    ClinicalContext,
    MedicalRiskTier,
    unified_reasoning_engine,
)
from app.schemas.medical_reasoning import (
    ClinicalContextRequest,
    ClinicalModifierSchema,
    ClinicalPersistRequest,
    FusionReasoning,
    ImagingSignal,
    SemanticSignal,
    UnifiedAnalysisSession,
    UnifiedExplainability,
    UnifiedRiskAssessment,
    UnifiedTrustReport,
)
from app.schemas.medical_assistant import MedicalAssistantRequest, MedicalAssistantResponse
from app.modules.ai.services.medical_assistant import MedicalAssistantService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/medical", tags=["medical"])

# Temperature from Phase 15 calibration
_CALIBRATION_TEMPERATURE: float = 0.4585
_CALIBRATION_ECE: float = 0.0389
_PIPELINE_VERSION: str = "v6_phase17"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_lung_mask_for_gradcam(
    service: VisionInferenceService,
    pil_image,
) -> tuple[object | None, str | None]:
    """
    Run lung segmentation on *pil_image* and return (mask_tensor, seg_quality).

    mask_tensor is an (H, W) float CPU tensor with values in {0.0, 1.0} in the
    ORIGINAL image coordinate space.  GradCAM's apply_lung_mask() will resize it
    to match the heatmap spatial size.

    Returns (None, None) on failure so callers degrade gracefully.
    """
    try:
        import torch
        import numpy as np
        seg_result = service._seg_pipeline.segmenter.segment(pil_image)
        # mask is H×W uint8 (0 = background, 255 = lung)
        mask_f32 = seg_result.mask.astype("float32") / 255.0
        return torch.from_numpy(mask_f32), seg_result.quality
    except Exception:
        logger.debug("Lung segmentation for GradCAM mask failed", exc_info=True)
        return None, None


def _compute_gradcam_heatmap(
    service: VisionInferenceService,
    pil_image,
    class_idx: int,
) -> tuple[object | None, object | None, dict | None]:
    """
    Compute pulmonary-focused GradCAM heatmap, bilateral spatial score, and
    raw GradCAM telemetry.

    Rendering is intentionally deferred so the caller can apply risk-tier-aware
    opacity after the reasoning engine has determined the final risk level.

    Returns (heatmap_tensor | None, BilateralSpatialScore | None, telemetry | None).
    """
    if service.model is None or service.architecture is None:
        return None, None, None
    try:
        from app.modules.vision.explainability.gradcam import build_gradcam
        from app.modules.vision.explainability.bilateral_scorer import compute_bilateral_score

        lung_mask_tensor, seg_quality = _get_lung_mask_for_gradcam(service, pil_image)

        tensor = service._pipeline.preprocess_for_inference(pil_image).to(service.device)
        with build_gradcam(service.model, service.architecture) as cam:
            heatmap = cam.generate_pulmonary_focused(
                tensor,
                class_idx=class_idx,
                lung_mask_tensor=lung_mask_tensor,
                seg_quality=seg_quality,
            )
        cam_telemetry = dict(cam.last_telemetry)

        bilateral_score = compute_bilateral_score(heatmap)
        return heatmap, bilateral_score, cam_telemetry
    except Exception:
        logger.warning("GradCAM/bilateral generation failed", exc_info=True)
        return None, None, None


def _compute_localization_confidence(telemetry: dict) -> float:
    """
    Derive focal pathology confidence from GradCAM telemetry in [0, 1].

    High when the CAM shows coherent, focal, lung-contained activation:
      - cam_trust_gain_effective >= 0.35  (covers strength + lung overlap + border)
      - cam_entropy              <= 0.78  (focal, not diffuse noise)
      - pathology_coherence_score >= 0.45 (coherent blobs, not speckle)

    Returns 0.0 on any gate failure; otherwise a product-scaled score.
    cam_trust_gain_effective already captures lung overlap and border ratio,
    so entropy and coherence are the only additional factors needed here.
    """
    cam_trust  = float(telemetry.get("cam_trust_gain_effective", 0.15))
    cam_entropy = float(telemetry.get("cam_entropy",              1.0))
    coherence   = float(telemetry.get("pathology_coherence_score", 0.0))

    if cam_trust  < 0.35:  return 0.0
    if cam_entropy > 0.78:  return 0.0
    if coherence   < 0.45:  return 0.0

    focal_factor = min(1.0, (0.78 - cam_entropy) / 0.28)   # best at entropy ≤ 0.50
    coh_factor   = min(1.0, (coherence - 0.45)   / 0.35)   # best at coherence ≥ 0.80

    return round(min(1.0, cam_trust * (focal_factor * coh_factor) ** 0.5), 4)


# Risk-tier → overlay alpha mapping.
# LOW gets deliberately soft opacity so a weak/uncertain heatmap does not look
# dramatic.  HIGH/CRITICAL get full emphasis to match clinical urgency.
_RISK_TIER_ALPHA: dict[str, float] = {
    "LOW":                     0.22,
    "MODERATE":                0.32,
    "HIGH_DIFFERENTIAL_RISK":  0.42,
    "CRITICAL_PULMONARY_RISK": 0.50,
}

# Classes that are inherently non-medical or synthetic.  If they somehow reach
# the renderer (OOD guard passed but classifier is uncertain), further dampen
# the heatmap so it doesn't look "beautifully medical".
_FAKE_OR_NONMEDICAL_CLASSES: frozenset[str] = frozenset(
    {"hard_negative", "fake_medical"}
)


def _render_heatmap_risk_aware(
    pil_image,
    heatmap,
    *,
    risk_tier: str,
    predicted_class: str,
) -> str | None:
    """
    Render a GradCAM heatmap overlay with risk-tier-aware opacity.

    LOW risk  → alpha 0.22 (soft, calm)
    MODERATE  → alpha 0.32
    HIGH      → alpha 0.42 (full emphasis)
    CRITICAL  → alpha 0.50

    Fake/non-medical predicted classes receive an additional 0.40× heatmap
    attenuation to prevent aesthetically convincing overlays on OOD content.
    """
    try:
        from app.modules.vision.explainability.heatmap import HeatmapRenderer

        alpha = _RISK_TIER_ALPHA.get(risk_tier, 0.40)
        cam = heatmap

        if predicted_class in _FAKE_OR_NONMEDICAL_CLASSES:
            cam = cam * 0.40

        overlay = HeatmapRenderer(alpha=alpha).overlay(pil_image, cam)
        buf = io.BytesIO()
        overlay.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        logger.warning("Risk-aware heatmap rendering failed", exc_info=True)
        return None


def _run_unified_analysis(
    *,
    file_bytes: bytes,
    original_filename: str,
    upload_handler: ImageUploadHandler,
    main_service: VisionInferenceService,
    clinical_ctx: ClinicalContext | None,
    include_gradcam: bool,
) -> UnifiedAnalysisSession:
    """Synchronous unified analysis pipeline. Called via asyncio.to_thread."""
    t0 = time.perf_counter()

    # ── 1. Validate + persist upload ──────────────────────────────────────────
    upload = upload_handler.handle(file_bytes, original_filename)
    if not upload.success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=upload.error or "Image validation failed.",
        )
    image = load_image_from_bytes(file_bytes)

    if not main_service.is_ready:
        raise RuntimeError(
            "Vision model not loaded. Load a v6 calibrated checkpoint first."
        )

    # ── 2. Semantic gate (CLIP) ───────────────────────────────────────────────
    semantic_result    = None
    gate_outcome       = None
    medical_refinement = None
    semantic_schema    = None
    is_ood             = False

    try:
        from app.modules.vision.semantic.semantic_analyzer import get_semantic_analyzer
        from app.modules.vision.semantic.semantic_gate import semantic_gate

        semantic_result = get_semantic_analyzer().analyze(image)
        gate_outcome    = semantic_gate.evaluate(semantic_result)
        is_ood          = not gate_outcome.passed

        rsn = gate_outcome.reasoning
        semantic_schema = SemanticSignal(
            label=semantic_result.top_semantic_label,
            medical_relevance_score=semantic_result.medical_relevance_score,
            ood_score=semantic_result.ood_score,
            gate_passed=gate_outcome.passed,
            rejection_code=gate_outcome.rejection_code,
            reasoning_decision=rsn.semantic_decision if rsn else None,
            reasoning_confidence=rsn.reasoning_confidence if rsn else None,
            top_matches=[
                {"label": m.label, "score": round(m.score, 4), "rank": m.rank}
                for m in semantic_result.top_matches[:3]
            ],
        )

        # ── 3. Medical refiner (gate-pass only) ───────────────────────────────
        if gate_outcome.passed:
            try:
                from app.modules.vision.semantic.medical_refiner import get_medical_refiner
                medical_refinement = get_medical_refiner().refine(image)
            except Exception:
                logger.warning("MedicalRefiner unavailable — skipping", exc_info=True)

    except Exception:
        logger.warning("Semantic gate unavailable — treating as medical", exc_info=True)

    # ── 4. EfficientNet classifier ────────────────────────────────────────────
    prediction = main_service.predict(image, source=original_filename)
    probabilities = {
        label: round(p, 4)
        for label, p in zip(main_service.class_names, prediction.probabilities)
    }

    # Recover raw (pre-temperature) confidence from calibrated confidence
    # raw_confidence = calibrated_confidence * T  (approximate — exact only if single class)
    raw_confidence = round(
        prediction.confidence * main_service.calibration_temperature, 4
    )

    # ── 5. Fusion Intelligence Layer (advisory) ───────────────────────────────
    fusion_result      = None
    semantic_alignment = "uncertain"
    agreement_score    = 0.50
    uncertainty_score  = 0.50
    fusion_delta       = 0.0

    if semantic_schema is not None and not is_ood:
        try:
            from app.modules.vision.fusion.intelligent_fusion import default_fusion

            ref = medical_refinement
            fusion_result = default_fusion.fuse(
                classifier_confidence=prediction.confidence,
                reasoning_decision=semantic_schema.reasoning_decision,
                reasoning_confidence=semantic_schema.reasoning_confidence,
                semantic_uncertainty=None,
                semantic_consistency=None,
                medical_plausibility=ref.medical_plausibility if ref else None,
                fake_medical_score=ref.fake_medical_score if ref else None,
                ood_score=semantic_schema.ood_score,
                medical_relevance_score=semantic_schema.medical_relevance_score,
            )
            semantic_alignment = fusion_result.semantic_alignment
            agreement_score    = fusion_result.agreement_score
            uncertainty_score  = fusion_result.uncertainty_score
            fusion_delta       = fusion_result.fusion_delta
        except Exception:
            logger.warning("Fusion Intelligence Layer unavailable — using defaults", exc_info=True)

    # ── 6. Calibration V2 (advisory trust tier) ───────────────────────────────
    trust_tier         = "moderate_trust"
    trust_score        = 0.60
    calibration_state  = "stable"
    uncertainty_reason = None
    semantic_warning   = None
    cal_threshold      = 0.50

    try:
        from app.modules.vision.explainability.calibration_v2 import build_calibration_v2

        ref = medical_refinement
        cal = build_calibration_v2(
            classifier_confidence=prediction.confidence,
            threshold=cal_threshold,
            fusion_confidence=fusion_result.fusion_confidence if fusion_result else prediction.confidence,
            fusion_delta=fusion_delta,
            agreement_score=agreement_score,
            uncertainty_score=uncertainty_score,
            semantic_alignment=semantic_alignment,
            reasoning_type=semantic_schema.reasoning_decision if semantic_schema else None,
            reasoning_decision=semantic_schema.reasoning_decision if semantic_schema else None,
            semantic_uncertainty=None,
            medical_plausibility=ref.medical_plausibility if ref else None,
            fake_medical_score=ref.fake_medical_score if ref else None,
            ood_score=semantic_schema.ood_score if semantic_schema else 0.0,
        )
        trust_tier         = cal.trust_tier
        trust_score        = cal.trust_score
        calibration_state  = cal.calibration_state
        uncertainty_reason = cal.uncertainty_reason
        semantic_warning   = cal.semantic_warning
    except Exception:
        logger.warning("Calibration V2 unavailable — using defaults", exc_info=True)

    # ── 7. GradCAM computation + bilateral spatial scoring ────────────────────
    # Rendering is deferred to step 7b (after reasoning) so opacity is keyed to
    # the final risk tier rather than being a fixed constant.
    gradcam_heatmap       = None
    gradcam_b64           = None
    bilateral_score       = None
    localization_confidence = None
    if not is_ood:
        gradcam_heatmap, bilateral_score, cam_telemetry = _compute_gradcam_heatmap(
            main_service, image, prediction.class_index
        )
        if cam_telemetry and prediction.class_label == "pneumonia_xray":
            localization_confidence = _compute_localization_confidence(cam_telemetry)

    # ── 8. Unified Reasoning Engine ───────────────────────────────────────────
    ref = medical_refinement
    result = unified_reasoning_engine.analyze(
        predicted_class=prediction.class_label,
        calibrated_confidence=prediction.confidence,
        probabilities=probabilities,
        is_ood=is_ood,
        ood_class=prediction.class_label if is_ood else None,
        trust_tier=trust_tier,
        trust_score=trust_score,
        calibration_state=calibration_state,
        uncertainty_reason=uncertainty_reason,
        semantic_warning=semantic_warning,
        semantic_alignment=semantic_alignment,
        agreement_score=agreement_score,
        uncertainty_score=uncertainty_score,
        fusion_delta=fusion_delta,
        clinical_context=clinical_ctx,
        bilateral_score=bilateral_score,
        model_version=f"{main_service.architecture or 'efficientnet_b0'}@{main_service.version or 'v6'}",
        medical_relevance_score=(
            semantic_schema.medical_relevance_score if semantic_schema else None
        ),
        medical_plausibility=ref.medical_plausibility if ref else None,
        fake_medical_score=ref.fake_medical_score if ref else None,
        semantic_margin=ref.semantic_margin if ref else None,
        refiner_top_type=ref.semantic_medical_type if ref else None,
        refiner_group_scores=ref.group_scores if ref else None,
        source_filename=original_filename,
        localization_confidence=localization_confidence,
    )

    # ── 7b. Risk-aware heatmap rendering ─────────────────────────────────────
    # Now that the reasoning engine has produced the final risk_tier, render the
    # heatmap overlay with the appropriate opacity for that tier.
    if include_gradcam and gradcam_heatmap is not None:
        gradcam_b64 = _render_heatmap_risk_aware(
            image,
            gradcam_heatmap,
            risk_tier=result.risk_tier.value,
            predicted_class=prediction.class_label,
        )

    # ── 9. Assemble unified response ──────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - t0) * 1000

    clin_mod = result.clinical_modifier_result
    clin_schema = ClinicalModifierSchema(
        provided=result.clinical_provided,
        clinical_delta=result.clinical_modifier,
        delta_direction=clin_mod.direction if clin_mod else "neutral",
        symptoms_flagged=clinical_ctx.symptoms if clinical_ctx else [],
        exposure_flagged=clinical_ctx.exposure_history if clinical_ctx else None,
        symptom_score=clin_mod.symptom_score if clin_mod else 0.0,
        exposure_score=clin_mod.exposure_score if clin_mod else 0.0,
        contradiction_detected=clin_mod.contradiction.detected if clin_mod else False,
        contradiction_severity=clin_mod.contradiction.severity if clin_mod else None,
        contradiction_note=result.contradiction_note,
        weight_applied=clin_mod.weight_factor if clin_mod else 1.0,
    )

    # Compute imaging weight (imaging always dominant; clinical provides remainder)
    imaging_weight  = round(1.0 - abs(result.clinical_modifier), 4)
    clinical_weight = round(abs(result.clinical_modifier), 4)

    imaging_schema = ImagingSignal(
        predicted_class=prediction.class_label,
        calibrated_confidence=round(prediction.confidence, 4),
        raw_confidence=raw_confidence,
        temperature_applied=main_service.calibration_temperature,
        class_probabilities=probabilities,
        ood_detected=is_ood,
        ood_class=prediction.class_label if is_ood else None,
        imaging_score=result.imaging_score,
        model_version=main_service.version or "v6",
        inference_ms=round(prediction.inference_ms, 2),
    )

    fusion_schema = FusionReasoning(
        imaging_weight=imaging_weight,
        clinical_weight=clinical_weight,
        semantic_alignment=semantic_alignment,
        agreement_score=result.agreement_score,
        uncertainty_score=result.uncertainty_score,
        fusion_delta=round(fusion_delta, 4),
        ood_guard_applied=result.ood_guard_applied,
    )

    trust_schema = UnifiedTrustReport(
        trust_tier=trust_tier,
        trust_score=trust_score,
        calibration_state=calibration_state,
        ece_at_training=_CALIBRATION_ECE,
        temperature_used=main_service.calibration_temperature,
        uncertainty_reason=uncertainty_reason,
        semantic_warning=semantic_warning,
    )

    risk_schema = UnifiedRiskAssessment(
        risk_tier=result.risk_tier.value,      # type: ignore[arg-type]
        final_score=result.final_score,
        imaging_score=result.imaging_score,
        clinical_modifier=result.clinical_modifier,
        near_boundary=result.near_boundary,
        boundary_proximity=result.boundary_proximity,
        requires_immediate_action=result.requires_immediate_action,
        differential_classes=result.differential_classes,
    )

    explain_schema = UnifiedExplainability(
        summary=result.final_summary,
        imaging_findings=result.imaging_findings,
        clinical_context_applied=result.clinical_summary,
        contradiction_note=result.contradiction_note,
        gradcam_base64=gradcam_b64,
        gradcam_target_class=prediction.class_label if gradcam_b64 else None,
        reasoning_chain=result.reasoning_chain,
        pipeline_warnings=result.pipeline_warnings,
    )

    logger.info(
        "MedicalAnalyze[%s]: class=%s score=%.4f tier=%s trust=%s elapsed=%.1fms",
        result.session_id, prediction.class_label, result.final_score,
        result.risk_tier.value, trust_tier, elapsed_ms,
    )

    return UnifiedAnalysisSession(
        session_id=result.session_id,
        timestamp=result.timestamp,
        imaging=imaging_schema,
        semantic=semantic_schema,
        clinical=clin_schema,
        fusion=fusion_schema,
        trust=trust_schema,
        risk=risk_schema,
        explainability=explain_schema,
        ood_guard_applied=result.ood_guard_applied,
        clinical_override_attempted=result.clinical_override_attempted,
        model_version=main_service.version or "v6",
        pipeline_version=_PIPELINE_VERSION,
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "/analyze",
    response_model=UnifiedAnalysisSession,
    summary="Unified medical image analysis (imaging-first, clinically-assisted)",
    responses={
        422: {"description": "Image validation failed."},
        503: {"description": "Vision model not loaded."},
    },
)
async def analyze(
    request: Request,
    file: UploadFile = File(..., description="Medical image file (jpg/png/bmp/tiff)."),
    clinical_context: str | None = Form(
        None,
        description=(
            "Optional JSON string conforming to ClinicalContextRequest. "
            "Clinical data can only shift the imaging score by ±0.15 and cannot override imaging findings."
        ),
    ),
    gradcam: bool = Form(True, description="Include GradCAM overlay in response."),
    main_service: VisionInferenceService = Depends(get_vision_inference_service),
    upload_handler: ImageUploadHandler   = Depends(get_image_upload_handler),
    current_user: User | None            = Depends(get_optional_user),
    db: AsyncSession                     = Depends(get_db),
) -> UnifiedAnalysisSession:
    """
    Unified imaging-first medical analysis.

    Signal hierarchy:
    - **Imaging** (EfficientNet, T*=0.4585) — always primary
    - **Semantic gate** (CLIP) — OOD veto; non-medical images are capped at LOW
    - **Clinical context** — bounded modifier (±0.15); never overrides imaging
    - **Fusion intelligence** — advisory semantic-classifier alignment
    - **Trust tier** — Calibration V2 post-fusion advisory

    Behavior by scenario:
    - `Clean image + strong symptoms` → imaging stays LOW if score is low; clinical pushes at most to MODERATE
    - `Severe image + no clinical data` → CRITICAL verdict unchanged; clinical absence does not downgrade
    - `Conflicting signals` → contradiction detected, clinical weight reduced, imaging governs tier
    - `Uncertain imaging` → trust_tier=uncertain flagged, MODERATE verdict with warning
    - `OOD image` → score capped at 0.15, risk=LOW, clinical ignored, ood_guard_applied=true
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty upload — 0 bytes received.",
        )

    # Parse optional clinical context JSON
    clinical_ctx: ClinicalContext | None = None
    if clinical_context:
        try:
            ctx_data = json.loads(clinical_context)
            ctx_req  = ClinicalContextRequest(**ctx_data)
            clinical_ctx = ClinicalContext(
                # Legacy fields
                symptoms=ctx_req.symptoms,
                exposure_history=ctx_req.exposure_history,
                duration_days=ctx_req.duration_days,
                severity=ctx_req.severity,
                immunocompromised=ctx_req.immunocompromised,
                age_group=ctx_req.age_group,
                notes=ctx_req.notes,
                # Phase 22 structured fields
                age_numeric=ctx_req.age,
                sex=ctx_req.sex,
                respiratory_severity=ctx_req.respiratory_severity,
                oxygenation_context=ctx_req.oxygenation_context,
                fever_severity=ctx_req.fever_severity,
                recent_worsening=ctx_req.recent_worsening,
                rodent_exposure_level=ctx_req.rodent_exposure_level,
                symptom_duration_tier=ctx_req.symptom_duration_tier,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid clinical_context JSON: {exc}",
            )

    try:
        session = await asyncio.to_thread(
            _run_unified_analysis,
            file_bytes=file_bytes,
            original_filename=file.filename or "upload",
            upload_handler=upload_handler,
            main_service=main_service,
            clinical_ctx=clinical_ctx,
            include_gradcam=gradcam,
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception:
        logger.exception("Unified medical analysis failed for file=%s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Medical analysis error. Check server logs.",
        )

    # Persist result — isolated so a DB failure never breaks the analysis response.
    try:
        svc = AnalysisService(db)
        await svc.save_from_session(
            session,
            user_id=current_user.id if current_user else None,
            duration_ms=session.imaging.inference_ms,
        )
        await db.commit()
    except Exception:
        logger.warning(
            "Failed to persist analysis session_id=%s — analysis result still returned",
            session.session_id,
            exc_info=True,
        )
        await db.rollback()

    return session


@router.post(
    "/persist-clinical",
    summary="Persist a frontend-computed clinical-only analysis result",
    status_code=200,
)
async def persist_clinical(
    body: ClinicalPersistRequest,
    current_user: User | None = Depends(get_optional_user),
    db: AsyncSession         = Depends(get_db),
) -> dict:
    """
    Called by the frontend after running the local clinical-only scorer.

    The scoring logic lives in the browser (clinical-analysis.ts).  This endpoint
    accepts the computed result and stores it identically to image analyses so the
    dashboard reflects all analysis types.

    Isolated so that a DB failure never breaks the frontend — the response is
    only advisory; the UI already shows the result before this call returns.
    """
    try:
        svc = AnalysisService(db)
        record = await svc.save_clinical_only(
            session_id=body.session_id,
            risk_tier=body.risk_tier,
            final_score=body.final_score,
            user_id=current_user.id if current_user else None,
            summary=body.summary,
            duration_ms=body.duration_ms,
        )
        await db.commit()
        return {"ok": True, "id": record.id}
    except Exception:
        logger.warning(
            "Failed to persist clinical-only session_id=%s", body.session_id, exc_info=True
        )
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clinical analysis persistence failed.",
        )


# ── Medical AI Assistant ──────────────────────────────────────────────────────

@router.post(
    "/assistant",
    summary="Klinik açıklama asistanı — mevcut analiz hakkında soru sor",
    responses={
        503: {"description": "AI servisi şu anda kullanılamıyor."},
    },
)
async def medical_assistant(
    request: Request,
    body: MedicalAssistantRequest,
    service: MedicalAssistantService = Depends(get_medical_assistant_service),
) -> MedicalAssistantResponse:
    """
    Context-aware AI assistant for the current analysis session.

    Accepts the curated analysis context (MedicalAnalysisContext) alongside
    the user question. Returns a Turkish clinical explanation grounded in THIS
    specific encounter — no diagnosis, no medication, no treatment plans.
    """
    from app.modules.ai.providers.base import ProviderError

    try:
        result = await service.ask(
            message=body.message,
            session_id=body.session_id,
            analysis_context=body.analysis_context,
        )
    except ProviderError as exc:
        logger.error("Medical assistant: provider error — %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI sağlayıcı erişilemiyor: {exc}",
        )

    return MedicalAssistantResponse(
        content=result.content,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        model=result.model,
        duration_ms=result.duration_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        timestamp=result.timestamp,
        session_id=body.session_id,
    )


@router.post(
    "/assistant/stream",
    summary="Klinik asistan — sunucu-taraflı olay akışı (SSE)",
    responses={
        503: {"description": "AI servisi şu anda kullanılamıyor."},
    },
)
async def medical_assistant_stream(
    request: Request,
    body: MedicalAssistantRequest,
    service: MedicalAssistantService = Depends(get_medical_assistant_service),
) -> StreamingResponse:
    """
    Streaming variant of /assistant using Server-Sent Events.

    Emits ``data: {"token": "..."}`` lines as Ollama generates tokens.
    Terminates with ``data: [DONE]``.
    On provider error emits ``data: {"error": "..."}`` before DONE.
    """
    import json as _json

    from app.modules.ai.providers.base import ProviderError

    async def event_generator():
        try:
            async for token in service.ask_stream(
                message=body.message,
                session_id=body.session_id,
                analysis_context=body.analysis_context,
            ):
                yield f"data: {_json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        except ProviderError as exc:
            logger.error("Medical assistant stream: provider error — %s", exc)
            yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
