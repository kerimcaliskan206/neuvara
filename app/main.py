# ── Apple Silicon OpenMP runtime fix (must run before any heavy imports) ────
# PyTorch, XGBoost, and LightGBM each ship their own libomp.  When torch
# loads first, subsequent ML libs cannot reliably claim libomp slots, and
# any unpickle / inference call SIGSEGVs on Apple Silicon.
#
# Fix has two halves:
#   1. KMP_DUPLICATE_LIB_OK + OMP_NUM_THREADS=1 — let multiple libomp
#      runtimes coexist and disable concurrent worker threads.
#   2. Pre-import xgboost / lightgbm / sklearn BEFORE torch ever loads
#      (vision routes pull torch in transitively via app.api.v1.router).
#      This pins the load order so the ML stack initialises libomp first.
#
# Verified end-to-end predict returns in ~40 ms with both halves; either
# half alone still SIGSEGVs.  Latency cost of OMP_NUM_THREADS=1 is
# negligible for our small ensemble.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: F401, E402  ── load order matters
import lightgbm  # noqa: F401, E402
import sklearn  # noqa: F401, E402
# ─────────────────────────────────────────────────────────────────────────────

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exception_handlers import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.core.logging import setup_logging
from app.middleware.request_logging import RequestLoggingMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio as _asyncio

    setup_logging(debug=settings.DEBUG, environment=settings.ENVIRONMENT)
    logger.info(
        "Starting %s v%s [%s]", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT
    )
    loop = _asyncio.get_running_loop()

    # ── ML model warm-up ─────────────────────────────────────────────────────
    # State is set immediately so routes return 503 (not AttributeError) while
    # the background thread loads the model.
    from app.modules.ml.inference.service import InferenceService
    service = InferenceService()
    app.state.inference_service = service

    if settings.ML_AUTO_LOAD_ON_STARTUP:
        def _load_ml():
            loaded = service.load_best()
            if loaded:
                logger.info(
                    "ML model ready: %s @ %s", service.model_name, service.model_version
                )
            else:
                logger.warning(
                    "No trained ML model found — /ml/predict will return 503. "
                    "Run: python scripts/train.py"
                )
        loop.run_in_executor(None, _load_ml)
    else:
        logger.warning(
            "ML_AUTO_LOAD_ON_STARTUP=false — skipping ML model warm-up; "
            "/ml/predict will return 503."
        )
    # ─────────────────────────────────────────────────────────────────────────

    # ── Vision model warm-up ─────────────────────────────────────────────────
    import os
    from pathlib import Path as _Path

    from app.modules.vision.inference.service import VisionInferenceService
    from app.modules.vision.upload.handler import ImageUploadHandler

    _V6_CKPT = _Path("models/vision/v6_medical/stage_c_bilateral/best.pt")
    _V6_TEMP = _Path("models/vision/v6_medical/calibration/temperature_config.json")

    vision_service = VisionInferenceService()
    app.state.vision_service = vision_service
    app.state.vision_gate_service = None
    app.state.image_upload_handler = ImageUploadHandler()

    if settings.VISION_AUTO_LOAD_ON_STARTUP:
        _v6_ckpt_exists = _V6_CKPT.exists()
        _v6_temp_path = _V6_TEMP if _V6_TEMP.exists() else None

        def _load_vision():
            loaded = False
            if _v6_ckpt_exists:
                loaded = vision_service.load_v6_calibrated(
                    checkpoint_path=_V6_CKPT,
                    temperature_config_path=_v6_temp_path,
                )
                if loaded:
                    logger.info(
                        "Vision model ready (v6 calibrated): %s @ %s | T=%.4f",
                        vision_service.architecture, vision_service.version,
                        vision_service.calibration_temperature,
                    )
                else:
                    logger.warning(
                        "v6 calibrated checkpoint found but failed to load — "
                        "falling back to VisionModelStore"
                    )
            if not loaded:
                loaded = vision_service.load_from_store()
                if loaded:
                    logger.info(
                        "Vision model ready (store): %s @ %s",
                        vision_service.architecture, vision_service.version,
                    )
                else:
                    logger.warning(
                        "No vision model available — /vision/predict and "
                        "/medical/analyze will return 503."
                    )
        loop.run_in_executor(None, _load_vision)
    else:
        logger.warning(
            "VISION_AUTO_LOAD_ON_STARTUP=false — skipping vision model warm-up."
        )

    gate_version = os.environ.get("VISION_GATE_VERSION")
    if gate_version:
        def _load_gate():
            candidate = VisionInferenceService()
            if candidate.load_from_store(version=gate_version):
                logger.info(
                    "Vision relevance gate loaded: %s @ %s",
                    candidate.architecture, candidate.version,
                )
                app.state.vision_gate_service = candidate
            else:
                logger.warning(
                    "VISION_GATE_VERSION=%s requested but failed to load.", gate_version
                )
        loop.run_in_executor(None, _load_gate)
    # ─────────────────────────────────────────────────────────────────────────

    # ── CLIP semantic analyzer warm-up ───────────────────────────────────────
    try:
        from app.modules.vision.semantic.semantic_analyzer import get_semantic_analyzer as _get_sem

        def _load_clip():
            try:
                _get_sem()
                logger.info("CLIP semantic analyzer ready (ViT-B-32/openai)")
            except Exception as _exc:  # noqa: BLE001 — non-fatal on startup
                logger.warning(
                    "CLIP semantic analyzer warm-up failed: %s — gate will be skipped", _exc
                )
        loop.run_in_executor(None, _load_clip)
    except Exception as _exc:  # noqa: BLE001 — non-fatal on startup
        logger.warning("CLIP import failed: %s — gate will be skipped", _exc)
    # ─────────────────────────────────────────────────────────────────────────

    # ── AI assistant warm-up (Ollama) ────────────────────────────────────────
    # Construct the provider + services once so each request reuses the
    # underlying httpx.AsyncClient.  Failure here is non-fatal: /ai endpoints
    # will return 503 until Ollama is reachable.
    from app.modules.ai.config import ai_config
    from app.modules.ai.providers.groq import GroqProvider
    from app.modules.ai.services.chat_service import AIChatService
    from app.modules.ai.services.health import AIHealthService
    from app.modules.ai.services.interpretation import (
        FusionInterpretationService,
        MLInterpretationService,
        VisionInterpretationService,
    )
    from app.modules.ai.services.medical_assistant import MedicalAssistantService

    app.state.ai_provider = None
    app.state.ai_chat_service = None
    app.state.ai_health_service = None
    app.state.ai_ml_interpretation_service = None
    app.state.ai_vision_interpretation_service = None
    app.state.ai_fusion_interpretation_service = None
    app.state.ai_medical_assistant_service = None

    if ai_config.enabled:
        try:
            ai_provider = GroqProvider(ai_config.groq)
            app.state.ai_provider = ai_provider
            app.state.ai_chat_service = AIChatService(
                provider=ai_provider, config=ai_config,
            )
            app.state.ai_health_service = AIHealthService(ai_provider, ai_config)
            app.state.ai_ml_interpretation_service = MLInterpretationService(
                ai_provider, ai_config,
            )
            app.state.ai_vision_interpretation_service = VisionInterpretationService(
                ai_provider, ai_config,
            )
            app.state.ai_fusion_interpretation_service = FusionInterpretationService(
                ai_provider, ai_config,
            )
            app.state.ai_medical_assistant_service = MedicalAssistantService(
                provider=ai_provider, config=ai_config,
            )
            logger.info(
                "AI assistant ready: provider=Groq model=%s",
                ai_config.groq.model,
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal on startup
            logger.warning("AI assistant init failed: %s — /ai endpoints will 503", exc)
    else:
        logger.info("AI assistant disabled via config — /ai endpoints will 503")
    # ─────────────────────────────────────────────────────────────────────────

    yield

    # ── AI assistant shutdown ────────────────────────────────────────────────
    provider = getattr(app.state, "ai_provider", None)
    if provider is not None:
        try:
            await provider.close()
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning("AI provider close failed: %s", exc)
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("Shutting down %s", settings.APP_NAME)


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        lifespan=lifespan,
    )

    # Middleware (order matters: last added = outermost = runs first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    # Exception handlers
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_application()
