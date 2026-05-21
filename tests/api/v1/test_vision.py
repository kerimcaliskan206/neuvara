"""
Vision API endpoint tests.

Strategy
────────
The real ``VisionInferenceService`` requires a trained model on disk.
Tests stub it out via FastAPI ``dependency_overrides`` so the suite
verifies API contract (status codes, response shape, validation, rejection
logic) rather than model accuracy.
"""
import io
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient
from PIL import Image

from app.core.dependencies import (
    get_image_upload_handler,
    get_vision_gate_service,
    get_vision_inference_service,
)
from app.main import app
from app.modules.vision.inference.predictor import VisionPrediction
from app.modules.vision.inference.service import VisionInferenceService
from app.modules.vision.upload.handler import (
    ImageMetadata,
    ImageUploadHandler,
    UploadResult,
)
from app.modules.vision.validation.validator import ValidationResult


# ── Helpers ──────────────────────────────────────────────────────────────────


def _image_bytes(width: int = 128, height: int = 128, color: str = "white") -> bytes:
    """Produce a tiny valid JPEG for upload tests."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_main_service(
    *,
    class_label: str = "related",
    class_index: int = 1,
    confidence: float = 0.91,
    class_names: list[str] | None = None,
) -> MagicMock:
    class_names = class_names or ["unrelated", "related"]
    svc = MagicMock(spec=VisionInferenceService)
    svc.is_ready = True
    svc.architecture = "efficientnet_b0"
    svc.version = "v20260514_120000"
    svc.class_names = class_names
    svc.metadata = {"metrics": {"f1": 0.92}}
    svc.model = MagicMock()
    svc.device = "cpu"

    probs = [0.02, 0.98] if class_index == 1 else [0.93, 0.07]
    if class_label == "unrelated" and class_index == 0:
        probs = [0.93, 0.07]
    if confidence < 0.5:
        probs = [1 - confidence, confidence] if class_index == 1 else [confidence, 1 - confidence]

    svc.predict.return_value = VisionPrediction(
        class_index=class_index,
        class_label=class_label,
        confidence=confidence,
        probabilities=probs,
        inference_ms=3.4,
    )
    svc.model_info.return_value = {
        "is_ready": True,
        "architecture": "efficientnet_b0",
        "model_version": "v20260514_120000",
        "class_names": class_names,
        "image_size": [224, 224],
        "metrics": {"f1": 0.92},
    }
    return svc


def _make_upload_handler() -> MagicMock:
    """Upload handler that accepts everything we throw at it."""
    handler = MagicMock(spec=ImageUploadHandler)

    def _handle(file_bytes: bytes, original_filename: str) -> UploadResult:
        metadata = ImageMetadata(
            original_filename=original_filename,
            safe_filename=f"abc123_{original_filename}",
            width=128,
            height=128,
            mode="RGB",
            format="JPEG",
            file_size_bytes=len(file_bytes),
            uploaded_at="2026-05-14T00:00:00+00:00",
            upload_path=f"/tmp/{original_filename}",
        )
        return UploadResult(
            success=True,
            original_filename=original_filename,
            validation=ValidationResult(passed=True),
            safe_filename=metadata.safe_filename,
            upload_path=metadata.upload_path,
            metadata=metadata,
        )

    handler.handle.side_effect = _handle
    return handler


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def main_service() -> MagicMock:
    return _make_main_service()


@pytest.fixture
def upload_handler() -> MagicMock:
    return _make_upload_handler()


@pytest.fixture
async def vision_client(client: AsyncClient, main_service, upload_handler):
    """Client with vision dependencies overridden (no gate)."""
    app.dependency_overrides[get_vision_inference_service] = lambda: main_service
    app.dependency_overrides[get_image_upload_handler] = lambda: upload_handler
    app.dependency_overrides[get_vision_gate_service] = lambda: None
    original_state = getattr(app.state, "vision_service", None)
    app.state.vision_service = main_service
    yield client
    app.dependency_overrides.pop(get_vision_inference_service, None)
    app.dependency_overrides.pop(get_image_upload_handler, None)
    app.dependency_overrides.pop(get_vision_gate_service, None)
    app.state.vision_service = original_state


@pytest.fixture
async def no_model_client(client: AsyncClient):
    """Client where no vision model is loaded — verify 503."""
    app.dependency_overrides.pop(get_vision_inference_service, None)
    original_state = getattr(app.state, "vision_service", None)
    app.state.vision_service = None
    yield client
    app.state.vision_service = original_state


# ── POST /vision/predict — happy path ────────────────────────────────────────


@pytest.mark.asyncio
async def test_predict_accepts_valid_image(vision_client: AsyncClient):
    response = await vision_client.post(
        "/api/v1/vision/predict",
        files={"file": ("sample.jpg", _image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["accepted"] is True
    assert data["predicted_class"] == "related"
    assert data["predicted_class_index"] == 1
    assert 0.0 <= data["confidence"] <= 1.0
    assert "related" in data["probabilities"]
    assert "unrelated" in data["probabilities"]
    assert data["model_name"] == "efficientnet_b0"
    assert data["model_version"].startswith("v")
    assert "inference_duration_ms" in data
    assert "timestamp" in data
    assert data["gradcam_base64"] is None  # not requested
    assert data["gate"]["enabled"] is False  # no gate fixture


@pytest.mark.asyncio
async def test_predict_response_schema_keys(vision_client: AsyncClient):
    response = await vision_client.post(
        "/api/v1/vision/predict",
        files={"file": ("a.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    for key in (
        "accepted", "predicted_class", "predicted_class_index",
        "confidence", "probabilities", "threshold", "rejection_reason",
        "gate", "image", "upload", "model_name", "model_version",
        "inference_duration_ms", "gradcam_base64", "timestamp",
    ):
        assert key in data, f"missing key: {key}"


# ── Threshold rejection ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_predict_rejects_low_confidence(client: AsyncClient, upload_handler):
    low_conf_service = _make_main_service(confidence=0.42)
    app.dependency_overrides[get_vision_inference_service] = lambda: low_conf_service
    app.dependency_overrides[get_image_upload_handler] = lambda: upload_handler
    app.dependency_overrides[get_vision_gate_service] = lambda: None
    app.state.vision_service = low_conf_service
    try:
        response = await client.post(
            "/api/v1/vision/predict?threshold=0.7",
            files={"file": ("low.jpg", _image_bytes(), "image/jpeg")},
        )
    finally:
        app.dependency_overrides.pop(get_vision_inference_service, None)
        app.dependency_overrides.pop(get_image_upload_handler, None)
        app.dependency_overrides.pop(get_vision_gate_service, None)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["accepted"] is False
    assert data["predicted_class"] is None
    assert data["rejection_reason"] is not None
    assert "confidence" in data["rejection_reason"].lower()


# ── Gate rejection (unrelated) ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_predict_gate_rejects_unrelated(client: AsyncClient, upload_handler):
    main_service = _make_main_service()
    gate_service = _make_main_service(
        class_label="unrelated", class_index=0, confidence=0.95,
    )

    app.dependency_overrides[get_vision_inference_service] = lambda: main_service
    app.dependency_overrides[get_image_upload_handler] = lambda: upload_handler
    app.dependency_overrides[get_vision_gate_service] = lambda: gate_service
    app.state.vision_service = main_service
    app.state.vision_gate_service = gate_service
    try:
        response = await client.post(
            "/api/v1/vision/predict",
            files={"file": ("off_topic.jpg", _image_bytes(), "image/jpeg")},
        )
    finally:
        app.dependency_overrides.pop(get_vision_inference_service, None)
        app.dependency_overrides.pop(get_image_upload_handler, None)
        app.dependency_overrides.pop(get_vision_gate_service, None)
        app.state.vision_gate_service = None

    assert response.status_code == 200
    data = response.json()
    assert data["accepted"] is False
    assert data["gate"]["enabled"] is True
    assert data["gate"]["predicted_class"] == "unrelated"
    assert data["rejection_reason"] is not None
    # Main classifier should NOT have been called when gate rejects
    assert main_service.predict.call_count == 0


# ── Validation failures ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_predict_rejects_unsupported_mime(vision_client: AsyncClient):
    response = await vision_client.post(
        "/api/v1/vision/predict",
        files={"file": ("doc.pdf", b"fake-pdf-bytes", "application/pdf")},
    )
    assert response.status_code == 415
    assert response.json()["error"]


@pytest.mark.asyncio
async def test_predict_rejects_empty_upload(vision_client: AsyncClient):
    response = await vision_client.post(
        "/api/v1/vision/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_rejects_oversize_upload(vision_client: AsyncClient):
    # 11 MiB > 10 MB default limit
    payload = b"x" * (11 * 1024 * 1024)
    response = await vision_client.post(
        "/api/v1/vision/predict",
        files={"file": ("huge.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 413


# ── No model loaded ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_predict_no_model_returns_503(no_model_client: AsyncClient):
    response = await no_model_client.post(
        "/api/v1/vision/predict",
        files={"file": ("a.jpg", _image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503


# ── GET /vision/models/current ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_models_current_returns_info(vision_client: AsyncClient):
    response = await vision_client.get("/api/v1/vision/models/current")
    assert response.status_code == 200
    data = response.json()
    assert data["is_ready"] is True
    assert data["architecture"] == "efficientnet_b0"
    assert data["model_version"].startswith("v")
    assert data["class_names"] == ["unrelated", "related"]
    assert data["gate_loaded"] is False


@pytest.mark.asyncio
async def test_models_current_no_model_returns_503(no_model_client: AsyncClient):
    response = await no_model_client.get("/api/v1/vision/models/current")
    assert response.status_code == 503