"""
Inference API tests.

Strategy
────────
Most tests use a mock InferenceService injected via FastAPI dependency override
so the test suite does not require trained model files on disk.

The mock produces deterministic, fixed responses — tests verify API contract
(status codes, response shape, validation) rather than model accuracy.
"""
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient

from app.core.dependencies import get_inference_service
from app.main import app
from app.modules.ml.inference.service import InferenceService

# ── Shared fixtures ───────────────────────────────────────────────────────────

_MOCK_SINGLE_RESULT = {
    "prediction": 1,
    "label": "positive",
    "probability": 0.87,
    "confidence": "high",
    "model_name": "weighted_voting_tuned",
    "model_version": "v20260514_000000",
    "inference_duration_ms": 5.2,
    "timestamp": "2026-05-14T00:00:00+00:00",
}

_MOCK_BATCH_RESULT = {
    "predictions": [
        {
            "prediction": 1,
            "label": "positive",
            "probability": 0.87,
            "confidence": "high",
        },
        {
            "prediction": 0,
            "label": "negative",
            "probability": 0.21,
            "confidence": "low",
        },
    ],
    "total": 2,
    "model_name": "weighted_voting_tuned",
    "model_version": "v20260514_000000",
    "inference_duration_ms": 8.4,
    "timestamp": "2026-05-14T00:00:00+00:00",
}

_VALID_PATIENT = {
    "age": 35.0,
    "gender": "M",
    "region": "north",
    "season": "spring",
    "rodent_contact": 1,
    "outdoor_work": 1,
    "fever": 1,
    "myalgia": 1,
    "headache": 0,
    "thrombocytopenia": 1,
    "rodent_density": 7.2,
    "precipitation_mm": 120.5,
    "humidity_pct": 72.3,
}


def _make_mock_service() -> MagicMock:
    svc = MagicMock(spec=InferenceService)
    svc.is_ready = True
    svc.model_name = "weighted_voting_tuned"
    svc.model_version = "v20260514_000000"
    svc.metadata = {"model_type": "weighted_voting", "tuned": True}
    svc.config = MagicMock()
    svc.predict_single.return_value = _MOCK_SINGLE_RESULT
    svc.predict_batch.return_value = _MOCK_BATCH_RESULT
    return svc


@pytest.fixture
async def ml_client(client: AsyncClient):
    """
    AsyncClient with the InferenceService dependency overridden by a mock.
    Yields the client and tears down the override afterward.
    """
    mock_svc = _make_mock_service()
    app.dependency_overrides[get_inference_service] = lambda: mock_svc
    # Also put the mock in app.state so /models/current works
    original_state = getattr(app.state, "inference_service", None)
    app.state.inference_service = mock_svc
    yield client
    app.dependency_overrides.pop(get_inference_service, None)
    app.state.inference_service = original_state


@pytest.fixture
async def no_model_client(client: AsyncClient):
    """Client where no model is loaded — verify 503 responses."""
    original_state = getattr(app.state, "inference_service", None)
    app.state.inference_service = None
    # Remove any override so the real dependency runs
    app.dependency_overrides.pop(get_inference_service, None)
    yield client
    app.state.inference_service = original_state


# ── POST /ml/predict ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_valid_patient(ml_client: AsyncClient):
    response = await ml_client.post(
        "/api/v1/ml/predict",
        json={"patient": _VALID_PATIENT},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert data["label"] in ("negative", "positive")
    assert "probability" in data
    assert "confidence" in data
    assert "model_name" in data
    assert "model_version" in data
    assert "inference_duration_ms" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_predict_all_fields_optional(ml_client: AsyncClient):
    """Empty patient dict is valid — pipeline imputes everything."""
    response = await ml_client.post(
        "/api/v1/ml/predict",
        json={"patient": {}},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_partial_fields(ml_client: AsyncClient):
    response = await ml_client.post(
        "/api/v1/ml/predict",
        json={"patient": {"age": 45, "fever": 1, "rodent_contact": 1}},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_rejects_unknown_field(ml_client: AsyncClient):
    """extra='forbid' on PatientInput must cause 422."""
    response = await ml_client.post(
        "/api/v1/ml/predict",
        json={"patient": {"unknown_field": "bad"}},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_rejects_age_out_of_range(ml_client: AsyncClient):
    response = await ml_client.post(
        "/api/v1/ml/predict",
        json={"patient": {"age": 999}},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_rejects_binary_field_out_of_range(ml_client: AsyncClient):
    response = await ml_client.post(
        "/api/v1/ml/predict",
        json={"patient": {"fever": 5}},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_returns_positive_label_for_prediction_1(ml_client: AsyncClient):
    response = await ml_client.post(
        "/api/v1/ml/predict",
        json={"patient": _VALID_PATIENT},
    )
    data = response.json()
    assert data["prediction"] == 1
    assert data["label"] == "positive"


@pytest.mark.asyncio
async def test_predict_no_model_returns_503(no_model_client: AsyncClient):
    response = await no_model_client.post(
        "/api/v1/ml/predict",
        json={"patient": _VALID_PATIENT},
    )
    assert response.status_code == 503


# ── POST /ml/predict/batch ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_predict_valid(ml_client: AsyncClient):
    response = await ml_client.post(
        "/api/v1/ml/predict/batch",
        json={"patients": [_VALID_PATIENT, {"age": 60, "fever": 0}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2
    assert "model_name" in data
    assert "inference_duration_ms" in data


@pytest.mark.asyncio
async def test_batch_predict_single_patient(ml_client: AsyncClient):
    response = await ml_client.post(
        "/api/v1/ml/predict/batch",
        json={"patients": [{}]},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_batch_predict_rejects_empty_list(ml_client: AsyncClient):
    response = await ml_client.post(
        "/api/v1/ml/predict/batch",
        json={"patients": []},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict_rejects_oversized_list(ml_client: AsyncClient):
    response = await ml_client.post(
        "/api/v1/ml/predict/batch",
        json={"patients": [{}] * 101},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict_items_have_correct_fields(ml_client: AsyncClient):
    response = await ml_client.post(
        "/api/v1/ml/predict/batch",
        json={"patients": [_VALID_PATIENT]},
    )
    data = response.json()
    item = data["predictions"][0]
    assert "prediction" in item
    assert "label" in item
    assert "probability" in item
    assert "confidence" in item


@pytest.mark.asyncio
async def test_batch_predict_no_model_returns_503(no_model_client: AsyncClient):
    response = await no_model_client.post(
        "/api/v1/ml/predict/batch",
        json={"patients": [_VALID_PATIENT]},
    )
    assert response.status_code == 503


# ── GET /ml/models/current ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_current_model_returns_info(ml_client: AsyncClient):
    response = await ml_client.get("/api/v1/ml/models/current")
    assert response.status_code == 200
    data = response.json()
    assert data["is_ready"] is True
    assert "model_name" in data
    assert "model_version" in data
    assert "metadata" in data


@pytest.mark.asyncio
async def test_current_model_no_model_returns_503(no_model_client: AsyncClient):
    response = await no_model_client.get("/api/v1/ml/models/current")
    assert response.status_code == 503
