import logging
import time
from datetime import datetime, timezone

import pandas as pd

from app.modules.ml.config import MLConfig, ml_config
from app.modules.ml.inference.registry import ModelRegistry
from app.modules.ml.inference.stabilizer import clinical_stabilizer
from app.modules.ml.persistence.model_store import ModelStore

logger = logging.getLogger(__name__)

_LABELS: dict[int, str] = {0: "negative", 1: "positive"}
_CONFIDENCE_THRESHOLDS = (0.85, 0.60)


def _confidence(probability: float | None) -> str:
    if probability is None:
        return "unknown"
    if probability >= 0.85:
        return "high"
    if probability >= 0.60:
        return "medium"
    return "low"


def _confidence_margin(probability: float | None) -> float | None:
    """Distance to the nearest confidence threshold (0.85 or 0.60)."""
    if probability is None:
        return None
    return round(min(abs(probability - t) for t in _CONFIDENCE_THRESHOLDS), 4)


def _near_threshold(probability: float | None, margin: float = 0.05) -> bool:
    """True when probability is within `margin` of any confidence threshold."""
    if probability is None:
        return False
    return any(abs(probability - t) <= margin for t in _CONFIDENCE_THRESHOLDS)


class InferenceService:
    """
    Production-grade inference service.

    Intended to be instantiated once at application startup and stored in
    app.state.inference_service.  All prediction requests reuse the same
    loaded model and pipeline — no disk I/O per request.

    Rules:
        - Never retrain here.
        - Never call fit() here.
        - Always log inference metadata (model, version, duration, outcome).
        - Raise RuntimeError (converted to HTTP 503 by the dependency) when not ready.
    """

    def __init__(self, config: MLConfig = ml_config) -> None:
        self.config = config
        self.store = ModelStore(config)
        self.registry = ModelRegistry(config)

        self._model = None
        self._pipeline = None
        self._model_name: str | None = None
        self._model_version: str | None = None
        self._metadata: dict = {}

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._pipeline is not None

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def model_version(self) -> str | None:
        return self._model_version

    @property
    def metadata(self) -> dict:
        return self._metadata

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_best(self) -> bool:
        """
        Discover and load the highest-priority available model.
        Returns True on success, False if nothing is on disk.
        """
        result = self.registry.best_available()
        if result is None:
            logger.warning(
                "InferenceService: no trained model found — "
                "prediction endpoints will return 503 until a model is trained."
            )
            return False
        model_name, version = result
        return self._load(model_name, version)

    def load(self, model_name: str, version: str | None = None) -> bool:
        """Load a specific model by name. Returns True on success."""
        resolved = self.registry.resolve(model_name, version)
        if resolved is None:
            logger.error(
                "InferenceService: cannot load '%s' @ %s — not found on disk.",
                model_name, version or "latest",
            )
            return False
        return self._load(*resolved)

    def reload_best(self) -> bool:
        """
        Reload the best model (e.g., after a new training run).
        Swaps atomically — a failed reload leaves the previous model in place.
        """
        result = self.registry.best_available()
        if result is None:
            return False
        model_name, version = result
        # Only reload if it's a different version
        if model_name == self._model_name and version == self._model_version:
            logger.info(
                "InferenceService: already on latest — %s @ %s",
                model_name, version,
            )
            return True
        return self._load(model_name, version)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_single(self, input_dict: dict) -> dict:
        """
        Run inference on one patient record.
        input_dict must contain the feature columns (no 'label' key).
        Returns a structured result dict.
        """
        self._require_ready()
        start = time.perf_counter()

        X = pd.DataFrame([input_dict])
        X_processed = self._pipeline.transform(X)

        prediction = int(self._model.predict(X_processed)[0])
        probability: float | None = None
        if hasattr(self._model, "predict_proba"):
            probability = float(self._model.predict_proba(X_processed)[0][1])

        # ── Clinical stabilization (post-Platt, pre-response) ────────────────
        # Applies a bounded adjustment (±0.060 max) in the direction of
        # evidence-consistent clinical signals. The ML ensemble is unchanged;
        # only the output probability is softly nudged.
        stab = None
        adjusted_probability = probability
        if probability is not None:
            stab = clinical_stabilizer.adjust(probability, input_dict)
            adjusted_probability = stab.adjusted_probability

        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Inference [single] model=%s version=%s prediction=%d "
            "p_raw=%.4f p_adj=%.4f Δ=%.4f confidence=%s duration_ms=%.1f",
            self._model_name,
            self._model_version,
            prediction,
            probability or 0.0,
            adjusted_probability or 0.0,
            stab.total_delta if stab else 0.0,
            _confidence(adjusted_probability),
            duration_ms,
        )

        return {
            "prediction": prediction,
            "label": _LABELS.get(prediction, "unknown"),
            "probability": (
                round(adjusted_probability, 4) if adjusted_probability is not None else None
            ),
            "ml_raw_probability": (
                round(probability, 4) if probability is not None else None
            ),
            "confidence": _confidence(adjusted_probability),
            "near_threshold": _near_threshold(adjusted_probability),
            "confidence_margin": _confidence_margin(adjusted_probability),
            "stability_delta": stab.total_delta if stab else 0.0,
            "stability_applied": stab.stabilization_applied if stab else False,
            "stability_contributions": [
                {
                    "feature": c.feature,
                    "value": c.value,
                    "raw_delta": c.raw_delta,
                    "effective_delta": c.effective_delta,
                }
                for c in (stab.contributions if stab else [])
            ],
            "model_name": self._model_name,
            "model_version": self._model_version,
            "inference_duration_ms": round(duration_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def predict_batch(self, input_list: list[dict]) -> dict:
        """
        Run inference on a list of patient records.
        Returns predictions list plus aggregate metadata.
        """
        self._require_ready()
        start = time.perf_counter()

        X = pd.DataFrame(input_list)
        X_processed = self._pipeline.transform(X)

        predictions = self._model.predict(X_processed).tolist()
        probabilities: list[float] | None = None
        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(X_processed)[:, 1].tolist()

        duration_ms = (time.perf_counter() - start) * 1000
        timestamp = datetime.now(timezone.utc).isoformat()

        items = [
            {
                "prediction": predictions[i],
                "label": _LABELS.get(predictions[i], "unknown"),
                "probability": (
                    round(probabilities[i], 4) if probabilities is not None else None
                ),
                "confidence": _confidence(
                    probabilities[i] if probabilities is not None else None
                ),
            }
            for i in range(len(predictions))
        ]

        logger.info(
            "Inference [batch] model=%s version=%s n=%d duration_ms=%.1f",
            self._model_name,
            self._model_version,
            len(predictions),
            duration_ms,
        )

        return {
            "predictions": items,
            "total": len(items),
            "model_name": self._model_name,
            "model_version": self._model_version,
            "inference_duration_ms": round(duration_ms, 2),
            "timestamp": timestamp,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self, model_name: str, version: str) -> bool:
        try:
            model = self.store.load(model_name, version)
            pipeline = self.store.load("preprocessing_pipeline", version)
        except FileNotFoundError as exc:
            logger.error("InferenceService: load failed — %s", exc)
            return False
        except Exception as exc:
            logger.exception("InferenceService: unexpected error loading model — %s", exc)
            return False

        self._model = model
        self._pipeline = pipeline
        self._model_name = model_name
        self._model_version = version
        self._metadata = self.store.load_metadata(model_name, version)
        logger.info(
            "InferenceService: ready — model=%s version=%s",
            model_name, version,
        )
        return True

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError(
                "InferenceService has no model loaded. "
                "Train a model first with: python scripts/train.py"
            )
