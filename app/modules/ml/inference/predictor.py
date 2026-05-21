import logging

import pandas as pd

from app.modules.ml.config import MLConfig, ml_config
from app.modules.ml.ensemble.models import ModelType
from app.modules.ml.persistence.model_store import ModelStore
from app.modules.ml.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)


class Predictor:
    """
    Loads a trained model + preprocessing pipeline and produces predictions.

    Usage:
        predictor = Predictor()
        predictor.load(ModelType.VOTING)
        result = predictor.predict(input_df)
    """

    def __init__(self, config: MLConfig = ml_config) -> None:
        self.config = config
        self.store = ModelStore(config)
        self._model = None
        self._pipeline: PreprocessingPipeline | None = None
        self._loaded_version: str | None = None

    def load(self, model_type: ModelType, version: str | None = None) -> None:
        resolved_version = version or self.store.latest_version(model_type.value)
        if resolved_version is None:
            raise RuntimeError(
                f"No saved model found for '{model_type}'. "
                "Train a model first."
            )
        self._model = self.store.load(model_type.value, resolved_version)
        self._pipeline = self.store.load("preprocessing_pipeline", resolved_version)
        self._loaded_version = resolved_version
        logger.info(
            "Predictor loaded: model=%s, version=%s", model_type, resolved_version
        )

    def predict(self, X: pd.DataFrame) -> dict:
        if self._model is None:
            raise RuntimeError("Predictor not loaded. Call load() first.")

        X_processed = self._pipeline.transform(X)
        predictions = self._model.predict(X_processed)
        result: dict = {"predictions": predictions.tolist(), "version": self._loaded_version}

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X_processed)
            result["probabilities"] = proba.tolist()

        return result
