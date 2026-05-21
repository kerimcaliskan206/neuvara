import logging
from pathlib import Path

import joblib
import pandas as pd

from app.modules.ml.config import MLConfig, ml_config
from app.modules.ml.preprocessing.cleaners import MissingValueHandler
from app.modules.ml.preprocessing.encoders import CategoricalEncoder
from app.modules.ml.preprocessing.feature_selection import FeatureSelector
from app.modules.ml.preprocessing.features import FeatureEngineer
from app.modules.ml.preprocessing.scalers import FeatureScaler
from app.modules.ml.preprocessing.splitters import DataSplit, DataSplitter

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Orchestrates the full preprocessing flow:

        clean → engineer → [select] → encode → scale → split

    The optional feature selection step is controlled by
    config.feature_selection.enabled.
    """

    def __init__(self, config: MLConfig = ml_config) -> None:
        self.config = config
        cfg = config.preprocessing
        fsel = config.feature_selection

        self.cleaner = MissingValueHandler(numeric_strategy=cfg.missing_strategy)
        self.feature_engineer = FeatureEngineer()
        self.selector: FeatureSelector | None = (
            FeatureSelector(
                variance_threshold=fsel.variance_threshold,
                correlation_threshold=fsel.correlation_threshold,
                top_k=fsel.top_k,
            )
            if fsel.enabled
            else None
        )
        self.encoder = CategoricalEncoder()
        self.scaler = FeatureScaler(scaler_type=cfg.scaler)
        self.splitter = DataSplitter(
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=cfg.stratify,
        )
        self._fitted = False

    # ── Public API ──────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> DataSplit:
        logger.info("Preprocessing started: %d rows × %d cols.", *df.shape)
        target = self.config.target_column
        y = df[target]
        X = df.drop(columns=[target])

        X = self.cleaner.fit_transform(X)
        X = self.feature_engineer.fit_transform(X)
        if self.selector is not None:
            X = self.selector.fit_transform(X)
        X = self.encoder.fit_transform(X)
        X = self.scaler.fit_transform(X)

        self._fitted = True
        processed = X.copy()
        processed[target] = y.values

        logger.info("Preprocessing complete: %d features ready for training.", X.shape[1])
        return self.splitter.split(processed, target)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(
                "Pipeline is not fitted. Call fit_transform() first."
            )
        X = self.cleaner.transform(X)
        X = self.feature_engineer.transform(X)
        if self.selector is not None:
            X = self.selector.transform(X)
        X = self.encoder.transform(X)
        X = self.scaler.transform(X)
        return X

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> Path:
        """Saves the fitted pipeline to a joblib file."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted pipeline.")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        logger.info("Pipeline saved → %s", dest)
        return dest

    @classmethod
    def load(cls, path: Path | str) -> "PreprocessingPipeline":
        """Loads a previously saved pipeline from a joblib file."""
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Pipeline file not found: {src}")
        pipeline: "PreprocessingPipeline" = joblib.load(src)
        logger.info("Pipeline loaded ← %s", src)
        return pipeline
