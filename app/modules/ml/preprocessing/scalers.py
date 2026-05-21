import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

_SCALER_REGISTRY: dict[str, type] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scales numeric features using a configurable scaler type."""

    def __init__(self, scaler_type: str = "standard") -> None:
        self.scaler_type = scaler_type
        self._scaler = None
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        scaler_cls = _SCALER_REGISTRY.get(self.scaler_type)
        if scaler_cls is None:
            raise ValueError(
                f"Unknown scaler: '{self.scaler_type}'. "
                f"Choose from: {list(_SCALER_REGISTRY)}"
            )
        self._columns = X.select_dtypes(include=["number"]).columns.tolist()
        self._scaler = scaler_cls()
        self._scaler.fit(X[self._columns])
        logger.info(
            "FeatureScaler fitted: type=%s, %d numeric columns.",
            self.scaler_type,
            len(self._columns),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        result[self._columns] = self._scaler.transform(result[self._columns])
        return result
