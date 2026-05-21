import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

logger = logging.getLogger(__name__)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns — OneHot (default) or Label encoding."""

    def __init__(self, strategy: str = "onehot", max_cardinality: int = 20) -> None:
        self.strategy = strategy          # "onehot" | "label"
        self.max_cardinality = max_cardinality
        self._encoders: dict = {}
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self._columns = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        for col in self._columns:
            n_unique = X[col].nunique()
            if n_unique > self.max_cardinality:
                logger.warning(
                    "Column '%s' has %d unique values — high cardinality.", col, n_unique
                )
            if self.strategy == "label":
                enc = LabelEncoder()
                enc.fit(X[col].astype(str))
            else:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                enc.fit(X[[col]])
            self._encoders[col] = enc

        logger.info(
            "CategoricalEncoder fitted: %d columns, strategy=%s.",
            len(self._columns),
            self.strategy,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        for col, enc in self._encoders.items():
            if col not in result.columns:
                continue
            if self.strategy == "label":
                result[col] = enc.transform(result[col].astype(str))
            else:
                encoded = enc.transform(result[[col]])
                new_cols = enc.get_feature_names_out([col])
                result = result.drop(columns=[col])
                result[new_cols] = encoded
        return result
