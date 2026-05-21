import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Imputes missing values — numeric and categorical columns separately."""

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
    ) -> None:
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self._numeric_imputer: SimpleImputer | None = None
        self._categorical_imputer: SimpleImputer | None = None
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self._numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        if self._numeric_cols:
            self._numeric_imputer = SimpleImputer(strategy=self.numeric_strategy)
            self._numeric_imputer.fit(X[self._numeric_cols])

        if self._categorical_cols:
            self._categorical_imputer = SimpleImputer(strategy=self.categorical_strategy)
            self._categorical_imputer.fit(X[self._categorical_cols])

        total_missing = int(X.isnull().sum().sum())
        if total_missing:
            logger.info(
                "MissingValueHandler fitted: %d missing values across %d columns.",
                total_missing,
                len(self._numeric_cols) + len(self._categorical_cols),
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        if self._numeric_cols and self._numeric_imputer:
            result[self._numeric_cols] = self._numeric_imputer.transform(
                result[self._numeric_cols]
            )
        if self._categorical_cols and self._categorical_imputer:
            result[self._categorical_cols] = self._categorical_imputer.transform(
                result[self._categorical_cols]
            )
        return result
