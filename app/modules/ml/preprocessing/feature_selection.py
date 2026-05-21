import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Removes uninformative features using two strategies:

    1. Low-variance removal  — drops columns where all (or nearly all) rows
       have the same value. These carry no signal.

    2. High-correlation removal — when two features are almost perfectly
       correlated (r > threshold), one is redundant. Drops the second one
       encountered in each correlated pair.

    3. Top-k selection (optional) — after the above, keeps only the k
       highest-variance numeric features.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        top_k: int | None = None,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.top_k = top_k
        self._selected_features: list[str] = []
        self._removed_low_variance: list[str] = []
        self._removed_high_correlation: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        numeric = X.select_dtypes(include=[np.number])
        non_numeric = [c for c in X.columns if c not in numeric.columns]

        # ── Step 1: variance threshold ──────────────────────────────────
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(numeric)
        passed_variance = numeric.columns[vt.get_support()].tolist()
        self._removed_low_variance = [
            c for c in numeric.columns if c not in passed_variance
        ]

        # ── Step 2: correlation filter ──────────────────────────────────
        if passed_variance:
            corr = numeric[passed_variance].corr().abs()
            upper = corr.where(
                np.triu(np.ones(corr.shape, dtype=bool), k=1)
            )
            to_drop = [
                col
                for col in upper.columns
                if any(upper[col] > self.correlation_threshold)
            ]
            self._removed_high_correlation = to_drop
            passed_corr = [f for f in passed_variance if f not in to_drop]
        else:
            self._removed_high_correlation = []
            passed_corr = []

        # ── Step 3: optional top-k ──────────────────────────────────────
        if self.top_k is not None and passed_corr:
            variances = numeric[passed_corr].var().sort_values(ascending=False)
            passed_corr = variances.head(self.top_k).index.tolist()

        self._selected_features = passed_corr + non_numeric

        logger.info(
            "FeatureSelector: %d → %d features "
            "(removed %d low-variance, %d high-correlation).",
            len(X.columns),
            len(self._selected_features),
            len(self._removed_low_variance),
            len(self._removed_high_correlation),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [f for f in self._selected_features if f in X.columns]
        return X[available].copy()

    def get_removed_features(self) -> dict[str, list[str]]:
        return {
            "low_variance": self._removed_low_variance,
            "high_correlation": self._removed_high_correlation,
        }

    @property
    def selected_features(self) -> list[str]:
        return list(self._selected_features)
