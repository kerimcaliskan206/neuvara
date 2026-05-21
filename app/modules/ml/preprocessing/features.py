import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering for hantavirus prediction.

    Planned features (implement when training data is available):
    - rodent_density_index   : population density of carrier species per km²
    - seasonal_risk_score    : risk multiplier based on month/season
    - geographic_risk_zone   : encoded regional risk level
    - rainfall_30d           : cumulative rainfall in prior 30 days
    - temp_variance_7d       : temperature variance over prior 7 days
    - human_exposure_index   : proximity to known rodent habitats
    """

    def fit(self, X: pd.DataFrame, y=None):
        logger.info(
            "FeatureEngineer fitted: %d input features (no transformations active yet).",
            X.shape[1],
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()
