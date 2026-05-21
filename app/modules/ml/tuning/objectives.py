"""
Optuna objective factory functions.

Each factory returns a closure that:
  1. Samples hyperparameters from the search space
  2. Builds a fresh model with those params
  3. Evaluates it via StratifiedKFold cross-validation
  4. Returns the mean CV score (Optuna maximizes this)

Using cross-validation keeps test data completely untouched during tuning.
"""
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from app.modules.ml.ensemble.models import build_lightgbm, build_random_forest, build_xgboost
from app.modules.ml.tuning.config import TuningConfig
from app.modules.ml.tuning.search_spaces import lgbm_search_space, rf_search_space, xgb_search_space

# Maps our metric names to sklearn scoring strings
_METRIC_TO_SCORING: dict[str, str] = {
    "f1": "f1_weighted",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
}


def _make_cv(config: TuningConfig) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )


def make_rf_objective(X, y, config: TuningConfig):
    scoring = _METRIC_TO_SCORING.get(config.metric, "f1_weighted")
    cv = _make_cv(config)

    def objective(trial: optuna.Trial) -> float:
        params = rf_search_space(trial)
        model = build_random_forest(random_state=config.random_state, **params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    return objective


def make_xgb_objective(X, y, config: TuningConfig):
    scoring = _METRIC_TO_SCORING.get(config.metric, "f1_weighted")
    cv = _make_cv(config)

    def objective(trial: optuna.Trial) -> float:
        params = xgb_search_space(trial)
        model = build_xgboost(**params, random_state=config.random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    return objective


def make_lgbm_objective(X, y, config: TuningConfig):
    scoring = _METRIC_TO_SCORING.get(config.metric, "f1_weighted")
    cv = _make_cv(config)

    def objective(trial: optuna.Trial) -> float:
        params = lgbm_search_space(trial)
        model = build_lightgbm(**params, random_state=config.random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    return objective
