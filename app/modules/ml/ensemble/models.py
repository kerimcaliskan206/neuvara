import logging
from enum import Enum

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    VOTING = "voting"
    WEIGHTED_VOTING = "weighted_voting"
    STACKING = "stacking"


def build_random_forest(random_state: int = 42, **kwargs) -> RandomForestClassifier:
    params = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    params.update(kwargs)
    logger.info("Building RandomForestClassifier: %s", params)
    return RandomForestClassifier(random_state=random_state, **params)


def build_xgboost(**kwargs):
    from xgboost import XGBClassifier

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)
    logger.info("Building XGBClassifier: %s", params)
    return XGBClassifier(**params)


def build_lightgbm(**kwargs):
    from lightgbm import LGBMClassifier

    params = {
        "n_estimators": 200,
        "max_depth": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }
    params.update(kwargs)
    logger.info("Building LGBMClassifier: %s", params)
    return LGBMClassifier(**params)


def build_voting_ensemble(
    estimators: list | None = None,
) -> VotingClassifier:
    if estimators is None:
        estimators = [
            ("rf", build_random_forest()),
            ("xgb", build_xgboost()),
            ("lgbm", build_lightgbm()),
        ]
    names = [name for name, _ in estimators]
    logger.info("Building VotingClassifier (soft, equal weights) with: %s", names)
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def build_weighted_voting_ensemble(
    estimators: list,
    weights: list[float] | None = None,
) -> VotingClassifier:
    """
    Soft-voting ensemble with explicit per-model weights.

    Weights are typically derived from held-out F1 or ROC-AUC scores so
    better-performing models have proportionally more influence.
    """
    names = [name for name, _ in estimators]
    logger.info(
        "Building weighted VotingClassifier with: %s | weights=%s",
        names,
        [round(w, 4) for w in weights] if weights else "equal",
    )
    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
        n_jobs=-1,
    )


class EnsembleFactory:
    _BUILDERS = {
        ModelType.RANDOM_FOREST: build_random_forest,
        ModelType.XGBOOST: build_xgboost,
        ModelType.LIGHTGBM: build_lightgbm,
        ModelType.VOTING: lambda **kw: build_voting_ensemble(),
        ModelType.WEIGHTED_VOTING: lambda **kw: build_weighted_voting_ensemble(
            estimators=[
                ("rf", build_random_forest()),
                ("xgb", build_xgboost()),
                ("lgbm", build_lightgbm()),
            ]
        ),
    }

    @classmethod
    def create(cls, model_type: ModelType, **kwargs):
        builder = cls._BUILDERS.get(model_type)
        if builder is None:
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Choose from: {list(ModelType)}"
            )
        return builder(**kwargs)
