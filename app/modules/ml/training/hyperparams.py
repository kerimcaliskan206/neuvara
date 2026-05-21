from pydantic import BaseModel


class RandomForestParams(BaseModel):
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: str = "balanced"
    n_jobs: int = -1


class XGBoostParams(BaseModel):
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    eval_metric: str = "logloss"
    n_jobs: int = -1


class LightGBMParams(BaseModel):
    n_estimators: int = 200
    max_depth: int = -1
    learning_rate: float = 0.05
    num_leaves: int = 31
    class_weight: str = "balanced"
    n_jobs: int = -1
    verbose: int = -1


class ModelHyperparams(BaseModel):
    random_forest: RandomForestParams = RandomForestParams()
    xgboost: XGBoostParams = XGBoostParams()
    lightgbm: LightGBMParams = LightGBMParams()
