from pathlib import Path

from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DataConfig(BaseModel):
    raw_dir: Path = _PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = _PROJECT_ROOT / "data" / "processed"


class ModelStorageConfig(BaseModel):
    models_dir: Path = _PROJECT_ROOT / "models"


class ColumnsConfig(BaseModel):
    required: list[str] = []          # validation fails if any are missing
    drop_on_load: list[str] = []      # removed immediately after loading


class PreprocessingConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    scaler: str = "standard"           # "standard" | "minmax" | "robust"
    missing_strategy: str = "median"   # "median" | "mean" | "most_frequent"
    missing_threshold: float = 0.5     # drop column if missing > this fraction


class FeatureSelectionConfig(BaseModel):
    enabled: bool = False
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    top_k: int | None = None


class RandomForestConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_split: int = 2
    class_weight: str = "balanced"


class XGBoostConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8


class LightGBMConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = -1
    learning_rate: float = 0.05
    num_leaves: int = 31
    class_weight: str = "balanced"


class ModelHyperparamsConfig(BaseModel):
    random_state: int = 42
    random_forest: RandomForestConfig = RandomForestConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()


class ExperimentsConfig(BaseModel):
    experiments_dir: Path = _PROJECT_ROOT / "experiments"


class MLConfig(BaseModel):
    data: DataConfig = DataConfig()
    storage: ModelStorageConfig = ModelStorageConfig()
    experiments: ExperimentsConfig = ExperimentsConfig()
    columns: ColumnsConfig = ColumnsConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()
    hyperparams: ModelHyperparamsConfig = ModelHyperparamsConfig()
    target_column: str = "label"


ml_config = MLConfig()
