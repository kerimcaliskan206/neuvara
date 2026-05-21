from pydantic import BaseModel, field_validator


class TuningConfig(BaseModel):
    n_trials: int = 50
    metric: str = "f1"           # "f1" | "roc_auc" | "accuracy" | "precision" | "recall"
    direction: str = "maximize"
    cv_folds: int = 3
    random_state: int = 42
    timeout: int | None = None   # seconds per model; None = no timeout
    n_jobs: int = 1              # parallel Optuna trials (1 = serial for reproducibility)
    show_progress: bool = True
    pruning: bool = True         # MedianPruner to discard bad trials early

    @field_validator("metric")
    @classmethod
    def _valid_metric(cls, v: str) -> str:
        allowed = {"f1", "roc_auc", "accuracy", "precision", "recall"}
        if v not in allowed:
            raise ValueError(f"metric must be one of {allowed}, got '{v}'")
        return v

    @field_validator("direction")
    @classmethod
    def _valid_direction(cls, v: str) -> str:
        if v not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'")
        return v
