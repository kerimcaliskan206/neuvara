from pydantic import BaseModel, field_validator


class VisionTrainingConfig(BaseModel):
    """All hyperparameters for a vision training run."""

    # Dataset
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Optimization
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"        # "adamw" | "adam" | "sgd"
    momentum: float = 0.9           # SGD only

    # Learning rate schedule
    scheduler: str = "cosine"       # "cosine" | "step" | "plateau" | "none"
    step_size: int = 10             # StepLR: decay every N epochs
    gamma: float = 0.1              # StepLR decay factor
    warmup_epochs: int = 2          # Linear warmup before main schedule

    # Regularization
    grad_clip: float | None = 1.0   # Max gradient norm (None = disabled)
    mixed_precision: bool = False   # Automatic Mixed Precision (CUDA only)
    label_smoothing: float = 0.1    # Prevents overconfident softmax

    # Two-phase transfer learning
    freeze_epochs: int = 5          # Phase A: backbone frozen, head trained
    unfreeze_lr: float = 1e-5       # Phase B: backbone learning rate
    differential_lr_factor: float = 5.0
    # Phase B head LR = unfreeze_lr * differential_lr_factor
    # Backbone is trained conservatively; head can be updated more aggressively.

    # Class imbalance
    use_weighted_sampler: bool = True   # WeightedRandomSampler in training DataLoader
    use_focal_loss: bool = False        # Focal Loss instead of CrossEntropyLoss
    focal_gamma: float = 2.0           # Focal Loss focusing parameter

    # Early stopping
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.001

    # Checkpointing
    save_best_only: bool = True
    monitor_metric: str = "val_f1"  # Metric to track for best-model checkpoint

    # Reproducibility
    random_state: int = 42

    @field_validator("optimizer")
    @classmethod
    def _valid_optimizer(cls, v: str) -> str:
        allowed = {"adamw", "adam", "sgd"}
        if v not in allowed:
            raise ValueError(f"optimizer must be one of {allowed}")
        return v

    @field_validator("scheduler")
    @classmethod
    def _valid_scheduler(cls, v: str) -> str:
        allowed = {"cosine", "step", "plateau", "none"}
        if v not in allowed:
            raise ValueError(f"scheduler must be one of {allowed}")
        return v
