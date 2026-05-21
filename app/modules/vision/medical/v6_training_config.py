"""
V6 medical training configuration — Phase 10.

Isolated configs for the first real medical specialization:
healthy_xray vs pneumonia_xray (Stage 2 of the 4-stage curriculum).

Three sub-stages (A → B → C) progressively unfreeze the EfficientNet backbone:
  A: fully frozen    — head only, high LR, fast convergence
  B: top-1 block     — last feature block + head, lower LR
  C: top-3 blocks    — selective fine-tuning, very low LR + focal loss

V5 production model is never touched.
All outputs isolated under models/vision/v6_medical/.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Class topology ────────────────────────────────────────────────────────────


V6_BINARY_MEDICAL_CLASSES: tuple[str, ...] = (
    "healthy_xray",
    "pneumonia_xray",
    "hard_negative",
    "fake_medical",
)

V6_POSITIVE_CLASSES: frozenset[str] = frozenset({"healthy_xray", "pneumonia_xray"})
V6_OOD_CLASSES: frozenset[str] = frozenset({"hard_negative", "fake_medical"})


# ── Sub-stage identifiers ─────────────────────────────────────────────────────


class V6SubStage(str, Enum):
    A_FROZEN    = "stage_a_frozen"
    B_TOP1      = "stage_b_top1"
    C_SELECTIVE = "stage_c_selective"


# ── Backbone policy ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BackboneUnfreezePolicy:
    """EfficientNet feature-block unfreezing policy."""
    freeze_all: bool = True
    unfreeze_top_n_blocks: int = 0   # 0=none, 1=Stage B, 3=Stage C


BACKBONE_A = BackboneUnfreezePolicy(freeze_all=True,  unfreeze_top_n_blocks=0)
BACKBONE_B = BackboneUnfreezePolicy(freeze_all=False, unfreeze_top_n_blocks=1)
BACKBONE_C = BackboneUnfreezePolicy(freeze_all=False, unfreeze_top_n_blocks=3)


def apply_backbone_policy(model, policy: BackboneUnfreezePolicy) -> None:
    """
    Apply freeze/unfreeze policy to an EfficientNetClassifier.

    Always freezes all features first, then selectively unfreezes the last N
    blocks. The classifier head is always trainable.
    """
    features = model._backbone.features

    for p in features.parameters():
        p.requires_grad = False
    for p in model._backbone.classifier.parameters():
        p.requires_grad = True

    if not policy.freeze_all and policy.unfreeze_top_n_blocks > 0:
        n = len(features)
        for i in range(max(0, n - policy.unfreeze_top_n_blocks), n):
            for p in features[i].parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "Backbone policy applied: freeze_all=%s unfreeze_top=%d | "
        "trainable=%d / total=%d",
        policy.freeze_all, policy.unfreeze_top_n_blocks, trainable, total,
    )


# ── Replay buffer config ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class V6ReplayConfig:
    """
    Experience replay for OOD forgetting prevention.

    When replay_dir is set, hard_negative images from the v5 training set
    are injected into the v6 training data (train split only) to prevent
    the model from forgetting OOD rejection learned in v5.
    """
    enabled: bool = True
    replay_dir: Path | None = None       # path to extra v5 hard_negative images
    fraction: float = 0.25              # fraction of training set to add as replay
    sampling: str = "hard_examples"     # "uniform" | "hard_examples"
    max_replay_images: int = 2_000      # cap to avoid replay domination


# ── Compatibility gates ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class V6CompatibilityConfig:
    """Pass/fail gates evaluated after each sub-stage before checkpoint is accepted."""

    check_ood_rejection_rate: bool = True
    min_ood_rejection_rate: float = 0.90     # hard_negative + fake_medical rejection

    check_positive_recall: bool = True
    min_positive_recall: float = 0.75        # healthy_xray + pneumonia_xray recall

    check_gradcam_sanity: bool = True        # activation maps must not be degenerate

    check_calibration: bool = True
    max_ece: float = 0.15                    # Expected Calibration Error ceiling

    check_semantic_conflict: bool = False    # advisory; requires CLIP at eval time
    max_semantic_conflict_rate: float = 0.12


# ── Per-sub-stage hyperparameters ─────────────────────────────────────────────


@dataclass(frozen=True)
class V6SubStageConfig:
    """Hyperparameters for one sub-stage of v6 training."""

    sub_stage: V6SubStage
    backbone_policy: BackboneUnfreezePolicy

    # Optimisation
    epochs: int
    learning_rate: float             # backbone LR (also head LR in Stage A)
    head_lr_factor: float = 5.0      # head_lr = learning_rate × head_lr_factor (B/C only)
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"        # "cosine" | "step" | "none"
    warmup_epochs: int = 2
    grad_clip: float = 1.0

    # Regularisation
    label_smoothing: float = 0.10
    dropout: float = 0.30
    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    # Batch
    batch_size: int = 32
    use_weighted_sampler: bool = True

    # Early stopping
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.001
    monitor_metric: str = "val_f1"   # "val_f1" | "val_loss"

    notes: str = ""


STAGE_A_CONFIG = V6SubStageConfig(
    sub_stage=V6SubStage.A_FROZEN,
    backbone_policy=BACKBONE_A,
    epochs=25,
    learning_rate=1e-3,
    label_smoothing=0.08,
    notes=(
        "Backbone fully frozen; head-only training. "
        "High LR safe — ImageNet features protected."
    ),
)

STAGE_B_CONFIG = V6SubStageConfig(
    sub_stage=V6SubStage.B_TOP1,
    backbone_policy=BACKBONE_B,
    epochs=20,
    learning_rate=2e-4,
    head_lr_factor=5.0,
    label_smoothing=0.10,
    notes=(
        "Top-1 EfficientNet block (features[-1]) unfrozen. "
        "Differential LR: backbone=2e-4, head=1e-3."
    ),
)

STAGE_C_CONFIG = V6SubStageConfig(
    sub_stage=V6SubStage.C_SELECTIVE,
    backbone_policy=BACKBONE_C,
    epochs=20,
    learning_rate=5e-5,
    head_lr_factor=4.0,
    label_smoothing=0.12,
    use_focal_loss=True,
    focal_gamma=2.0,
    early_stopping_patience=10,
    notes=(
        "Top-3 EfficientNet blocks unfrozen. Focal loss for class imbalance. "
        "Very low LR with cosine decay. "
        "MixUp between pneumonia/opacity classes is FORBIDDEN."
    ),
)

ALL_SUB_STAGES: tuple[V6SubStageConfig, ...] = (
    STAGE_A_CONFIG,
    STAGE_B_CONFIG,
    STAGE_C_CONFIG,
)

SUB_STAGE_MAP: dict[str, V6SubStageConfig] = {
    "a": STAGE_A_CONFIG,
    "b": STAGE_B_CONFIG,
    "c": STAGE_C_CONFIG,
    V6SubStage.A_FROZEN.value:    STAGE_A_CONFIG,
    V6SubStage.B_TOP1.value:      STAGE_B_CONFIG,
    V6SubStage.C_SELECTIVE.value: STAGE_C_CONFIG,
}


# ── Full training config ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class V6TrainingConfig:
    """Complete configuration for a v6 medical training run."""

    run_name: str
    classes: tuple[str, ...]
    positive_classes: frozenset[str]
    ood_classes: frozenset[str]

    sub_stages: tuple[V6SubStageConfig, ...]
    replay: V6ReplayConfig
    compatibility: V6CompatibilityConfig

    output_dir: Path = Path("models/vision/v6_medical")
    random_seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False

    notes: str = ""

    def class_to_idx(self) -> dict[str, int]:
        return {cls: i for i, cls in enumerate(self.classes)}

    def idx_to_class(self) -> dict[int, str]:
        return {i: cls for i, cls in enumerate(self.classes)}

    def get_sub_stage(self, key: str | V6SubStage) -> V6SubStageConfig:
        k = key.value if isinstance(key, V6SubStage) else key
        cfg = SUB_STAGE_MAP.get(k)
        if cfg is None or cfg not in self.sub_stages:
            raise KeyError(f"Sub-stage '{key}' not in this config.")
        return cfg

    def stage_output_dir(self, sub_stage: V6SubStageConfig) -> Path:
        return self.output_dir / sub_stage.sub_stage.value

    @property
    def num_classes(self) -> int:
        return len(self.classes)


V6_BINARY_MEDICAL_CONFIG = V6TrainingConfig(
    run_name="v6_binary_medical",
    classes=V6_BINARY_MEDICAL_CLASSES,
    positive_classes=V6_POSITIVE_CLASSES,
    ood_classes=V6_OOD_CLASSES,
    sub_stages=ALL_SUB_STAGES,
    replay=V6ReplayConfig(enabled=True, fraction=0.25, sampling="hard_examples"),
    compatibility=V6CompatibilityConfig(),
    notes=(
        "Phase 10: First real medical specialization. "
        "Positive: healthy_xray, pneumonia_xray. "
        "OOD: hard_negative, fake_medical. "
        "V5 production model is never modified."
    ),
)


# ── Checkpoint metadata schema ────────────────────────────────────────────────


def build_checkpoint_meta(
    *,
    run_name: str,
    sub_stage: str,
    classes: list[str],
    epoch: int,
    best_val_f1: float,
    val_f1: float,
    val_loss: float,
    ood_rejection_rate: float,
    compatibility_passed: bool,
    architecture: str = "efficientnet_b0",
    notes: str = "",
) -> dict:
    """Return a dict that is embedded in every v6 checkpoint under 'v6_meta'."""
    from datetime import datetime, timezone
    return {
        "v6_phase": "10",
        "run_name": run_name,
        "sub_stage": sub_stage,
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "num_classes": len(classes),
        "architecture": architecture,
        "epoch": epoch,
        "best_val_f1": round(best_val_f1, 4),
        "val_f1": round(val_f1, 4),
        "val_loss": round(val_loss, 5),
        "ood_rejection_rate": round(ood_rejection_rate, 4),
        "compatibility_passed": compatibility_passed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }
