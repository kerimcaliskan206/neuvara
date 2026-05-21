"""
Staged training plan — Phase 8.

Defines the four-stage curriculum for transitioning from the v5 binary
(related / unrelated / hard_negative) classifier toward a fully-specialized
medical multiclass classifier.

Stage overview
--------------
  Stage 1  Baseline retention   — retrain v5 topology with cleaned data;
                                   frozen backbone, head-only.  Establishes
                                   a reproducible performance floor.
  Stage 2  Binary medical       — introduce healthy_xray vs pneumonia_xray
                                   as separate classes; replace `related`.
                                   Unfreeze top 1 EfficientNet block.
  Stage 3  Subtle distinction   — add opacity_pattern + infiltrate_pattern;
                                   highest confusion risk; unfreeze top 3 blocks.
  Stage 4  Full specialization  — add microscopy subclasses, hantavirus_candidate,
                                   fake_medical, ai_generated_medical;
                                   full fine-tuning at low LR.

OOD preservation is enforced at every stage: hard_negative and unrelated
must be present throughout and their rejection rate must not fall below the
floor set in OOD_PRESERVATION_POLICY.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────


class TrainingStage(str, Enum):
    STAGE_1_BASELINE          = "stage_1_baseline"
    STAGE_2_BINARY_MEDICAL    = "stage_2_binary_medical"
    STAGE_3_SUBTLE_CLASSES    = "stage_3_subtle_classes"
    STAGE_4_FULL_SPECIALIZATION = "stage_4_full_specialization"


class BackboneStrategy(str, Enum):
    """EfficientNet backbone freezing policy for each stage."""
    FULLY_FROZEN          = "fully_frozen"
    TOP_1_BLOCK_UNFROZEN  = "top_1_block_unfrozen"
    TOP_3_BLOCKS_UNFROZEN = "top_3_blocks_unfrozen"
    FULLY_UNFROZEN        = "fully_unfrozen"


class AugmentationIntensity(str, Enum):
    LIGHT    = "light"
    MODERATE = "moderate"
    HEAVY    = "heavy"


class DataBalancingStrategy(str, Enum):
    CLASS_WEIGHTS   = "class_weights"
    OVERSAMPLING    = "oversampling"
    UNDERSAMPLING   = "undersampling"
    MIXED           = "mixed"


# ── Stage descriptor ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageDescriptor:
    """Complete training configuration for a single curriculum stage."""

    stage: TrainingStage
    display_name: str
    description: str

    # Classes active in this stage
    target_classes: tuple[str, ...]        # positive medical classes
    ood_classes: tuple[str, ...]           # always-present OOD rejection classes

    # Backbone
    backbone_strategy: BackboneStrategy
    unfrozen_blocks: int                   # how many top blocks to unfreeze

    # Hyperparameters
    max_epochs: int
    base_lr: float
    warmup_epochs: int
    weight_decay: float
    label_smoothing: float

    # Data
    min_samples_per_class: int
    augmentation: AugmentationIntensity
    balancing_strategy: DataBalancingStrategy
    use_class_weights: bool

    # Training controls
    early_stopping_patience: int
    min_delta: float                       # min improvement to count as progress

    # Prerequisite + notes
    prerequisite: TrainingStage | None
    notes: str = ""


# ── Confusion risk ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfusionRiskGroup:
    """
    Classes that the model may confuse with each other.

    Risk level guides data collection priority and evaluation effort:
      high    — similar texture/density; hard even for radiologists
      medium  — distinguishable but requires fine-grained features
      low     — usually separable by modality or large morphological diff
    """
    name: str
    classes: tuple[str, ...]
    risk_level: str      # "high" | "medium" | "low"
    introduced_at: TrainingStage
    mitigation: str


# ── OOD preservation ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OODPreservationPolicy:
    """
    Constraints enforced at every stage to prevent erosion of OOD rejection.

    The semantic gate (CLIP) acts as a first-line OOD filter; the classifier
    provides a second line.  Both must be preserved.
    """
    hard_negative_min_fraction: float      # minimum fraction in train set
    unrelated_min_fraction: float
    ood_eval_every_n_epochs: int           # run OOD rejection eval this often
    min_hard_negative_rejection_rate: float  # acceptance.py hard-reject must hold
    min_unrelated_rejection_rate: float
    semantic_gate_always_active: bool      # CLIP gate cannot be disabled
    replay_fraction_from_prev_stage: float # v5 data kept in replay buffer
    notes: str = ""


# ── Dataset balancing plan ────────────────────────────────────────────────────


@dataclass
class DatasetBalancingPlan:
    """Per-stage dataset size and balancing targets."""

    stage: TrainingStage
    strategy: DataBalancingStrategy
    target_per_class: dict[str, int]       # desired sample count per class
    hard_negative_cap: int                 # hard cap prevents OOD class dominating
    oversample_below_threshold: int        # oversample classes below this count
    undersample_above_threshold: int       # undersample classes above this count
    synthetic_augmentation_allowed: bool   # allow MixUp/CutMix for rare classes
    notes: str = ""


# ── Curriculum stages ─────────────────────────────────────────────────────────


_OOD_CLASSES = ("unrelated", "hard_negative")

STAGE_PLAN: tuple[StageDescriptor, ...] = (

    StageDescriptor(
        stage=TrainingStage.STAGE_1_BASELINE,
        display_name="Stage 1 — Baseline Retention",
        description=(
            "Retrain with the v5 class set (related / unrelated / hard_negative) "
            "on a cleaned, deduplicated dataset.  Backbone fully frozen.  "
            "Goal: establish a reproducible v5 performance floor before any "
            "class-topology changes."
        ),
        target_classes=("related",),
        ood_classes=_OOD_CLASSES,
        backbone_strategy=BackboneStrategy.FULLY_FROZEN,
        unfrozen_blocks=0,
        max_epochs=30,
        base_lr=1e-3,
        warmup_epochs=3,
        weight_decay=1e-4,
        label_smoothing=0.05,
        min_samples_per_class=500,
        augmentation=AugmentationIntensity.MODERATE,
        balancing_strategy=DataBalancingStrategy.CLASS_WEIGHTS,
        use_class_weights=True,
        early_stopping_patience=7,
        min_delta=0.001,
        prerequisite=None,
        notes=(
            "Run dataset audit (scripts/medical_dataset_audit.py) before this stage. "
            "Must achieve ≥ 0.92 hard_negative rejection rate to proceed to Stage 2."
        ),
    ),

    StageDescriptor(
        stage=TrainingStage.STAGE_2_BINARY_MEDICAL,
        display_name="Stage 2 — Binary Medical (Healthy vs Pneumonia)",
        description=(
            "Replace the single `related` class with two distinct medical classes: "
            "healthy_xray and pneumonia_xray.  Unfreeze the top EfficientNet block "
            "to allow the feature extractor to adapt to finer medical distinctions. "
            "OOD classes (unrelated, hard_negative) retained at full proportion."
        ),
        target_classes=("healthy_xray", "pneumonia_xray"),
        ood_classes=_OOD_CLASSES,
        backbone_strategy=BackboneStrategy.TOP_1_BLOCK_UNFROZEN,
        unfrozen_blocks=1,
        max_epochs=40,
        base_lr=5e-4,
        warmup_epochs=5,
        weight_decay=1e-4,
        label_smoothing=0.08,
        min_samples_per_class=800,
        augmentation=AugmentationIntensity.MODERATE,
        balancing_strategy=DataBalancingStrategy.MIXED,
        use_class_weights=True,
        early_stopping_patience=8,
        min_delta=0.001,
        prerequisite=TrainingStage.STAGE_1_BASELINE,
        notes=(
            "Initialise from Stage 1 checkpoint. "
            "Use knowledge distillation on OOD classes to prevent forgetting. "
            "Minimum healthy_xray / pneumonia_xray split: 60/40 ratio max."
        ),
    ),

    StageDescriptor(
        stage=TrainingStage.STAGE_3_SUBTLE_CLASSES,
        display_name="Stage 3 — Subtle Pattern Classes",
        description=(
            "Introduce opacity_pattern and infiltrate_pattern alongside Stage 2 classes. "
            "These classes are visually similar to pneumonia_xray — highest confusion risk. "
            "Unfreeze top 3 EfficientNet blocks.  Increase label smoothing to reduce "
            "overconfidence on ambiguous radiological patterns."
        ),
        target_classes=("healthy_xray", "pneumonia_xray", "opacity_pattern", "infiltrate_pattern"),
        ood_classes=_OOD_CLASSES,
        backbone_strategy=BackboneStrategy.TOP_3_BLOCKS_UNFROZEN,
        unfrozen_blocks=3,
        max_epochs=50,
        base_lr=2e-4,
        warmup_epochs=8,
        weight_decay=2e-4,
        label_smoothing=0.12,
        min_samples_per_class=600,
        augmentation=AugmentationIntensity.HEAVY,
        balancing_strategy=DataBalancingStrategy.MIXED,
        use_class_weights=True,
        early_stopping_patience=10,
        min_delta=0.0005,
        prerequisite=TrainingStage.STAGE_2_BINARY_MEDICAL,
        notes=(
            "Initialise from Stage 2 checkpoint. "
            "Run confusion matrix analysis per epoch — flag if pneumonia/opacity confusion "
            "rate exceeds 25 %. "
            "Consider focal loss (gamma=2) to handle hard pneumonia/opacity boundary cases."
        ),
    ),

    StageDescriptor(
        stage=TrainingStage.STAGE_4_FULL_SPECIALIZATION,
        display_name="Stage 4 — Full Medical Specialization",
        description=(
            "Add hantavirus_candidate, normal_microscopy, infected_microscopy, "
            "fake_medical, and ai_generated_medical.  Full backbone fine-tuning "
            "at very low LR.  Microscopy classes require separate preprocessing "
            "pipeline (stain normalisation before augmentation)."
        ),
        target_classes=(
            "healthy_xray", "pneumonia_xray", "opacity_pattern", "infiltrate_pattern",
            "hantavirus_candidate", "normal_microscopy", "infected_microscopy",
        ),
        ood_classes=("unrelated", "hard_negative", "fake_medical", "ai_generated_medical"),
        backbone_strategy=BackboneStrategy.FULLY_UNFROZEN,
        unfrozen_blocks=-1,     # all blocks
        max_epochs=60,
        base_lr=1e-5,
        warmup_epochs=10,
        weight_decay=3e-4,
        label_smoothing=0.10,
        min_samples_per_class=400,
        augmentation=AugmentationIntensity.HEAVY,
        balancing_strategy=DataBalancingStrategy.MIXED,
        use_class_weights=True,
        early_stopping_patience=12,
        min_delta=0.0003,
        prerequisite=TrainingStage.STAGE_3_SUBTLE_CLASSES,
        notes=(
            "Initialise from Stage 3 checkpoint. "
            "hantavirus_candidate requires ≥ 300 samples to train; use Stage 3 model "
            "in inference mode until threshold met. "
            "Microscopy classes need stain normalisation (Macenko/Vahadane) "
            "before augmentation pipeline. "
            "fake_medical and ai_generated_medical move to ood_classes — "
            "they are rejection classes, not diagnostic targets."
        ),
    ),
)


# ── Confusion risk groups ─────────────────────────────────────────────────────


CONFUSION_RISK_GROUPS: tuple[ConfusionRiskGroup, ...] = (
    ConfusionRiskGroup(
        name="pneumonia_opacity_boundary",
        classes=("pneumonia_xray", "opacity_pattern"),
        risk_level="high",
        introduced_at=TrainingStage.STAGE_3_SUBTLE_CLASSES,
        mitigation=(
            "Focal loss (gamma ≥ 2), class-specific confidence thresholds, "
            "Grad-CAM audit on boundary cases, radiologist review of confusion pairs."
        ),
    ),
    ConfusionRiskGroup(
        name="opacity_infiltrate_boundary",
        classes=("opacity_pattern", "infiltrate_pattern"),
        risk_level="high",
        introduced_at=TrainingStage.STAGE_3_SUBTLE_CLASSES,
        mitigation=(
            "Heavy augmentation on infiltrate class, "
            "semantic consistency check via CLIP reasoning layer."
        ),
    ),
    ConfusionRiskGroup(
        name="hantavirus_pneumonia_overlap",
        classes=("hantavirus_candidate", "pneumonia_xray", "opacity_pattern"),
        risk_level="medium",
        introduced_at=TrainingStage.STAGE_4_FULL_SPECIALIZATION,
        mitigation=(
            "Clinical metadata (bilateral vs unilateral pattern) required for true separation. "
            "Model predicts pattern similarity; final diagnosis requires clinical context."
        ),
    ),
    ConfusionRiskGroup(
        name="microscopy_subclasses",
        classes=("normal_microscopy", "infected_microscopy"),
        risk_level="medium",
        introduced_at=TrainingStage.STAGE_4_FULL_SPECIALIZATION,
        mitigation=(
            "Stain normalisation before inference, "
            "high-magnification crops preferred, "
            "separate validation set per staining protocol."
        ),
    ),
    ConfusionRiskGroup(
        name="fake_vs_real_medical",
        classes=("fake_medical", "healthy_xray"),
        risk_level="low",
        introduced_at=TrainingStage.STAGE_4_FULL_SPECIALIZATION,
        mitigation=(
            "Medical refiner (Phase 4) provides advisory fake_medical_score. "
            "Calibration V2 flags suspicious trust_tier before classifier decision."
        ),
    ),
)


# ── OOD preservation policy ───────────────────────────────────────────────────


OOD_PRESERVATION_POLICY = OODPreservationPolicy(
    hard_negative_min_fraction=0.15,
    unrelated_min_fraction=0.15,
    ood_eval_every_n_epochs=5,
    min_hard_negative_rejection_rate=0.92,
    min_unrelated_rejection_rate=0.90,
    semantic_gate_always_active=True,
    replay_fraction_from_prev_stage=0.20,
    notes=(
        "Hard negative and unrelated must each constitute ≥ 15 % of train batches. "
        "If rejection rate drops below floor, halt training and review OOD data quality. "
        "CLIP semantic gate cannot be disabled even during backbone fine-tuning — "
        "it runs as a pre-inference filter independent of the EfficientNet classifier."
    ),
)


# ── Balancing plans ───────────────────────────────────────────────────────────


BALANCING_PLANS: dict[TrainingStage, DatasetBalancingPlan] = {

    TrainingStage.STAGE_1_BASELINE: DatasetBalancingPlan(
        stage=TrainingStage.STAGE_1_BASELINE,
        strategy=DataBalancingStrategy.CLASS_WEIGHTS,
        target_per_class={"related": 2000, "unrelated": 600, "hard_negative": 600},
        hard_negative_cap=800,
        oversample_below_threshold=300,
        undersample_above_threshold=3000,
        synthetic_augmentation_allowed=False,
    ),

    TrainingStage.STAGE_2_BINARY_MEDICAL: DatasetBalancingPlan(
        stage=TrainingStage.STAGE_2_BINARY_MEDICAL,
        strategy=DataBalancingStrategy.MIXED,
        target_per_class={
            "healthy_xray": 1500,
            "pneumonia_xray": 1500,
            "unrelated": 500,
            "hard_negative": 500,
        },
        hard_negative_cap=700,
        oversample_below_threshold=400,
        undersample_above_threshold=2500,
        synthetic_augmentation_allowed=True,
        notes="Oversample healthy_xray if fewer than 400 samples available at annotation time.",
    ),

    TrainingStage.STAGE_3_SUBTLE_CLASSES: DatasetBalancingPlan(
        stage=TrainingStage.STAGE_3_SUBTLE_CLASSES,
        strategy=DataBalancingStrategy.MIXED,
        target_per_class={
            "healthy_xray": 1200,
            "pneumonia_xray": 1200,
            "opacity_pattern": 800,
            "infiltrate_pattern": 800,
            "unrelated": 500,
            "hard_negative": 500,
        },
        hard_negative_cap=700,
        oversample_below_threshold=400,
        undersample_above_threshold=2000,
        synthetic_augmentation_allowed=True,
        notes="MixUp between pneumonia and opacity classes explicitly FORBIDDEN — would blur hard boundary.",
    ),

    TrainingStage.STAGE_4_FULL_SPECIALIZATION: DatasetBalancingPlan(
        stage=TrainingStage.STAGE_4_FULL_SPECIALIZATION,
        strategy=DataBalancingStrategy.MIXED,
        target_per_class={
            "healthy_xray": 1000,
            "pneumonia_xray": 1000,
            "opacity_pattern": 700,
            "infiltrate_pattern": 700,
            "hantavirus_candidate": 300,
            "normal_microscopy": 600,
            "infected_microscopy": 600,
            "unrelated": 400,
            "hard_negative": 400,
            "fake_medical": 300,
            "ai_generated_medical": 300,
        },
        hard_negative_cap=600,
        oversample_below_threshold=200,
        undersample_above_threshold=1500,
        synthetic_augmentation_allowed=True,
        notes=(
            "hantavirus_candidate is minority — oversample aggressively with heavy augmentation. "
            "Do not mix microscopy and radiology augmentation pipelines."
        ),
    ),
}


# ── Query helpers ─────────────────────────────────────────────────────────────


def get_stage_descriptor(stage: TrainingStage) -> StageDescriptor:
    for s in STAGE_PLAN:
        if s.stage == stage:
            return s
    raise KeyError(f"Unknown stage: {stage}")


def get_all_classes_at_stage(stage: TrainingStage) -> tuple[str, ...]:
    """Return the combined target + OOD class set for a given stage."""
    desc = get_stage_descriptor(stage)
    return desc.target_classes + desc.ood_classes


def get_high_risk_confusion_pairs() -> list[tuple[str, str]]:
    """Return all class pairs in high-risk confusion groups."""
    pairs: list[tuple[str, str]] = []
    for group in CONFUSION_RISK_GROUPS:
        if group.risk_level == "high":
            classes = group.classes
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    pairs.append((classes[i], classes[j]))
    return pairs
