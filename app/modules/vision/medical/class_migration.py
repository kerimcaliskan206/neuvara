"""
Class migration strategy — Phase 8.

Defines the complete mapping from the current v5 class topology
(related / unrelated / hard_negative) to the planned v6 medical
multiclass topology, together with safety rails that prevent
catastrophic forgetting and OOD quality regression.

v5 topology
-----------
  related       — any medical image the pipeline should accept
  unrelated     — clearly non-medical (wildlife, objects, scenes)
  hard_negative — looks medical but should be rejected (duplicate-colour
                  radiograph, low-quality scan, adversarial input)

v6 topology (staged)
--------------------
  Radiology  : healthy_xray, pneumonia_xray, opacity_pattern,
               infiltrate_pattern, hantavirus_candidate
  Microscopy : normal_microscopy, infected_microscopy
  OOD / rejection: hard_negative, unrelated, fake_medical, ai_generated_medical

Migration challenge
-------------------
`related` is a coarse umbrella label.  Every image in v5/related must be
re-annotated (or sourced fresh) with a v6 subclass label.  This cannot be
done by the model itself — it requires either human annotation, public
dataset alignment, or CLIP-guided weak supervision followed by human QA.

The migration mappings below encode:
  1. Which v5 label maps to which v6 label(s)
  2. Whether re-annotation is required
  3. The splitting strategy if one-to-many
  4. The risk level for forgetting / regression
  5. Catastrophic-forgetting guards per class transition

acceptance.py compatibility note
---------------------------------
NON_TARGET_CLASSES in acceptance.py must be updated at each stage
transition.  The ACCEPTANCE_POLICY_BY_STAGE dict below provides the
new NON_TARGET_CLASSES set for each stage so acceptance.py can be
updated safely during rollout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── Label enums ───────────────────────────────────────────────────────────────


class V5Label(str, Enum):
    """Current production class labels (v5 EfficientNet classifier)."""
    RELATED       = "related"
    UNRELATED     = "unrelated"
    HARD_NEGATIVE = "hard_negative"


class V6Label(str, Enum):
    """Target v6 medical multiclass labels."""
    # Radiology — positive diagnostic targets
    HEALTHY_XRAY          = "healthy_xray"
    PNEUMONIA_XRAY        = "pneumonia_xray"
    OPACITY_PATTERN       = "opacity_pattern"
    INFILTRATE_PATTERN    = "infiltrate_pattern"
    HANTAVIRUS_CANDIDATE  = "hantavirus_candidate"
    # Microscopy — positive diagnostic targets
    NORMAL_MICROSCOPY     = "normal_microscopy"
    INFECTED_MICROSCOPY   = "infected_microscopy"
    # OOD / rejection classes
    HARD_NEGATIVE         = "hard_negative"
    UNRELATED             = "unrelated"
    FAKE_MEDICAL          = "fake_medical"
    AI_GENERATED_MEDICAL  = "ai_generated_medical"


class MigrationRisk(str, Enum):
    LOW      = "low"      # direct 1:1 carry-over, no data change
    MEDIUM   = "medium"   # requires dataset collection or annotation
    HIGH     = "high"     # class splitting with ambiguity risk
    CRITICAL = "critical" # insufficient data; must not train until resolved


class SplittingStrategy(str, Enum):
    """How to split a coarse v5 label into v6 subclasses."""
    DIRECT          = "direct"          # 1:1 mapping, no splitting needed
    MANUAL_RELABEL  = "manual_relabel"  # human annotation of existing images
    PUBLIC_DATASET  = "public_dataset"  # source new images from public datasets
    CLIP_WEAK_SUPER = "clip_weak_super" # CLIP semantic scores guide pre-annotation; human QA required
    NEW_COLLECTION  = "new_collection"  # new data must be collected (e.g. hantavirus)


# ── Migration mapping ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassMigrationMapping:
    """Migration record for a single v5 → v6 class transition."""

    v5_label: V5Label
    v6_labels: tuple[V6Label, ...]
    splitting_strategy: SplittingStrategy
    requires_relabeling: bool        # human annotation needed
    requires_new_data: bool          # new images must be sourced/collected
    migration_risk: MigrationRisk
    minimum_v6_samples: dict[str, int] = field(default_factory=dict)
    public_dataset_sources: tuple[str, ...] = ()
    notes: str = ""


V5_TO_V6_MIGRATION: dict[V5Label, ClassMigrationMapping] = {

    V5Label.RELATED: ClassMigrationMapping(
        v5_label=V5Label.RELATED,
        v6_labels=(
            V6Label.HEALTHY_XRAY,
            V6Label.PNEUMONIA_XRAY,
            V6Label.OPACITY_PATTERN,
            V6Label.INFILTRATE_PATTERN,
            V6Label.HANTAVIRUS_CANDIDATE,
            V6Label.NORMAL_MICROSCOPY,
            V6Label.INFECTED_MICROSCOPY,
        ),
        splitting_strategy=SplittingStrategy.CLIP_WEAK_SUPER,
        requires_relabeling=True,
        requires_new_data=True,
        migration_risk=MigrationRisk.HIGH,
        minimum_v6_samples={
            "healthy_xray":          800,
            "pneumonia_xray":        800,
            "opacity_pattern":       500,
            "infiltrate_pattern":    500,
            "hantavirus_candidate":  300,
            "normal_microscopy":     500,
            "infected_microscopy":   500,
        },
        public_dataset_sources=(
            "NIH ChestX-ray14 (112,120 images, 14 conditions)",
            "CheXpert (224,316 images, Stanford)",
            "PneumoniaMNIST (5,856 pediatric CXR, Kaggle)",
            "RSNA Pneumonia Detection Challenge (26,684 CXR)",
            "Malaria Cell Images Dataset (27,558 microscopy, NIH)",
        ),
        notes=(
            "CLIP semantic refiner (Phase 4) scores can guide pre-annotation: "
            "high medical_plausibility + healthy_xray sub-group → candidate for healthy_xray label. "
            "All CLIP-guided assignments MUST be reviewed by a human annotator before training. "
            "Hantavirus CXR images require verified clinical provenance — do not use CLIP alone."
        ),
    ),

    V5Label.UNRELATED: ClassMigrationMapping(
        v5_label=V5Label.UNRELATED,
        v6_labels=(V6Label.UNRELATED,),
        splitting_strategy=SplittingStrategy.DIRECT,
        requires_relabeling=False,
        requires_new_data=False,
        migration_risk=MigrationRisk.LOW,
        minimum_v6_samples={"unrelated": 400},
        notes=(
            "Direct carry-over. Existing unrelated images are already clean OOD samples. "
            "Optionally expand with additional wildlife/object images for robustness."
        ),
    ),

    V5Label.HARD_NEGATIVE: ClassMigrationMapping(
        v5_label=V5Label.HARD_NEGATIVE,
        v6_labels=(
            V6Label.HARD_NEGATIVE,
            V6Label.FAKE_MEDICAL,
            V6Label.AI_GENERATED_MEDICAL,
        ),
        splitting_strategy=SplittingStrategy.MANUAL_RELABEL,
        requires_relabeling=True,
        requires_new_data=True,
        migration_risk=MigrationRisk.MEDIUM,
        minimum_v6_samples={
            "hard_negative":        400,
            "fake_medical":         200,
            "ai_generated_medical": 200,
        },
        notes=(
            "Most existing hard_negative images stay as hard_negative. "
            "Images generated by AI (GAN, diffusion) are relabeled as ai_generated_medical. "
            "Synthetic procedural textures are relabeled as fake_medical. "
            "New fake/AI images can be generated programmatically for robustness training."
        ),
    ),
}


# ── Acceptance policy by stage ────────────────────────────────────────────────


# acceptance.py NON_TARGET_CLASSES must be updated at each stage.
# This dict provides the correct frozenset for each transition.
ACCEPTANCE_POLICY_BY_STAGE: dict[str, frozenset[str]] = {
    "stage_1_baseline": frozenset({"unrelated", "hard_negative"}),
    "stage_2_binary_medical": frozenset({"unrelated", "hard_negative"}),
    "stage_3_subtle_classes": frozenset({"unrelated", "hard_negative"}),
    "stage_4_full_specialization": frozenset({
        "unrelated", "hard_negative", "fake_medical", "ai_generated_medical"
    }),
}

# Target classes accepted as positive predictions per stage
POSITIVE_CLASSES_BY_STAGE: dict[str, frozenset[str]] = {
    "stage_1_baseline": frozenset({"related"}),
    "stage_2_binary_medical": frozenset({"healthy_xray", "pneumonia_xray"}),
    "stage_3_subtle_classes": frozenset({
        "healthy_xray", "pneumonia_xray", "opacity_pattern", "infiltrate_pattern"
    }),
    "stage_4_full_specialization": frozenset({
        "healthy_xray", "pneumonia_xray", "opacity_pattern", "infiltrate_pattern",
        "hantavirus_candidate", "normal_microscopy", "infected_microscopy",
    }),
}


# ── Catastrophic forgetting prevention ───────────────────────────────────────


@dataclass(frozen=True)
class ForgettingPreventionConfig:
    """
    Configuration for techniques that prevent catastrophic forgetting
    when transitioning between training stages.

    Each technique is independently toggleable so the training script
    can enable the minimal necessary set for each stage transition.
    """

    # Knowledge distillation from previous stage checkpoint
    use_distillation: bool
    distillation_temperature: float          # KD softmax temperature (2–5 typical)
    distillation_alpha: float                # weight of KD loss vs CE loss [0, 1]

    # Experience replay: keep a fraction of previous-stage training data
    use_replay: bool
    replay_buffer_fraction: float            # fraction of prev-stage images to replay
    replay_sampling: str                     # "uniform" | "hard_examples"

    # Elastic Weight Consolidation (Fisher-based parameter importance)
    use_ewc: bool
    ewc_lambda: float                        # regularisation strength

    # Performance floor from previous stage (halt if violated)
    performance_floors: dict[str, float]     # metric → minimum acceptable value

    notes: str = ""


FORGETTING_PREVENTION: dict[str, ForgettingPreventionConfig] = {

    "stage_1_to_stage_2": ForgettingPreventionConfig(
        use_distillation=True,
        distillation_temperature=3.0,
        distillation_alpha=0.3,
        use_replay=True,
        replay_buffer_fraction=0.25,
        replay_sampling="hard_examples",
        use_ewc=False,
        ewc_lambda=0.0,
        performance_floors={
            "hard_negative_rejection_rate": 0.92,
            "unrelated_rejection_rate": 0.90,
        },
        notes="Light distillation: backbone is frozen, only head changes.",
    ),

    "stage_2_to_stage_3": ForgettingPreventionConfig(
        use_distillation=True,
        distillation_temperature=4.0,
        distillation_alpha=0.4,
        use_replay=True,
        replay_buffer_fraction=0.30,
        replay_sampling="hard_examples",
        use_ewc=True,
        ewc_lambda=500.0,
        performance_floors={
            "hard_negative_rejection_rate": 0.92,
            "unrelated_rejection_rate": 0.90,
            "healthy_vs_pneumonia_auc": 0.88,
        },
        notes=(
            "Strongest forgetting risk: 3 backbone blocks unfreeze. "
            "EWC protects OOD-critical parameter directions."
        ),
    ),

    "stage_3_to_stage_4": ForgettingPreventionConfig(
        use_distillation=True,
        distillation_temperature=4.0,
        distillation_alpha=0.35,
        use_replay=True,
        replay_buffer_fraction=0.20,
        replay_sampling="uniform",
        use_ewc=True,
        ewc_lambda=300.0,
        performance_floors={
            "hard_negative_rejection_rate": 0.90,
            "unrelated_rejection_rate": 0.88,
            "healthy_vs_pneumonia_auc": 0.86,
            "opacity_vs_infiltrate_auc": 0.82,
        },
        notes="Full backbone unfreezing — use very low LR (1e-5) with cosine annealing.",
    ),
}


# ── CLIP weak-supervision annotation workflow ─────────────────────────────────


@dataclass(frozen=True)
class CLIPAnnotationWorkflow:
    """
    Describes how to use the medical refiner (Phase 4) output to
    pre-annotate v5/related images for v6 subclass labeling.

    This is a suggestion protocol for human annotators, NOT an automated pipeline.
    All CLIP-guided pre-annotations require human QA before training use.
    """
    name: str
    clip_signal: str        # which SemanticInfo / MedicalRefinement field drives the rule
    threshold: float
    candidate_v6_label: V6Label
    confidence: str         # "high" | "medium" | "low" — annotator QA workload
    notes: str


CLIP_ANNOTATION_WORKFLOWS: tuple[CLIPAnnotationWorkflow, ...] = (
    CLIPAnnotationWorkflow(
        name="healthy_xray_candidate",
        clip_signal="medical_refinement.semantic_medical_type == 'healthy_xray' AND medical_plausibility >= 0.65",
        threshold=0.65,
        candidate_v6_label=V6Label.HEALTHY_XRAY,
        confidence="high",
        notes="High precision signal. Radiologist spot-check recommended at 10% sample rate.",
    ),
    CLIPAnnotationWorkflow(
        name="pneumonia_xray_candidate",
        clip_signal="medical_refinement.semantic_medical_type IN ('pneumonia_xray', 'lung_opacity') AND medical_plausibility >= 0.60",
        threshold=0.60,
        candidate_v6_label=V6Label.PNEUMONIA_XRAY,
        confidence="medium",
        notes="Overlap with opacity_pattern likely. Full radiologist review required.",
    ),
    CLIPAnnotationWorkflow(
        name="microscopy_candidate",
        clip_signal="semantic.reasoning_type == 'microscopy_candidate' AND semantic.reasoning_confidence >= 0.70",
        threshold=0.70,
        candidate_v6_label=V6Label.NORMAL_MICROSCOPY,
        confidence="medium",
        notes="Subtype (normal vs infected) cannot be determined from CLIP alone — requires pathologist.",
    ),
    CLIPAnnotationWorkflow(
        name="fake_medical_candidate",
        clip_signal="medical_refinement.fake_medical_score >= 0.45",
        threshold=0.45,
        candidate_v6_label=V6Label.FAKE_MEDICAL,
        confidence="high",
        notes="High precision for obviously fake content. Review borderline 0.35–0.45 range manually.",
    ),
)


# ── Query helpers ─────────────────────────────────────────────────────────────


def get_v6_labels_for(v5_label: V5Label) -> tuple[V6Label, ...]:
    return V5_TO_V6_MIGRATION[v5_label].v6_labels


def get_non_target_classes(stage_name: str) -> frozenset[str]:
    """
    Return the correct NON_TARGET_CLASSES frozenset for acceptance.py
    at a given training stage.
    """
    return ACCEPTANCE_POLICY_BY_STAGE.get(stage_name, frozenset({"unrelated", "hard_negative"}))


def get_positive_classes(stage_name: str) -> frozenset[str]:
    return POSITIVE_CLASSES_BY_STAGE.get(stage_name, frozenset({"related"}))


def requires_relabeling(v5_label: V5Label) -> bool:
    return V5_TO_V6_MIGRATION[v5_label].requires_relabeling


def migration_critical_path() -> list[V5Label]:
    """Return v5 labels in order of migration complexity (highest risk first)."""
    return sorted(
        V5_TO_V6_MIGRATION.keys(),
        key=lambda v: list(MigrationRisk).index(V5_TO_V6_MIGRATION[v].migration_risk),
        reverse=True,
    )
