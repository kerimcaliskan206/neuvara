"""
Medical category registry + disease-class abstraction — Phase 7.

Single source of truth for all medical image categories the HantaProject
vision pipeline will support in current and future training rounds.

Two abstraction levels
----------------------
MedicalCategory  — fine-grained class label (image-level ground truth).
DiseaseGroup     — coarser training label that one or more categories map to.
                   A DiseaseGroup is what the model actually predicts; the
                   MedicalCategory is the annotation-time label.

Example mapping:
  DiseaseGroup.INFILTRATE maps to
      MedicalCategory.PNEUMONIA_XRAY + MedicalCategory.OPACITY_PATTERN
  This lets the annotator keep fine-grained labels while the classifier
  trains on the merged group.

Source alignment
----------------
MedicalCategory.semantic_group corresponds to the sub-group keys in
MedicalRefiner._SUBGROUPS (Phase 4), enabling future cross-layer
consistency checks between CLIP semantic analysis and the classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── Fine-grained category enum ────────────────────────────────────────────────


class MedicalCategory(str, Enum):
    """All supported medical image categories for current and future training."""

    HEALTHY_XRAY           = "healthy_xray"
    PNEUMONIA_XRAY         = "pneumonia_xray"
    HANTAVIRUS_CANDIDATE   = "hantavirus_candidate"
    OPACITY_PATTERN        = "opacity_pattern"
    NORMAL_MICROSCOPY      = "normal_microscopy"
    INFECTED_MICROSCOPY    = "infected_microscopy"
    FAKE_MEDICAL           = "fake_medical"
    AI_GENERATED_MEDICAL   = "ai_generated_medical"


class ImageModality(str, Enum):
    """Imaging modality for each category."""

    CHEST_XRAY  = "chest_xray"
    MICROSCOPY  = "microscopy"
    CT_SCAN     = "ct_scan"
    SYNTHETIC   = "synthetic"


# ── Coarse disease-group enum ─────────────────────────────────────────────────


class DiseaseGroup(str, Enum):
    """
    Higher-level disease abstraction for binary / multiclass training.

    A single DiseaseGroup can span multiple MedicalCategory entries.
    Groups marked is_trainable=False need more annotated data before a
    standalone classifier can be trained.
    """

    NORMAL                = "normal"
    BACTERIAL_PNEUMONIA   = "bacterial_pneumonia"
    VIRAL_PNEUMONIA       = "viral_pneumonia"
    HANTAVIRUS            = "hantavirus"
    INFILTRATE            = "infiltrate"
    OPACITY               = "opacity"
    MICROSCOPY_POSITIVE   = "microscopy_positive"
    MICROSCOPY_NEGATIVE   = "microscopy_negative"
    NON_MEDICAL           = "non_medical"


# ── Per-category metadata ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class CategoryMetadata:
    """Training-time metadata for a single MedicalCategory."""

    category: MedicalCategory
    modality: ImageModality
    is_pathological: bool             # True when the category represents disease
    is_real_medical: bool             # False for synthetic / fake classes
    semantic_group: str               # aligns with MedicalRefiner sub-group keys
    description: str
    training_priority: int            # 1 = primary, 2 = secondary, 3 = auxiliary
    min_recommended_samples: int = 500
    hard_negative_compatible: bool = False   # usable as hard-negative training examples


@dataclass(frozen=True)
class DiseaseClassMapping:
    """Maps a DiseaseGroup to its underlying MedicalCategory members."""

    disease_group: DiseaseGroup
    categories: tuple[MedicalCategory, ...]
    binary_positive: bool    # True = positive class in a binary healthy-vs-sick task
    is_trainable: bool       # False when insufficient confirmed cases exist
    notes: str = ""


# ── Registry ──────────────────────────────────────────────────────────────────


CATEGORY_REGISTRY: dict[MedicalCategory, CategoryMetadata] = {
    MedicalCategory.HEALTHY_XRAY: CategoryMetadata(
        category=MedicalCategory.HEALTHY_XRAY,
        modality=ImageModality.CHEST_XRAY,
        is_pathological=False,
        is_real_medical=True,
        semantic_group="healthy_xray",
        description="Normal chest radiograph without pathological findings.",
        training_priority=1,
        min_recommended_samples=1000,
    ),
    MedicalCategory.PNEUMONIA_XRAY: CategoryMetadata(
        category=MedicalCategory.PNEUMONIA_XRAY,
        modality=ImageModality.CHEST_XRAY,
        is_pathological=True,
        is_real_medical=True,
        semantic_group="pneumonia_xray",
        description="Chest X-ray with pneumonia — infiltrate, consolidation, or opacity.",
        training_priority=1,
        min_recommended_samples=1000,
    ),
    MedicalCategory.HANTAVIRUS_CANDIDATE: CategoryMetadata(
        category=MedicalCategory.HANTAVIRUS_CANDIDATE,
        modality=ImageModality.CHEST_XRAY,
        is_pathological=True,
        is_real_medical=True,
        semantic_group="lung_opacity",
        description=(
            "Chest X-ray with bilateral interstitial infiltrates or opacities "
            "consistent with hantavirus pulmonary syndrome."
        ),
        training_priority=1,
        min_recommended_samples=300,
    ),
    MedicalCategory.OPACITY_PATTERN: CategoryMetadata(
        category=MedicalCategory.OPACITY_PATTERN,
        modality=ImageModality.CHEST_XRAY,
        is_pathological=True,
        is_real_medical=True,
        semantic_group="lung_opacity",
        description="Focal or diffuse pulmonary opacity, non-specific etiology.",
        training_priority=2,
        min_recommended_samples=500,
    ),
    MedicalCategory.NORMAL_MICROSCOPY: CategoryMetadata(
        category=MedicalCategory.NORMAL_MICROSCOPY,
        modality=ImageModality.MICROSCOPY,
        is_pathological=False,
        is_real_medical=True,
        semantic_group="medical_microscopy",
        description="Microscopy slide with no pathological findings.",
        training_priority=2,
        min_recommended_samples=500,
    ),
    MedicalCategory.INFECTED_MICROSCOPY: CategoryMetadata(
        category=MedicalCategory.INFECTED_MICROSCOPY,
        modality=ImageModality.MICROSCOPY,
        is_pathological=True,
        is_real_medical=True,
        semantic_group="medical_microscopy",
        description="Microscopy slide with pathological findings (viral inclusions, altered morphology).",
        training_priority=2,
        min_recommended_samples=500,
    ),
    MedicalCategory.FAKE_MEDICAL: CategoryMetadata(
        category=MedicalCategory.FAKE_MEDICAL,
        modality=ImageModality.SYNTHETIC,
        is_pathological=False,
        is_real_medical=False,
        semantic_group="fake_medical_texture",
        description="Synthetic or procedurally generated medical-looking image.",
        training_priority=3,
        hard_negative_compatible=True,
    ),
    MedicalCategory.AI_GENERATED_MEDICAL: CategoryMetadata(
        category=MedicalCategory.AI_GENERATED_MEDICAL,
        modality=ImageModality.SYNTHETIC,
        is_pathological=False,
        is_real_medical=False,
        semantic_group="ai_generated_medical",
        description="AI-synthesised radiograph or microscopy slide.",
        training_priority=3,
        hard_negative_compatible=True,
    ),
}

DISEASE_MAPPINGS: dict[DiseaseGroup, DiseaseClassMapping] = {
    DiseaseGroup.NORMAL: DiseaseClassMapping(
        disease_group=DiseaseGroup.NORMAL,
        categories=(MedicalCategory.HEALTHY_XRAY,),
        binary_positive=False,
        is_trainable=True,
    ),
    DiseaseGroup.BACTERIAL_PNEUMONIA: DiseaseClassMapping(
        disease_group=DiseaseGroup.BACTERIAL_PNEUMONIA,
        categories=(MedicalCategory.PNEUMONIA_XRAY,),
        binary_positive=True,
        is_trainable=True,
        notes="Typically lobar or segmental consolidation on CXR.",
    ),
    DiseaseGroup.VIRAL_PNEUMONIA: DiseaseClassMapping(
        disease_group=DiseaseGroup.VIRAL_PNEUMONIA,
        categories=(MedicalCategory.PNEUMONIA_XRAY, MedicalCategory.HANTAVIRUS_CANDIDATE),
        binary_positive=True,
        is_trainable=False,
        notes="Requires radiologist sub-labeling to distinguish from bacterial pneumonia.",
    ),
    DiseaseGroup.HANTAVIRUS: DiseaseClassMapping(
        disease_group=DiseaseGroup.HANTAVIRUS,
        categories=(MedicalCategory.HANTAVIRUS_CANDIDATE,),
        binary_positive=True,
        is_trainable=False,
        notes="Insufficient confirmed hantavirus CXR cases for standalone training.",
    ),
    DiseaseGroup.INFILTRATE: DiseaseClassMapping(
        disease_group=DiseaseGroup.INFILTRATE,
        categories=(MedicalCategory.PNEUMONIA_XRAY, MedicalCategory.OPACITY_PATTERN),
        binary_positive=True,
        is_trainable=True,
    ),
    DiseaseGroup.OPACITY: DiseaseClassMapping(
        disease_group=DiseaseGroup.OPACITY,
        categories=(MedicalCategory.OPACITY_PATTERN, MedicalCategory.HANTAVIRUS_CANDIDATE),
        binary_positive=True,
        is_trainable=True,
    ),
    DiseaseGroup.MICROSCOPY_POSITIVE: DiseaseClassMapping(
        disease_group=DiseaseGroup.MICROSCOPY_POSITIVE,
        categories=(MedicalCategory.INFECTED_MICROSCOPY,),
        binary_positive=True,
        is_trainable=True,
    ),
    DiseaseGroup.MICROSCOPY_NEGATIVE: DiseaseClassMapping(
        disease_group=DiseaseGroup.MICROSCOPY_NEGATIVE,
        categories=(MedicalCategory.NORMAL_MICROSCOPY,),
        binary_positive=False,
        is_trainable=True,
    ),
    DiseaseGroup.NON_MEDICAL: DiseaseClassMapping(
        disease_group=DiseaseGroup.NON_MEDICAL,
        categories=(MedicalCategory.FAKE_MEDICAL, MedicalCategory.AI_GENERATED_MEDICAL),
        binary_positive=False,
        is_trainable=True,
        notes="Hard-negative class for out-of-distribution robustness training.",
    ),
}


# ── Query helpers ─────────────────────────────────────────────────────────────


def get_real_medical_categories() -> list[MedicalCategory]:
    """Return categories that represent genuine clinical imagery."""
    return [cat for cat, meta in CATEGORY_REGISTRY.items() if meta.is_real_medical]


def get_hard_negative_categories() -> list[MedicalCategory]:
    """Return categories suitable for hard-negative training examples."""
    return [cat for cat, meta in CATEGORY_REGISTRY.items() if meta.hard_negative_compatible]


def get_trainable_disease_groups() -> list[DiseaseGroup]:
    """Return DiseaseGroups that have enough data for standalone training."""
    return [dg for dg, m in DISEASE_MAPPINGS.items() if m.is_trainable]


def categories_for_modality(modality: ImageModality) -> list[MedicalCategory]:
    """Return all categories for a given imaging modality."""
    return [cat for cat, meta in CATEGORY_REGISTRY.items() if meta.modality == modality]


def categories_for_disease_group(group: DiseaseGroup) -> tuple[MedicalCategory, ...]:
    """Return the MedicalCategory members that compose a DiseaseGroup."""
    return DISEASE_MAPPINGS[group].categories
