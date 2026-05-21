"""
Dataset metadata schema.

Every image that enters the dataset is represented by an ImageRecord.
Records are persisted in the dataset manifest (manifest.json) and form
the single source of truth for provenance, quality, and split assignment.

Design goals
------------
- Immutable identity: content_hash never changes after ingestion.
- Reproducible splits: split field is set once and persists across runs.
- Multimodal readiness: multimodal_case_id links an image to a structured
  case record for future symptom+image fusion.
- Audit trail: source_type, source_url, collector, and ingestion_date
  provide full provenance.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ImageClass(str, Enum):
    """The three semantic categories of this dataset."""
    RELATED = "related"
    UNRELATED = "unrelated"
    HARD_NEGATIVE = "hard_negative"


class SourceType(str, Enum):
    """How the image was acquired."""
    COLLECTED = "collected"        # manually collected from a source URL
    LICENSED = "licensed"         # purchased or licensed dataset
    SYNTHETIC = "synthetic"       # GAN / diffusion-generated
    AUGMENTED = "augmented"       # derived from another image via augmentation
    INSTITUTIONAL = "institutional"  # provided by a hospital / lab / institution


class Split(str, Enum):
    """Dataset partition."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    UNASSIGNED = "unassigned"     # ingested but not yet split


class QualityFlag(str, Enum):
    """Specific quality issues detected during validation."""
    BLURRY = "blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    LOW_CONTRAST = "low_contrast"
    GRAYSCALE = "grayscale"
    SMALL_RESOLUTION = "small_resolution"
    EXTREME_ASPECT_RATIO = "extreme_aspect_ratio"
    POSSIBLE_DUPLICATE = "possible_duplicate"
    FORMAT_MISMATCH = "format_mismatch"


class ImageRecord(BaseModel):
    """
    Full metadata record for a single dataset image.

    Fields that cannot be computed from the file (source_type, source_url,
    collector, tags, multimodal_case_id, notes) are optional at ingestion
    but should be filled in before the image is promoted to a split.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    image_id: str = Field(
        description="SHA-256 hex digest of the raw file bytes. Stable unique identifier."
    )
    filename: str = Field(description="Original filename at ingestion time.")
    class_name: ImageClass

    # ── Content fingerprints (deduplication) ──────────────────────────────────

    content_hash: str = Field(
        description="SHA-256 of raw file bytes. Identical files share this hash."
    )
    perceptual_hash: str = Field(
        description=(
            "64-bit difference hash (dhash) encoded as 16 hex chars. "
            "Near-duplicate images have a Hamming distance < 8."
        )
    )

    # ── Provenance ────────────────────────────────────────────────────────────

    source_type: SourceType = SourceType.COLLECTED
    source_url: Optional[str] = None
    source_institution: Optional[str] = None
    collector: Optional[str] = None
    acquisition_date: Optional[date] = None
    ingestion_date: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    # ── Image properties ──────────────────────────────────────────────────────

    width: int
    height: int
    channels: int = 3
    format: str = Field(description="e.g. 'JPEG', 'PNG'")
    file_size_bytes: int

    # ── Quality ───────────────────────────────────────────────────────────────

    quality_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Composite quality score [0, 1]. Computed from blur, brightness, "
            "contrast, and resolution sub-scores. Threshold for acceptance: >= 0.5."
        )
    )
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    blur_score: Optional[float] = Field(
        None, description="Laplacian variance. Higher = sharper. Threshold: >= 100."
    )
    brightness_mean: Optional[float] = Field(
        None, description="Mean pixel brightness in [0, 255]."
    )
    contrast_std: Optional[float] = Field(
        None, description="Pixel value std-dev. Low = low contrast."
    )
    validated: bool = False

    # ── Split assignment ──────────────────────────────────────────────────────

    split: Split = Split.UNASSIGNED

    # ── Domain annotations ────────────────────────────────────────────────────

    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Free-form domain tags. Examples: 'rodent', 'droppings', "
            "'burrow', 'ct_scan', 'hps_lesion', 'rural_environment'."
        )
    )
    notes: Optional[str] = None

    # ── Multimodal fusion link ────────────────────────────────────────────────

    multimodal_case_id: Optional[str] = Field(
        None,
        description=(
            "Links this image to a structured patient/case record for future "
            "symptom + image fusion. Null until a matching case record exists."
        )
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("image_id", "content_hash")
    @classmethod
    def _is_hex64(cls, v: str) -> str:
        if len(v) != 64 or not all(c in "0123456789abcdef" for c in v):
            raise ValueError("Expected 64-character lowercase hex string (SHA-256).")
        return v

    @field_validator("perceptual_hash")
    @classmethod
    def _is_hex16(cls, v: str) -> str:
        if len(v) != 16 or not all(c in "0123456789abcdef" for c in v):
            raise ValueError("Expected 16-character lowercase hex string (dhash).")
        return v

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def resolution_mp(self) -> float:
        return (self.width * self.height) / 1_000_000

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    @property
    def is_acceptable_quality(self) -> bool:
        return self.quality_score >= 0.5 and QualityFlag.POSSIBLE_DUPLICATE not in self.quality_flags

    def summary(self) -> str:
        flags = ", ".join(f.value for f in self.quality_flags) or "none"
        return (
            f"{self.filename} | {self.class_name.value} | "
            f"{self.width}x{self.height} | q={self.quality_score:.2f} | "
            f"split={self.split.value} | flags=[{flags}]"
        )


class DatasetManifestMeta(BaseModel):
    """Header metadata stored at the top of manifest.json."""

    version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    description: str = ""
    parent_version: Optional[str] = None
    git_commit: Optional[str] = None
    notes: str = ""

    class_definitions: dict[str, str] = Field(
        default_factory=lambda: {
            "related": (
                "Hantavirus-relevant imagery: rodents (deer mouse, cotton rat, "
                "rice rat), burrows, droppings, nesting sites, affected environments, "
                "hantavirus pulmonary syndrome (HPS) radiological findings."
            ),
            "unrelated": (
                "Completely off-domain imagery: landscapes, unrelated animals, "
                "household objects, human portraits, vehicles, food, etc."
            ),
            "hard_negative": (
                "Medically similar but hantavirus-unrelated imagery: other "
                "respiratory diseases (COVID, influenza pneumonia), other rodent "
                "species not known to carry hantavirus, similar-looking lesions "
                "from other causes, other zoonotic diseases (leptospirosis, "
                "plague). Designed to prevent the model from classifying any "
                "'medical-looking' image as hantavirus-related."
            ),
        }
    )