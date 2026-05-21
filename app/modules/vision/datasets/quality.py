"""
Image quality validation for dataset ingestion.

Quality is assessed along five axes:
  1. Resolution   — minimum and maximum pixel dimensions
  2. Sharpness    — Laplacian variance (blur detection)
  3. Brightness   — mean pixel value (too dark / too bright)
  4. Contrast     — pixel value standard deviation
  5. Channel mode — grayscale vs. RGB

Each axis produces a sub-score in [0, 1]. The composite quality_score is
the geometric mean of the sub-scores, weighted by their importance.
Images with quality_score < 0.5 are rejected from the dataset.

Thresholds are calibrated for clinical / field photography. Microscopy
images may require adjusted thresholds (lower min_blur_score because
some specimens legitimately have smooth regions).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from app.modules.vision.datasets.schema import QualityFlag

logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """Configurable quality thresholds."""

    min_width: int = 100
    min_height: int = 100
    max_width: int = 8192
    max_height: int = 8192

    min_blur_score: float = 80.0      # Laplacian variance
    min_brightness: float = 20.0      # mean pixel in [0, 255]
    max_brightness: float = 235.0
    min_contrast_std: float = 15.0    # pixel std-dev

    max_aspect_ratio: float = 8.0     # width/height or height/width

    # Weights for composite quality score (must sum to 1.0)
    weight_resolution: float = 0.2
    weight_blur: float = 0.4
    weight_brightness: float = 0.2
    weight_contrast: float = 0.2


@dataclass
class QualityReport:
    """Result of quality validation for a single image."""

    passed: bool
    quality_score: float
    blur_score: float
    brightness_mean: float
    contrast_std: float
    width: int
    height: int
    channels: int
    flags: list[QualityFlag] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        flag_str = ", ".join(f.value for f in self.flags) or "none"
        return (
            f"[{status}] q={self.quality_score:.3f} | "
            f"blur={self.blur_score:.1f} | "
            f"brightness={self.brightness_mean:.1f} | "
            f"contrast={self.contrast_std:.1f} | "
            f"size={self.width}x{self.height} | "
            f"flags=[{flag_str}]"
        )


class ImageQualityValidator:
    """
    Validates image quality along five axes and produces a composite score.

    Parameters
    ----------
    thresholds : QualityThresholds
        Configurable acceptance thresholds.
    min_quality_score : float
        Images with score below this value are marked as failed.
    """

    def __init__(
        self,
        thresholds: QualityThresholds | None = None,
        min_quality_score: float = 0.5,
    ) -> None:
        self.thresholds = thresholds or QualityThresholds()
        self.min_quality_score = min_quality_score

    def validate(self, image: Image.Image) -> QualityReport:
        """
        Run all quality checks on a PIL Image.

        Returns a QualityReport with passed=True/False, composite score,
        per-axis scores, and a list of QualityFlag issues.
        """
        t = self.thresholds
        flags: list[QualityFlag] = []
        notes: list[str] = []

        # ── Convert to arrays ─────────────────────────────────────────────────

        rgb_img = image.convert("RGB")
        rgb = np.array(rgb_img, dtype=np.float32)
        gray = np.mean(rgb, axis=2)           # (H, W) float

        h, w = gray.shape
        channels = 3

        # ── 1. Resolution ─────────────────────────────────────────────────────

        resolution_score = 1.0
        if w < t.min_width or h < t.min_height:
            flags.append(QualityFlag.SMALL_RESOLUTION)
            notes.append(f"Image too small: {w}×{h} < {t.min_width}×{t.min_height}")
            resolution_score = 0.0
        elif w > t.max_width or h > t.max_height:
            notes.append(f"Image very large: {w}×{h} (will be downsampled at training)")
            resolution_score = 0.8

        ar = max(w, h) / max(min(w, h), 1)
        if ar > t.max_aspect_ratio:
            flags.append(QualityFlag.EXTREME_ASPECT_RATIO)
            notes.append(f"Extreme aspect ratio: {ar:.1f}:1")
            resolution_score = min(resolution_score, 0.3)

        # ── 2. Sharpness (Laplacian variance) ────────────────────────────────

        gray_uint8 = gray.clip(0, 255).astype(np.uint8)
        lap = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        blur_score = float(lap.var())

        if blur_score < t.min_blur_score:
            flags.append(QualityFlag.BLURRY)
            notes.append(f"Image appears blurry: Laplacian variance={blur_score:.1f} < {t.min_blur_score}")

        blur_normalized = min(blur_score / (t.min_blur_score * 5), 1.0)

        # ── 3. Brightness ─────────────────────────────────────────────────────

        brightness_mean = float(gray.mean())

        if brightness_mean < t.min_brightness:
            flags.append(QualityFlag.TOO_DARK)
            notes.append(f"Image too dark: mean={brightness_mean:.1f} < {t.min_brightness}")
        elif brightness_mean > t.max_brightness:
            flags.append(QualityFlag.TOO_BRIGHT)
            notes.append(f"Image too bright (overexposed): mean={brightness_mean:.1f} > {t.max_brightness}")

        # Parabolic score: 1.0 at 127 (mid), 0 at extremes
        brightness_normalized = 1.0 - (abs(brightness_mean - 127.5) / 127.5) ** 2
        brightness_normalized = max(brightness_normalized, 0.0)

        if QualityFlag.TOO_DARK in flags or QualityFlag.TOO_BRIGHT in flags:
            brightness_normalized = min(brightness_normalized, 0.3)

        # ── 4. Contrast (std-dev) ─────────────────────────────────────────────

        contrast_std = float(gray.std())

        if contrast_std < t.min_contrast_std:
            flags.append(QualityFlag.LOW_CONTRAST)
            notes.append(f"Low contrast: std={contrast_std:.1f} < {t.min_contrast_std}")

        contrast_normalized = min(contrast_std / (t.min_contrast_std * 4), 1.0)

        # ── 5. Channel mode check ─────────────────────────────────────────────

        original_arr = np.array(image)
        if original_arr.ndim == 2:
            flags.append(QualityFlag.GRAYSCALE)
            notes.append("Image is grayscale — will be converted to 3-channel RGB.")
            channels = 1

        # ── Composite score ───────────────────────────────────────────────────

        quality_score = (
            t.weight_resolution * resolution_score
            + t.weight_blur * blur_normalized
            + t.weight_brightness * brightness_normalized
            + t.weight_contrast * contrast_normalized
        )
        quality_score = float(min(max(quality_score, 0.0), 1.0))

        passed = quality_score >= self.min_quality_score

        report = QualityReport(
            passed=passed,
            quality_score=round(quality_score, 4),
            blur_score=round(blur_score, 2),
            brightness_mean=round(brightness_mean, 2),
            contrast_std=round(contrast_std, 2),
            width=w,
            height=h,
            channels=channels,
            flags=flags,
            notes=notes,
        )

        logger.debug("Quality: %s", report.summary())
        return report

    def validate_bytes(self, data: bytes) -> QualityReport:
        """Validate quality directly from raw file bytes."""
        import io
        image = Image.open(io.BytesIO(data))
        return self.validate(image)


def quick_quality_score(image: Image.Image) -> float:
    """Convenience wrapper — returns only the composite quality score."""
    return ImageQualityValidator().validate(image).quality_score