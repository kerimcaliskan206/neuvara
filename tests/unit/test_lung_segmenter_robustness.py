"""
Phase 28 — Lung segmentation robustness tests.

Synthetic chest-X-ray-ish images exercise each branch:
  - bright-lung-on-dark (Otsu direct wins)
  - dark-lung-on-bright (Otsu inverted wins)
  - low contrast (adaptive threshold rescues)
  - black border (border crop + segmentation)
  - pure noise (all candidates fail → fallback fires with reason)

Runs without conftest because the repo's conftest pulls in app.main (xgboost).
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.modules.vision.segmentation.lung_segmenter import (
    LungSegmenter,
    SegmentationResult,
)
from app.modules.vision.segmentation.mask_utils import (
    compute_bilateral_balance,
    compute_black_border_crop,
    count_contours,
    create_center_mask,
)


# ───────────────────────────── synthetic-image factories ──────────────────────


def _two_lungs_bright_on_dark(size: int = 256) -> Image.Image:
    img = np.full((size, size), 25, dtype=np.uint8)
    # Two bright "lung" regions with a wide mediastinal gap so morph closing
    # doesn't merge them.
    img[55:200,  35:110] = 190
    img[55:200, 146:221] = 190
    # Brighter spine/mediastinum strip in the middle
    img[40:210, 118:138] = 60
    return Image.fromarray(img, mode="L").convert("RGB")


def _two_lungs_dark_on_bright(size: int = 256) -> Image.Image:
    """Reversed polarity — lungs appear dark on bright background."""
    img = np.full((size, size), 200, dtype=np.uint8)
    img[55:200,  35:110] = 35
    img[55:200, 146:221] = 35
    img[40:210, 118:138] = 160
    return Image.fromarray(img, mode="L").convert("RGB")


def _low_contrast(size: int = 256) -> Image.Image:
    """Lungs only ~15 gray levels darker than the surrounding tissue."""
    img = np.full((size, size), 130, dtype=np.uint8)
    img[55:200,  35:110] = 110
    img[55:200, 146:221] = 110
    return Image.fromarray(img, mode="L").convert("RGB")


def _black_border(size: int = 256, border: int = 35) -> Image.Image:
    img = _two_lungs_bright_on_dark(size - 2 * border)
    arr = np.array(img.convert("L"))
    framed = np.zeros((size, size), dtype=np.uint8)
    framed[border:size - border, border:size - border] = arr
    return Image.fromarray(framed, mode="L").convert("RGB")


def _pure_noise(size: int = 256, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (size, size), dtype=np.uint8)
    return Image.fromarray(img, mode="L").convert("RGB")


# ───────────────────────────── mask_utils helpers ─────────────────────────────


def test_compute_black_border_crop_detects_uniform_frame():
    arr = np.full((100, 100), 200, dtype=np.uint8)
    arr[:15, :] = 0
    arr[-15:, :] = 0
    arr[:, :15] = 0
    arr[:, -15:] = 0
    top, bottom, left, right = compute_black_border_crop(arr)
    assert top == 15 and bottom == 15 and left == 15 and right == 15


def test_compute_black_border_crop_capped_for_all_dark_image():
    """An image that is genuinely dark all over should not be eaten entirely."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    top, bottom, left, right = compute_black_border_crop(arr, max_crop_frac=0.35)
    assert top <= 35 and bottom <= 35 and left <= 35 and right <= 35


def test_compute_bilateral_balance_perfect_and_lopsided():
    m_balanced = np.zeros((64, 64), dtype=np.uint8)
    m_balanced[20:40, 5:25]   = 255
    m_balanced[20:40, 39:59]  = 255
    assert compute_bilateral_balance(m_balanced) == 1.0

    m_lopsided = np.zeros((64, 64), dtype=np.uint8)
    m_lopsided[20:40, 5:25] = 255
    assert compute_bilateral_balance(m_lopsided) == 0.0

    m_empty = np.zeros((64, 64), dtype=np.uint8)
    assert compute_bilateral_balance(m_empty) == 0.0


def test_count_contours_finds_separate_blobs():
    m = np.zeros((64, 64), dtype=np.uint8)
    m[10:20, 10:20] = 255
    m[40:50, 40:50] = 255
    assert count_contours(m) == 2


# ───────────────────────────── full-pipeline checks ───────────────────────────


def test_segment_bright_lungs_recovers_without_fallback():
    seg = LungSegmenter().segment(_two_lungs_bright_on_dark())
    assert seg.quality != "fallback", (
        f"clean bright-lung synthetic should not fall back, got {seg.fallback_reason}"
    )
    assert seg.mask_confidence > 0.35
    assert seg.fallback_reason is None
    assert seg.lung_area_ratio == seg.lung_area_pct


def test_segment_dark_lungs_recovers_via_inverted_otsu():
    seg = LungSegmenter().segment(_two_lungs_dark_on_bright())
    assert seg.quality != "fallback"
    assert seg.mask_confidence > 0.35


def test_segment_low_contrast_recovers_via_adaptive():
    """Low-contrast input that would defeat plain Otsu should still pass
    via the adaptive-threshold candidate."""
    seg = LungSegmenter().segment(_low_contrast())
    assert seg.quality != "fallback", (
        f"low-contrast should be rescued by adaptive threshold, got "
        f"fallback_reason={seg.fallback_reason}"
    )


def test_segment_black_border_image_crops_then_segments():
    """A black-framed image should crop and still segment cleanly."""
    seg = LungSegmenter().segment(_black_border())
    assert seg.quality != "fallback"
    # Mask coordinates must be in the FULL image frame, not the cropped frame.
    assert seg.mask.shape == (256, 256)


def test_segment_pure_noise_falls_back_with_reason():
    seg = LungSegmenter().segment(_pure_noise())
    assert seg.quality == "fallback"
    assert seg.fallback_reason is not None
    assert seg.mask_confidence < 0.35


def test_segment_telemetry_fields_present():
    seg = LungSegmenter().segment(_two_lungs_bright_on_dark())
    # All requested debug metrics should be set on the result.
    assert isinstance(seg.lung_area_ratio, float)
    assert isinstance(seg.contour_count, int)
    assert isinstance(seg.bilateral_balance, float)
    assert isinstance(seg.mask_confidence, float)
    assert seg.fallback_reason is None  # good-quality result


def test_fallback_mask_still_returns_full_size_mask():
    """Even when fallback fires, the returned mask must match input dims."""
    seg = LungSegmenter().segment(_pure_noise())
    assert seg.mask.shape == (256, 256)
    # Center mask covers the central 80% in each axis → ~64% by area.
    frac = (seg.mask > 127).sum() / seg.mask.size
    assert 0.55 < frac < 0.70


def test_min_accept_confidence_threshold_protects_against_garbage():
    """A mask with extreme bilateral imbalance must be rejected even if
    its area happens to land in the plausible range."""
    seg = LungSegmenter()
    # Construct a lopsided "mask" mid-pipeline and score it.
    lop = np.zeros((256, 256), dtype=np.uint8)
    lop[60:200, 30:120] = 255   # left half only, ~12% area (in band)
    conf = seg._score_candidate(lop, frac=0.12)
    assert conf < seg._MIN_ACCEPT_CONFIDENCE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
