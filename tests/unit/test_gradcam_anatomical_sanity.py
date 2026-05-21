"""
Phase 27 — GradCAM anatomical sanity tests.

Pure-tensor tests for the four new metric helpers, the Gaussian smoother,
and the trust-gain math. No model load — synthetic CAMs are constructed
directly with torch so the suite runs in milliseconds.
"""
from __future__ import annotations

import pytest
import torch

from app.modules.vision.explainability.anatomical_constraint import (
    compute_border_activation_ratio,
    compute_cam_entropy,
    compute_central_bias_score,
    compute_lung_overlap_score,
)
from app.modules.vision.explainability.gradcam import _smooth_cam


# ───────────────────────────── lung_overlap_score ─────────────────────────────


def test_lung_overlap_score_full_inside_returns_one():
    cam = torch.zeros(64, 64)
    cam[20:40, 20:40] = 1.0
    mask = torch.zeros(64, 64)
    mask[10:50, 10:50] = 1.0
    score = compute_lung_overlap_score(cam, mask)
    assert score == 1.0


def test_lung_overlap_score_fully_outside_returns_zero():
    cam = torch.zeros(64, 64)
    cam[0:5, 0:5] = 1.0   # top-left corner, outside lung
    mask = torch.zeros(64, 64)
    mask[20:40, 20:40] = 1.0
    score = compute_lung_overlap_score(cam, mask)
    assert score == 0.0


def test_lung_overlap_score_missing_mask_defaults_to_one():
    """Graceful fallback when no segmentation is available."""
    cam = torch.rand(32, 32)
    assert compute_lung_overlap_score(cam, None) == 1.0


def test_lung_overlap_score_resizes_mismatched_mask():
    cam = torch.ones(64, 64)
    mask = torch.zeros(32, 32)
    mask[8:24, 8:24] = 1.0
    score = compute_lung_overlap_score(cam, mask)
    # Resized mask covers ~25% of the area at the cam resolution → ~0.25.
    assert 0.20 < score < 0.32


# ─────────────────────────── border_activation_ratio ──────────────────────────


def test_border_activation_ratio_center_only_is_zero():
    cam = torch.zeros(64, 64)
    cam[28:36, 28:36] = 1.0   # central 8×8 patch
    ratio = compute_border_activation_ratio(cam)
    assert ratio == 0.0


def test_border_activation_ratio_pure_border_is_high():
    """A frame-only activation should produce a high border ratio."""
    cam = torch.zeros(64, 64)
    cam[:6, :]  = 1.0   # top
    cam[-6:, :] = 1.0   # bottom
    cam[:, :6]  = 1.0   # left
    cam[:, -6:] = 1.0   # right
    ratio = compute_border_activation_ratio(cam)
    assert ratio >= 0.95


def test_border_activation_ratio_uniform_matches_band_area():
    cam = torch.ones(100, 100)
    ratio = compute_border_activation_ratio(cam, border_frac=0.10)
    # Border band area is approximately 4*W*B - 4*B^2 = 4*100*10 - 4*100 = 3600/10000 = 0.36
    assert 0.34 < ratio < 0.38


# ─────────────────────────────── cam_entropy ──────────────────────────────────


def test_cam_entropy_single_pixel_is_low():
    cam = torch.zeros(32, 32)
    cam[15, 15] = 1.0
    h = compute_cam_entropy(cam)
    assert h == 0.0


def test_cam_entropy_uniform_is_near_one():
    cam = torch.ones(32, 32)
    h = compute_cam_entropy(cam)
    assert h >= 0.99


def test_cam_entropy_localised_is_intermediate():
    cam = torch.zeros(64, 64)
    cam[20:30, 20:30] = 1.0   # 100 pixels uniformly active
    h = compute_cam_entropy(cam)
    # Uniform over 100 of 4096 pixels: H/Hmax = log(100)/log(4096) ≈ 0.553.
    assert 0.50 < h < 0.60


def test_cam_entropy_all_zero_is_zero():
    cam = torch.zeros(16, 16)
    assert compute_cam_entropy(cam) == 0.0


# ────────────────────────────── _smooth_cam ───────────────────────────────────


def test_smooth_cam_spreads_a_point_source():
    x = torch.zeros(64, 64)
    x[32, 32] = 1.0
    y = _smooth_cam(x, sigma=1.5)
    # Energy is conserved (close to it — within a few percent for a 5×5 kernel).
    assert abs(float(y.sum()) - 1.0) < 0.05
    # Peak is no longer at value 1.0.
    assert float(y.max()) < 0.20
    # Some neighbourhood is now non-zero.
    assert int((y > 0.001).sum()) > 9


def test_smooth_cam_preserves_uniform_input():
    x = torch.ones(32, 32)
    y = _smooth_cam(x, sigma=1.5)
    # Interior pixels stay close to 1.0 (edge effects are minor with padding).
    assert abs(float(y[16, 16]) - 1.0) < 0.05


def test_smooth_cam_zero_sigma_is_identity():
    x = torch.rand(32, 32)
    y = _smooth_cam(x, sigma=0.0)
    assert torch.allclose(x, y)


# ────────────────────────────── trust-gain math ───────────────────────────────
# Validates the composed gain matches the desired demo behaviour:
#   healthy   → near-zero raw_max → low gain (calm output)
#   pneumonia → high raw_max, lungs, low border → near-1 gain (full output)
#   fake      → moderate raw_max, low lungs, high border → low gain


def _trust_gain(raw_max: float, lung_overlap: float, border_ratio: float) -> float:
    """Replica of the inline math at the end of generate_pulmonary_focused()."""
    strength_gain     = min(1.0, max(0.0, (raw_max - 0.05) / 0.45))
    localisation_gain = min(1.0, max(0.0, (lung_overlap - 0.30) / 0.50))
    border_penalty    = max(0.0, 1.0 - max(0.0, (border_ratio - 0.10) / 0.25))
    return round(strength_gain * localisation_gain * border_penalty, 4)


def test_trust_gain_healthy_lung_stays_calm():
    # Weak raw CAM (healthy lung typically peaks low), well-localised, no border.
    gain = _trust_gain(raw_max=0.08, lung_overlap=0.85, border_ratio=0.05)
    # Strength gain ≈ 0.066, so total ≈ 0.066.
    assert gain < 0.15


def test_trust_gain_localised_pneumonia_is_full():
    gain = _trust_gain(raw_max=0.80, lung_overlap=0.85, border_ratio=0.05)
    assert gain > 0.95


def test_trust_gain_fake_image_is_low():
    # Fake image: CAM might be strong but localisation is poor and border
    # is heavy.
    gain = _trust_gain(raw_max=0.60, lung_overlap=0.20, border_ratio=0.40)
    # lung_overlap < 0.30 → localisation_gain = 0 → product = 0.
    assert gain == 0.0


def test_trust_gain_border_heavy_attenuates():
    # Strong CAM, lungs OK, but border dominates → trust low.
    gain = _trust_gain(raw_max=0.80, lung_overlap=0.85, border_ratio=0.50)
    # Border penalty ≈ 1 - (0.50-0.10)/0.25 = 1 - 1.6 → clamped 0 → gain = 0.
    assert gain == 0.0


def test_trust_gain_monotonic_in_strength():
    """For fixed lung/border, gain rises monotonically with raw_cam_max."""
    g1 = _trust_gain(raw_max=0.10, lung_overlap=0.85, border_ratio=0.05)
    g2 = _trust_gain(raw_max=0.30, lung_overlap=0.85, border_ratio=0.05)
    g3 = _trust_gain(raw_max=0.60, lung_overlap=0.85, border_ratio=0.05)
    assert g1 < g2 < g3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
