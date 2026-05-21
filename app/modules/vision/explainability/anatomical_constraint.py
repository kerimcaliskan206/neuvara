"""
Anatomical constraint post-processing for GradCAM heatmaps.

Keeps GradCAM attention inside segmented lung regions by applying:
  - Hard lung ROI masking  (zeros non-lung activations)
  - Edge / corner penalty  (attenuates border activations based on violation flags)
  - Gaussian center prior  (softly boosts anatomically central regions)
  - Fallback quality penalty (smoothing + strength reduction when seg quality is poor)

All operations are pure PyTorch — MPS-compatible, no CPU-only ops.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ── Hard lung mask ────────────────────────────────────────────────────────────

def apply_lung_mask(
    cam: torch.Tensor,
    lung_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Zero out activations outside the segmented lung boundary.

    Parameters
    ----------
    cam : (H, W) float tensor on CPU
    lung_mask : (H, W) float tensor in [0, 1] — may differ in spatial size.

    Returns
    -------
    (masked_cam, telemetry_dict)
        masked_cam    : cam with non-lung pixels zeroed
        telemetry_dict: outside_lung_pct, masked_activation_pct
    """
    if cam.shape != lung_mask.shape:
        lung_mask = F.interpolate(
            lung_mask.unsqueeze(0).unsqueeze(0).float(),
            size=cam.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        lung_mask = (lung_mask > 0.5).float()

    total_before = float(cam.sum().item()) + 1e-8
    outside_energy = float((cam * (1.0 - lung_mask)).sum().item())

    masked_cam = cam * lung_mask
    total_after = float(masked_cam.sum().item()) + 1e-8

    return masked_cam, {
        "outside_lung_pct":       round(100.0 * outside_energy / total_before, 2),
        "masked_activation_pct":  round(100.0 * total_after / total_before, 2),
    }


# ── Gaussian center prior ─────────────────────────────────────────────────────

def build_gaussian_center_prior(
    H: int,
    W: int,
    *,
    center_y: float = 0.50,
    center_x: float = 0.50,
    sigma_y: float = 0.30,
    sigma_x: float = 0.34,
    floor: float = 0.12,
) -> torch.Tensor:
    """
    Gaussian prior centered on the pulmonary field.

    Returns (H, W) float tensor in [floor, 1.0] on CPU.
    Multiplying a heatmap by this prior boosts central anatomical regions
    without zeroing any activation (floor > 0 preserves peripheral pneumonia).
    """
    ys = torch.linspace(0.0, 1.0, H).unsqueeze(1).expand(H, W)
    xs = torch.linspace(0.0, 1.0, W).unsqueeze(0).expand(H, W)

    gauss = torch.exp(
        -(((ys - center_y) ** 2) / (2.0 * sigma_y ** 2)
          + ((xs - center_x) ** 2) / (2.0 * sigma_x ** 2))
    )
    return torch.clamp(gauss, floor, 1.0)


# ── Edge / corner penalty ─────────────────────────────────────────────────────

def apply_edge_corner_penalty(
    cam: torch.Tensor,
    *,
    max_x_frac: float,
    max_y_frac: float,
    edge_pct: float,
    edge_dominant: bool,
    edge_band_frac: float = 0.10,
    corner_radius_frac: float = 0.16,
) -> tuple[torch.Tensor, dict]:
    """
    Apply a spatially-graded penalty to edge and corner regions.

    Violation conditions (per Step-2 spec):
      edge_pct > 15  OR  max_x_frac > 0.85  OR  max_x_frac < 0.15
                     OR  max_y_frac > 0.90  OR  max_y_frac < 0.10
        → attention_penalty *= 0.60
      edge_dominant
        → attention_penalty *= 0.75

    The scalar penalty is turned into a *spatial map*: the border band receives
    ``attention_penalty`` weight while the image center receives 1.0, with a
    smooth ramp between them.
    """
    H, W = cam.shape

    attention_penalty = 1.0
    edge_penalty_applied = False
    border_activation_detected = False
    anatomical_violation_score = 0.0

    violations = 0
    if edge_pct > 15.0:
        violations += 1
    if max_x_frac > 0.85 or max_x_frac < 0.15:
        violations += 1
        border_activation_detected = True
    if max_y_frac > 0.90 or max_y_frac < 0.10:
        violations += 1
        border_activation_detected = True

    if violations > 0:
        attention_penalty *= 0.60
        edge_penalty_applied = True
        anatomical_violation_score += min(1.0, violations * 0.20)

    if edge_dominant:
        attention_penalty *= 0.75
        edge_penalty_applied = True
        anatomical_violation_score = min(1.0, anatomical_violation_score + 0.30)

    telemetry = {
        "attention_penalty":           round(attention_penalty, 4),
        "anatomical_violation_score":  round(anatomical_violation_score, 3),
        "edge_penalty_applied":        bool(edge_penalty_applied),
        "border_activation_detected":  bool(border_activation_detected),
    }

    if not edge_penalty_applied:
        return cam, telemetry

    # Build spatial penalty map: 0 at image boundary → attention_penalty;
    # rising to 1.0 at edge_band_frac inward from each side.
    ys = torch.linspace(0.0, 1.0, H).unsqueeze(1).expand(H, W)
    xs = torch.linspace(0.0, 1.0, W).unsqueeze(0).expand(H, W)

    left_ramp   = torch.clamp(xs / max(1e-4, edge_band_frac), 0.0, 1.0)
    right_ramp  = torch.clamp((1.0 - xs) / max(1e-4, edge_band_frac), 0.0, 1.0)
    top_ramp    = torch.clamp(ys / max(1e-4, edge_band_frac), 0.0, 1.0)
    bottom_ramp = torch.clamp((1.0 - ys) / max(1e-4, edge_band_frac), 0.0, 1.0)

    # Minimum of all four → 0 anywhere near any border, 1 well inside
    edge_dist = torch.minimum(
        torch.minimum(left_ramp, right_ramp),
        torch.minimum(top_ramp, bottom_ramp),
    )

    # Corner suppression: extra penalty near the four image corners
    dist_tl = torch.sqrt(ys ** 2 + xs ** 2)
    dist_tr = torch.sqrt(ys ** 2 + (1.0 - xs) ** 2)
    dist_bl = torch.sqrt((1.0 - ys) ** 2 + xs ** 2)
    dist_br = torch.sqrt((1.0 - ys) ** 2 + (1.0 - xs) ** 2)
    min_corner_dist = torch.stack([dist_tl, dist_tr, dist_bl, dist_br]).min(dim=0).values
    corner_ramp = torch.clamp(min_corner_dist / max(1e-4, corner_radius_frac), 0.0, 1.0)

    spatial_dist = torch.minimum(edge_dist, corner_ramp)

    # penalty_map: attention_penalty at border, 1.0 at center
    penalty_map = attention_penalty + (1.0 - attention_penalty) * spatial_dist

    return cam * penalty_map, telemetry


# ── Fallback quality penalty ──────────────────────────────────────────────────

def apply_fallback_penalty(
    cam: torch.Tensor,
    *,
    quality: str,
    strength_factor: float = 0.72,
    kernel_size: int = 5,
) -> tuple[torch.Tensor, bool]:
    """
    When segmentation quality is 'fallback', reduce GradCAM trust.

    - Blends toward the mean to reduce sharp spurious peaks.
    - Applies average-pool smoothing as a proxy for Gaussian blur (MPS-safe).

    Returns (penalized_cam, fallback_penalty_applied).
    """
    if quality != "fallback":
        return cam, False

    # Blend toward mean (reduces peak sharpness without zeroing)
    mean_val = cam.mean()
    cam = strength_factor * cam + (1.0 - strength_factor) * mean_val

    # Smooth via average pooling (kernel_size must be odd)
    padding = kernel_size // 2
    cam_4d = cam.unsqueeze(0).unsqueeze(0)
    cam_4d = F.avg_pool2d(cam_4d, kernel_size=kernel_size, stride=1, padding=padding)
    cam = cam_4d.squeeze(0).squeeze(0)

    return cam, True


# ── Telemetry helpers ─────────────────────────────────────────────────────────

def compute_bilateral_balance_score(
    cam: torch.Tensor,
    lung_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Symmetry of activation between left and right halves.
    1.0 = perfect balance, 0.0 = all activation on one side.
    """
    H, W = cam.shape
    mid = W // 2
    eff = cam if lung_mask is None else cam * lung_mask
    left  = float(eff[:, :mid].sum().item()) + 1e-8
    right = float(eff[:, mid:].sum().item()) + 1e-8
    return round(1.0 - abs(left - right) / (left + right), 4)


def compute_central_bias_score(
    cam: torch.Tensor,
    lung_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Fraction of activation in the central 50%×60% window.
    1.0 = all attention in anatomical centre.
    """
    H, W = cam.shape
    cy0, cy1 = int(H * 0.25), int(H * 0.75)
    cx0, cx1 = int(W * 0.20), int(W * 0.80)
    eff = cam if lung_mask is None else cam * lung_mask
    total = float(eff.sum().item()) + 1e-8
    return round(float(eff[cy0:cy1, cx0:cx1].sum().item()) / total, 4)


# ── Phase 27 — Anatomical sanity metrics ──────────────────────────────────────
# Lightweight, pure-function metrics surfaced in GradCAM telemetry. Used by
# the adaptive trust gain at the end of generate_pulmonary_focused() and as
# debug fields for the demo UI.


def compute_lung_overlap_score(
    cam: torch.Tensor,
    lung_mask: Optional[torch.Tensor],
) -> float:
    """
    Clean ratio of CAM energy that falls inside the segmented lung mask.

    1.0 = all attention inside lungs (medically believable);
    0.0 = all attention outside lungs (unreliable, likely shortcut).
    Returns 1.0 if no lung mask is available (graceful fallback).
    """
    if lung_mask is None:
        return 1.0
    if lung_mask.shape != cam.shape:
        # Resize mask via bilinear → threshold at 0.5 (consistent with apply_lung_mask).
        m = F.interpolate(
            lung_mask.float().unsqueeze(0).unsqueeze(0),
            size=cam.shape, mode="bilinear", align_corners=False,
        ).squeeze()
        lung_mask = (m > 0.5).float()
    total = float(cam.sum().item()) + 1e-8
    in_lung = float((cam * lung_mask).sum().item())
    return round(in_lung / total, 4)


def compute_border_activation_ratio(
    cam: torch.Tensor,
    border_frac: float = 0.10,
) -> float:
    """
    Clean ratio of CAM energy that falls inside a uniform border band.

    Default border is 10% of the smaller dimension. Includes the four edges
    and (implicitly) the corners. High values indicate the model is leaning
    on dataset-specific border / annotation artefacts rather than anatomy.
    """
    H, W = cam.shape
    bh = max(1, int(H * border_frac))
    bw = max(1, int(W * border_frac))
    border_mask = torch.zeros_like(cam)
    border_mask[:bh, :]      = 1.0
    border_mask[H - bh:, :]  = 1.0
    border_mask[:, :bw]      = 1.0
    border_mask[:, W - bw:]  = 1.0
    total = float(cam.sum().item()) + 1e-8
    border = float((cam * border_mask).sum().item())
    return round(border / total, 4)


def compute_cam_entropy(cam: torch.Tensor) -> float:
    """
    Shannon entropy of the CAM treated as a probability distribution.

    Normalized to [0, 1] by dividing by log(N) where N = H*W.
      0.0 = perfectly focused (a single pixel); good — sharply localized.
      1.0 = perfectly uniform; bad — diffuse, no real signal.

    Used as one diffuseness signal for the adaptive trust gain.
    """
    flat = cam.flatten().clamp(min=0.0)
    s = float(flat.sum().item())
    if s < 1e-8:
        return 0.0
    p = flat / s
    # ε guard against log(0); only nonzero terms contribute.
    nonzero = p[p > 1e-12]
    if nonzero.numel() == 0:
        return 0.0
    H_nats = float(-(nonzero * nonzero.log()).sum().item())
    H_max  = float(torch.log(torch.tensor(float(flat.numel()))).item())
    if H_max <= 0:
        return 0.0
    return round(H_nats / H_max, 4)
