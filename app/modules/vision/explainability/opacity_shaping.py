"""
Opacity-sensitive pulmonary attention shaping — Phase 2.

Applied after Stage-3 anatomical constraints. Takes the anatomically-masked
heatmap and modulates it to favour diffuse pulmonary opacity/infiltration
patterns while suppressing responses associated with skeletal structures,
sharp boundary artifacts, and isolated noise speckles.

Pipeline (in order)
-------------------
  1. Speckle removal          — isolated hot pixels → neighbourhood average
  2. Pathology coherence      — blob-aware coherence weighting (penalises speckles)
  3. Texture amplification    — boosts cam in soft-opacity, low-edge lung regions
  4. Bilateral opacity boost  — raises diffuse floor when both lungs show signal
  5. Skeletal suppression     — clavicle, lateral border, sharp rib edges
  6. Lung-mask re-application — safety: no leakage
  7. Coherence smoothing      — gentle 7×7 smooth, 60/40 blend

All operations are pure PyTorch — MPS-compatible.  No scipy, no numpy masks,
no CPU-only ops.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ── Grayscale extraction ──────────────────────────────────────────────────────

def _extract_grayscale(
    image_tensor: torch.Tensor,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    """
    Convert (1, C, H, W) model input tensor → normalised (H, W) grayscale on CPU.

    The model-normalised tensor preserves relative texture patterns (the
    preprocessing is monotone within each channel), so Sobel and variance
    maps computed on this grayscale are spatially correct.
    """
    t = image_tensor.detach().cpu().float()
    if t.dim() == 4:
        t = t.squeeze(0)          # (C, H, W)
    gray = (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]) if t.shape[0] == 3 else t.mean(0)

    g_min, g_max = gray.min(), gray.max()
    if (g_max - g_min) > 1e-6:
        gray = (gray - g_min) / (g_max - g_min)
    else:
        gray = torch.zeros_like(gray)

    if gray.shape != torch.Size(target_hw):
        gray = F.interpolate(
            gray.unsqueeze(0).unsqueeze(0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    return gray.clamp(0.0, 1.0)


# ── Edge magnitude ────────────────────────────────────────────────────────────

def _edge_magnitude(gray: torch.Tensor) -> torch.Tensor:
    """
    Sobel magnitude — (H, W) float in [0, 1].

    Normalised at the 99th-percentile to prevent a single bright edge
    from dominating the entire suppression map.
    """
    g4d = gray.unsqueeze(0).unsqueeze(0)
    kx = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32
    ).div(8.0).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()

    gx = F.conv2d(g4d, kx, padding=1)
    gy = F.conv2d(g4d, ky, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).squeeze(0).squeeze(0)

    flat = mag.flatten()
    k99  = max(1, int(len(flat) * 0.99))
    cap  = float(flat.sort()[0][k99 - 1].item())
    return (mag / max(cap, 1e-6)).clamp(0.0, 1.0)


# ── Local variance ────────────────────────────────────────────────────────────

def _local_variance(gray: torch.Tensor, kernel: int = 11) -> torch.Tensor:
    """E[X²] − E[X]² via average pooling → variance map in [0, 1]."""
    g4d = gray.unsqueeze(0).unsqueeze(0)
    p   = kernel // 2
    ex  = F.avg_pool2d(g4d,       kernel_size=kernel, stride=1, padding=p)
    ex2 = F.avg_pool2d(g4d ** 2,  kernel_size=kernel, stride=1, padding=p)
    var = (ex2 - ex ** 2).clamp(0.0).squeeze(0).squeeze(0)

    flat = var.flatten()
    k99  = max(1, int(len(flat) * 0.99))
    cap  = float(flat.sort()[0][k99 - 1].item())
    return (var / max(cap, 1e-6)).clamp(0.0, 1.0)


# ── Opacity map ───────────────────────────────────────────────────────────────

def _opacity_map(gray: torch.Tensor, kernel: int = 21) -> torch.Tensor:
    """
    Low-frequency intensity via heavy average pooling.

    On X-rays (post-normalisation), consolidated/opacified regions tend to
    be smoother and brighter than air-filled parenchyma.  This map captures
    those broad intensity regions.
    """
    p = kernel // 2
    return F.avg_pool2d(
        gray.unsqueeze(0).unsqueeze(0), kernel_size=kernel, stride=1, padding=p
    ).clamp(0.0, 1.0).squeeze(0).squeeze(0)


# ── Texture map ───────────────────────────────────────────────────────────────

def _texture_map(
    gray: torch.Tensor,
    lung_mask: torch.Tensor,
    *,
    variance_kernel: int = 11,
    edge_power: float = 2.5,
) -> torch.Tensor:
    """
    Pulmonary soft-texture score inside the lung mask.

    texture = local_variance × (1 − edge_magnitude)^edge_power

    High where: moderate variance AND low Sobel response → diffuse soft-tissue.
    Low where:  high Sobel (bone/rib) or very low variance (uniform air).

    Normalised within the lung mask to [0, 1].
    """
    var  = _local_variance(gray, kernel=variance_kernel)
    edge = _edge_magnitude(gray)

    tex = var * (1.0 - edge).clamp(0.0, 1.0) ** edge_power

    # Restrict to lung and normalise
    tex = tex * lung_mask
    t_max = float((tex * lung_mask).max().item())
    return (tex / max(t_max, 1e-8)).clamp(0.0, 1.0)


# ── Speckle removal ───────────────────────────────────────────────────────────

def _remove_speckle(cam: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """
    Suppress isolated hot pixels that are not part of coherent regions.

    alpha = clamp(local_avg / cam, 0, 1):
      • cam >> local_avg → alpha → 0 → replace with neighbourhood average
      • cam ≈ local_avg  → alpha → 1 → keep original
    """
    c4d      = cam.unsqueeze(0).unsqueeze(0)
    p        = kernel // 2
    local_avg = F.avg_pool2d(c4d, kernel_size=kernel, stride=1, padding=p).squeeze(0).squeeze(0)
    alpha    = (local_avg / (cam + 1e-6)).clamp(0.0, 1.0)
    return alpha * cam + (1.0 - alpha) * local_avg


# ── Coherence weighting ───────────────────────────────────────────────────────

def _coherence_weight(cam: torch.Tensor) -> torch.Tensor:
    """
    Penalise pixels that are isolated relative to a larger neighbourhood.

    weight = large_avg / (small_avg + ε), clamped to [0.1, 1.0].

    Blob pixel:   large_avg ≈ small_avg → weight ≈ 1 (no change).
    Isolated spike: large_avg << small_avg → weight → 0.1 (suppressed).
    """
    c4d       = cam.unsqueeze(0).unsqueeze(0)
    small_avg = F.avg_pool2d(c4d, kernel_size=5,  stride=1, padding=2).squeeze(0).squeeze(0)
    large_avg = F.avg_pool2d(c4d, kernel_size=17, stride=1, padding=8).squeeze(0).squeeze(0)
    return (large_avg / (small_avg + 1e-6)).clamp(0.1, 1.0)


# ── Bilateral opacity boost ───────────────────────────────────────────────────

def _bilateral_opacity_boost(
    cam: torch.Tensor,
    lung_mask: torch.Tensor,
    *,
    diffuse_kernel: int = 21,
    boost_factor: float = 0.08,
    threshold: float = 0.35,
) -> tuple[torch.Tensor, float]:
    """
    If both lung halves have meaningful diffuse activation, add back a small
    fraction of the diffuse signal to prevent over-suppression of bilateral haze.

    The additive term is derived from the CAM itself (heavily smoothed), so
    no energy from outside existing activation is injected.

    Returns (boosted_cam, bilateral_opacity_score).
    """
    H, W = cam.shape
    mid  = W // 2
    eff  = cam * lung_mask

    p       = diffuse_kernel // 2
    diffuse = F.avg_pool2d(
        eff.unsqueeze(0).unsqueeze(0),
        kernel_size=diffuse_kernel, stride=1, padding=p,
    ).squeeze(0).squeeze(0)

    left_e  = float(diffuse[:, :mid].mean().item())
    right_e = float(diffuse[:, mid:].mean().item())

    top = max(left_e, right_e)
    bilateral_sym     = (min(left_e, right_e) / top) if top > 1e-8 else 0.0
    diffuse_strength  = (left_e + right_e) / 2.0
    # Score: product of symmetry and (bounded) diffuse strength
    bilateral_opacity_score = round(float(bilateral_sym * min(1.0, diffuse_strength * 12.0)), 4)

    if bilateral_opacity_score >= threshold:
        # Add a diffuse floor — only inside lungs, only from existing cam signal
        cam = (cam + boost_factor * diffuse * lung_mask).clamp(0.0)

    return cam, bilateral_opacity_score


# ── Skeletal / clavicle / lateral suppression ─────────────────────────────────

def _skeletal_suppression(
    gray: torch.Tensor,
    *,
    clavicle_frac: float = 0.18,
    lateral_frac: float = 0.12,
    clavicle_weight: float = 0.60,
    lateral_weight: float = 0.70,
    sharp_edge_threshold: float = 0.65,
    sharp_edge_weight: float = 0.55,
) -> torch.Tensor:
    """
    Suppression map in [0, 1].  Multiply into cam to attenuate skeletal regions.

    • Top ``clavicle_frac`` rows → clavicle_weight
    • Outer ``lateral_frac`` columns → lateral_weight
    • Pixels where Sobel > sharp_edge_threshold → sharp_edge_weight
    """
    H, W = gray.shape
    edge = _edge_magnitude(gray)

    sup = torch.ones(H, W, dtype=torch.float32)

    # Clavicle / upper-chest band
    cr = max(1, int(H * clavicle_frac))
    sup[:cr, :].clamp_max_(clavicle_weight)

    # Lateral image borders
    lc = max(1, int(W * lateral_frac))
    sup[:, :lc].clamp_max_(lateral_weight)
    sup[:, W - lc:].clamp_max_(lateral_weight)

    # Sharp structural edges (ribs, diaphragm border)
    bone_penalty = torch.where(
        edge > sharp_edge_threshold,
        torch.full_like(edge, sharp_edge_weight),
        torch.ones_like(edge),
    )
    sup = sup * bone_penalty

    return sup.clamp(0.0, 1.0)


# ── Coherence smoothing ───────────────────────────────────────────────────────

def _coherence_smooth(cam: torch.Tensor, kernel: int = 7) -> torch.Tensor:
    """7×7 average-pool smoothing, 60/40 blended with original."""
    c4d     = cam.unsqueeze(0).unsqueeze(0)
    p       = kernel // 2
    smoothed = F.avg_pool2d(c4d, kernel_size=kernel, stride=1, padding=p).squeeze(0).squeeze(0)
    return (0.60 * smoothed + 0.40 * cam).clamp(0.0)


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_opacity_shaping(
    cam: torch.Tensor,
    image_tensor: torch.Tensor,
    lung_mask: torch.Tensor,
    *,
    texture_amplify_weight: float = 0.40,
    enable_skeletal_suppression: bool = True,
    enable_coherence_smoothing: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Apply opacity-sensitive pulmonary attention shaping.

    Parameters
    ----------
    cam : (H, W) float tensor — Stage-3 normalised output
    image_tensor : (1, C, H, W) on any device — model input
    lung_mask : (H, W) float in [0, 1]

    Returns
    -------
    (shaped_cam, telemetry_dict)
    """
    H, W = cam.shape

    # Align lung_mask to heatmap dimensions
    if lung_mask.shape != cam.shape:
        lung_mask = F.interpolate(
            lung_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode="nearest",
        ).squeeze(0).squeeze(0).clamp(0.0, 1.0)

    # Extract grayscale from model input (moved to CPU inside the function)
    gray = _extract_grayscale(image_tensor, target_hw=(H, W))

    cam_in     = cam                         # keep original for telemetry
    total_pre  = float(cam.sum().item()) + 1e-8
    shaped     = cam.clone()

    # ── 1. Speckle removal ────────────────────────────────────────────────
    shaped = _remove_speckle(shaped, kernel=3)

    # ── 2. Pathology coherence weighting ──────────────────────────────────
    shaped = (shaped * _coherence_weight(shaped)).clamp(0.0)

    # ── 3. Pulmonary texture amplification ────────────────────────────────
    tex_map = _texture_map(gray, lung_mask)
    shaped  = shaped * (1.0 + texture_amplify_weight * tex_map)

    # ── 4. Bilateral opacity boost ─────────────────────────────────────────
    shaped, bilateral_opacity_score = _bilateral_opacity_boost(shaped, lung_mask)

    # ── 5. Skeletal suppression ────────────────────────────────────────────
    if enable_skeletal_suppression:
        sup_map = _skeletal_suppression(gray)
        # Record how much pre-shaped energy lived in skeletal zones BEFORE sup
        skeletal_overlap_score = round(
            float((cam_in * (1.0 - sup_map)).sum().item()) / total_pre, 4
        )
        shaped = shaped * sup_map
    else:
        sup_map = torch.ones_like(gray)
        skeletal_overlap_score = 0.0

    # ── 6. Lung-mask re-application (safety) ─────────────────────────────
    shaped = shaped * lung_mask

    # ── 7. Coherence smoothing ─────────────────────────────────────────────
    if enable_coherence_smoothing:
        shaped = _coherence_smooth(shaped, kernel=7)

    # ── 8. Final lung-mask pass (smoothing may leak 1-2px outside mask) ──
    shaped = shaped * lung_mask

    # ── Telemetry ──────────────────────────────────────────────────────────
    total_post = float(shaped.sum().item()) + 1e-8

    # opacity_response_score: fraction of shaped energy over high-opacity regions
    op_map               = _opacity_map(gray)
    high_opacity_mask    = (op_map > 0.55).float() * lung_mask
    opacity_response_score = round(
        float((shaped * high_opacity_mask).sum().item()) / total_post, 4
    )

    # diffuse_pattern_score: ratio of large-pool to total (high → diffuse blob)
    diffuse_4d = F.avg_pool2d(
        shaped.unsqueeze(0).unsqueeze(0), kernel_size=15, stride=1, padding=7
    ).squeeze(0).squeeze(0)
    diffuse_pattern_score = round(
        float(diffuse_4d.sum().item()) / total_post, 4
    )

    # pulmonary_texture_score: mean texture map value inside lung
    lung_px = float(lung_mask.sum().item()) + 1e-8
    pulmonary_texture_score = round(
        float((tex_map * lung_mask).sum().item()) / lung_px, 4
    )

    # speckle_noise_score: fraction of original energy in isolated spikes
    speckle_after = float(_remove_speckle(cam_in, kernel=3).sum().item())
    speckle_noise_score = round(max(0.0, 1.0 - speckle_after / total_pre), 4)

    # pathology_coherence_score: fraction of shaped energy in large coherent blobs
    large_pooled = F.avg_pool2d(
        shaped.unsqueeze(0).unsqueeze(0), kernel_size=17, stride=1, padding=8
    ).squeeze(0).squeeze(0)
    pathology_coherence_score = round(
        float(large_pooled.sum().item()) / total_post, 4
    )

    return shaped, {
        "opacity_response_score":    opacity_response_score,
        "diffuse_pattern_score":     diffuse_pattern_score,
        "pulmonary_texture_score":   pulmonary_texture_score,
        "speckle_noise_score":       speckle_noise_score,
        "skeletal_overlap_score":    skeletal_overlap_score,
        "pathology_coherence_score": pathology_coherence_score,
        "bilateral_opacity_score":   bilateral_opacity_score,
    }
