"""
Bilateral pulmonary spatial scorer — Phase 21.

Analyzes a GradCAM heatmap to estimate bilateral pulmonary involvement.
Pure spatial analysis: no model calls, no preprocessing required.

Metrics
-------
  left_ratio      : fraction of activation mass in the left half
  right_ratio     : fraction in the right half
  symmetry        : min/max ratio [0,1] — 1.0 = perfectly bilateral
  lower_density   : fraction of activation in the lower 60% (gravity-dependent)
  diffuse_spread  : fraction of pixels above 0.30 threshold — wide = diffuse
  bilateral_burden: composite severity index [0,1]

Composite formula
-----------------
  bilateral_burden = symmetry * 0.45 + lower_density * 0.35 + diffuse_spread * 0.20

Interpretation
--------------
  ≥ 0.70 : strong bilateral/ARDS-compatible pattern → imaging_score boosted in reasoning
  0.40–0.70 : moderate bilateral involvement
  < 0.30  : unilateral or low-burden → clinical cap tightened for healthy class
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class BilateralSpatialScore:
    left_ratio:            float  # activation mass fraction in left half [0,1]
    right_ratio:           float  # activation mass fraction in right half [0,1]
    symmetry:              float  # bilateral symmetry [0,1] — 1.0 = perfectly balanced
    lower_density:         float  # activation fraction in lower 60% of image [0,1]
    diffuse_spread:        float  # fraction of pixels > 0.30 threshold [0,1]
    activation_area_ratio: float  # fraction of pixels with any activation > 0.05 [0,1]
    bilateral_burden:      float  # composite index [0,1]


def compute_bilateral_score(
    heatmap: "Union[np.ndarray, torch.Tensor]",
) -> BilateralSpatialScore:
    """
    Compute bilateral pulmonary burden from a GradCAM heatmap.

    Parameters
    ----------
    heatmap : 2D array / tensor (H, W) with values in [0, 1].
              Typically 224×224 from GradCAM.generate().

    Returns
    -------
    BilateralSpatialScore with all spatial metrics.
    """
    try:
        import torch as _torch
        if isinstance(heatmap, _torch.Tensor):
            arr = heatmap.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(heatmap, dtype=np.float32)
    except ImportError:
        arr = np.asarray(heatmap, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {arr.shape}")

    H, W = arr.shape
    total_mass = float(arr.sum()) or 1e-9

    # Left / right split (for PA chest X-ray, image left ≈ anatomical right)
    mid = W // 2
    left_mass   = float(arr[:, :mid].sum())
    right_mass  = float(arr[:, mid:].sum())
    left_ratio  = left_mass  / total_mass
    right_ratio = right_mass / total_mass

    # Bilateral symmetry: how balanced is activation between the two halves?
    denom = max(left_ratio, right_ratio)
    symmetry = (min(left_ratio, right_ratio) / denom) if denom > 1e-9 else 0.0

    # Lower-lung density: lower 60% of image height
    lower_start   = int(H * 0.40)
    lower_density = float(arr[lower_start:, :].sum()) / total_mass

    # Diffuse spread: fraction of pixels above the 0.30 activation threshold
    n_pixels      = H * W
    diffuse_spread = float((arr > 0.30).sum()) / n_pixels

    # Activation area ratio: any non-trivial pixel (> 0.05)
    activation_area_ratio = float((arr > 0.05).sum()) / n_pixels

    # Composite bilateral burden
    bilateral_burden = round(
        min(1.0, symmetry * 0.45 + lower_density * 0.35 + diffuse_spread * 0.20),
        4,
    )

    return BilateralSpatialScore(
        left_ratio=round(left_ratio, 4),
        right_ratio=round(right_ratio, 4),
        symmetry=round(symmetry, 4),
        lower_density=round(lower_density, 4),
        diffuse_spread=round(diffuse_spread, 4),
        activation_area_ratio=round(activation_area_ratio, 4),
        bilateral_burden=bilateral_burden,
    )
