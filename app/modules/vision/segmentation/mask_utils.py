"""
Low-level mask operations for chest X-ray lung segmentation.

All functions operate on uint8 numpy arrays where non-zero pixels are
foreground (lung). Caller is responsible for type/shape contracts.
"""
from __future__ import annotations

import cv2
import numpy as np


# ── Morphological helpers ──────────────────────────────────────────────────────


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes in a binary mask by flood-filling from the image
    boundary and inverting — anything unreachable from the border is interior.
    """
    filled = mask.copy()
    h, w = filled.shape
    flood_seed = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(filled, flood_seed, (0, 0), 255)
    exterior = cv2.bitwise_not(filled)
    return mask | exterior


def keep_largest_components(mask: np.ndarray, n: int = 2) -> tuple[np.ndarray, int]:
    """
    Keep only the n largest connected components (by pixel area).

    Returns
    -------
    result : np.ndarray
        Binary mask containing only the top-n components.
    kept : int
        Actual number of components kept (may be < n if fewer exist).
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if n_labels <= 1:
        return mask, 0

    # stats rows: [left, top, width, height, area]; row 0 = background
    areas = stats[1:, cv2.CC_STAT_AREA]
    kept = min(n, len(areas))
    top_idx = np.argsort(areas)[::-1][:kept] + 1  # +1: re-index past background

    result = np.zeros_like(mask)
    for idx in top_idx:
        result[labels == idx] = 255

    return result, int(kept)


def apply_morphological_cleanup(
    mask: np.ndarray,
    kernel_size: int = 15,
    iterations: int = 2,
) -> np.ndarray:
    """Close small gaps and smooth jagged edges on a binary mask."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


# ── Geometry ───────────────────────────────────────────────────────────────────


def compute_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Tight bounding box of all non-zero pixels.

    Returns (x1, y1, x2, y2) or None when the mask is empty.
    """
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def lung_area_metrics(mask: np.ndarray) -> dict:
    """Return basic area statistics for a lung mask."""
    h, w = mask.shape
    lung_px = int((mask > 127).sum())
    total_px = h * w
    return {
        "lung_area_pct": round(lung_px / max(total_px, 1), 4),
        "lung_px": lung_px,
        "mask_h": h,
        "mask_w": w,
    }


# ── Border detection ───────────────────────────────────────────────────────────


def detect_black_border(
    arr: np.ndarray,
    margin: int = 15,
    threshold: int = 15,
) -> bool:
    """
    Return True if any image edge (top/bottom/left/right) has a mean pixel
    value below *threshold*, indicating a black border artifact.

    Parameters
    ----------
    arr : H×W uint8 grayscale array
    margin : pixels to sample from each edge
    threshold : mean brightness cut-off (0-255)
    """
    edges = (
        float(np.mean(arr[:margin, :])),
        float(np.mean(arr[-margin:, :])),
        float(np.mean(arr[:, :margin])),
        float(np.mean(arr[:, -margin:])),
    )
    return any(v < threshold for v in edges)


# ── Fallback mask ──────────────────────────────────────────────────────────────


def create_center_mask(h: int, w: int, frac: float = 0.80) -> np.ndarray:
    """
    Rectangular center-region mask used when segmentation fails.

    The mask covers the central *frac* fraction of each axis.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mh = int(h * (1.0 - frac) / 2.0)
    mw = int(w * (1.0 - frac) / 2.0)
    mask[mh: h - mh, mw: w - mw] = 255
    return mask


# ── Phase 28 — Robustness helpers ──────────────────────────────────────────────


def compute_black_border_crop(
    arr: np.ndarray,
    threshold: int = 12,
    max_crop_frac: float = 0.35,
) -> tuple[int, int, int, int]:
    """
    Detect a black border (DICOM padding / letterbox) and return crop offsets.

    Walks inward from each edge until the mean row/column brightness rises
    above ``threshold``. Capped at ``max_crop_frac`` of the dimension so an
    image that is genuinely dark all over is not eaten entirely.

    Parameters
    ----------
    arr : H×W uint8 grayscale array
    threshold : mean brightness below which a row/column is "black border"
    max_crop_frac : safety cap on how much we'll trim from each side

    Returns
    -------
    (top, bottom, left, right) — number of pixels to crop from each side.
    """
    h, w = arr.shape
    max_v = max(1, int(h * max_crop_frac))
    max_h = max(1, int(w * max_crop_frac))

    row_means = arr.mean(axis=1)
    col_means = arr.mean(axis=0)

    top = 0
    while top < max_v and row_means[top] < threshold:
        top += 1
    bottom = 0
    while bottom < max_v and row_means[h - 1 - bottom] < threshold:
        bottom += 1
    left = 0
    while left < max_h and col_means[left] < threshold:
        left += 1
    right = 0
    while right < max_h and col_means[w - 1 - right] < threshold:
        right += 1

    return top, bottom, left, right


def compute_bilateral_balance(mask: np.ndarray) -> float:
    """
    Symmetry of lung mask between left and right halves.

    1.0 = perfectly balanced (good bilateral lung pair).
    0.0 = entirely on one side (single-lung or shortcut artefact).

    Empty mask → 0.0 (no signal).
    """
    if mask.size == 0:
        return 0.0
    h, w = mask.shape
    mid = w // 2
    left = int((mask[:, :mid] > 127).sum())
    right = int((mask[:, mid:] > 127).sum())
    total = left + right
    if total == 0:
        return 0.0
    return round(1.0 - abs(left - right) / total, 4)


def count_contours(mask: np.ndarray) -> int:
    """
    Count distinct external contours in the mask (before largest-N filtering).

    Different from n_components-after-keep-largest: this surfaces the raw
    number of separate blobs the thresholder produced, which is a useful
    "noisiness" signal.
    """
    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)
