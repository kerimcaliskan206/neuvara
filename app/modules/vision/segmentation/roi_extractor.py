"""
Lung ROI extraction from a segmentation mask.

Takes the raw PIL image + binary lung mask and returns a cropped PIL Image
that contains both lung fields with a configurable padding margin.

Design goals
------------
- Preserve aspect ratio (no squashing).
- Remove black borders that confuse the classifier.
- Add 5-10 % padding so anatomical edges are visible.
- Keep the crop inside the image boundary (clamp, never pad with black).
- Return rich metadata for downstream telemetry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.modules.vision.segmentation.mask_utils import (
    compute_bounding_box,
    detect_black_border,
)

logger = logging.getLogger(__name__)


@dataclass
class ROIResult:
    """Output of ROIExtractor.extract()."""

    roi_image: Image.Image
    roi_width: int
    roi_height: int
    # fraction of original image area covered by the ROI
    crop_ratio: float
    border_removed: bool
    # tight bounding box in original image coordinate space (x1, y1, x2, y2)
    bbox: tuple[int, int, int, int]
    padding_pct: float


class ROIExtractor:
    """
    Extract the bounding-box ROI from a lung mask with aspect-ratio-preserving
    padding.

    Parameters
    ----------
    padding_frac : float
        Fraction of the ROI side length added as padding on each side.
        Default 0.07 → 7 % padding (total ≈ 14 % enlargement per axis).
    border_margin : int
        Pixel margin sampled from each image edge when detecting black borders.
    border_threshold : int
        Mean brightness below which an edge is classified as a black border.
    """

    def __init__(
        self,
        padding_frac: float = 0.07,
        border_margin: int = 15,
        border_threshold: int = 15,
    ) -> None:
        if not 0.0 <= padding_frac <= 0.3:
            raise ValueError(f"padding_frac must be in [0, 0.3], got {padding_frac}")
        self.padding_frac = padding_frac
        self.border_margin = border_margin
        self.border_threshold = border_threshold

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, image: Image.Image, mask: np.ndarray) -> ROIResult:
        """
        Crop the lung ROI from *image* guided by *mask*.

        Parameters
        ----------
        image : PIL.Image (RGB)
        mask  : H×W uint8 binary mask (non-zero = lung)

        Returns
        -------
        ROIResult with the cropped image and crop metadata.
        """
        orig_w, orig_h = image.size
        gray_arr = np.array(image.convert("L"))

        border_removed = detect_black_border(
            gray_arr,
            margin=self.border_margin,
            threshold=self.border_threshold,
        )

        bbox = compute_bounding_box(mask)
        if bbox is None:
            logger.warning(
                "ROIExtractor: mask is empty — returning full image as ROI"
            )
            return ROIResult(
                roi_image=image,
                roi_width=orig_w,
                roi_height=orig_h,
                crop_ratio=1.0,
                border_removed=border_removed,
                bbox=(0, 0, orig_w, orig_h),
                padding_pct=0.0,
            )

        x1, y1, x2, y2 = bbox
        roi_w = x2 - x1
        roi_h = y2 - y1

        # Add padding (fraction of each ROI axis)
        pad_x = int(roi_w * self.padding_frac)
        pad_y = int(roi_h * self.padding_frac)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(orig_w, x2 + pad_x)
        y2 = min(orig_h, y2 + pad_y)

        cropped = image.crop((x1, y1, x2, y2))
        crop_w = x2 - x1
        crop_h = y2 - y1
        crop_ratio = (crop_w * crop_h) / max(orig_w * orig_h, 1)

        logger.debug(
            "ROIExtractor: bbox=(%d,%d,%d,%d) pad_x=%d pad_y=%d "
            "crop=%dx%d ratio=%.3f border_removed=%s",
            x1, y1, x2, y2, pad_x, pad_y, crop_w, crop_h,
            crop_ratio, border_removed,
        )

        return ROIResult(
            roi_image=cropped,
            roi_width=crop_w,
            roi_height=crop_h,
            crop_ratio=round(crop_ratio, 4),
            border_removed=border_removed,
            bbox=(x1, y1, x2, y2),
            padding_pct=round(self.padding_frac * 100, 1),
        )
