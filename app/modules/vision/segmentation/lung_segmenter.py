"""
Lung segmentation for chest X-rays using classical computer vision.

Algorithm
---------
1. Convert to grayscale and apply CLAHE (contrast normalization).
2. Gaussian blur to suppress high-frequency noise.
3. Otsu thresholding — tried on both the enhanced image and its inverse,
   so the algorithm is robust to datasets that store lungs as dark vs. light.
4. Morphological closing to merge small gaps in the lung fields.
5. Hole filling via boundary flood-fill + inversion.
6. Connected-component analysis — keep the two largest regions (left / right lung).
7. Validate lung area fraction; fall back to a center-region mask when the
   segmentation result is implausible.

No additional pretrained model is required — only OpenCV and NumPy, which are
already project dependencies.

Public surface
--------------
    LungSegmenter          – segments a PIL image, returns SegmentationResult
    LungSegmentationPipeline – full pipeline: segment → ROI extract → telemetry
                               → optional debug saves → ROI PIL image
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.modules.vision.segmentation.mask_utils import (
    apply_morphological_cleanup,
    compute_bilateral_balance,
    compute_black_border_crop,
    count_contours,
    create_center_mask,
    fill_holes,
    keep_largest_components,
    lung_area_metrics,
)
from app.modules.vision.segmentation.roi_extractor import ROIExtractor, ROIResult

logger = logging.getLogger(__name__)

# ── Segmentation result ────────────────────────────────────────────────────────


@dataclass
class SegmentationResult:
    """Output of LungSegmenter.segment()."""

    mask: np.ndarray            # H×W uint8 (0 = background, 255 = lung)
    lung_area_pct: float        # fraction of image pixels that are lung
    n_components: int           # number of connected lung regions found
    quality: str                # "good" | "fallback" | "single_lung"
    width: int                  # original image width
    height: int                 # original image height
    segmentation_ms: float      # wall-clock time for this call

    # Phase 28 — robustness debug metrics. All optional / default-safe so
    # existing callers that construct SegmentationResult without these still
    # work (none do today, but defensive).
    lung_area_ratio: float = 0.0      # alias of lung_area_pct under the canonical name
    contour_count: int = 0            # raw external contour count (pre largest-N filter)
    bilateral_balance: float = 0.0    # left/right symmetry of the chosen mask [0, 1]
    fallback_reason: Optional[str] = None  # None when not fallback; otherwise short tag
    mask_confidence: float = 0.0      # composite trust scalar [0, 1] for the mask


# ── Telemetry record ──────────────────────────────────────────────────────────


@dataclass
class SegmentationTelemetry:
    """All metrics logged per inference pass through the pipeline."""

    lung_area_pct: float
    roi_width: int
    roi_height: int
    crop_ratio: float
    border_removed: bool
    quality: str
    n_components: int
    segmentation_ms: float
    # Phase 28 — robustness debug metrics.
    lung_area_ratio: float = 0.0
    contour_count: int = 0
    bilateral_balance: float = 0.0
    fallback_reason: Optional[str] = None
    mask_confidence: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "lung_area_pct": self.lung_area_pct,
            "lung_area_ratio": self.lung_area_ratio,
            "roi_width": self.roi_width,
            "roi_height": self.roi_height,
            "crop_ratio": self.crop_ratio,
            "border_removed": self.border_removed,
            "quality": self.quality,
            "n_components": self.n_components,
            "contour_count": self.contour_count,
            "bilateral_balance": self.bilateral_balance,
            "fallback_reason": self.fallback_reason,
            "mask_confidence": self.mask_confidence,
            "segmentation_ms": round(self.segmentation_ms, 2),
            **self.extra,
        }


# ── LungSegmenter ─────────────────────────────────────────────────────────────


class LungSegmenter:
    """
    Segments lung fields from a chest X-ray using CLAHE + Otsu + morphology.

    Parameters
    ----------
    clahe_clip : float
        CLAHE clip limit for contrast enhancement.
    clahe_tile : tuple[int, int]
        Tile grid size for CLAHE.
    kernel_size : int
        Morphological structuring element diameter (pixels).
    morph_iterations : int
        Number of morphological closing passes.
    min_lung_frac : float
        Minimum plausible lung area fraction. Below this → fallback.
    max_lung_frac : float
        Maximum plausible lung area fraction. Above this → fallback.
    """

    # Plausible range for lung area on a standard PA chest X-ray
    _MIN_LUNG_FRAC: float = 0.08
    _MAX_LUNG_FRAC: float = 0.68

    def __init__(
        self,
        clahe_clip: float = 2.0,
        clahe_tile: tuple[int, int] = (8, 8),
        kernel_size: int = 15,
        morph_iterations: int = 2,
        min_lung_frac: float = _MIN_LUNG_FRAC,
        max_lung_frac: float = _MAX_LUNG_FRAC,
    ) -> None:
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.kernel_size = kernel_size
        self.morph_iterations = morph_iterations
        self.min_lung_frac = min_lung_frac
        self.max_lung_frac = max_lung_frac

        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=clahe_tile
        )

    # ── Public ────────────────────────────────────────────────────────────────

    # Phase 28 — minimum mask-confidence to accept a candidate over the
    # center-mask fallback. Tuned so a single-blob, lopsided, or border-only
    # mask still loses to the center fallback, but a moderate bilateral
    # segmentation wins.
    _MIN_ACCEPT_CONFIDENCE: float = 0.35

    def segment(self, image: Image.Image) -> SegmentationResult:
        """
        Segment lung fields from a PIL Image.

        Phase 28 pipeline:
          1. Crop any black border (DICOM padding / letterbox) so it does
             not poison the Otsu histogram.
          2. CLAHE + Gaussian blur.
          3. Three candidate masks:
             A. Otsu on enhanced image
             B. Otsu on inverted enhanced image
             C. Adaptive (mean) threshold — robust to low-contrast cases
          4. Score each candidate by area-plausibility × bilateral balance ×
             contour sanity → mask_confidence in [0, 1].
          5. Pick the best candidate. Only fall back to a center mask when
             every candidate scores below ``_MIN_ACCEPT_CONFIDENCE``.
          6. Re-pad the chosen mask back to the original image dimensions.
        """
        t0 = time.perf_counter()
        gray = np.array(image.convert("L"))
        h, w = gray.shape

        # 1. Border crop — record offsets so we can re-pad the final mask.
        top, bottom, left, right = compute_black_border_crop(gray)
        border_cropped = top + bottom + left + right > 0
        if border_cropped:
            cropped = gray[top:h - bottom if bottom else h, left:w - right if right else w]
        else:
            cropped = gray
        ch, cw = cropped.shape

        # 2. Preprocess.
        enhanced = self._clahe.apply(cropped)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # 3. Three candidate masks (all in cropped coordinates).
        candidates: list[tuple[str, np.ndarray, float]] = []
        mask_a, frac_a = self._threshold_and_clean(blurred, ch, cw, invert=False)
        candidates.append(("otsu_direct", mask_a, frac_a))
        mask_b, frac_b = self._threshold_and_clean(blurred, ch, cw, invert=True)
        candidates.append(("otsu_inverted", mask_b, frac_b))
        # Adaptive threshold: robust to global illumination drift / low
        # contrast. Block size scales with image; C=3 is a gentle bias.
        block = max(31, (min(ch, cw) // 16) | 1)  # ensure odd
        adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, 3,
        )
        mask_c = self._post_threshold(adapt)
        frac_c = int((mask_c > 127).sum()) / max(ch * cw, 1)
        candidates.append(("adaptive_mean", mask_c, frac_c))

        # 4. Score each candidate.
        scored: list[tuple[str, np.ndarray, float, float]] = []
        for name, m, frac in candidates:
            conf = self._score_candidate(m, frac)
            scored.append((name, m, frac, conf))
            logger.debug(
                "LungSegmenter candidate %s: frac=%.3f confidence=%.3f",
                name, frac, conf,
            )

        # 5. Pick the best, fall back only when ALL candidates score low.
        scored.sort(key=lambda t: t[3], reverse=True)
        best_name, best_mask, best_frac, best_conf = scored[0]
        fallback_reason: Optional[str] = None

        if best_conf < self._MIN_ACCEPT_CONFIDENCE:
            best_mask = create_center_mask(ch, cw, frac=0.80)
            quality = "fallback"
            fallback_reason = self._diagnose_fallback(scored)
            logger.warning(
                "LungSegmenter: all 3 candidates scored below %.2f "
                "(best=%s conf=%.3f) — using centre-mask fallback (reason=%s)",
                self._MIN_ACCEPT_CONFIDENCE, best_name, best_conf, fallback_reason,
            )
            mask_confidence = round(best_conf, 4)
        else:
            quality = "good"
            mask_confidence = round(best_conf, 4)
            logger.debug(
                "LungSegmenter: selected candidate=%s frac=%.3f confidence=%.3f",
                best_name, best_frac, mask_confidence,
            )

        # 6. Re-pad to original dimensions if we cropped a border.
        if border_cropped:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[top:h - bottom if bottom else h, left:w - right if right else w] = best_mask
        else:
            mask = best_mask

        _, n_components = keep_largest_components(mask, n=2)
        if n_components == 1 and quality == "good":
            quality = "single_lung"

        metrics = lung_area_metrics(mask)
        contour_n = count_contours(mask)
        bilateral = compute_bilateral_balance(mask)
        elapsed = (time.perf_counter() - t0) * 1000.0

        logger.debug(
            "LungSegmenter: quality=%s lung_area_pct=%.3f n_components=%d "
            "contour_count=%d bilateral=%.2f mask_conf=%.3f "
            "border_cropped=%s elapsed_ms=%.1f",
            quality, metrics["lung_area_pct"], n_components,
            contour_n, bilateral, mask_confidence,
            border_cropped, elapsed,
        )

        return SegmentationResult(
            mask=mask,
            lung_area_pct=metrics["lung_area_pct"],
            n_components=n_components,
            quality=quality,
            width=w,
            height=h,
            segmentation_ms=elapsed,
            lung_area_ratio=metrics["lung_area_pct"],
            contour_count=contour_n,
            bilateral_balance=bilateral,
            fallback_reason=fallback_reason,
            mask_confidence=mask_confidence,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _post_threshold(self, thresh: np.ndarray) -> np.ndarray:
        """Shared post-threshold cleanup: close → fill holes → keep top-2."""
        closed = apply_morphological_cleanup(
            thresh, kernel_size=self.kernel_size, iterations=self.morph_iterations
        )
        filled = fill_holes(closed)
        mask, _ = keep_largest_components(filled, n=2)
        return mask

    def _threshold_and_clean(
        self,
        blurred: np.ndarray,
        h: int,
        w: int,
        invert: bool,
    ) -> tuple[np.ndarray, float]:
        """
        Apply Otsu threshold (optionally on inverted image), then morphological
        cleanup + hole fill + component selection. Return (mask, area_frac).
        """
        source = cv2.bitwise_not(blurred) if invert else blurred
        _, thresh = cv2.threshold(
            source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask = self._post_threshold(thresh)
        lung_px = int((mask > 127).sum())
        frac = lung_px / max(h * w, 1)
        return mask, frac

    def _score_candidate(self, mask: np.ndarray, frac: float) -> float:
        """
        Composite anatomical plausibility score in [0, 1].

        Components (each in [0, 1], multiplied):
          area_gain        — 1.0 inside plausible range, ramps to 0 outside
          bilateral_gain   — left/right symmetry (0 = single-blob, 1 = balanced)
          contour_gain     — 1.0 for 1-4 contours, fades for noisy/empty masks

        An empty mask scores 0. A frame-filling mask scores 0. A single
        large blob in one half scores low. A balanced two-lung mask with a
        plausible area scores ~1.0.
        """
        # Area gain — triangle ramp peaked at the centre of the plausible band.
        target = (self.min_lung_frac + self.max_lung_frac) / 2.0
        if self.min_lung_frac <= frac <= self.max_lung_frac:
            half_band = (self.max_lung_frac - self.min_lung_frac) / 2.0
            area_gain = 1.0 - abs(frac - target) / max(half_band, 1e-6)
            area_gain = max(0.4, area_gain)  # in-band always scores ≥ 0.4
        else:
            # Linear decay outside the band, zero past ±50% of the band width.
            band = self.max_lung_frac - self.min_lung_frac
            if frac < self.min_lung_frac:
                gap = self.min_lung_frac - frac
            else:
                gap = frac - self.max_lung_frac
            area_gain = max(0.0, 1.0 - gap / max(band / 2.0, 1e-6))

        bilateral_gain = compute_bilateral_balance(mask)

        n_contours = count_contours(mask)
        if 1 <= n_contours <= 4:
            contour_gain = 1.0
        elif n_contours == 0:
            contour_gain = 0.0
        else:
            # Many tiny blobs → noisy threshold result.
            contour_gain = max(0.2, 1.0 - (n_contours - 4) * 0.10)

        return round(area_gain * bilateral_gain * contour_gain, 4)

    @staticmethod
    def _diagnose_fallback(
        scored: list[tuple[str, np.ndarray, float, float]],
    ) -> str:
        """Produce a short fallback_reason tag from the scored candidates."""
        best_name, best_mask, best_frac, best_conf = scored[0]
        if (best_mask > 127).sum() == 0:
            return "empty_mask"
        bal = compute_bilateral_balance(best_mask)
        if bal < 0.20:
            return "no_bilateral_symmetry"
        if best_conf < 0.10:
            return "no_plausible_candidate"
        # In-range but still rejected: typically noisy / fragmented.
        return f"low_confidence({best_name}:{best_conf:.2f})"


# ── Full pipeline ─────────────────────────────────────────────────────────────


class LungSegmentationPipeline:
    """
    End-to-end pipeline: Raw X-ray → segmented lung ROI (PIL Image).

    Usage
    -----
        pipeline = LungSegmentationPipeline()
        roi_image, telemetry = pipeline.process(pil_image)
        # roi_image is a cropped PIL Image ready for the classifier transforms.

    Parameters
    ----------
    padding_frac : float
        Padding added around the lung bounding box (fraction of ROI side).
        Default 0.07 (7 %).
    save_debug : bool
        When True, debug visualisations are written to *debug_dir* on every
        call. Keep False in production to avoid I/O overhead.
    debug_dir : Path | str | None
        Directory for debug images. Defaults to data/vision/debug relative
        to the project root.
    segmenter : LungSegmenter | None
        Custom segmenter instance. Creates a default one if None.
    extractor : ROIExtractor | None
        Custom extractor instance. Creates a default one if None.
    """

    _DEFAULT_DEBUG_DIR = (
        Path(__file__).resolve().parents[5] / "data" / "vision" / "debug"
    )

    def __init__(
        self,
        padding_frac: float = 0.07,
        save_debug: bool = False,
        debug_dir: Optional[Path | str] = None,
        segmenter: Optional[LungSegmenter] = None,
        extractor: Optional[ROIExtractor] = None,
    ) -> None:
        self.segmenter = segmenter or LungSegmenter()
        self.extractor = extractor or ROIExtractor(padding_frac=padding_frac)
        self.save_debug = save_debug
        self.debug_dir = Path(debug_dir) if debug_dir else self._DEFAULT_DEBUG_DIR

        if self.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "LungSegmentationPipeline: debug output enabled → %s",
                self.debug_dir,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self, image: Image.Image
    ) -> tuple[Image.Image, SegmentationTelemetry]:
        """
        Segment lung ROI from *image*.

        Parameters
        ----------
        image : PIL.Image (any mode; will be converted to RGB internally)

        Returns
        -------
        roi_image : PIL.Image.Image
            Cropped lung region, RGB, suitable for classifier preprocessing.
        telemetry : SegmentationTelemetry
            Metrics logged for every inference pass.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        seg: SegmentationResult = self.segmenter.segment(image)
        roi: ROIResult = self.extractor.extract(image, seg.mask)

        telemetry = SegmentationTelemetry(
            lung_area_pct=seg.lung_area_pct,
            roi_width=roi.roi_width,
            roi_height=roi.roi_height,
            crop_ratio=roi.crop_ratio,
            border_removed=roi.border_removed,
            quality=seg.quality,
            n_components=seg.n_components,
            segmentation_ms=seg.segmentation_ms,
        )

        self._log_telemetry(telemetry)

        if self.save_debug:
            self._save_debug_images(image, seg.mask, roi)

        return roi.roi_image, telemetry

    # ── Telemetry ─────────────────────────────────────────────────────────────

    @staticmethod
    def _log_telemetry(tel: SegmentationTelemetry) -> None:
        logger.info(
            "seg_telemetry "
            "lung_area_pct=%.4f "
            "roi_width=%d "
            "roi_height=%d "
            "crop_ratio=%.4f "
            "border_removed=%s "
            "quality=%s "
            "n_components=%d "
            "segmentation_ms=%.1f",
            tel.lung_area_pct,
            tel.roi_width,
            tel.roi_height,
            tel.crop_ratio,
            tel.border_removed,
            tel.quality,
            tel.n_components,
            tel.segmentation_ms,
        )

    # ── Debug visualisation ───────────────────────────────────────────────────

    def _save_debug_images(
        self,
        original: Image.Image,
        mask: np.ndarray,
        roi: ROIResult,
    ) -> None:
        """
        Save four debug images to self.debug_dir:
          <id>_original.png
          <id>_mask.png
          <id>_roi_crop.png
          <id>_overlay.png
        """
        run_id = uuid.uuid4().hex[:8]
        try:
            # 1. Original
            orig_path = self.debug_dir / f"{run_id}_original.png"
            original.save(orig_path)

            # 2. Lung mask (single channel stored as RGB for readability)
            mask_img = Image.fromarray(mask, mode="L").convert("RGB")
            mask_path = self.debug_dir / f"{run_id}_mask.png"
            mask_img.save(mask_path)

            # 3. ROI crop
            roi_path = self.debug_dir / f"{run_id}_roi_crop.png"
            roi.roi_image.save(roi_path)

            # 4. Overlay: semi-transparent green mask + red ROI bounding box
            overlay = self._build_overlay(original, mask, roi)
            overlay_path = self.debug_dir / f"{run_id}_overlay.png"
            overlay.save(overlay_path)

            logger.debug(
                "LungSegmentationPipeline: debug images saved with id=%s to %s",
                run_id, self.debug_dir,
            )
        except Exception:
            logger.warning(
                "LungSegmentationPipeline: failed to save debug images for id=%s",
                run_id, exc_info=True,
            )

    @staticmethod
    def _build_overlay(
        original: Image.Image,
        mask: np.ndarray,
        roi: ROIResult,
    ) -> Image.Image:
        """
        Compose the overlay visualisation:
        - green semi-transparent lung mask
        - red bounding box for the ROI crop
        """
        overlay = original.copy().convert("RGBA")
        w, h = overlay.size

        # Green mask layer
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        lung_pixels = mask > 127
        mask_rgba[lung_pixels] = [0, 200, 0, 100]  # green, 40 % opacity
        mask_layer = Image.fromarray(mask_rgba, mode="RGBA")
        overlay = Image.alpha_composite(overlay, mask_layer)

        # Red ROI bounding box
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = roi.bbox
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 220), width=3)

        return overlay.convert("RGB")
