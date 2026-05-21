"""
Segmentation-aware preprocessing pipeline.

Wraps the standard ImagePreprocessingPipeline with a lung segmentation step
so the classifier only ever sees the lung ROI — never the full raw image.

Pipeline stages
---------------
    Raw X-ray (PIL Image)
        │
        ▼
    LungSegmentationPipeline      ← CLAHE + Otsu + morphology + ROI crop
        │  (returns PIL Image, SegmentationTelemetry)
        ▼
    ImagePreprocessingPipeline    ← Resize → Normalize → Tensor
        │  (returns (1, C, H, W) or (C, H, W) tensor)
        ▼
    Classifier input

The segmentation step is transparent to the downstream model — output shape
and normalisation statistics are identical to the non-segmented pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from app.modules.vision.config import VisionConfig, vision_config
from app.modules.vision.preprocessing.pipeline import ImagePreprocessingPipeline
from app.modules.vision.segmentation import (
    LungSegmentationPipeline,
    SegmentationTelemetry,
)
from app.modules.vision.utils.io import load_image

logger = logging.getLogger(__name__)


class SegmentedPreprocessingPipeline:
    """
    Drop-in replacement for ImagePreprocessingPipeline that routes every image
    through lung segmentation before the standard transforms.

    Parameters
    ----------
    config : VisionConfig
        Standard vision configuration (image size, normalization, etc.).
    padding_frac : float
        Padding added around the lung bounding box (fraction of ROI side).
    save_debug : bool
        When True, debug images are saved to data/vision/debug/ on every call.
        Keep False in production.
    debug_dir : Path | str | None
        Override for the debug output directory.
    segmentation_pipeline : LungSegmentationPipeline | None
        Custom segmentation pipeline. A default one is created if None.
    """

    def __init__(
        self,
        config: VisionConfig = vision_config,
        padding_frac: float = 0.07,
        save_debug: bool = False,
        debug_dir: Optional[Path | str] = None,
        segmentation_pipeline: Optional[LungSegmentationPipeline] = None,
    ) -> None:
        self._base = ImagePreprocessingPipeline(config)
        self._seg = segmentation_pipeline or LungSegmentationPipeline(
            padding_frac=padding_frac,
            save_debug=save_debug,
            debug_dir=debug_dir,
        )
        logger.debug(
            "SegmentedPreprocessingPipeline ready: "
            "target_size=%dx%d padding_frac=%.2f save_debug=%s",
            config.image_size.width,
            config.image_size.height,
            padding_frac,
            save_debug,
        )

    # ── Single image ──────────────────────────────────────────────────────────

    def preprocess_for_inference(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, SegmentationTelemetry]:
        """
        Segment + preprocess a PIL Image for inference.

        Returns
        -------
        tensor : (1, C, H, W) float tensor, normalised, on CPU.
        telemetry : SegmentationTelemetry — metrics for this image.
        """
        roi, telemetry = self._seg.process(image)
        tensor = self._base.preprocess_for_inference(roi)
        return tensor, telemetry

    def preprocess_for_training(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, SegmentationTelemetry]:
        """
        Segment + preprocess a PIL Image for training (with augmentation).

        Returns
        -------
        tensor : (C, H, W) float tensor with stochastic augmentation.
        telemetry : SegmentationTelemetry — metrics for this image.
        """
        roi, telemetry = self._seg.process(image)
        tensor = self._base.preprocess_for_training(roi)
        return tensor, telemetry

    def preprocess_for_validation(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, SegmentationTelemetry]:
        """
        Segment + preprocess a PIL Image for validation (deterministic).

        Returns
        -------
        tensor : (C, H, W) float tensor, no augmentation.
        telemetry : SegmentationTelemetry — metrics for this image.
        """
        roi, telemetry = self._seg.process(image)
        tensor = self._base.preprocess_for_validation(roi)
        return tensor, telemetry

    # ── From path ─────────────────────────────────────────────────────────────

    def preprocess_path(
        self, path: Path | str
    ) -> tuple[torch.Tensor, SegmentationTelemetry]:
        """Load from disk, segment, and preprocess for inference."""
        image = load_image(Path(path))
        return self.preprocess_for_inference(image)

    # ── Batch ─────────────────────────────────────────────────────────────────

    def preprocess_batch(
        self, images: list[Image.Image]
    ) -> tuple[torch.Tensor, list[SegmentationTelemetry]]:
        """
        Process a list of PIL Images → (N, C, H, W) tensor.

        Segmentation is applied sequentially; telemetry is returned per image.
        """
        tensors: list[torch.Tensor] = []
        telemetries: list[SegmentationTelemetry] = []

        for img in images:
            tensor, tel = self.preprocess_for_inference(img)
            tensors.append(tensor.squeeze(0))
            telemetries.append(tel)

        return torch.stack(tensors), telemetries

    # ── ROI-only helper ───────────────────────────────────────────────────────

    def extract_roi(
        self, image: Image.Image
    ) -> tuple[Image.Image, SegmentationTelemetry]:
        """
        Return the lung ROI as a PIL Image without applying classifier transforms.

        Useful for visualisation and dataset preprocessing scripts.
        """
        return self._seg.process(image)
