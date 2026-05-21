"""
Lung segmentation sub-package for chest X-ray preprocessing.

Exposes a clean public API for the rest of the vision module.

Quick start
-----------
    from app.modules.vision.segmentation import LungSegmentationPipeline

    pipeline = LungSegmentationPipeline(padding_frac=0.07, save_debug=False)
    roi_image, telemetry = pipeline.process(pil_image)
    # roi_image → pass to ImagePreprocessingPipeline.preprocess_for_inference()
    # telemetry → SegmentationTelemetry with lung_area_pct, roi_width, …
"""
from app.modules.vision.segmentation.lung_segmenter import (
    LungSegmentationPipeline,
    LungSegmenter,
    SegmentationResult,
    SegmentationTelemetry,
)
from app.modules.vision.segmentation.mask_utils import (
    apply_morphological_cleanup,
    compute_bounding_box,
    create_center_mask,
    detect_black_border,
    fill_holes,
    keep_largest_components,
    lung_area_metrics,
)
from app.modules.vision.segmentation.roi_extractor import ROIExtractor, ROIResult

__all__ = [
    # Pipeline (primary entry-point)
    "LungSegmentationPipeline",
    # Components (for custom assemblies)
    "LungSegmenter",
    "ROIExtractor",
    # Result types
    "SegmentationResult",
    "SegmentationTelemetry",
    "ROIResult",
    # Mask utilities (for tests / custom logic)
    "fill_holes",
    "keep_largest_components",
    "apply_morphological_cleanup",
    "compute_bounding_box",
    "lung_area_metrics",
    "create_center_mask",
    "detect_black_border",
]
