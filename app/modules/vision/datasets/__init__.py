from app.modules.vision.datasets.schema import (
    ImageClass,
    ImageRecord,
    QualityFlag,
    SourceType,
    Split,
    DatasetManifestMeta,
)
from app.modules.vision.datasets.dataset import (
    ImageFolderDataset,
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CLASSES_3,
    CLASS_TO_IDX_3,
    IDX_TO_CLASS_3,
)
from app.modules.vision.datasets.manifest import DatasetManifest
from app.modules.vision.datasets.quality import ImageQualityValidator, QualityThresholds
from app.modules.vision.datasets.deduplication import (
    DuplicateDetector,
    content_hash,
    perceptual_hash,
    hamming_distance,
)
from app.modules.vision.datasets.balancer import (
    compute_class_weights,
    compute_class_weights_tensor,
    build_weighted_sampler,
    imbalance_report,
)
from app.modules.vision.datasets.versioning import DatasetVersionManager
from app.modules.vision.datasets.segmented_dataset import (
    SegmentedROIDataset,
    DatasetTelemetrySummary,
)

__all__ = [
    "ImageClass", "ImageRecord", "QualityFlag", "SourceType", "Split",
    "DatasetManifestMeta",
    "ImageFolderDataset",
    "CLASSES", "CLASS_TO_IDX", "IDX_TO_CLASS",
    "CLASSES_3", "CLASS_TO_IDX_3", "IDX_TO_CLASS_3",
    "DatasetManifest",
    "ImageQualityValidator", "QualityThresholds",
    "DuplicateDetector", "content_hash", "perceptual_hash", "hamming_distance",
    "compute_class_weights", "compute_class_weights_tensor",
    "build_weighted_sampler", "imbalance_report",
    "DatasetVersionManager",
    "SegmentedROIDataset",
    "DatasetTelemetrySummary",
]
