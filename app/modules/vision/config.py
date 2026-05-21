from pathlib import Path

from pydantic import BaseModel, field_validator

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ImageSizeConfig(BaseModel):
    width: int = 224
    height: int = 224

    @property
    def as_tuple(self) -> tuple[int, int]:
        return (self.height, self.width)


class AugmentationConfig(BaseModel):
    """Training-time augmentation controls."""
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_degrees: int = 15
    color_jitter: bool = True
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.1
    color_jitter_hue: float = 0.05
    random_erasing: bool = False
    random_erasing_prob: float = 0.1
    random_resized_crop: bool = False
    random_resized_crop_scale_min: float = 0.7
    grayscale_prob: float = 0.0


class NormalizationConfig(BaseModel):
    # ImageNet statistics — standard for pretrained transfer learning
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]


class UploadConfig(BaseModel):
    max_file_size_mb: float = 10.0
    allowed_extensions: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    allowed_mime_types: set[str] = {
        "image/jpeg", "image/png", "image/bmp", "image/tiff"
    }
    min_dimension_px: int = 32    # reject tiny/corrupt images
    max_dimension_px: int = 8192  # reject absurdly large images


class VisionStorageConfig(BaseModel):
    uploads_dir: Path = _PROJECT_ROOT / "data" / "vision" / "uploads"
    processed_dir: Path = _PROJECT_ROOT / "data" / "vision" / "processed"
    dataset_dir: Path = _PROJECT_ROOT / "data" / "vision" / "datasets"
    models_dir: Path = _PROJECT_ROOT / "models" / "vision"
    checkpoints_dir: Path = _PROJECT_ROOT / "models" / "vision" / "checkpoints"


class VisionModelConfig(BaseModel):
    architecture: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 2          # negative / positive
    dropout: float = 0.3
    freeze_backbone: bool = True  # train only the classification head initially

    @field_validator("architecture")
    @classmethod
    def _valid_arch(cls, v: str) -> str:
        supported = {
            "resnet18", "resnet34", "resnet50", "resnet101",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b4", "efficientnet_b7",
        }
        if v not in supported:
            raise ValueError(f"Unsupported architecture '{v}'. Choose from: {supported}")
        return v


class VisionConfig(BaseModel):
    storage: VisionStorageConfig = VisionStorageConfig()
    model: VisionModelConfig = VisionModelConfig()
    image_size: ImageSizeConfig = ImageSizeConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    upload: UploadConfig = UploadConfig()
    device: str = "auto"   # "auto" | "cpu" | "cuda" | "mps"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


vision_config = VisionConfig()
