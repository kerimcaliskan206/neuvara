"""
Image preprocessing pipeline.

Wraps the transform factories in a high-level API that:
  - handles PIL Image → Tensor conversion for inference
  - handles file path loading
  - produces batched tensors (N, C, H, W) for model input
"""
import logging
from pathlib import Path

import torch
from PIL import Image

from app.modules.vision.config import VisionConfig, vision_config
from app.modules.vision.preprocessing.transforms import (
    get_inference_transforms,
    get_train_transforms,
    get_val_transforms,
)
from app.modules.vision.utils.io import load_image

logger = logging.getLogger(__name__)


class ImagePreprocessingPipeline:
    """
    Stateless preprocessing pipeline — no fitting required.

    Unlike the tabular ML pipeline, image transforms use fixed
    ImageNet statistics (not dataset-specific statistics) so the
    pipeline does not need to be serialized alongside the model.
    """

    def __init__(self, config: VisionConfig = vision_config) -> None:
        self.config = config
        self._train_tf = get_train_transforms(config)
        self._val_tf = get_val_transforms(config)
        self._infer_tf = get_inference_transforms(config)
        logger.debug(
            "ImagePreprocessingPipeline ready: target size=%dx%d",
            config.image_size.width, config.image_size.height,
        )

    # ── Single image ──────────────────────────────────────────────────────────

    def preprocess_for_inference(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → (1, C, H, W) tensor ready for model.forward()."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self._infer_tf(image)         # (C, H, W)
        return tensor.unsqueeze(0)             # (1, C, H, W)

    def preprocess_for_training(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → (C, H, W) tensor with stochastic augmentation."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self._train_tf(image)

    def preprocess_for_validation(self, image: Image.Image) -> torch.Tensor:
        """PIL Image → (C, H, W) tensor (deterministic, no augmentation)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self._val_tf(image)

    # ── From path ─────────────────────────────────────────────────────────────

    def preprocess_path(self, path: Path | str) -> torch.Tensor:
        """Load image from disk and preprocess for inference → (1, C, H, W)."""
        image = load_image(Path(path))
        return self.preprocess_for_inference(image)

    # ── Batch ─────────────────────────────────────────────────────────────────

    def preprocess_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """List of PIL Images → (N, C, H, W) tensor for batch inference."""
        tensors = [self.preprocess_for_inference(img).squeeze(0) for img in images]
        return torch.stack(tensors)

    def preprocess_paths_batch(self, paths: list[Path | str]) -> torch.Tensor:
        """List of paths → (N, C, H, W) tensor."""
        return self.preprocess_batch([load_image(Path(p)) for p in paths])
