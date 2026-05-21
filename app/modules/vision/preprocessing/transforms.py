"""
Transform pipeline factories.

Three distinct transform pipelines:
  - train   : resize → augmentations → ToTensor → normalize
  - val     : resize → center crop → ToTensor → normalize
  - inference: identical to val (no stochastic ops)

All pipelines produce tensors normalized with ImageNet statistics, matching
what the pretrained ResNet/EfficientNet backbones were trained on.
"""
import logging

import torchvision.transforms as T

from app.modules.vision.config import VisionConfig, vision_config

logger = logging.getLogger(__name__)


def get_train_transforms(config: VisionConfig = vision_config) -> T.Compose:
    """
    Stochastic transforms for training.
    Augmentations are controlled by config.augmentation.
    """
    aug = config.augmentation
    norm = config.normalization
    h, w = config.image_size.height, config.image_size.width

    if aug.random_resized_crop:
        # Replaces Resize+RandomCrop: simulates varying subject distance and zoom.
        # scale_min=0.7 keeps ≥83 % of each image dimension, preventing diagnostic
        # content loss in radiological and wildlife images.
        steps = [
            T.RandomResizedCrop(
                (h, w),
                scale=(aug.random_resized_crop_scale_min, 1.0),
                ratio=(0.75, 1.33),
            )
        ]
    else:
        steps = [
            T.Resize((int(h * 1.15), int(w * 1.15))),
            T.RandomCrop((h, w)),
        ]

    if aug.horizontal_flip:
        steps.append(T.RandomHorizontalFlip(p=0.5))

    if aug.vertical_flip:
        steps.append(T.RandomVerticalFlip(p=0.5))

    if aug.rotation_degrees > 0:
        steps.append(T.RandomRotation(degrees=aug.rotation_degrees))

    if aug.color_jitter:
        steps.append(
            T.ColorJitter(
                brightness=aug.color_jitter_brightness,
                contrast=aug.color_jitter_contrast,
                saturation=aug.color_jitter_saturation,
                hue=aug.color_jitter_hue,
            )
        )

    if aug.grayscale_prob > 0:
        steps.append(T.RandomGrayscale(p=aug.grayscale_prob))

    steps += [
        T.ToTensor(),
        T.Normalize(mean=norm.mean, std=norm.std),
    ]

    if aug.random_erasing:
        steps.append(T.RandomErasing(p=aug.random_erasing_prob))

    logger.debug("Train transforms: %d steps", len(steps))
    return T.Compose(steps)


def get_val_transforms(config: VisionConfig = vision_config) -> T.Compose:
    """
    Deterministic transforms for validation/test.
    No random operations — results are reproducible.
    """
    norm = config.normalization
    h, w = config.image_size.height, config.image_size.width

    return T.Compose([
        T.Resize((int(h * 1.15), int(w * 1.15))),
        T.CenterCrop((h, w)),
        T.ToTensor(),
        T.Normalize(mean=norm.mean, std=norm.std),
    ])


def get_inference_transforms(config: VisionConfig = vision_config) -> T.Compose:
    """
    Inference transforms — identical to val (deterministic, no augmentation).
    Kept as a separate function so callers are explicit about their intent.
    """
    return get_val_transforms(config)


def get_denormalize_transform(config: VisionConfig = vision_config) -> T.Normalize:
    """
    Inverse normalization — used to reconstruct a displayable image from a tensor
    (e.g., for Grad-CAM overlays).
    """
    mean = config.normalization.mean
    std = config.normalization.std
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    return T.Normalize(mean=inv_mean, std=inv_std)
