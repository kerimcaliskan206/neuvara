"""
Lung ROI spatial robustness augmentations.

Designed specifically for pre-segmented lung crops where:
- The full image frame IS the lung ROI (borders removed).
- Any operation that pads with black would reintroduce border artifacts.
- Random erasing / cutout would occlude diagnostically critical lung tissue.
- Vertical flip is anatomically invalid (diaphragm should stay at bottom).
- Large rotations are invalid (PA/AP views are near-upright).

What we ADD vs. the general "pulmonary_bilateral" preset:
  - Small affine translation (±5% of image side) — simulates positioning offset
  - Gaussian noise injection — simulates film grain / sensor noise
  - Mild Gaussian blur — simulates focus variation
  - Tighter rotation range (±6°)

What we EXPLICITLY EXCLUDE vs. general augmentation:
  - RandomErasing, vertical flip, large rotations, padding-based crops

Public API
----------
    get_lung_roi_train_transforms(config)   → torchvision Compose (train)
    get_lung_roi_val_transforms(config)     → torchvision Compose (val / test)
    LUNG_ROI_PRESET                         → AugmentationPreset for augmentation.py
"""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torchvision.transforms as T
from PIL import Image

from app.modules.vision.config import VisionConfig, vision_config
from app.modules.vision.preprocessing.augmentation import AugmentationPreset

logger = logging.getLogger(__name__)


# ── Augmentation preset (registers into PRESETS dict on import) ───────────────

LUNG_ROI_PRESET = AugmentationPreset(
    name="lung_roi",
    description=(
        "Phase 2 — Lung ROI spatial robustness augmentation. "
        "Applied to pre-segmented lung crops. "
        "Small affine translation + tight rotation. "
        "Contrast/brightness shifts for scanner variation. "
        "No RandomErasing. No vertical flip. No black-padding crops."
    ),
    params={
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation_degrees": 6,
        "color_jitter": True,
        "color_jitter_brightness": 0.35,
        "color_jitter_contrast": 0.40,
        "color_jitter_saturation": 0.10,
        "color_jitter_hue": 0.02,
        "random_erasing": False,
        "random_resized_crop": True,
        "random_resized_crop_scale_min": 0.88,
        "grayscale_prob": 0.30,
    },
)


# ── Gaussian noise transform ───────────────────────────────────────────────────


class _GaussianNoise:
    """
    Add zero-mean Gaussian noise to a tensor.

    Applied after ToTensor() so values are in [0, 1]. The noise standard
    deviation is drawn uniformly from [0, max_std] each call, so some samples
    are noisier than others (curriculum-style).
    """

    def __init__(self, max_std: float = 0.04) -> None:
        self.max_std = max_std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = torch.empty(1).uniform_(0, self.max_std).item()
        return torch.clamp(tensor + torch.randn_like(tensor) * std, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"GaussianNoise(max_std={self.max_std})"


# ── Transform factory ─────────────────────────────────────────────────────────


def get_lung_roi_train_transforms(
    config: VisionConfig = vision_config,
    *,
    noise_std: float = 0.03,
    blur_kernel: int = 3,
    blur_prob: float = 0.25,
    affine_translate_pct: float = 0.05,
) -> T.Compose:
    """
    Return the training-time transform pipeline for lung ROI crops.

    Parameters
    ----------
    config : VisionConfig
        Used for target image size and ImageNet normalisation stats.
    noise_std : float
        Maximum Gaussian noise standard deviation (post-ToTensor, [0, 1]).
    blur_kernel : int
        Gaussian blur kernel size (must be odd). Applied stochastically.
    blur_prob : float
        Probability of applying Gaussian blur on a given sample.
    affine_translate_pct : float
        Maximum translation as a fraction of image side. Simulates positioning
        offsets without introducing black-padded borders via fill=0.
        Translation is applied via RandomAffine with fill=pixel-mean.
    """
    norm = config.normalization
    h, w = config.image_size.height, config.image_size.width

    # Translate fraction → pixel bounds for RandomAffine
    translate = (affine_translate_pct, affine_translate_pct)

    steps: list[Callable] = [
        # Resize: slightly oversized so the crop doesn't hit zero-padded edges
        T.Resize((int(h * 1.10), int(w * 1.10))),

        # Random scale crop within the lung ROI — preserves content, no zero padding
        T.RandomResizedCrop(
            (h, w),
            scale=(0.88, 1.0),
            ratio=(0.80, 1.25),
        ),

        # Horizontal flip — valid; PA CXR can be flipped L/R without losing info
        T.RandomHorizontalFlip(p=0.5),

        # Small rotation — PA/AP views are near-upright; ±6° is the clinical range
        T.RandomRotation(degrees=6),

        # Affine translation — small positioning offsets; gray fill avoids black borders
        T.RandomAffine(
            degrees=0,
            translate=translate,
            fill=128,
        ),

        # Colour variation — simulates scanner-to-scanner differences and exposure variation
        T.ColorJitter(
            brightness=0.35,
            contrast=0.40,
            saturation=0.10,
            hue=0.02,
        ),

        # Grayscale conversion with 30% probability
        # (chest X-rays are inherently grayscale; prevents the model relying on colour)
        T.RandomGrayscale(p=0.30),

        # ToTensor before noise (noise operates in [0, 1])
        T.ToTensor(),

        # Gaussian noise — film grain / sensor noise simulation
        _GaussianNoise(max_std=noise_std),

        # Mild blur — focus variation simulation
        T.RandomApply([T.GaussianBlur(kernel_size=blur_kernel)], p=blur_prob),

        # ImageNet normalisation
        T.Normalize(mean=norm.mean, std=norm.std),
    ]

    logger.debug(
        "lung_roi train transforms: %d steps | target=%dx%d | "
        "noise_std=%.2f blur_prob=%.2f translate=%.2f",
        len(steps), w, h, noise_std, blur_prob, affine_translate_pct,
    )
    return T.Compose(steps)


def get_lung_roi_val_transforms(
    config: VisionConfig = vision_config,
) -> T.Compose:
    """
    Deterministic transforms for validation/test on lung ROI crops.

    Identical to get_val_transforms() — no augmentation.
    """
    norm = config.normalization
    h, w = config.image_size.height, config.image_size.width
    return T.Compose([
        T.Resize((int(h * 1.05), int(w * 1.05))),
        T.CenterCrop((h, w)),
        T.ToTensor(),
        T.Normalize(mean=norm.mean, std=norm.std),
    ])
