"""
Augmentation strategy definitions.

Augmentation synthetically expands a small dataset by applying random
but realistic transformations during training. The key constraint:
augmentations must be medically plausible — a microscopy slide is never
upside-down, but it can be rotated, slightly over/under-exposed, or zoomed.

Augmentations are ONLY applied during training, never at validation or inference.
"""
from dataclasses import dataclass


@dataclass
class AugmentationPreset:
    name: str
    description: str
    params: dict


# ── Preset library ────────────────────────────────────────────────────────────

PRESETS: dict[str, AugmentationPreset] = {
    "minimal": AugmentationPreset(
        name="minimal",
        description="Safe defaults — only horizontal flip and small rotations.",
        params={
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation_degrees": 10,
            "color_jitter": False,
            "random_erasing": False,
        },
    ),
    "standard": AugmentationPreset(
        name="standard",
        description="Recommended for most hantavirus image tasks.",
        params={
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation_degrees": 15,
            "color_jitter": True,
            "color_jitter_brightness": 0.2,
            "color_jitter_contrast": 0.2,
            "color_jitter_saturation": 0.1,
            "color_jitter_hue": 0.05,
            "random_erasing": False,
        },
    ),
    "aggressive": AugmentationPreset(
        name="aggressive",
        description="Stronger augmentation for very small datasets (< 500 images).",
        params={
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation_degrees": 30,
            "color_jitter": True,
            "color_jitter_brightness": 0.4,
            "color_jitter_contrast": 0.4,
            "color_jitter_saturation": 0.2,
            "color_jitter_hue": 0.1,
            "random_erasing": True,
            "random_erasing_prob": 0.2,
        },
    ),
    "aggressive_medical": AugmentationPreset(
        name="aggressive_medical",
        description=(
            "Aggressive augmentation for small mixed medical/wildlife datasets (< 500 images). "
            "Same as 'aggressive' but random_erasing is disabled to avoid occluding "
            "diagnostically important regions in radiological and pathology images."
        ),
        params={
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation_degrees": 30,
            "color_jitter": True,
            "color_jitter_brightness": 0.4,
            "color_jitter_contrast": 0.4,
            "color_jitter_saturation": 0.2,
            "color_jitter_hue": 0.1,
            "random_erasing": False,
            "random_resized_crop": True,
            "random_resized_crop_scale_min": 0.7,
            "grayscale_prob": 0.15,
        },
    ),
    "microscopy": AugmentationPreset(
        name="microscopy",
        description=(
            "Tailored for microscopy slides — aggressive rotation (any angle is valid), "
            "color variation for staining differences, no erasing."
        ),
        params={
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation_degrees": 180,
            "color_jitter": True,
            "color_jitter_brightness": 0.3,
            "color_jitter_contrast": 0.3,
            "color_jitter_saturation": 0.3,
            "color_jitter_hue": 0.05,
            "random_erasing": False,
        },
    ),
    "pulmonary_bilateral": AugmentationPreset(
        name="pulmonary_bilateral",
        description=(
            "Phase 21 — Bilateral/ARDS specialization for chest X-ray fine-tuning. "
            "Strong contrast variation simulates edema vs. clear lung differences. "
            "No vertical flip (anatomical orientation preserved). "
            "No random erasing (must not occlude lung fields). "
            "High grayscale probability (chest X-rays are grayscale). "
            "Tight rotation and crop range (PA/AP views are near-upright)."
        ),
        params={
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation_degrees": 8,
            "color_jitter": True,
            "color_jitter_brightness": 0.50,
            "color_jitter_contrast": 0.50,
            "color_jitter_saturation": 0.10,
            "color_jitter_hue": 0.02,
            "random_erasing": False,
            "random_resized_crop": True,
            "random_resized_crop_scale_min": 0.78,
            "grayscale_prob": 0.35,
        },
    ),
    "lung_roi": AugmentationPreset(
        name="lung_roi",
        description=(
            "Phase 2 — Lung ROI spatial robustness. "
            "Applied to pre-segmented lung crops only. "
            "Small affine translation, tight rotation, Gaussian noise via "
            "get_lung_roi_train_transforms(). No RandomErasing or vertical flip."
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
    ),
}


def get_preset(name: str) -> AugmentationPreset:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown augmentation preset '{name}'. "
            f"Available: {list(PRESETS)}"
        )
    return PRESETS[name]
