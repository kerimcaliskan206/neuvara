"""
EfficientNet transfer learning wrapper.

EfficientNet scales width, depth, and resolution simultaneously using a
compound coefficient. B0 is the smallest and fastest; B7 is the largest
and most accurate. B0 and B4 are the recommended options for most tasks.

Key difference from ResNet: EfficientNet's classifier is a Sequential
block at model.classifier[1], not model.fc.
"""
import logging

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B7_Weights,
)

from app.modules.vision.models.base import BaseVisionModel

logger = logging.getLogger(__name__)

_VARIANTS: dict[str, tuple] = {
    "efficientnet_b0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b1": (models.efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1),
    "efficientnet_b4": (models.efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
    "efficientnet_b7": (models.efficientnet_b7, EfficientNet_B7_Weights.IMAGENET1K_V1),
}


class EfficientNetClassifier(BaseVisionModel):
    """
    EfficientNet backbone + custom classification head.

    Architecture:
        Pretrained EfficientNet features (Conv + MBConv blocks)
            → AdaptiveAvgPool (built-in)
            → Dropout(dropout)
            → Linear(in_features → num_classes)
    """

    def __init__(
        self,
        variant: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(num_classes=num_classes, architecture=variant)

        if variant not in _VARIANTS:
            raise ValueError(
                f"Unknown EfficientNet variant '{variant}'. "
                f"Available: {list(_VARIANTS)}"
            )

        factory, weights = _VARIANTS[variant]
        self._backbone: nn.Module = factory(
            weights=weights if pretrained else None
        )

        # EfficientNet's classifier: Sequential(Dropout, Linear)
        # Replace the Linear layer while keeping the existing Dropout structure
        in_features: int = self._backbone.classifier[1].in_features
        self._backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        logger.info(
            "EfficientNetClassifier: %s | pretrained=%s | in_features=%d | num_classes=%d",
            variant, pretrained, in_features, num_classes,
        )

    # ── nn.Module ─────────────────────────────────────────────────────────────

    def forward(self, x):
        return self._backbone(x)

    # ── BaseVisionModel ───────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze the feature extractor (model.features). Keep classifier trainable."""
        for param in self._backbone.features.parameters():
            param.requires_grad = False
        for param in self._backbone.classifier.parameters():
            param.requires_grad = True
        self.log_parameter_summary()

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self._backbone.parameters():
            param.requires_grad = True
        self.log_parameter_summary()

    def get_backbone(self) -> nn.Module:
        return self._backbone.features

    def get_classifier(self) -> nn.Module:
        return self._backbone.classifier


def build_efficientnet(
    variant: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze: bool = True,
) -> EfficientNetClassifier:
    """Factory function — builds and optionally freezes the backbone."""
    model = EfficientNetClassifier(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
    if freeze:
        model.freeze_backbone()
    return model
