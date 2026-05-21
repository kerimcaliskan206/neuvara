"""
ResNet transfer learning wrapper.

Uses torchvision's pretrained ResNet with the final FC layer replaced
by a custom classification head sized to num_classes.
"""
import logging

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)

from app.modules.vision.models.base import BaseVisionModel

logger = logging.getLogger(__name__)

_VARIANTS: dict[str, tuple] = {
    "resnet18":  (models.resnet18,  ResNet18_Weights.IMAGENET1K_V1),
    "resnet34":  (models.resnet34,  ResNet34_Weights.IMAGENET1K_V1),
    "resnet50":  (models.resnet50,  ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
}


class ResNetClassifier(BaseVisionModel):
    """
    ResNet backbone + custom classification head.

    Architecture:
        Pretrained ResNet (feature extractor)
            → GlobalAveragePool (built into ResNet)
            → Dropout(dropout)
            → Linear(in_features → num_classes)

    Transfer learning strategy:
        1. Call freeze_backbone() — train only the head on a few epochs.
        2. Call unfreeze_backbone() — fine-tune everything with a small LR.
    """

    def __init__(
        self,
        variant: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(num_classes=num_classes, architecture=variant)

        if variant not in _VARIANTS:
            raise ValueError(
                f"Unknown ResNet variant '{variant}'. "
                f"Available: {list(_VARIANTS)}"
            )

        factory, weights = _VARIANTS[variant]
        self._backbone: nn.Module = factory(
            weights=weights if pretrained else None
        )

        # Replace the final fully-connected layer
        in_features: int = self._backbone.fc.in_features
        self._backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        logger.info(
            "ResNetClassifier: %s | pretrained=%s | in_features=%d | num_classes=%d",
            variant, pretrained, in_features, num_classes,
        )

    # ── nn.Module ─────────────────────────────────────────────────────────────

    def forward(self, x):
        return self._backbone(x)

    # ── BaseVisionModel ───────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all layers except the final classification head."""
        for name, param in self._backbone.named_parameters():
            param.requires_grad = "fc" in name
        self.log_parameter_summary()

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self._backbone.parameters():
            param.requires_grad = True
        self.log_parameter_summary()

    def get_backbone(self) -> nn.Module:
        return self._backbone

    def get_classifier(self) -> nn.Module:
        return self._backbone.fc


def build_resnet(
    variant: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze: bool = True,
) -> ResNetClassifier:
    """Factory function — builds and optionally freezes the backbone."""
    model = ResNetClassifier(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
    if freeze:
        model.freeze_backbone()
    return model
