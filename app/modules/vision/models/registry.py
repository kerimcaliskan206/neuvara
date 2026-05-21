"""
Vision model registry.

Single entry point for building any supported architecture.
New architectures are added here without touching training or inference code.
"""
import logging
from typing import Callable

from app.modules.vision.models.base import BaseVisionModel
from app.modules.vision.models.efficientnet import build_efficientnet
from app.modules.vision.models.resnet import build_resnet

logger = logging.getLogger(__name__)

# Maps architecture name → builder function
_BUILDERS: dict[str, Callable[..., BaseVisionModel]] = {
    "resnet18":         lambda **kw: build_resnet(variant="resnet18", **kw),
    "resnet34":         lambda **kw: build_resnet(variant="resnet34", **kw),
    "resnet50":         lambda **kw: build_resnet(variant="resnet50", **kw),
    "resnet101":        lambda **kw: build_resnet(variant="resnet101", **kw),
    "efficientnet_b0":  lambda **kw: build_efficientnet(variant="efficientnet_b0", **kw),
    "efficientnet_b1":  lambda **kw: build_efficientnet(variant="efficientnet_b1", **kw),
    "efficientnet_b4":  lambda **kw: build_efficientnet(variant="efficientnet_b4", **kw),
    "efficientnet_b7":  lambda **kw: build_efficientnet(variant="efficientnet_b7", **kw),
}

# Recommended architectures with brief rationale
RECOMMENDATIONS: dict[str, str] = {
    "efficientnet_b0": "Fast, small (5M params). Good starting point for limited data.",
    "resnet50":        "Balanced accuracy/speed (25M params). Good general-purpose choice.",
    "efficientnet_b4": "Higher accuracy, moderate size (19M params). Use when > 2000 images.",
    "resnet101":       "Highest accuracy in this list. Requires large dataset and GPU.",
}


class VisionModelRegistry:
    """Builds and describes vision models."""

    @classmethod
    def build(
        cls,
        architecture: str,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze: bool = True,
    ) -> BaseVisionModel:
        """
        Build a vision model by architecture name.

        Parameters
        ----------
        architecture : str
            One of the supported architecture names.
        num_classes : int
            Output classes (2 for binary hantavirus detection).
        pretrained : bool
            Load ImageNet weights. Almost always True for transfer learning.
        dropout : float
            Dropout rate in the classification head.
        freeze : bool
            Freeze the backbone at initialization (Phase A training).
        """
        builder = _BUILDERS.get(architecture)
        if builder is None:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Available: {cls.list_available()}"
            )

        model = builder(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze=freeze,
        )

        logger.info(
            "VisionModelRegistry: built %s | classes=%d | pretrained=%s | frozen=%s",
            architecture, num_classes, pretrained, freeze,
        )
        return model

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(_BUILDERS.keys())

    @classmethod
    def recommendations(cls) -> dict[str, str]:
        return RECOMMENDATIONS.copy()
