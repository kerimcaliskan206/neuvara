"""
Abstract base class for all vision classifiers.

All model wrappers must implement freeze_backbone() / unfreeze_backbone()
so the training loop can switch between fine-tuning the head only
(fast, few-shot) and full fine-tuning (slower, more data required).
"""
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseVisionModel(nn.Module, ABC):
    """
    Common interface for vision classifiers used in HantaProject.

    Training strategy (two-phase):
      Phase A — freeze_backbone():
        Only the classification head is trainable.
        Use a higher learning rate (1e-3) for a few epochs.
        Suitable when dataset is small (< 1000 images).

      Phase B — unfreeze_backbone():
        All layers are trainable.
        Use a much smaller learning rate (1e-5) to avoid destroying
        pretrained weights. Apply gradient clipping.
    """

    def __init__(self, num_classes: int, architecture: str) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture

    @abstractmethod
    def freeze_backbone(self) -> None:
        """Freeze all pretrained backbone layers. Only the head remains trainable."""

    @abstractmethod
    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers for full fine-tuning."""

    @abstractmethod
    def get_backbone(self) -> nn.Module:
        """Return the backbone (feature extractor) module."""

    @abstractmethod
    def get_classifier(self) -> nn.Module:
        """Return the classification head module."""

    # ── Shared utilities ──────────────────────────────────────────────────────

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → softmax probabilities. Shape: (N, num_classes)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → predicted class indices. Shape: (N,)."""
        return self.predict_proba(x).argmax(dim=1)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def log_parameter_summary(self) -> None:
        trainable = self.count_trainable_params()
        total = self.count_total_params()
        frozen = total - trainable
        logger.info(
            "[%s] Parameters: %d total | %d trainable | %d frozen",
            self.architecture, total, trainable, frozen,
        )

    def to_device(self, device: str) -> "BaseVisionModel":
        """Move model to device and return self (for chaining)."""
        return self.to(torch.device(device))
