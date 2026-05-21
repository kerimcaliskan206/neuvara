"""
Focal Loss for class-imbalanced multi-class classification.

Reference: Lin et al. 2017, https://arxiv.org/abs/1708.02002

Focal Loss down-weights easy examples (high-confidence correct predictions)
and concentrates gradient signal on hard, misclassified examples. Useful
when class imbalance cannot be fully corrected by sampling alone.

gamma=0 → identical to CrossEntropyLoss
gamma=2 → standard paper value (recommended starting point)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss compatible with CrossEntropyLoss interface.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values focus more on hard examples.
    weight : Tensor | None
        Per-class weights, same semantics as CrossEntropyLoss(weight=...).
    label_smoothing : float
        Label smoothing in [0, 1). Applied before focal weighting.
    reduction : str
        "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE with optional label smoothing — reduction="none" for per-sample loss
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # p_t ≈ model's confidence for the correct class (exp(-CE) is a stable proxy)
        p_t = torch.exp(-ce)
        focal = (1.0 - p_t) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, label_smoothing={self.label_smoothing}"
