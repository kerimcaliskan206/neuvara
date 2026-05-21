"""
Fusion weight configuration.

ML is always primary (α >> β). Vision is supporting evidence only.
All weights are normalised inside FusionWeightPolicy so they always sum to 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusionWeightPolicy:
    """
    Defines how ML and vision signals are blended.

    Invariants enforced at blend time:
      - When vision is unavailable or rejected, β=0 and α=1.0 automatically.
      - ml_weight > vision_weight is strongly recommended (ML is primary).
      - Low-confidence vision is treated as unavailable.
    """

    ml_weight: float = 0.75
    vision_weight: float = 0.25

    # Minimum vision confidence to allow any vision contribution.
    # Below this, vision is treated as "low_confidence" → ignored.
    min_vision_confidence: float = 0.55

    # Risk thresholds for labelling final score.
    high_risk_threshold: float = 0.65
    medium_risk_threshold: float = 0.40

    def __post_init__(self) -> None:
        if self.ml_weight <= 0 or self.vision_weight < 0:
            raise ValueError("ml_weight must be > 0; vision_weight must be >= 0")
        if self.ml_weight <= self.vision_weight:
            raise ValueError(
                "ml_weight must be strictly greater than vision_weight "
                "(ML is always the primary signal)"
            )

    def normalised(self) -> tuple[float, float]:
        """Return (α, β) guaranteed to sum to 1.0."""
        total = self.ml_weight + self.vision_weight
        return self.ml_weight / total, self.vision_weight / total

    def ml_only(self) -> tuple[float, float]:
        """(1.0, 0.0) — used when vision is unavailable or rejected."""
        return 1.0, 0.0


# Singleton used by the engine unless overridden.
DEFAULT_WEIGHT_POLICY = FusionWeightPolicy()
