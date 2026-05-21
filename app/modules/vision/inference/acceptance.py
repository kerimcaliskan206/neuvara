"""
Multi-tier vision acceptance.

The raw confidence filter is too coarse for clinical demo: a 0.55-confident
``unrelated`` prediction still passes a 0.50 threshold, and we never want
that surfaced as accepted.  This module layers three rules:

  1. **Hard reject** when the predicted class is itself non-target
     (``unrelated`` or ``hard_negative``).  The model is telling us this
     image isn't what the pipeline is for.
  2. **Sanity reject** when the combined non-target probability mass meets
     or exceeds the predicted class probability.  Catches the case where
     the top class is ``related`` but barely, and unrelated/hard_negative
     together account for most of the distribution.
  3. **Tier the remainder** by confidence:

       ``conf >= STRONG_ACCEPT``  → ``ACCEPTED``                  (Grad-CAM on)
       ``conf >= threshold``      → ``ACCEPTED_LOW_CONFIDENCE``   (warning, Grad-CAM optional)
       ``conf <  threshold``      → ``REJECTED``                  (no Grad-CAM)

Backwards compatibility: ``AcceptanceResult.accepted`` is true for both
``ACCEPTED`` and ``ACCEPTED_LOW_CONFIDENCE`` so existing boolean callers
keep working.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

#: Class labels the pipeline is NOT meant to surface as a positive
#: prediction.  Frozen so callers can't mutate.
NON_TARGET_CLASSES: frozenset[str] = frozenset({"unrelated", "hard_negative"})

#: Confidence at or above which we treat the prediction as fully accepted
#: and enable Grad-CAM by default.  Tuned to match the threshold the model
#: behaves stably on for genuine microscopy in the demo dataset.
STRONG_ACCEPT_THRESHOLD: float = 0.70


class AcceptanceLevel(str, Enum):
    """Tri-state acceptance outcome surfaced to callers and the API."""

    ACCEPTED = "accepted"
    ACCEPTED_LOW_CONFIDENCE = "accepted_low_confidence"
    REJECTED = "rejected"


@dataclass(frozen=True)
class AcceptanceResult:
    """Result of running the acceptance pipeline on one prediction."""

    level: AcceptanceLevel
    reason: str | None
    confidence: float
    predicted_class: str | None

    @property
    def accepted(self) -> bool:
        """True when level is either ACCEPTED or ACCEPTED_LOW_CONFIDENCE."""
        return self.level is not AcceptanceLevel.REJECTED

    @property
    def is_low_confidence(self) -> bool:
        return self.level is AcceptanceLevel.ACCEPTED_LOW_CONFIDENCE


def decide_acceptance(
    *,
    predicted_class: str,
    confidence: float,
    probabilities: Sequence[float],
    class_names: Sequence[str],
    threshold: float,
    strong_threshold: float = STRONG_ACCEPT_THRESHOLD,
) -> AcceptanceResult:
    """
    Apply the three-tier acceptance pipeline.

    Parameters
    ----------
    predicted_class
        Top-1 class label from the model.
    confidence
        Softmax probability of ``predicted_class``.
    probabilities
        Full softmax vector, aligned with ``class_names``.
    class_names
        Class names, aligned with ``probabilities``.
    threshold
        Minimum confidence required to be considered for acceptance.
    strong_threshold
        Confidence at or above which the prediction is fully accepted
        rather than flagged as low-confidence.  Must be >= ``threshold``;
        if a caller passes a smaller value we clamp.
    """
    strong_threshold = max(strong_threshold, threshold)
    probs = dict(zip(class_names, probabilities))

    # Rule 1: hard reject when the model itself predicted a non-target class.
    if predicted_class in NON_TARGET_CLASSES:
        return AcceptanceResult(
            level=AcceptanceLevel.REJECTED,
            reason=(
                f"Görüntü, hedef alan dışı sınıfa atandı ({predicted_class}). "
                "Lütfen mikroskobik/tıbbi bir görüntü yükleyin."
            ),
            confidence=confidence,
            predicted_class=predicted_class,
        )

    # Rule 2a: below the floor — reject before sanity check (cheaper).
    if confidence < threshold:
        return AcceptanceResult(
            level=AcceptanceLevel.REJECTED,
            reason=(
                f"Güven skoru {confidence:.2f}, kabul eşiğinin ({threshold:.2f}) altında."
            ),
            confidence=confidence,
            predicted_class=predicted_class,
        )

    # Rule 2b: sanity layer — even with predicted=target, the combined
    # non-target probability mass shouldn't dominate the predicted class.
    p_predicted = probs.get(predicted_class, confidence)
    p_non_target = sum(
        p for name, p in probs.items() if name in NON_TARGET_CLASSES
    )
    if p_non_target >= p_predicted:
        return AcceptanceResult(
            level=AcceptanceLevel.REJECTED,
            reason=(
                f"İlgisiz sınıf olasılığı ({p_non_target:.2f}) hedef sınıfı "
                f"({p_predicted:.2f}) baskıladığı için reddedildi."
            ),
            confidence=confidence,
            predicted_class=predicted_class,
        )

    # Rule 3: tier the accept.
    if confidence >= strong_threshold:
        return AcceptanceResult(
            level=AcceptanceLevel.ACCEPTED,
            reason=None,
            confidence=confidence,
            predicted_class=predicted_class,
        )

    return AcceptanceResult(
        level=AcceptanceLevel.ACCEPTED_LOW_CONFIDENCE,
        reason=(
            f"Güven sınırda ({confidence:.2f}); sonuç düşük güvenle kabul edildi. "
            "Lütfen sonucu bir uzmana danışın."
        ),
        confidence=confidence,
        predicted_class=predicted_class,
    )
