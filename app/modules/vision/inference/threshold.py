"""
Confidence threshold filtering for vision inference.

Predictions whose max-class probability is below the threshold are
"rejected" — returned as None rather than a low-confidence guess.

This prevents the system from silently serving uncertain predictions.
The caller decides what to do with rejections (ask for a better image,
flag for manual review, etc.).
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Outcome of applying a confidence filter to one prediction."""

    accepted: bool
    class_index: int | None          # None when rejected
    class_label: str | None          # None when rejected
    confidence: float                 # always set (even for rejections)
    probabilities: list[float]        # always set
    inference_ms: float               # always set
    rejection_reason: str | None      # None when accepted

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "class_index": self.class_index,
            "class_label": self.class_label,
            "confidence": round(self.confidence, 4),
            "probabilities": [round(p, 4) for p in self.probabilities],
            "inference_ms": round(self.inference_ms, 2),
            "rejection_reason": self.rejection_reason,
        }


class ConfidenceFilter:
    """
    Wraps raw model predictions with a configurable acceptance threshold.

    Parameters
    ----------
    threshold : float
        Minimum confidence (max-class softmax probability) required to
        accept a prediction.  Typical values: 0.6–0.9.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.threshold = threshold

    # ── Single prediction ─────────────────────────────────────────────────────

    def apply(self, prediction) -> FilterResult:
        """
        Apply the threshold to a VisionPrediction.

        Parameters
        ----------
        prediction : VisionPrediction
            Result from VisionPredictor.predict_single().

        Returns
        -------
        FilterResult with accepted=True if confidence >= threshold.
        """
        accepted = prediction.confidence >= self.threshold

        if not accepted:
            logger.debug(
                "ConfidenceFilter: rejected prediction "
                "(confidence=%.4f < threshold=%.4f, label=%s)",
                prediction.confidence, self.threshold, prediction.class_label,
            )

        return FilterResult(
            accepted=accepted,
            class_index=prediction.class_index if accepted else None,
            class_label=prediction.class_label if accepted else None,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            inference_ms=prediction.inference_ms,
            rejection_reason=(
                None if accepted
                else f"confidence {prediction.confidence:.4f} < threshold {self.threshold:.4f}"
            ),
        )

    # ── Batch ─────────────────────────────────────────────────────────────────

    def filter_batch(self, predictions: list) -> list[FilterResult]:
        """Apply the threshold to each prediction in a list."""
        return [self.apply(p) for p in predictions]

    # ── Aggregate stats ───────────────────────────────────────────────────────

    def acceptance_rate(self, results: list[FilterResult]) -> float:
        """Fraction of results that were accepted (0.0–1.0)."""
        if not results:
            return 0.0
        return sum(r.accepted for r in results) / len(results)

    def is_confident(self, confidence: float) -> bool:
        """Convenience check without constructing a FilterResult."""
        return confidence >= self.threshold
