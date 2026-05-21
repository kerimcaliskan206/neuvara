from dataclasses import dataclass


@dataclass
class PredictionResponse:
    prediction: int
    probability: float | None
    label: str
    confidence: str

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "probability": round(self.probability, 4) if self.probability is not None else None,
            "label": self.label,
            "confidence": self.confidence,
        }


class PredictionFormatter:
    """Converts raw model output into human-readable API responses."""

    LABELS: dict[int, str] = {0: "negative", 1: "positive"}

    def format_single(
        self, prediction: int, probability: float | None = None
    ) -> PredictionResponse:
        confidence = self._confidence_level(probability)
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            label=self.LABELS.get(prediction, "unknown"),
            confidence=confidence,
        )

    def format_batch(
        self,
        predictions: list[int],
        probabilities: list[float] | None = None,
    ) -> list[dict]:
        return [
            self.format_single(
                pred,
                probabilities[i] if probabilities else None,
            ).to_dict()
            for i, pred in enumerate(predictions)
        ]

    @staticmethod
    def _confidence_level(probability: float | None) -> str:
        if probability is None:
            return "unknown"
        if probability >= 0.85:
            return "high"
        if probability >= 0.60:
            return "medium"
        return "low"
