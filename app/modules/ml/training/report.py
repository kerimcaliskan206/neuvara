import logging
from dataclasses import dataclass, field

from app.modules.ml.evaluation.metrics import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    timestamp: str
    dataset: str
    version: str
    target_column: str
    n_samples_train: int
    n_samples_test: int
    n_features: int
    class_distribution: dict[str, dict]
    imbalance_ratio: float
    model_results: dict[str, EvaluationResult]
    best_model_name: str

    def log(self) -> None:
        best = self.model_results[self.best_model_name]
        logger.info("═" * 56)
        logger.info("  HANTAPROJECT — TRAINING REPORT")
        logger.info("  Timestamp : %s", self.timestamp)
        logger.info("  Version   : %s", self.version)
        logger.info("  Dataset   : %s", self.dataset)
        logger.info("  Target    : %s", self.target_column)
        logger.info("  Train / Test: %d / %d samples", self.n_samples_train, self.n_samples_test)
        logger.info("  Features  : %d", self.n_features)
        logger.info("  Imbalance : %.2f:1", self.imbalance_ratio)
        logger.info("  ─ Class distribution ─")
        for cls, info in self.class_distribution.items():
            logger.info(
                "    %-10s %5d  (%.1f%%)",
                cls, info["count"], info["percentage"],
            )
        logger.info("  ─ Best model ─")
        logger.info("    Name      : %s", self.best_model_name)
        logger.info("    Accuracy  : %.4f", best.accuracy)
        logger.info("    Precision : %.4f", best.precision)
        logger.info("    Recall    : %.4f", best.recall)
        logger.info("    F1        : %.4f", best.f1)
        logger.info("    ROC-AUC   : %.4f", best.roc_auc)
        logger.info("═" * 56)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "dataset": self.dataset,
            "version": self.version,
            "target_column": self.target_column,
            "n_samples_train": self.n_samples_train,
            "n_samples_test": self.n_samples_test,
            "n_features": self.n_features,
            "imbalance_ratio": self.imbalance_ratio,
            "best_model": self.best_model_name,
            "models": {
                name: metrics.to_dict()
                for name, metrics in self.model_results.items()
            },
        }
