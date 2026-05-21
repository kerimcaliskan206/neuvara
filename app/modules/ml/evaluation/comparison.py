import logging
from dataclasses import dataclass

from app.modules.ml.evaluation.metrics import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    rankings: list[tuple[str, EvaluationResult]]   # sorted by F1 descending
    best_model_name: str
    best_metrics: EvaluationResult


class ModelComparer:
    """Ranks trained models by F1 score and logs a side-by-side comparison table."""

    def compare(self, results: dict[str, EvaluationResult]) -> ComparisonResult:
        ranked = sorted(results.items(), key=lambda x: x[1].f1, reverse=True)
        best_name, best_metrics = ranked[0]
        return ComparisonResult(
            rankings=ranked,
            best_model_name=best_name,
            best_metrics=best_metrics,
        )

    def log_comparison(self, result: ComparisonResult) -> None:
        logger.info("─── Model Comparison (ranked by F1) ─────────────────────")
        logger.info(
            "  %-22s %7s %7s %7s %7s %7s",
            "Model", "Acc", "Prec", "Recall", "F1", "AUC",
        )
        logger.info("  " + "─" * 58)
        for i, (name, m) in enumerate(result.rankings):
            tag = "★ " if i == 0 else "  "
            logger.info(
                "  %s%-20s %7.4f %7.4f %7.4f %7.4f %7.4f",
                tag, name, m.accuracy, m.precision, m.recall, m.f1, m.roc_auc,
            )
        logger.info("  " + "─" * 58)
        logger.info(
            "  Best: %s  (F1=%.4f, AUC=%.4f)",
            result.best_model_name,
            result.best_metrics.f1,
            result.best_metrics.roc_auc,
        )
        logger.info("──────────────────────────────────────────────────────────")
