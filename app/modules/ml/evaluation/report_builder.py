import logging
from dataclasses import dataclass

from app.modules.ml.evaluation.metrics import EvaluationResult

logger = logging.getLogger(__name__)

_COL_W = 24
_METRIC_COLS = ["Acc", "Prec", "Recall", "F1", "AUC"]


@dataclass
class EvaluationReport:
    """Structured report over multiple models' evaluation results."""
    model_results: dict[str, EvaluationResult]
    best_model_name: str
    context: dict  # free-form metadata (dataset, n_samples, version, …)

    def log(self) -> None:
        logger.info("═" * 70)
        logger.info("  HANTAPROJECT — ADVANCED EVALUATION REPORT")
        for k, v in self.context.items():
            logger.info("  %-18s %s", f"{k}:", v)
        logger.info("─" * 70)
        logger.info(
            "  %-26s %7s %7s %7s %7s %7s",
            "Model", *_METRIC_COLS,
        )
        logger.info("  " + "─" * 62)
        for name, m in sorted(
            self.model_results.items(), key=lambda x: x[1].f1, reverse=True
        ):
            tag = "★ " if name == self.best_model_name else "  "
            logger.info(
                "  %s%-24s %7.4f %7.4f %7.4f %7.4f %7.4f",
                tag, name, m.accuracy, m.precision, m.recall, m.f1, m.roc_auc,
            )
        logger.info("  " + "─" * 62)
        best = self.model_results[self.best_model_name]
        logger.info("  Best model  : %s", self.best_model_name)
        logger.info("  F1          : %.4f", best.f1)
        logger.info("  ROC-AUC     : %.4f", best.roc_auc)
        logger.info("  Precision   : %.4f", best.precision)
        logger.info("  Recall      : %.4f", best.recall)
        logger.info("═" * 70)

    def to_dict(self) -> dict:
        return {
            "context": self.context,
            "best_model": self.best_model_name,
            "models": {
                name: m.to_dict() for name, m in self.model_results.items()
            },
        }

    def comparison_table(self) -> str:
        """Returns a plain-text comparison table (useful for CLI output)."""
        header = (
            f"{'Model':<{_COL_W + 2}}"
            + "".join(f"{c:>8}" for c in _METRIC_COLS)
        )
        sep = "─" * len(header)
        lines = [sep, header, sep]
        for name, m in sorted(
            self.model_results.items(), key=lambda x: x[1].f1, reverse=True
        ):
            tag = "★" if name == self.best_model_name else " "
            row = (
                f"{tag} {name:<{_COL_W + 1}}"
                f"{m.accuracy:>8.4f}"
                f"{m.precision:>8.4f}"
                f"{m.recall:>8.4f}"
                f"{m.f1:>8.4f}"
                f"{m.roc_auc:>8.4f}"
            )
            lines.append(row)
        lines.append(sep)
        return "\n".join(lines)


class EvaluationReportBuilder:
    """Assembles an EvaluationReport from model results and optional context."""

    def build(
        self,
        model_results: dict[str, EvaluationResult],
        best_model_name: str | None = None,
        context: dict | None = None,
    ) -> EvaluationReport:
        if not model_results:
            raise ValueError("model_results must not be empty")
        if best_model_name is None:
            best_model_name = max(model_results, key=lambda n: model_results[n].f1)
        return EvaluationReport(
            model_results=model_results,
            best_model_name=best_model_name,
            context=context or {},
        )
