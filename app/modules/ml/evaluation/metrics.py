import logging
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}

    def log(self, model_name: str = "") -> None:
        label = f" [{model_name}]" if model_name else ""
        logger.info("─── Metrics%s ────────────────", label)
        for metric, value in self.to_dict().items():
            bar = "█" * int(value * 20)
            logger.info("  %-12s %.4f  %s", metric, value, bar)


class ModelEvaluator:
    def evaluate(
        self,
        y_true,
        y_pred,
        y_prob=None,
        model_name: str = "",
    ) -> EvaluationResult:
        roc_auc = 0.0
        if y_prob is not None:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                logger.warning(
                    "ROC-AUC could not be computed — only one class present in y_true."
                )

        result = EvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            recall=float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            roc_auc=roc_auc,
        )
        result.log(model_name)
        return result

    def log_confusion_matrix(self, y_true, y_pred, model_name: str = "") -> np.ndarray:
        cm = confusion_matrix(y_true, y_pred)
        label = f" [{model_name}]" if model_name else ""
        labels = sorted(set(y_true))
        logger.info("─── Confusion Matrix%s ──────────────────", label)
        logger.info("  Predicted →  %s", "  ".join(f"{l:>4}" for l in labels))
        for i, row in enumerate(cm):
            logger.info("  Actual %-4s  %s", labels[i], "  ".join(f"{v:>4}" for v in row))
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        if cm.shape == (2, 2):
            logger.info("  TN=%d  FP=%d  FN=%d  TP=%d", tn, fp, fn, tp)
        return cm

    def classification_report(self, y_true, y_pred) -> str:
        return classification_report(y_true, y_pred)

    def confusion_matrix(self, y_true, y_pred) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)
