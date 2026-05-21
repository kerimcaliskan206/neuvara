"""
Vision model evaluation metrics.

Computes accuracy, per-class precision/recall/F1, macro-averaged F1,
and the confusion matrix using sklearn.  All computation happens on CPU;
the model runs on whatever device it was placed on.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Full metrics for one evaluation pass."""

    accuracy: float
    precision: float          # macro average
    recall: float             # macro average
    f1: float                 # macro average
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float]]   # class → {precision, recall, f1, support}
    n_samples: int
    class_names: list[str] = field(default_factory=list)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary_line(self) -> str:
        return (
            f"acc={self.accuracy:.4f}  "
            f"prec={self.precision:.4f}  "
            f"rec={self.recall:.4f}  "
            f"f1={self.f1:.4f}  "
            f"n={self.n_samples}"
        )

    def log(self, prefix: str = "") -> None:
        tag = f"[{prefix}] " if prefix else ""
        logger.info("%sEvaluation: %s", tag, self.summary_line())
        for cls, m in self.per_class.items():
            logger.info(
                "%s  %-12s  prec=%.3f  rec=%.3f  f1=%.3f  n=%d",
                tag, cls, m["precision"], m["recall"], m["f1"], int(m["support"]),
            )
        cm = self.confusion_matrix
        logger.info("%s  Confusion matrix:", tag)
        for row in cm:
            logger.info("%s    %s", tag, "  ".join(f"{v:5d}" for v in row))

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "n_samples": self.n_samples,
            "class_names": self.class_names,
            "confusion_matrix": self.confusion_matrix,
            "per_class": self.per_class,
        }


class VisionEvaluator:
    """
    Evaluates a trained model on a DataLoader.

    Parameters
    ----------
    class_names : list[str]
        Human-readable class labels in index order.
    """

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names

    # ── Main entry point ──────────────────────────────────────────────────────

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device | str,
        threshold: float | None = None,
    ) -> EvaluationResult:
        """
        Run a full evaluation pass.

        Parameters
        ----------
        model : nn.Module
            Trained model (will be set to eval mode).
        dataloader : DataLoader
            Labeled dataset loader (returns image tensors and integer labels).
        device : str | torch.device
            Device to run inference on.
        threshold : float | None
            Confidence threshold.  Samples whose max-softmax probability is
            below this value are excluded from metric computation.
            Pass None to include all samples.

        Returns
        -------
        EvaluationResult
        """
        device = torch.device(device) if isinstance(device, str) else device
        model.eval()

        all_preds: list[int] = []
        all_labels: list[int] = []
        all_confidences: list[float] = []
        n_rejected = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probs = F.softmax(logits, dim=1).cpu()
                confidences, preds = probs.max(dim=1)

                for pred, label, conf in zip(
                    preds.tolist(), labels.tolist(), confidences.tolist()
                ):
                    if threshold is not None and conf < threshold:
                        n_rejected += 1
                        continue
                    all_preds.append(pred)
                    all_labels.append(label)
                    all_confidences.append(conf)

        if not all_preds:
            logger.warning(
                "VisionEvaluator: all %d samples were rejected by threshold=%.2f",
                n_rejected, threshold or 0,
            )
            return self._empty_result(n_rejected)

        if n_rejected:
            logger.info(
                "VisionEvaluator: threshold=%.2f rejected %d/%d samples (%.1f%%)",
                threshold, n_rejected,
                n_rejected + len(all_preds),
                100 * n_rejected / (n_rejected + len(all_preds)),
            )

        return self._compute(all_preds, all_labels)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, preds: list[int], labels: list[int]) -> EvaluationResult:
        y_true = np.array(labels)
        y_pred = np.array(preds)

        acc = float(accuracy_score(y_true, y_pred))

        prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
            y_true, y_pred,
            average=None,
            labels=list(range(len(self.class_names))),
            zero_division=0,
        )
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        cm = confusion_matrix(
            y_true, y_pred, labels=list(range(len(self.class_names)))
        ).tolist()

        per_class = {
            cls: {
                "precision": round(float(prec_arr[i]), 4),
                "recall": round(float(rec_arr[i]), 4),
                "f1": round(float(f1_arr[i]), 4),
                "support": int(sup_arr[i]),
            }
            for i, cls in enumerate(self.class_names)
        }

        return EvaluationResult(
            accuracy=round(acc, 4),
            precision=round(float(prec_macro), 4),
            recall=round(float(rec_macro), 4),
            f1=round(float(f1_macro), 4),
            confusion_matrix=cm,
            per_class=per_class,
            n_samples=len(preds),
            class_names=self.class_names,
        )

    def _empty_result(self, n_samples: int) -> EvaluationResult:
        n = len(self.class_names)
        return EvaluationResult(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            confusion_matrix=[[0] * n for _ in range(n)],
            per_class={c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for c in self.class_names},
            n_samples=n_samples,
            class_names=self.class_names,
        )
