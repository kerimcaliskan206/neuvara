"""
Ensemble weight computation utilities.

Weights are used by sklearn's VotingClassifier(weights=...) to give
better-performing models more influence over the final probability estimate.
"""
import logging
import math
from dataclasses import dataclass

from app.modules.ml.evaluation.metrics import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ModelWeight:
    model_name: str
    weight: float
    metric_score: float


# ── Weight strategies ────────────────────────────────────────────────────────

def compute_f1_weights(results: dict[str, EvaluationResult]) -> list[ModelWeight]:
    """Weights proportional to each model's F1 score."""
    total = sum(r.f1 for r in results.values())
    if total == 0:
        return _equal_weights(results, "f1")
    return [
        ModelWeight(name, round(r.f1 / total, 4), r.f1)
        for name, r in results.items()
    ]


def compute_roc_auc_weights(results: dict[str, EvaluationResult]) -> list[ModelWeight]:
    """Weights proportional to each model's ROC-AUC score."""
    total = sum(r.roc_auc for r in results.values())
    if total == 0:
        return _equal_weights(results, "roc_auc")
    return [
        ModelWeight(name, round(r.roc_auc / total, 4), r.roc_auc)
        for name, r in results.items()
    ]


def compute_softmax_weights(
    results: dict[str, EvaluationResult],
    metric: str = "f1",
    temperature: float = 1.0,
) -> list[ModelWeight]:
    """
    Softmax-scaled weights.
    Lower temperature → sharper (winner-take-more) distribution.
    Higher temperature → flatter (closer to equal) distribution.
    """
    scores = {name: getattr(r, metric, r.f1) for name, r in results.items()}
    scaled = {name: s / temperature for name, s in scores.items()}
    max_val = max(scaled.values())
    exps = {name: math.exp(s - max_val) for name, s in scaled.items()}
    total = sum(exps.values())
    return [
        ModelWeight(name, round(exps[name] / total, 4), scores[name])
        for name in results
    ]


def weights_as_list(weights: list[ModelWeight], model_order: list[str]) -> list[float]:
    """Return weights in the order expected by sklearn VotingClassifier."""
    weight_map = {w.model_name: w.weight for w in weights}
    return [weight_map[name] for name in model_order]


def log_weights(weights: list[ModelWeight], strategy: str = "") -> None:
    label = f" ({strategy})" if strategy else ""
    logger.info("─── Ensemble Weights%s ──────────────────────────────", label)
    for w in sorted(weights, key=lambda x: x.weight, reverse=True):
        bar = "█" * int(w.weight * 30)
        logger.info("  %-22s %.4f  %s", w.model_name, w.weight, bar)
    logger.info("────────────────────────────────────────────────────")


# ── Internal ─────────────────────────────────────────────────────────────────

def _equal_weights(results: dict[str, EvaluationResult], attr: str) -> list[ModelWeight]:
    equal = round(1.0 / len(results), 4)
    return [
        ModelWeight(name, equal, getattr(r, attr, 0.0))
        for name, r in results.items()
    ]
