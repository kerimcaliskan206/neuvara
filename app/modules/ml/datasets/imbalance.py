import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ImbalanceReport:
    class_counts: dict[str, int]
    class_ratios: dict[str, float]
    majority_class: str
    minority_class: str
    imbalance_ratio: float       # majority_count / minority_count
    is_imbalanced: bool
    severity: str                # "balanced" | "mild" | "moderate" | "severe"
    recommendation: str


class ImbalanceAnalyzer:
    """
    Detects and classifies class imbalance in a binary or multiclass target.

    Imbalance ratio thresholds:
        ≤ 1.5   → balanced
        1.5–3.0 → mild
        3.0–10  → moderate
        > 10    → severe
    """

    def __init__(self, imbalance_threshold: float = 1.5) -> None:
        self.imbalance_threshold = imbalance_threshold

    def analyze(self, y: pd.Series) -> ImbalanceReport:
        counts = y.value_counts()
        total = len(y)

        majority_class = str(counts.index[0])
        minority_class = str(counts.index[-1])
        majority_count = int(counts.iloc[0])
        minority_count = int(counts.iloc[-1])

        ratio = (
            majority_count / minority_count
            if minority_count > 0
            else float("inf")
        )

        severity, recommendation = self._classify(ratio)

        return ImbalanceReport(
            class_counts={str(k): int(v) for k, v in counts.items()},
            class_ratios={
                str(k): round(int(v) / total * 100, 2)
                for k, v in counts.items()
            },
            majority_class=majority_class,
            minority_class=minority_class,
            imbalance_ratio=round(ratio, 2),
            is_imbalanced=ratio > self.imbalance_threshold,
            severity=severity,
            recommendation=recommendation,
        )

    def log_report(self, report: ImbalanceReport) -> None:
        logger.info("─── Class Imbalance Report ──────────────")
        for cls, cnt in report.class_counts.items():
            pct = report.class_ratios[cls]
            bar = "█" * int(pct / 5)
            logger.info(
                "  Class %-10s  %5d  (%5.1f%%)  %s",
                cls, cnt, pct, bar,
            )
        logger.info(
            "  Imbalance ratio: %.2f:1  (majority:minority)", report.imbalance_ratio
        )
        logger.info("  Severity: %s", report.severity.upper())
        if report.is_imbalanced:
            logger.warning("  ⚠  %s", report.recommendation)
        else:
            logger.info("  ✓  %s", report.recommendation)
        logger.info("────────────────────────────────────────")

    @staticmethod
    def _classify(ratio: float) -> tuple[str, str]:
        if ratio <= 1.5:
            return (
                "balanced",
                "No action needed. Dataset is well-balanced.",
            )
        if ratio <= 3.0:
            return (
                "mild",
                "Use class_weight='balanced' in your model.",
            )
        if ratio <= 10.0:
            return (
                "moderate",
                "Use class_weight='balanced'. Evaluate with F1/ROC-AUC, not accuracy.",
            )
        return (
            "severe",
            "Consider SMOTE oversampling. Use class_weight='balanced'. "
            "Optimize threshold for recall on the minority class.",
        )
