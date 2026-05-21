import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# String values that look like valid data but actually represent missing/corrupted values
_CORRUPTED_SENTINELS = frozenset(
    {"nan", "null", "none", "na", "n/a", "?", "-", "--", "missing", "unknown", ""}
)


@dataclass
class ValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def log(self) -> None:
        status = "PASSED" if self.passed else "FAILED"
        logger.info("─── Dataset Validation: %s ─────────────", status)
        for w in self.warnings:
            logger.warning("  ⚠  %s", w)
        for issue in self.issues:
            logger.error("  ✗  %s", issue)
        if self.passed and not self.warnings:
            logger.info("  ✓  All checks passed.")
        logger.info("────────────────────────────────────────")


class DatasetValidator:
    def __init__(self, missing_threshold: float = 0.5) -> None:
        self.missing_threshold = missing_threshold

    def validate(
        self,
        df: pd.DataFrame,
        target_column: str,
        required_columns: list[str] | None = None,
    ) -> bool:
        result = self.full_validate(df, target_column, required_columns)
        result.log()
        return result.passed

    def full_validate(
        self,
        df: pd.DataFrame,
        target_column: str,
        required_columns: list[str] | None = None,
    ) -> ValidationResult:
        issues: list[str] = []
        warnings: list[str] = []

        # ── Critical checks (block training) ──────────────────────────────
        if df.empty:
            issues.append("Dataset is empty.")
            return ValidationResult(passed=False, issues=issues)

        if target_column not in df.columns:
            issues.append(
                f"Target column '{target_column}' not found. "
                f"Available: {df.columns.tolist()}"
            )

        if required_columns:
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                issues.append(f"Required columns missing: {missing_cols}")

        # ── Missing value checks ───────────────────────────────────────────
        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > self.missing_threshold]
        if not high_missing.empty:
            pcts = {col: f"{pct:.0%}" for col, pct in high_missing.items()}
            issues.append(f"Columns exceed {self.missing_threshold:.0%} missing: {pcts}")

        # ── Warnings (non-blocking) ────────────────────────────────────────
        n_duplicates = int(df.duplicated().sum())
        if n_duplicates:
            warnings.append(
                f"{n_duplicates} duplicate rows detected "
                f"({n_duplicates / len(df):.1%} of dataset)."
            )

        corrupted = self._find_corrupted_strings(df)
        if corrupted:
            warnings.append(
                f"Corrupted string sentinels found in columns: {corrupted}. "
                "Use DatasetLoader to clean these automatically."
            )

        if target_column in df.columns:
            n_classes = df[target_column].nunique()
            if n_classes < 2:
                issues.append(
                    f"Target column '{target_column}' has only {n_classes} unique "
                    "value(s). Need at least 2 for classification."
                )

        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            warnings=warnings,
        )

    @staticmethod
    def _find_corrupted_strings(df: pd.DataFrame) -> list[str]:
        affected: list[str] = []
        for col in df.select_dtypes(include="object").columns:
            lowered = df[col].dropna().astype(str).str.strip().str.lower()
            if lowered.isin(_CORRUPTED_SENTINELS).any():
                affected.append(col)
        return affected
