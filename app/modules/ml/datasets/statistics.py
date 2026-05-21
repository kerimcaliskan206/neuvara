import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NullEntry:
    column: str
    null_count: int
    null_pct: float


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    n_duplicates: int
    memory_mb: float
    columns: list[str]
    dtypes: dict[str, str]
    null_counts: dict[str, int]
    null_percentages: dict[str, float]

    @property
    def has_missing(self) -> bool:
        return any(v > 0 for v in self.null_counts.values())

    @property
    def has_duplicates(self) -> bool:
        return self.n_duplicates > 0


class DatasetStatistics:
    """Produces descriptive reports about a raw or processed DataFrame."""

    def summary(self, df: pd.DataFrame) -> DatasetSummary:
        null_counts = {col: int(df[col].isnull().sum()) for col in df.columns}
        n = len(df)
        null_pcts = {
            col: round(cnt / n * 100, 2) if n > 0 else 0.0
            for col, cnt in null_counts.items()
        }
        return DatasetSummary(
            n_rows=n,
            n_cols=len(df.columns),
            n_duplicates=int(df.duplicated().sum()),
            memory_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 4),
            columns=df.columns.tolist(),
            dtypes={col: str(dt) for col, dt in df.dtypes.items()},
            null_counts=null_counts,
            null_percentages=null_pcts,
        )

    def null_report(self, df: pd.DataFrame) -> list[NullEntry]:
        """Returns columns with missing values, sorted worst-first."""
        entries = []
        n = len(df)
        for col in df.columns:
            cnt = int(df[col].isnull().sum())
            if cnt > 0:
                entries.append(
                    NullEntry(
                        column=col,
                        null_count=cnt,
                        null_pct=round(cnt / n * 100, 2) if n > 0 else 0.0,
                    )
                )
        return sorted(entries, key=lambda e: e.null_pct, reverse=True)

    def dtype_report(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Groups column names by their inferred dtype category."""
        report: dict[str, list[str]] = {
            "numeric": df.select_dtypes(include="number").columns.tolist(),
            "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "boolean": df.select_dtypes(include="bool").columns.tolist(),
            "datetime": df.select_dtypes(include="datetime").columns.tolist(),
        }
        return {k: v for k, v in report.items() if v}

    def class_distribution(self, y: pd.Series) -> dict[str, dict]:
        """Returns per-class count and percentage for a target series."""
        counts = y.value_counts()
        total = len(y)
        return {
            str(cls): {
                "count": int(cnt),
                "percentage": round(cnt / total * 100, 2),
            }
            for cls, cnt in counts.items()
        }

    def log_full(self, df: pd.DataFrame, target_column: str | None = None) -> None:
        s = self.summary(df)
        logger.info("─── Dataset Summary ─────────────────────")
        logger.info("  Rows:        %d", s.n_rows)
        logger.info("  Columns:     %d", s.n_cols)
        logger.info("  Duplicates:  %d", s.n_duplicates)
        logger.info("  Memory:      %.4f MB", s.memory_mb)

        null_entries = self.null_report(df)
        if null_entries:
            logger.info("  Missing values:")
            for entry in null_entries:
                logger.info(
                    "    %-30s %5d  (%5.1f%%)",
                    entry.column, entry.null_count, entry.null_pct,
                )
        else:
            logger.info("  Missing values: none")

        dtype_groups = self.dtype_report(df)
        logger.info("  Dtypes:      %s", {k: len(v) for k, v in dtype_groups.items()})

        if target_column and target_column in df.columns:
            dist = self.class_distribution(df[target_column])
            logger.info("  Class distribution ('%s'):", target_column)
            for cls, info in dist.items():
                bar = "█" * int(info["percentage"] / 5)
                logger.info(
                    "    %-10s %5d  (%5.1f%%)  %s",
                    cls, info["count"], info["percentage"], bar,
                )
        logger.info("─────────────────────────────────────────")
