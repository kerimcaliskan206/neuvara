import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.modules.ml.config import MLConfig, ml_config

logger = logging.getLogger(__name__)

# String values that masquerade as real data but mean "missing"
_CORRUPTED_SENTINELS = ["nan", "null", "none", "na", "n/a", "?", "-", "--", "missing", "unknown"]

_READERS: dict[str, Any] = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}


class DatasetLoader:
    def __init__(self, config: MLConfig = ml_config) -> None:
        self.config = config

    def load_raw(self, filename: str, sep: str = ",", clean: bool = True) -> pd.DataFrame:
        df = self._load(self.config.data.raw_dir / filename, sep=sep)
        if clean:
            df = self._clean_sentinels(df)
        if self.config.columns.drop_on_load:
            df = self._drop_columns(df, self.config.columns.drop_on_load)
        return df

    def load_processed(self, filename: str) -> pd.DataFrame:
        return self._load(self.config.data.processed_dir / filename)

    def save_processed(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.config.data.processed_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved processed dataset → %s (%d rows)", path.name, len(df))
        return path

    # ── Internal helpers ────────────────────────────────────────────────────

    def _load(self, path: Path, sep: str = ",") -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                f"Place your CSV file in: {path.parent}"
            )
        reader = _READERS.get(path.suffix.lower())
        if reader is None:
            raise ValueError(
                f"Unsupported format: '{path.suffix}'. "
                f"Supported: {list(_READERS)}"
            )
        kwargs: dict[str, Any] = {}
        if path.suffix.lower() == ".csv":
            kwargs["sep"] = sep

        try:
            df = reader(path, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to read dataset '{path.name}': {exc}") from exc

        logger.info(
            "Loaded → %s | %d rows × %d cols", path.name, *df.shape
        )
        return df

    @staticmethod
    def _clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
        """Replace corrupted string sentinels with NaN."""
        sentinel_set = frozenset(_CORRUPTED_SENTINELS)
        str_cols = df.select_dtypes(include="object").columns
        if str_cols.empty:
            return df
        cleaned = df.copy()
        replaced = 0
        for col in str_cols:
            # normalize to lowercase stripped string, then check membership
            mask = cleaned[col].astype(str).str.strip().str.lower().isin(sentinel_set)
            replaced += int(mask.sum())
            cleaned.loc[mask, col] = float("nan")
        if replaced:
            logger.info("Cleaned %d corrupted sentinel values → NaN.", replaced)
        return cleaned

    @staticmethod
    def _drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        existing = [c for c in columns if c in df.columns]
        if existing:
            logger.info("Dropped configured columns: %s", existing)
            return df.drop(columns=existing)
        return df
