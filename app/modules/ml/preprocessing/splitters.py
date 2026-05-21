import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)


class DataSplitter:
    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self, df: pd.DataFrame, target_column: str) -> DataSplit:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        stratify_arg = y if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_arg,
        )
        logger.info(
            "Split: train=%d (%.0f%%), test=%d (%.0f%%)",
            len(X_train),
            (1 - self.test_size) * 100,
            len(X_test),
            self.test_size * 100,
        )
        return DataSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
