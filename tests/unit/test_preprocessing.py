"""
Unit tests for the ML preprocessing and dataset utilities.

All tests are synchronous and use synthetic DataFrames —
no database, no disk IO, no HTTP.
"""
import numpy as np
import pandas as pd
import pytest

from app.modules.ml.datasets.imbalance import ImbalanceAnalyzer
from app.modules.ml.datasets.statistics import DatasetStatistics
from app.modules.ml.datasets.validator import DatasetValidator
from app.modules.ml.preprocessing.cleaners import MissingValueHandler
from app.modules.ml.preprocessing.encoders import CategoricalEncoder
from app.modules.ml.preprocessing.feature_selection import FeatureSelector
from app.modules.ml.preprocessing.scalers import FeatureScaler
from app.modules.ml.preprocessing.splitters import DataSplitter


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def numeric_df_with_nulls() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25.0, None, 35.0, 40.0, None],
        "temp": [36.5, 37.0, None, 38.0, 36.8],
        "bp": [120.0, 130.0, 125.0, None, 118.0],
    })


@pytest.fixture
def categorical_df_with_nulls() -> pd.DataFrame:
    # Use np.nan (not None) — matches what pandas produces when loading from CSV
    return pd.DataFrame({
        "region": ["urban", np.nan, "rural", "urban", "rural"],
        "season": ["summer", "winter", np.nan, "spring", "winter"],
    })


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        "region": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def imbalanced_series() -> pd.Series:
    # 95:5 = 19:1 ratio → classified as "severe" (>10:1 threshold)
    return pd.Series([0] * 95 + [1] * 5, name="label")


@pytest.fixture
def balanced_series() -> pd.Series:
    return pd.Series([0] * 50 + [1] * 50, name="label")


# ── MissingValueHandler ───────────────────────────────────────────────────────

class TestMissingValueHandler:
    def test_fills_numeric_nulls(self, numeric_df_with_nulls):
        handler = MissingValueHandler()
        result = handler.fit_transform(numeric_df_with_nulls)
        assert result.isnull().sum().sum() == 0

    def test_numeric_strategy_median(self, numeric_df_with_nulls):
        handler = MissingValueHandler(numeric_strategy="median")
        result = handler.fit_transform(numeric_df_with_nulls)
        # median of [25, 35, 40] = 35 — NaN in 'age' should become 35
        assert result["age"].iloc[1] == pytest.approx(35.0)

    def test_fills_categorical_nulls(self, categorical_df_with_nulls):
        handler = MissingValueHandler()
        result = handler.fit_transform(categorical_df_with_nulls)
        assert result.isnull().sum().sum() == 0

    def test_transform_without_fit_uses_fitted_values(self, numeric_df_with_nulls):
        handler = MissingValueHandler()
        handler.fit(numeric_df_with_nulls)
        new_data = pd.DataFrame({
            "age": [None],
            "temp": [37.0],
            "bp": [None],
        })
        result = handler.transform(new_data)
        assert result.isnull().sum().sum() == 0

    def test_no_change_when_no_nulls(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        handler = MissingValueHandler()
        result = handler.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)


# ── CategoricalEncoder ────────────────────────────────────────────────────────

class TestCategoricalEncoder:
    def test_onehot_creates_binary_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
        enc = CategoricalEncoder(strategy="onehot")
        result = enc.fit_transform(df)
        # original column gone, new columns created
        assert "color" not in result.columns
        assert any("color" in c for c in result.columns)
        assert result.shape[1] == 3  # red, blue, green

    def test_onehot_values_are_binary(self):
        df = pd.DataFrame({"color": ["red", "blue", "red"]})
        enc = CategoricalEncoder(strategy="onehot")
        result = enc.fit_transform(df)
        numeric_vals = result.values.flatten()
        assert set(numeric_vals).issubset({0.0, 1.0})

    def test_label_encoding_produces_integers(self):
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        enc = CategoricalEncoder(strategy="label")
        result = enc.fit_transform(df)
        assert result["status"].dtype != object

    def test_label_encoding_preserves_column(self):
        df = pd.DataFrame({"status": ["a", "b", "a"]})
        enc = CategoricalEncoder(strategy="label")
        result = enc.fit_transform(df)
        assert "status" in result.columns

    def test_handles_unseen_categories_gracefully(self):
        df_train = pd.DataFrame({"color": ["red", "blue"]})
        df_test = pd.DataFrame({"color": ["green"]})
        enc = CategoricalEncoder(strategy="onehot")
        enc.fit(df_train)
        result = enc.transform(df_test)
        # unseen → all zeros (handle_unknown="ignore")
        assert result.isnull().sum().sum() == 0

    def test_no_categorical_columns_passthrough(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        enc = CategoricalEncoder()
        result = enc.fit_transform(df)
        pd.testing.assert_frame_equal(result, df)


# ── FeatureScaler ─────────────────────────────────────────────────────────────

class TestFeatureScaler:
    def test_standard_scaler_zero_mean(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        scaler = FeatureScaler(scaler_type="standard")
        result = scaler.fit_transform(df)
        assert abs(result["x"].mean()) < 1e-10

    def test_standard_scaler_unit_std(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        scaler = FeatureScaler(scaler_type="standard")
        result = scaler.fit_transform(df)
        # sklearn StandardScaler uses population std (ddof=0); pandas .std() defaults to ddof=1
        assert abs(result["x"].std(ddof=0) - 1.0) < 1e-10

    def test_minmax_scaler_range_zero_to_one(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        scaler = FeatureScaler(scaler_type="minmax")
        result = scaler.fit_transform(df)
        assert result["x"].min() == pytest.approx(0.0)
        assert result["x"].max() == pytest.approx(1.0)

    def test_robust_scaler_runs_without_error(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 100.0, 3.0, 4.0]})  # outlier at 100
        scaler = FeatureScaler(scaler_type="robust")
        result = scaler.fit_transform(df)
        assert result.isnull().sum().sum() == 0

    def test_unknown_scaler_raises_value_error(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        scaler = FeatureScaler(scaler_type="nonexistent")
        with pytest.raises(ValueError, match="Unknown scaler"):
            scaler.fit(df)

    def test_transform_uses_fit_statistics(self):
        train = pd.DataFrame({"x": [0.0, 10.0]})
        test = pd.DataFrame({"x": [5.0]})
        scaler = FeatureScaler(scaler_type="minmax")
        scaler.fit(train)
        result = scaler.transform(test)
        assert result["x"].iloc[0] == pytest.approx(0.5)


# ── DataSplitter ──────────────────────────────────────────────────────────────

class TestDataSplitter:
    def test_correct_row_counts(self, mixed_df):
        splitter = DataSplitter(test_size=0.25)
        split = splitter.split(mixed_df, "label")
        assert split.train_size + split.test_size == len(mixed_df)

    def test_test_size_ratio(self, mixed_df):
        splitter = DataSplitter(test_size=0.25)
        split = splitter.split(mixed_df, "label")
        assert split.test_size == 2
        assert split.train_size == 6

    def test_target_not_in_features(self, mixed_df):
        splitter = DataSplitter()
        split = splitter.split(mixed_df, "label")
        assert "label" not in split.X_train.columns
        assert "label" not in split.X_test.columns

    def test_stratified_preserves_class_ratio(self):
        df = pd.DataFrame({
            "x": range(100),
            "label": [0] * 70 + [1] * 30,
        })
        splitter = DataSplitter(test_size=0.2, stratify=True, random_state=42)
        split = splitter.split(df, "label")
        train_ratio = split.y_train.mean()
        test_ratio = split.y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05

    def test_reproducible_with_same_seed(self, mixed_df):
        s1 = DataSplitter(random_state=42).split(mixed_df, "label")
        s2 = DataSplitter(random_state=42).split(mixed_df, "label")
        pd.testing.assert_frame_equal(s1.X_train, s2.X_train)


# ── DatasetValidator ──────────────────────────────────────────────────────────

class TestDatasetValidator:
    def test_rejects_empty_dataframe(self):
        validator = DatasetValidator()
        assert not validator.validate(pd.DataFrame(), "label")

    def test_rejects_missing_target_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        validator = DatasetValidator()
        assert not validator.validate(df, "label")

    def test_rejects_high_missing_values(self):
        df = pd.DataFrame({
            "a": [None] * 9 + [1.0],  # 90% missing
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        validator = DatasetValidator(missing_threshold=0.5)
        assert not validator.validate(df, "label")

    def test_rejects_single_class_target(self):
        df = pd.DataFrame({"x": [1, 2, 3], "label": [0, 0, 0]})
        validator = DatasetValidator()
        assert not validator.validate(df, "label")

    def test_accepts_valid_dataset(self):
        df = pd.DataFrame({
            "age": [25.0, 30.0, 35.0],
            "region": ["A", "B", "A"],
            "label": [0, 1, 0],
        })
        validator = DatasetValidator()
        assert validator.validate(df, "label")

    def test_warns_on_duplicates(self, capsys):
        df = pd.DataFrame({"x": [1, 1, 2], "label": [0, 0, 1]})
        validator = DatasetValidator()
        result = validator.full_validate(df, "label")
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_warns_on_corrupted_sentinels(self):
        df = pd.DataFrame({
            "region": ["urban", "?", "rural"],
            "label": [0, 1, 0],
        })
        validator = DatasetValidator()
        result = validator.full_validate(df, "label")
        assert any("corrupted" in w.lower() for w in result.warnings)

    def test_required_columns_check(self):
        df = pd.DataFrame({"x": [1, 2], "label": [0, 1]})
        validator = DatasetValidator()
        result = validator.full_validate(df, "label", required_columns=["x", "y", "z"])
        assert not result.passed
        assert any("missing" in issue.lower() for issue in result.issues)


# ── DatasetStatistics ─────────────────────────────────────────────────────────

class TestDatasetStatistics:
    def test_summary_row_count(self, mixed_df):
        stats = DatasetStatistics()
        s = stats.summary(mixed_df)
        assert s.n_rows == len(mixed_df)
        assert s.n_cols == len(mixed_df.columns)

    def test_summary_detects_duplicates(self):
        df = pd.DataFrame({"x": [1, 1, 2], "y": [1, 1, 3]})
        stats = DatasetStatistics()
        s = stats.summary(df)
        assert s.n_duplicates == 1

    def test_null_report_sorted_worst_first(self):
        df = pd.DataFrame({
            "a": [None] * 8 + [1.0, 1.0],     # 80% missing
            "b": [None] * 3 + [1.0] * 7,       # 30% missing
            "c": [1.0] * 10,                   # 0% missing
        })
        stats = DatasetStatistics()
        report = stats.null_report(df)
        assert report[0].column == "a"
        assert report[0].null_pct == pytest.approx(80.0)
        assert len(report) == 2  # 'c' has no nulls

    def test_class_distribution_sums_to_100(self):
        y = pd.Series([0, 0, 0, 1, 1, 1, 1])
        stats = DatasetStatistics()
        dist = stats.class_distribution(y)
        total_pct = sum(v["percentage"] for v in dist.values())
        assert total_pct == pytest.approx(100.0)

    def test_dtype_report_groups_correctly(self, mixed_df):
        stats = DatasetStatistics()
        report = stats.dtype_report(mixed_df)
        assert "age" in report["numeric"]
        assert "region" in report["categorical"]


# ── ImbalanceAnalyzer ─────────────────────────────────────────────────────────

class TestImbalanceAnalyzer:
    def test_detects_severe_imbalance(self, imbalanced_series):
        analyzer = ImbalanceAnalyzer()
        report = analyzer.analyze(imbalanced_series)
        assert report.severity == "severe"
        assert report.is_imbalanced is True
        assert report.imbalance_ratio == pytest.approx(19.0)

    def test_balanced_dataset_not_flagged(self, balanced_series):
        analyzer = ImbalanceAnalyzer()
        report = analyzer.analyze(balanced_series)
        assert report.severity == "balanced"
        assert report.is_imbalanced is False

    def test_majority_minority_correct(self, imbalanced_series):
        analyzer = ImbalanceAnalyzer()
        report = analyzer.analyze(imbalanced_series)
        assert report.majority_class == "0"
        assert report.minority_class == "1"

    def test_class_ratios_sum_to_100(self, imbalanced_series):
        analyzer = ImbalanceAnalyzer()
        report = analyzer.analyze(imbalanced_series)
        total = sum(report.class_ratios.values())
        assert total == pytest.approx(100.0)


# ── FeatureSelector ───────────────────────────────────────────────────────────

class TestFeatureSelector:
    def test_removes_zero_variance_column(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "constant": [7.0, 7.0, 7.0, 7.0, 7.0],  # zero variance
        })
        selector = FeatureSelector(variance_threshold=0.01)
        result = selector.fit_transform(df)
        assert "constant" not in result.columns
        assert "x" in result.columns

    def test_removes_highly_correlated_column(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        df = pd.DataFrame({
            "x": x,
            "x_copy": x * 1.0001,  # nearly perfect correlation
            "y": rng.standard_normal(100),
        })
        selector = FeatureSelector(
            variance_threshold=0.0,
            correlation_threshold=0.99,
        )
        result = selector.fit_transform(df)
        # x and x_copy are correlated — one should be dropped
        assert "x_copy" not in result.columns

    def test_top_k_limits_features(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"f{i}": rng.standard_normal(50) for i in range(10)})
        selector = FeatureSelector(variance_threshold=0.0, top_k=3)
        result = selector.fit_transform(df)
        assert result.shape[1] == 3

    def test_non_numeric_columns_preserved(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "c"],
        })
        selector = FeatureSelector(variance_threshold=0.0)
        result = selector.fit_transform(df)
        assert "cat" in result.columns

    def test_get_removed_features_reports_correctly(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "const": [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        selector = FeatureSelector(variance_threshold=0.01)
        selector.fit(df)
        removed = selector.get_removed_features()
        assert "const" in removed["low_variance"]
