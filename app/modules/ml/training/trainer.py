import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sklearn.calibration import CalibratedClassifierCV

from app.modules.ml.config import MLConfig, ml_config
from app.modules.ml.datasets.imbalance import ImbalanceAnalyzer
from app.modules.ml.datasets.loader import DatasetLoader
from app.modules.ml.datasets.statistics import DatasetStatistics
from app.modules.ml.datasets.validator import DatasetValidator
from app.modules.ml.ensemble.models import (
    EnsembleFactory,
    ModelType,
    build_lightgbm,
    build_random_forest,
    build_voting_ensemble,
    build_xgboost,
)
from app.modules.ml.evaluation.comparison import ComparisonResult, ModelComparer
from app.modules.ml.evaluation.metrics import EvaluationResult, ModelEvaluator
from app.modules.ml.persistence.model_store import ModelStore
from app.modules.ml.preprocessing.pipeline import PreprocessingPipeline
from app.modules.ml.preprocessing.splitters import DataSplit
from app.modules.ml.training.report import TrainingReport
from app.modules.ml.utils.logging import log_duration

logger = logging.getLogger(__name__)

# Models trained in every full run
_DEFAULT_BASE_MODELS = [
    ModelType.RANDOM_FOREST,
    ModelType.XGBOOST,
    ModelType.LIGHTGBM,
]


@dataclass
class SingleTrainingResult:
    model_type: str
    version: str
    model_path: str
    metrics: EvaluationResult


class ModelTrainer:
    """
    Orchestrates the complete training lifecycle:
      load → validate → analyze → preprocess → train → evaluate → compare → save → report
    """

    def __init__(self, config: MLConfig = ml_config) -> None:
        self.config = config
        self.loader = DatasetLoader(config)
        self.validator = DatasetValidator(
            missing_threshold=config.preprocessing.missing_threshold
        )
        self.stats = DatasetStatistics()
        self.imbalance_analyzer = ImbalanceAnalyzer()
        self.pipeline = PreprocessingPipeline(config)
        self.evaluator = ModelEvaluator()
        self.comparer = ModelComparer()
        self.store = ModelStore(config)

    # ── Public API ───────────────────────────────────────────────────────────

    def train_all(self, dataset_filename: str) -> TrainingReport:
        """
        Trains RF, XGBoost, LightGBM, and a Voting ensemble.
        All models share the same version timestamp and preprocessing pipeline.
        Returns a full TrainingReport.
        """
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logger.info("Training run started — version: %s", version)

        with log_duration("full training pipeline"):
            # ── 1. Load ──────────────────────────────────────────────────
            df = self.loader.load_raw(dataset_filename)

            # ── 2. Validate ──────────────────────────────────────────────
            validation = self.validator.full_validate(
                df,
                self.config.target_column,
                self.config.columns.required or None,
            )
            validation.log()
            if not validation.passed:
                raise ValueError(
                    f"Dataset '{dataset_filename}' failed validation. "
                    "Fix the issues above before training."
                )

            # ── 3. Analyze ───────────────────────────────────────────────
            self.stats.log_full(df, self.config.target_column)
            imbalance_report = self.imbalance_analyzer.analyze(df[self.config.target_column])
            self.imbalance_analyzer.log_report(imbalance_report)
            class_dist = self.stats.class_distribution(df[self.config.target_column])

            # ── 4. Preprocess ────────────────────────────────────────────
            split = self.pipeline.fit_transform(df)
            logger.info(
                "Split: %d train / %d test | %d features",
                split.train_size, split.test_size, split.X_train.shape[1],
            )

            # ── 5. Train base models + evaluate each ─────────────────────
            all_results: dict[str, EvaluationResult] = {}
            fitted_models: dict[str, object] = {}

            for model_type in _DEFAULT_BASE_MODELS:
                model, metrics = self._train_single(
                    split, model_type, version
                )
                all_results[model_type.value] = metrics
                fitted_models[model_type.value] = model

            # ── 6. Voting ensemble (fresh estimators — sklearn requirement) ─
            logger.info("Building Voting ensemble from fresh estimators...")
            voting = self._build_voting_from_config()
            voting_metrics = self._fit_and_evaluate(
                voting, split, ModelType.VOTING.value
            )
            # Platt calibration on held-out test set.
            # cv="prefit" because voting is already fitted on training data;
            # the test set provides a clean calibration signal never seen during fit.
            cal_voting = CalibratedClassifierCV(
                estimator=voting, cv="prefit", method="sigmoid"
            )
            cal_voting.fit(split.X_test, split.y_test)
            self.store.save(cal_voting, ModelType.VOTING.value, version=version)
            all_results[ModelType.VOTING.value] = voting_metrics

            # ── 7. Save preprocessing pipeline ───────────────────────────
            pipeline_path = (
                self.config.storage.models_dir
                / f"v{version}"
                / "preprocessing_pipeline.joblib"
            )
            self.pipeline.save(pipeline_path)

            # ── 8. Compare ───────────────────────────────────────────────
            comparison = self.comparer.compare(all_results)
            self.comparer.log_comparison(comparison)

            # ── 9. Report ────────────────────────────────────────────────
            report = TrainingReport(
                timestamp=version,
                dataset=dataset_filename,
                version=version,
                target_column=self.config.target_column,
                n_samples_train=split.train_size,
                n_samples_test=split.test_size,
                n_features=split.X_train.shape[1],
                class_distribution=class_dist,
                imbalance_ratio=imbalance_report.imbalance_ratio,
                model_results=all_results,
                best_model_name=comparison.best_model_name,
            )
            report.log()

        return report

    def train_single(
        self, dataset_filename: str, model_type: ModelType
    ) -> SingleTrainingResult:
        """Trains and evaluates a single model type."""
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        df = self.loader.load_raw(dataset_filename)
        if not self.validator.validate(df, self.config.target_column):
            raise ValueError("Dataset failed validation.")
        split = self.pipeline.fit_transform(df)
        model, metrics = self._train_single(split, model_type, version)
        pipeline_path = (
            self.config.storage.models_dir
            / f"v{version}"
            / "preprocessing_pipeline.joblib"
        )
        self.pipeline.save(pipeline_path)
        return SingleTrainingResult(
            model_type=model_type.value,
            version=version,
            model_path=str(
                self.config.storage.models_dir / f"v{version}" / f"{model_type.value}.joblib"
            ),
            metrics=metrics,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _train_single(
        self, split: DataSplit, model_type: ModelType, version: str
    ) -> tuple:
        hp = self.config.hyperparams
        rs = hp.random_state

        if model_type == ModelType.RANDOM_FOREST:
            model = build_random_forest(
                random_state=rs, **hp.random_forest.model_dump(exclude_none=True)
            )
        elif model_type == ModelType.XGBOOST:
            model = build_xgboost(
                random_state=rs, **hp.xgboost.model_dump(exclude_none=True)
            )
        elif model_type == ModelType.LIGHTGBM:
            model = build_lightgbm(
                random_state=rs, **hp.lightgbm.model_dump(exclude_none=True)
            )
        else:
            model = EnsembleFactory.create(model_type)

        metrics = self._fit_and_evaluate(model, split, model_type.value)
        self.store.save(model, model_type.value, version=version)
        return model, metrics

    def _fit_and_evaluate(self, model, split: DataSplit, name: str) -> EvaluationResult:
        with log_duration(f"fit [{name}]"):
            model.fit(split.X_train, split.y_train)

        y_pred = model.predict(split.X_test)
        y_prob = (
            model.predict_proba(split.X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        metrics = self.evaluator.evaluate(
            split.y_test, y_pred, y_prob, model_name=name
        )
        self.evaluator.log_confusion_matrix(split.y_test, y_pred, model_name=name)
        return metrics

    def _build_voting_from_config(self):
        hp = self.config.hyperparams
        rs = hp.random_state
        return build_voting_ensemble([
            ("rf", build_random_forest(random_state=rs, **hp.random_forest.model_dump(exclude_none=True))),
            ("xgb", build_xgboost(random_state=rs, **hp.xgboost.model_dump(exclude_none=True))),
            ("lgbm", build_lightgbm(random_state=rs, **hp.lightgbm.model_dump(exclude_none=True))),
        ])
