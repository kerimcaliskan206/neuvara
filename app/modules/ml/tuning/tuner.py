import logging
from dataclasses import dataclass

from app.modules.ml.config import MLConfig, ml_config
from app.modules.ml.ensemble.models import (
    ModelType,
    build_lightgbm,
    build_random_forest,
    build_xgboost,
)
from app.modules.ml.evaluation.metrics import EvaluationResult, ModelEvaluator
from app.modules.ml.persistence.model_store import ModelStore
from app.modules.ml.preprocessing.splitters import DataSplit
from app.modules.ml.tuning.config import TuningConfig
from app.modules.ml.tuning.objectives import (
    make_lgbm_objective,
    make_rf_objective,
    make_xgb_objective,
)
from app.modules.ml.tuning.optimizer import HyperparameterOptimizer, OptimizationResult
from app.modules.ml.utils.logging import log_duration

logger = logging.getLogger(__name__)

_TUNABLE_MODELS = [
    (ModelType.RANDOM_FOREST, make_rf_objective),
    (ModelType.XGBOOST, make_xgb_objective),
    (ModelType.LIGHTGBM, make_lgbm_objective),
]

_OBJECTIVE_MAP = {
    ModelType.RANDOM_FOREST: make_rf_objective,
    ModelType.XGBOOST: make_xgb_objective,
    ModelType.LIGHTGBM: make_lgbm_objective,
}


@dataclass
class TuningRunResult:
    version: str
    model_type: str
    optimization: OptimizationResult
    metrics: EvaluationResult
    best_params: dict


@dataclass
class TuningSession:
    version: str
    results: dict[str, TuningRunResult]   # model_name → TuningRunResult
    best_model_name: str
    best_metrics: EvaluationResult


class ModelTuner:
    """
    Orchestrates Optuna hyperparameter tuning for base models.

    Workflow per model:
      1. Run Optuna study (cross-validation on training data only)
      2. Retrain on the full training set using best params
      3. Evaluate on held-out test set
      4. Save tuned model via ModelStore
    """

    def __init__(
        self,
        ml_config: MLConfig = ml_config,
        tuning_config: TuningConfig | None = None,
    ) -> None:
        self.ml_config = ml_config
        self.tuning_config = tuning_config or TuningConfig()
        self.optimizer = HyperparameterOptimizer(self.tuning_config)
        self.evaluator = ModelEvaluator()
        self.store = ModelStore(ml_config)

    # ── Public API ───────────────────────────────────────────────────────────

    def tune_all(self, split: DataSplit, version: str) -> TuningSession:
        """Tune RF, XGBoost, and LightGBM; return a complete TuningSession."""
        results: dict[str, TuningRunResult] = {}

        for model_type, objective_factory in _TUNABLE_MODELS:
            name = model_type.value
            with log_duration(f"tune [{name}]"):
                run_result = self._tune_one(model_type, objective_factory, split, version)
            results[name] = run_result

        best_name = self._pick_best(results)
        return TuningSession(
            version=version,
            results=results,
            best_model_name=best_name,
            best_metrics=results[best_name].metrics,
        )

    def tune_single(
        self, model_type: ModelType, split: DataSplit, version: str
    ) -> TuningRunResult:
        """Tune a single model type."""
        objective_factory = _OBJECTIVE_MAP.get(model_type)
        if objective_factory is None:
            raise ValueError(
                f"Tuning is not supported for '{model_type}'. "
                f"Supported: {list(_OBJECTIVE_MAP)}"
            )
        return self._tune_one(model_type, objective_factory, split, version)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _tune_one(
        self, model_type: ModelType, objective_factory, split: DataSplit, version: str
    ) -> TuningRunResult:
        name = model_type.value
        rs = self.ml_config.hyperparams.random_state

        objective = objective_factory(split.X_train, split.y_train, self.tuning_config)
        opt_result = self.optimizer.optimize(name, objective)

        # Retrain on the full training set with the winning hyperparameters
        model = self.build_with_params(model_type, opt_result.best_params, rs)
        model.fit(split.X_train, split.y_train)

        y_pred = model.predict(split.X_test)
        y_prob = (
            model.predict_proba(split.X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        metrics = self.evaluator.evaluate(
            split.y_test, y_pred, y_prob, model_name=f"{name} (tuned)"
        )
        self.evaluator.log_confusion_matrix(
            split.y_test, y_pred, model_name=f"{name} (tuned)"
        )

        self.store.save(model, f"{name}_tuned", version=version)
        self.store.save_metadata(
            f"{name}_tuned",
            version,
            {
                "model_type": name,
                "tuned": True,
                "best_params": opt_result.best_params,
                "cv_score": opt_result.best_score,
                "test_metrics": metrics.to_dict(),
            },
        )

        return TuningRunResult(
            version=version,
            model_type=name,
            optimization=opt_result,
            metrics=metrics,
            best_params=opt_result.best_params,
        )

    def _pick_best(self, results: dict[str, TuningRunResult]) -> str:
        metric_attr = self.tuning_config.metric
        return max(
            results,
            key=lambda n: getattr(results[n].metrics, metric_attr, results[n].metrics.f1),
        )

    @staticmethod
    def build_with_params(model_type: ModelType, params: dict, random_state: int):
        """Build a fresh (unfitted) model with the given hyperparameters."""
        if model_type == ModelType.RANDOM_FOREST:
            return build_random_forest(random_state=random_state, **params)
        if model_type == ModelType.XGBOOST:
            return build_xgboost(**params, random_state=random_state)
        if model_type == ModelType.LIGHTGBM:
            return build_lightgbm(**params, random_state=random_state)
        raise ValueError(f"Cannot build model for type: {model_type}")
