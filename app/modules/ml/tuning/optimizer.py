import logging
from dataclasses import dataclass, field

import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler

from app.modules.ml.tuning.config import TuningConfig

logger = logging.getLogger(__name__)

# Keep Optuna from flooding the log with per-trial INFO messages
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationResult:
    model_name: str
    best_params: dict
    best_score: float
    n_trials_completed: int
    n_trials_pruned: int
    all_trials: list[dict] = field(default_factory=list)


class HyperparameterOptimizer:
    """Runs an Optuna study for a single model and returns a structured result."""

    def __init__(self, config: TuningConfig) -> None:
        self.config = config

    def optimize(self, model_name: str, objective_fn) -> OptimizationResult:
        sampler = TPESampler(seed=self.config.random_state)
        pruner = (
            MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            if self.config.pruning
            else NopPruner()
        )

        study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=f"{model_name}_tuning",
        )

        logger.info(
            "Optuna study [%s]: %d trials | metric=%s | cv=%d folds",
            model_name,
            self.config.n_trials,
            self.config.metric,
            self.config.cv_folds,
        )

        study.optimize(
            objective_fn,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=self.config.show_progress,
            n_jobs=self.config.n_jobs,
        )

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ]

        all_trial_records = [
            {
                "trial": t.number,
                "score": t.value,
                "params": t.params,
                "state": t.state.name,
                "duration_sec": (
                    t.duration.total_seconds() if t.duration else None
                ),
            }
            for t in study.trials
        ]

        logger.info(
            "[%s] Best CV score: %.4f | Completed: %d | Pruned: %d",
            model_name,
            study.best_value,
            len(completed),
            len(pruned),
        )
        logger.info("[%s] Best params: %s", model_name, study.best_params)

        return OptimizationResult(
            model_name=model_name,
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials_completed=len(completed),
            n_trials_pruned=len(pruned),
            all_trials=all_trial_records,
        )
