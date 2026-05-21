"""
HantaProject — Hyperparameter Tuning CLI

Tunes RF, XGBoost, and LightGBM with Optuna (cross-validation based),
builds a weighted voting ensemble from the winners, and saves all artifacts.

Usage:
    python scripts/tune.py --dataset hantavirus_sample.csv
    python scripts/tune.py --dataset hantavirus_sample.csv --trials 100 --metric roc_auc
    python scripts/tune.py --dataset hantavirus_sample.csv --model random_forest --trials 30
    python scripts/tune.py --dataset hantavirus_sample.csv --no-progress

Options:
    --dataset           Dataset filename in data/raw/ (required)
    --target            Target column name (default: label)
    --trials            Optuna trials per model (default: 50)
    --metric            Optimization metric: f1 | roc_auc | accuracy (default: f1)
    --cv-folds          Cross-validation folds (default: 3)
    --timeout           Per-model tuning timeout in seconds (default: none)
    --seed              Random seed (default: 42)
    --test-size         Test set fraction (default: 0.2)
    --model             Which model to tune: all | random_forest | xgboost | lightgbm
    --experiment-name   Experiment name for artifact directories (default: hanta_tuning)
    --no-progress       Suppress Optuna progress bar
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.logging import setup_logging  # noqa: E402
from app.modules.ml.config import (  # noqa: E402
    ExperimentsConfig,
    MLConfig,
    ModelHyperparamsConfig,
    PreprocessingConfig,
)
from app.modules.ml.datasets.loader import DatasetLoader  # noqa: E402
from app.modules.ml.datasets.validator import DatasetValidator  # noqa: E402
from app.modules.ml.ensemble.models import (  # noqa: E402
    ModelType,
    build_weighted_voting_ensemble,
)
from app.modules.ml.ensemble.weighting import (  # noqa: E402
    compute_f1_weights,
    log_weights,
    weights_as_list,
)
from app.modules.ml.evaluation.metrics import ModelEvaluator  # noqa: E402
from app.modules.ml.evaluation.report_builder import EvaluationReportBuilder  # noqa: E402
from app.modules.ml.experiments.artifacts import ExperimentArtifactStore  # noqa: E402
from app.modules.ml.experiments.report import ModelTuningRecord, TuningReport  # noqa: E402
from app.modules.ml.experiments.tracker import ExperimentTracker  # noqa: E402
from app.modules.ml.persistence.model_store import ModelStore  # noqa: E402
from app.modules.ml.preprocessing.pipeline import PreprocessingPipeline  # noqa: E402
from app.modules.ml.tuning.config import TuningConfig  # noqa: E402
from app.modules.ml.tuning.tuner import ModelTuner, TuningSession  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HantaProject — Hyperparameter Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset filename inside data/raw/ (e.g. hantavirus_sample.csv)",
    )
    parser.add_argument("--target", default="label", help="Target column name")
    parser.add_argument(
        "--trials", type=int, default=50, help="Optuna trials per model"
    )
    parser.add_argument(
        "--metric",
        default="f1",
        choices=["f1", "roc_auc", "accuracy"],
        help="Metric to optimize",
    )
    parser.add_argument("--cv-folds", type=int, default=3, help="Cross-validation folds")
    parser.add_argument(
        "--timeout", type=int, default=None, help="Per-model timeout in seconds"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "random_forest", "xgboost", "lightgbm"],
        help="Which model to tune",
    )
    parser.add_argument(
        "--experiment-name",
        default="hanta_tuning",
        help="Experiment name (used for artifact directory naming)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress Optuna progress bar",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(debug=True, environment="development")

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 62)
    logger.info("  HantaProject — Hyperparameter Tuning")
    logger.info("  Dataset    : %s", args.dataset)
    logger.info("  Model(s)   : %s", args.model)
    logger.info("  Trials     : %d per model", args.trials)
    logger.info("  Metric     : %s", args.metric)
    logger.info("  CV folds   : %d", args.cv_folds)
    logger.info("  Seed       : %d", args.seed)
    logger.info("  Version    : %s", version)
    logger.info("=" * 62)

    ml_config = MLConfig(
        preprocessing=PreprocessingConfig(
            test_size=args.test_size,
            random_state=args.seed,
        ),
        hyperparams=ModelHyperparamsConfig(random_state=args.seed),
        target_column=args.target,
    )
    tuning_config = TuningConfig(
        n_trials=args.trials,
        metric=args.metric,
        cv_folds=args.cv_folds,
        timeout=args.timeout,
        random_state=args.seed,
        show_progress=not args.no_progress,
    )

    # ── 1. Load & validate ───────────────────────────────────────────────────
    loader = DatasetLoader(ml_config)
    validator = DatasetValidator(
        missing_threshold=ml_config.preprocessing.missing_threshold
    )

    try:
        df = loader.load_raw(args.dataset)
    except FileNotFoundError as exc:
        logger.error("Dataset not found: %s", exc)
        logger.error("Run first: python scripts/generate_sample_data.py")
        sys.exit(1)

    validation = validator.full_validate(df, ml_config.target_column)
    validation.log()
    if not validation.passed:
        logger.error("Dataset failed validation — aborting.")
        sys.exit(1)

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    pipeline = PreprocessingPipeline(ml_config)
    split = pipeline.fit_transform(df)
    logger.info(
        "Split: %d train / %d test | %d features",
        split.train_size, split.test_size, split.X_train.shape[1],
    )

    pipeline_path = (
        ml_config.storage.models_dir / f"v{version}" / "preprocessing_pipeline.joblib"
    )
    pipeline.save(pipeline_path)

    # ── 3. Start experiment tracking ─────────────────────────────────────────
    tracker = ExperimentTracker()
    artifact_store = ExperimentArtifactStore(ml_config.experiments.experiments_dir)
    run = tracker.start_run(
        experiment_name=args.experiment_name,
        dataset=args.dataset,
        metric=args.metric,
    )

    # ── 4. Tune ──────────────────────────────────────────────────────────────
    tuner = ModelTuner(ml_config=ml_config, tuning_config=tuning_config)

    if args.model == "all":
        session: TuningSession = tuner.tune_all(split, version)
    else:
        model_type = ModelType(args.model)
        single = tuner.tune_single(model_type, split, version)
        session = TuningSession(
            version=version,
            results={args.model: single},
            best_model_name=args.model,
            best_metrics=single.metrics,
        )

    # Record Optuna trials in the experiment tracker
    for model_name, result in session.results.items():
        tracker.record_trials_from_optuna(
            run, model_name, result.optimization.all_trials
        )
    run.finish()

    # ── 5. Weighted voting ensemble (only when all 3 models were tuned) ───────
    weighted_voting_metrics = None
    if args.model == "all" and len(session.results) == 3:
        logger.info("Building weighted voting ensemble from tuned models...")
        metric_results = {name: r.metrics for name, r in session.results.items()}
        model_weights = compute_f1_weights(metric_results)
        log_weights(model_weights, strategy="F1-proportional")

        ordered_names = list(session.results.keys())
        estimators = [
            (
                name,
                ModelTuner.build_with_params(
                    ModelType(name),
                    session.results[name].best_params,
                    ml_config.hyperparams.random_state,
                ),
            )
            for name in ordered_names
        ]
        weight_values = weights_as_list(model_weights, ordered_names)
        voting_model = build_weighted_voting_ensemble(estimators, weights=weight_values)
        voting_model.fit(split.X_train, split.y_train)

        evaluator = ModelEvaluator()
        y_pred = voting_model.predict(split.X_test)
        y_prob = voting_model.predict_proba(split.X_test)[:, 1]
        weighted_voting_metrics = evaluator.evaluate(
            split.y_test, y_pred, y_prob, model_name="weighted_voting (tuned)"
        )

        store = ModelStore(ml_config)
        store.save(voting_model, "weighted_voting_tuned", version=version)
        store.save_metadata(
            "weighted_voting_tuned",
            version,
            {
                "model_type": "weighted_voting",
                "tuned": True,
                "weights": {w.model_name: w.weight for w in model_weights},
                "test_metrics": weighted_voting_metrics.to_dict(),
            },
        )
        logger.info(
            "Weighted voting → F1=%.4f  AUC=%.4f",
            weighted_voting_metrics.f1,
            weighted_voting_metrics.roc_auc,
        )

    # ── 6. Build & save reports ───────────────────────────────────────────────
    model_records = {
        name: ModelTuningRecord(
            model_name=name,
            best_params=r.best_params,
            best_cv_score=r.optimization.best_score,
            n_trials_completed=r.optimization.n_trials_completed,
            n_trials_pruned=r.optimization.n_trials_pruned,
            test_metrics=r.metrics,
        )
        for name, r in session.results.items()
    }

    tuning_report = TuningReport(
        experiment_name=args.experiment_name,
        timestamp=version,
        version=version,
        dataset=args.dataset,
        metric=args.metric,
        n_trials=args.trials,
        cv_folds=args.cv_folds,
        models=model_records,
        best_model_name=session.best_model_name,
        all_trials={
            name: r.optimization.all_trials for name, r in session.results.items()
        },
    )
    tuning_report.log()

    artifact_store.save_tuning_report(tuning_report)
    artifact_store.save_experiment_run(run)
    artifact_store.save_best_params(
        args.experiment_name,
        version,
        {name: r.best_params for name, r in session.results.items()},
    )
    artifact_store.save_metrics_history(
        args.experiment_name,
        version,
        run.metrics_history,
    )

    # ── 7. Advanced evaluation report ────────────────────────────────────────
    all_eval_results = {name: r.metrics for name, r in session.results.items()}
    if weighted_voting_metrics is not None:
        all_eval_results["weighted_voting (tuned)"] = weighted_voting_metrics

    report_builder = EvaluationReportBuilder()
    eval_report = report_builder.build(
        model_results=all_eval_results,
        context={
            "dataset": args.dataset,
            "version": version,
            "n_train": split.train_size,
            "n_test": split.test_size,
            "n_features": split.X_train.shape[1],
            "tuning_metric": args.metric,
            "trials_per_model": args.trials,
            "cv_folds": args.cv_folds,
        },
    )
    eval_report.log()

    # ── 8. Console summary ────────────────────────────────────────────────────
    best = session.results[session.best_model_name]
    print("\n" + "─" * 62)
    print(f"  Tuning complete — version: {version}")
    print(f"  Best base model : {session.best_model_name}")
    print(f"  CV Score        : {best.optimization.best_score:.4f}")
    print(f"  Test F1         : {best.metrics.f1:.4f}")
    print(f"  Test ROC-AUC    : {best.metrics.roc_auc:.4f}")
    print(f"\n  Best hyperparameters:\n{json.dumps(best.best_params, indent=4)}")
    if weighted_voting_metrics is not None:
        print(
            f"\n  Weighted voting → F1={weighted_voting_metrics.f1:.4f}"
            f"  AUC={weighted_voting_metrics.roc_auc:.4f}"
        )
    print(f"\n  Comparison table:\n{eval_report.comparison_table()}")
    print(f"\n  Models saved    → {ml_config.storage.models_dir}")
    print(
        f"  Experiments     → experiments/"
        f"{args.experiment_name}_{version}/"
    )
    print("─" * 62 + "\n")


if __name__ == "__main__":
    main()
