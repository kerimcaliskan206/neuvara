"""
HantaProject ML Training Script

Trains Random Forest, XGBoost, LightGBM, and a Voting ensemble
on a dataset from data/raw/ and saves all models to models/.

Usage:
    python scripts/train.py --dataset hantavirus_sample.csv
    python scripts/train.py --dataset hantavirus_sample.csv --target label
    python scripts/train.py --dataset hantavirus_sample.csv --test-size 0.25

Options:
    --dataset     Dataset filename (must be in data/raw/)
    --target      Target column name (default: label)
    --test-size   Test fraction, 0.0–1.0 (default: 0.2)
    --scaler      Scaler type: standard | minmax | robust (default: standard)
    --seed        Random seed (default: 42)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.logging import setup_logging  # noqa: E402
from app.modules.ml.config import (  # noqa: E402
    MLConfig,
    PreprocessingConfig,
    ModelHyperparamsConfig,
)
from app.modules.ml.training.trainer import ModelTrainer  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HantaProject — Train hantavirus prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset filename inside data/raw/ (e.g. hantavirus_sample.csv)",
    )
    parser.add_argument("--target", default="label", help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument(
        "--scaler",
        default="standard",
        choices=["standard", "minmax", "robust"],
        help="Feature scaling method",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save training report as JSON alongside models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup_logging(debug=True, environment="development")
    logger.info("=" * 56)
    logger.info("  HantaProject — ML Training")
    logger.info("  Dataset : %s", args.dataset)
    logger.info("  Target  : %s", args.target)
    logger.info("  Seed    : %d", args.seed)
    logger.info("=" * 56)

    config = MLConfig(
        preprocessing=PreprocessingConfig(
            test_size=args.test_size,
            random_state=args.seed,
            scaler=args.scaler,
        ),
        hyperparams=ModelHyperparamsConfig(random_state=args.seed),
        target_column=args.target,
    )

    trainer = ModelTrainer(config=config)

    try:
        report = trainer.train_all(args.dataset)
    except FileNotFoundError as exc:
        logger.error("Dataset not found: %s", exc)
        logger.error(
            "Run first: python scripts/generate_sample_data.py"
        )
        sys.exit(1)
    except ValueError as exc:
        logger.error("Training aborted: %s", exc)
        sys.exit(1)

    if args.save_report:
        report_path = (
            config.storage.models_dir / f"v{report.version}" / "training_report.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("Report saved → %s", report_path)

    print("\n" + "─" * 56)
    print(f"  Training complete — version: {report.version}")
    print(f"  Best model : {report.best_model_name}")
    best = report.model_results[report.best_model_name]
    print(f"  F1         : {best.f1:.4f}")
    print(f"  ROC-AUC    : {best.roc_auc:.4f}")
    print(f"  Models saved → {config.storage.models_dir}")
    print("─" * 56 + "\n")


if __name__ == "__main__":
    main()
