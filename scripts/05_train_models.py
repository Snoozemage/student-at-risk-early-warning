"""CLI script to execute Phase 6 Model Training and Cross-Validation."""
import argparse
import sys
from pathlib import Path

# Add src to pythonpath if executing as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from student_retention.logging_config import get_logger
from student_retention.modeling.train import train_and_evaluate_all_models
from student_retention.paths import (
    PROCESSED_FEATURES_DIR,
)

logger = get_logger("train_models")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute Phase 6 Machine Learning Model Training and Cross-Validation."
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet",
        help="Path to Day 35 feature matrix Parquet (default: data/processed/features/feature_matrix_policy_c.parquet)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible splitting and model training (default: 42)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.features_path.exists():
        logger.error("Feature matrix not found at %s. Please run feature extraction first.", args.features_path)
        return 1

    logger.info("Loading feature matrix from %s", args.features_path)
    df = pd.read_parquet(args.features_path)

    results = train_and_evaluate_all_models(
        feature_df=df,
        save_artifacts=True,
        random_state=args.random_state,
    )

    print("\nPhase 6 Model Training and Evaluation Completed Successfully!")
    print("\nCross-Validation Summary (5-Fold GroupKFold):")
    cv_table = results["cv_summary"][
        ["model_key", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean", "validation_optimal_threshold", "validation_threshold_f1"]
    ]
    print(cv_table.to_string(index=False))

    print("\nHeld-Out Test Evaluation Summary (Unseen Students):")
    test_table = results["test_summary"][
        ["model_key", "pr_auc", "roc_auc", "brier_score", "decision_threshold", "precision", "recall", "f1"]
    ]
    print(test_table.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
