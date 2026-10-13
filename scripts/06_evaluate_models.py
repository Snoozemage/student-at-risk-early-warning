"""CLI script to generate evaluation figures, curves, and error analysis for Phase 6."""
import sys
from pathlib import Path

# Add src to pythonpath if executing as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from student_retention.features.feature_selection import select_primary_features
from student_retention.logging_config import get_logger
from student_retention.modeling.evaluate import generate_evaluation_figures
from student_retention.modeling.predict import load_model
from student_retention.modeling.splits import split_by_student_group
from student_retention.paths import (
    FIGURES_DIR,
    MODELS_BASELINE_DIR,
    PROCESSED_FEATURES_DIR,
    ensure_directories,
)

logger = get_logger("evaluate_models")


def main() -> int:
    ensure_directories()
    features_path = PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet"
    if not features_path.exists():
        logger.error("Features not found at %s", features_path)
        return 1

    df = pd.read_parquet(features_path)
    clean_df = select_primary_features(df, include_target=True, include_identifiers=True)
    _, test_df = split_by_student_group(clean_df, test_size=0.20, random_state=42)

    # Load baseline models
    models = {}
    dummy_path = MODELS_BASELINE_DIR / "dummy_baseline.joblib"
    lr_path = MODELS_BASELINE_DIR / "logistic_regression.joblib"
    rf_path = MODELS_BASELINE_DIR / "random_forest.joblib"

    if dummy_path.exists():
        models["dummy_baseline"] = load_model(dummy_path)
    if lr_path.exists():
        models["logistic_regression"] = load_model(lr_path)
    if rf_path.exists():
        models["random_forest"] = load_model(rf_path)

    if not models:
        logger.error("No models found in %s. Please run scripts/05_train_models.py first.", MODELS_BASELINE_DIR)
        return 1

    generate_evaluation_figures(test_df, models, FIGURES_DIR)
    print(f"Evaluation figures successfully generated in {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
