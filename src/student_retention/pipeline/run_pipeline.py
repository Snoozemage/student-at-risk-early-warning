"""Master pipeline execution runner orchestrating end-to-end retention system workflows."""

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

from student_retention.logging_config import get_logger
from student_retention.reporting.generate_report import generate_all_deliverables

logger = get_logger("student_retention.pipeline")

# Stage definitions
STAGES: dict[str, str] = {
    "audit_sources": "Audit raw OULAD source CSV files and generate manifest",
    "clean_and_validate": "Clean raw sources, normalize types, and validate quality rules",
    "build_cohort": "Construct eligible Day-35 student cohort and withdrawal target",
    "run_eda": "Run exploratory data analysis and generate early signal figures",
    "build_features": "Engineer temporal VLE, assessment, and registration features",
    "train_models": "Fit baseline models, execute 5-fold CV, and tune thresholds",
    "evaluate_models": "Evaluate models on held-out test data and generate curves",
    "fairness_and_decisions": "Execute probability calibration, fairness audits, and support queue",
    "generate_deliverables": "Synthesize executive summary, technical report, and recommendations",
}


def run_stage_audit_sources() -> bool:
    """Run Stage 1: Audit sources."""
    from student_retention.ingestion.manifest import generate_source_manifest
    from student_retention.paths import QUALITY_REPORTS_DIR

    logger.info("Executing Stage: audit_sources...")
    generate_source_manifest(output_path=QUALITY_REPORTS_DIR / "source_inventory.json")
    return True


def run_stage_clean_and_validate() -> bool:
    """Run Stage 2: Clean and validate."""
    from student_retention.cleaning.cleaning_pipeline import run_cleaning_pipeline
    from student_retention.validation.report_generator import generate_quality_report

    logger.info("Executing Stage: clean_and_validate...")
    results = run_cleaning_pipeline()
    missing_reports = {name: report["missing_report"] for name, report in results["reports"].items()}
    identifier_reports = {
        name: report["identifier_report"] for name, report in results["reports"].items()
    }
    report = generate_quality_report(
        tables=results["tables"],
        missing_reports=missing_reports,
        identifier_reports=identifier_reports,
    )
    return report["status"] == "PASSED"


def run_stage_build_cohort() -> bool:
    """Run Stage 3: Build cohort."""
    from student_retention.transformation.build_cohort import CohortBuilder

    logger.info("Executing Stage: build_cohort...")
    builder = CohortBuilder(policy="policy_c", cutoff_day=35)
    builder.run()
    return True


def run_stage_run_eda() -> bool:
    """Run Stage 4: Run EDA."""
    from student_retention.eda.eda_runner import EDARunner

    logger.info("Executing Stage: run_eda...")
    runner = EDARunner(cutoff_day=35)
    runner.run()
    return True


def run_stage_build_features() -> bool:
    """Run Stage 5: Build features."""
    from student_retention.transformation.build_feature_table import FeatureTableBuilder

    logger.info("Executing Stage: build_features...")
    pipeline = FeatureTableBuilder(cutoff_day=35)
    pipeline.run_pipeline(policy_version="Policy C")
    return True


def run_stage_train_models() -> bool:
    """Run Stage 6: Train models."""
    import pandas as pd

    from student_retention.modeling.train import train_and_evaluate_all_models
    from student_retention.paths import PROCESSED_FEATURES_DIR

    logger.info("Executing Stage: train_models...")
    feature_path = PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {feature_path}")
    feature_df = pd.read_parquet(feature_path)
    train_and_evaluate_all_models(feature_df=feature_df, save_artifacts=True, random_state=42)
    return True


def run_stage_evaluate_models() -> bool:
    """Run Stage 6b: Evaluate models."""
    import pandas as pd

    from student_retention.features.feature_selection import select_primary_features
    from student_retention.modeling.evaluate import generate_evaluation_figures
    from student_retention.modeling.predict import load_model
    from student_retention.modeling.splits import split_by_student_group
    from student_retention.paths import (
        FIGURES_DIR,
        MODELS_BASELINE_DIR,
        PROCESSED_FEATURES_DIR,
        ensure_directories,
    )

    logger.info("Executing Stage: evaluate_models...")
    ensure_directories()
    feature_path = PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {feature_path}")
    clean_df = select_primary_features(
        pd.read_parquet(feature_path), include_target=True, include_identifiers=True
    )
    _, test_df = split_by_student_group(clean_df, test_size=0.20, random_state=42)
    models = {
        name: load_model(MODELS_BASELINE_DIR / f"{name}.joblib")
        for name in ("dummy_baseline", "logistic_regression", "random_forest")
        if (MODELS_BASELINE_DIR / f"{name}.joblib").exists()
    }
    if not models:
        raise FileNotFoundError(f"No trained model artifacts found in {MODELS_BASELINE_DIR}")
    generate_evaluation_figures(test_df, models, FIGURES_DIR)
    return True


def run_stage_fairness_and_decisions() -> bool:
    """Run Stage 7: Fairness and decision support."""
    import runpy

    logger.info("Executing Stage: fairness_and_decisions...")
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "07_run_fairness_and_decision_support.py"
    runpy.run_path(str(script_path), run_name="__main__")
    return True


def run_stage_generate_deliverables() -> bool:
    """Run Stage 8: Generate deliverables."""
    logger.info("Executing Stage: generate_deliverables...")
    try:
        paths = generate_all_deliverables()
        for name, p in paths.items():
            logger.info("Deliverable ready: %s -> %s", name, p)
        return True
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
        logger.error("Failed to execute Stage generate_deliverables: %s", e)
        return False


STAGE_RUNNERS: dict[str, Callable[[], bool]] = {
    "audit_sources": run_stage_audit_sources,
    "clean_and_validate": run_stage_clean_and_validate,
    "build_cohort": run_stage_build_cohort,
    "run_eda": run_stage_run_eda,
    "build_features": run_stage_build_features,
    "train_models": run_stage_train_models,
    "evaluate_models": run_stage_evaluate_models,
    "fairness_and_decisions": run_stage_fairness_and_decisions,
    "generate_deliverables": run_stage_generate_deliverables,
}


def run_pipeline(selected_stages: list[str] | None = None) -> bool:
    """Execute all or selected stages in sequence.

    Args:
        selected_stages: Optional list of stage keys to execute. If None, executes all stages.

    Returns:
        bool: True if all selected stages completed successfully, False otherwise.
    """
    stages_to_run = selected_stages or list(STAGES.keys())
    total_stages = len(stages_to_run)
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("STARTING MASTER RETENTION PIPELINE (%d stages)", total_stages)
    logger.info("=" * 70)

    for idx, stage_name in enumerate(stages_to_run, 1):
        if stage_name not in STAGE_RUNNERS:
            logger.error("Unknown stage requested: '%s'. Available: %s", stage_name, list(STAGES.keys()))
            return False

        logger.info("\n[%d/%d] >>> STAGE: %s", idx, total_stages, stage_name)
        stage_start = time.time()
        try:
            success = STAGE_RUNNERS[stage_name]()
            elapsed = time.time() - stage_start
            if not success:
                logger.error("Stage '%s' reported failure after %.2fs", stage_name, elapsed)
                return False
            logger.info("[%d/%d] <<< STAGE COMPLETED: %s (%.2fs)", idx, total_stages, stage_name, elapsed)
        except Exception:
            elapsed = time.time() - stage_start
            logger.exception("Stage '%s' crashed with error after %.2fs", stage_name, elapsed)
            return False

    total_elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("ALL %d PIPELINE STAGES COMPLETED SUCCESSFULLY (Total: %.2fs)", total_stages, total_elapsed)
    logger.info("=" * 70)
    return True


def main() -> None:
    """CLI entrypoint for running the master pipeline."""
    parser = argparse.ArgumentParser(
        description="Master execution pipeline for Smart Student Retention Early Warning System."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES.keys()),
        default=None,
        help="Specific pipeline stages to execute in order. Default: execute all stages.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List all available pipeline stages with descriptions and exit.",
    )

    args = parser.parse_args()

    if args.list_stages:
        print("\nAvailable Pipeline Stages:")
        for name, desc in STAGES.items():
            print(f"  - {name:<25}: {desc}")
        sys.exit(0)

    success = run_pipeline(selected_stages=args.stages)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
