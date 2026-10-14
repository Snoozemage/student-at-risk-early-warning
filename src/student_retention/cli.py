"""Command-line interface dispatcher for student-retention package."""
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from student_retention.cleaning.cleaning_pipeline import CleaningPipeline
from student_retention.eda.eda_runner import EDARunner
from student_retention.ingestion.manifest import SourceManifest
from student_retention.logging_config import get_logger
from student_retention.paths import (
    DATA_RAW_DIR,
    FIGURES_DIR,
    INTERIM_NORMALIZED_DIR,
    MODELS_BASELINE_DIR,
    PREDICTIONS_DIR,
    PROCESSED_COHORT_DIR,
    PROCESSED_FEATURES_DIR,
    QUALITY_REPORTS_DIR,
    REPORTS_DIR,
    TABLES_DIR,
)
from student_retention.transformation.build_cohort import CohortBuilder
from student_retention.transformation.build_feature_table import FeatureTableBuilder
from student_retention.validation.report_generator import generate_quality_report

logger = get_logger("student_retention.cli")



def run_audit(args: argparse.Namespace) -> int:
    """Execute raw source inventory audit."""
    logger.info("Starting source inventory audit from %s", args.raw_dir)
    manifest_gen = SourceManifest(raw_data_dir=args.raw_dir)
    manifest = manifest_gen.generate()
    output_path = manifest_gen.save(output_path=args.output, manifest_data=manifest)
    print(f"Source inventory report successfully written to {output_path}")
    print(f"Total raw rows: {manifest['total_raw_rows']:,}")
    return 0


def run_clean_and_validate(args: argparse.Namespace) -> int:
    """Execute cleaning, normalization, and quality validation pipeline."""
    logger.info("Executing cleaning and validation pipeline...")
    pipeline = CleaningPipeline(output_dir=args.interim_dir)
    results = pipeline.run(save_interim=True, save_format=args.format)

    tables = results["tables"]
    reports = results["reports"]
    missing_reports = {t: r["missing_report"] for t, r in reports.items()}
    id_reports = {t: r["identifier_report"] for t, r in reports.items()}

    quality_report = generate_quality_report(
        tables=tables,
        missing_reports=missing_reports,
        identifier_reports=id_reports,
        output_path=args.report_path,
    )

    print(f"Data quality report saved to {args.report_path} (Status: {quality_report['status']})")
    if quality_report["status"] != "PASSED":
        print("Validation violations detected:")
        for v in quality_report["all_violations"]:
            print(f"  - {v}")
        return 1
    return 0


def run_build_cohort(args: argparse.Namespace) -> int:
    """Execute cohort construction pipeline."""
    logger.info("Executing cohort construction under policy: %s (cutoff: Day %d)", args.policy, args.cutoff)
    builder = CohortBuilder(
        cutoff_day=args.cutoff,
        policy=args.policy,
        data_dir=args.data_dir,
    )
    results = builder.run(save_outputs=not args.no_save, output_dir=args.output_dir)
    audit = results["audit_report"]
    reconciliation = audit["reconciliation"]
    target_dist = audit["target_distribution"]

    print(f"\nCohort constructed under {args.policy.upper()} at Day {args.cutoff}:")
    print(f"  - Total source records: {reconciliation['total_source_records']:,}")
    print(f"  - Excluded records:     {reconciliation['total_excluded_records']:,} ({reconciliation['pct_excluded']}%)")
    print(f"  - Eligible cohort:      {reconciliation['eligible_cohort_records']:,} ({reconciliation['pct_eligible']}%)")
    print(f"  - Positive cases (1):   {target_dist['positive_cases']:,} ({target_dist['positive_base_rate_pct']}%)")
    print(f"  - Negative cases (0):   {target_dist['negative_cases']:,}")
    if "saved_paths" in results:
        print(f"  - Output saved to:      {results['saved_paths']['cohort_parquet']}")
    return 0


def run_eda(args: argparse.Namespace) -> int:
    """Execute exploratory data analysis & early warning signal discovery pipeline."""
    logger.info("Executing EDA pipeline...")
    runner = EDARunner(
        cohort_path=args.cohort_path,
        normalized_dir=args.data_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        reports_dir=args.reports_dir,
    )
    results = runner.run()
    print("\nEDA Pipeline Execution Completed Successfully!")
    print(f"  - Cohort records analyzed: {results['cohort_rows']:,}")
    print(f"  - Withdrawal base rate:    {results['withdrawal_rate']:.2%}")
    print(f"  - Tables exported to:      {results['tables_dir']}")
    print(f"  - Figures exported to:     {results['figures_dir']}")
    print(f"  - Report generated:        {results['report_path']}")
    return 0


def run_build_features(args: argparse.Namespace) -> int:
    """Execute temporal feature engineering pipeline."""
    logger.info("Executing temporal feature engineering pipeline (cutoff: Day %d, policy: %s)", args.cutoff, args.policy)
    builder = FeatureTableBuilder(cutoff_day=args.cutoff)
    feature_df, report = builder.run_pipeline(
        cohort_path=args.cohort_path,
        interim_dir=args.interim_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        policy_version=args.policy,
    )
    print("\nTemporal Feature Engineering Completed Successfully!")
    print(f"  - Eligible observations:   {len(feature_df):,}")
    print(f"  - Features extracted:      {len(feature_df.columns):,}")
    print(f"  - Validation status:       {report['metrics']['validation_status']}")
    print(f"  - Output directory:        {args.output_dir}")
    return 0


def run_train_models(args: argparse.Namespace) -> int:
    """Execute model training, cross-validation, and evaluation pipeline."""
    import pandas as pd

    from student_retention.modeling.train import train_and_evaluate_all_models

    logger.info("Executing Phase 6 Model Training Pipeline from %s", args.features_path)
    if not args.features_path.exists():
        logger.error("Features not found at %s", args.features_path)
        return 1

    df = pd.read_parquet(args.features_path)
    results = train_and_evaluate_all_models(
        feature_df=df,
        save_artifacts=True,
        random_state=args.random_state,
    )
    print("\nPhase 6 Model Training and Evaluation Completed Successfully!")
    cv_table = results["cv_summary"][
        ["model_key", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean", "validation_optimal_threshold", "validation_threshold_f1"]
    ]
    print("\nCross-Validation Summary (5-Fold GroupKFold):")
    print(cv_table.to_string(index=False))

    test_table = results["test_summary"][
        ["model_key", "pr_auc", "roc_auc", "brier_score", "decision_threshold", "precision", "recall", "f1"]
    ]
    print("\nHeld-Out Test Evaluation Summary:")
    print(test_table.to_string(index=False))
    return 0


def run_evaluate_models(args: argparse.Namespace) -> int:
    """Execute model evaluation and figure generation."""
    import pandas as pd

    from student_retention.features.feature_selection import select_primary_features
    from student_retention.modeling.evaluate import generate_evaluation_figures
    from student_retention.modeling.predict import load_model
    from student_retention.modeling.splits import split_by_student_group

    logger.info("Executing Phase 6 Model Evaluation and Figure Generation...")
    if not args.features_path.exists():
        logger.error("Features not found at %s", args.features_path)
        return 1

    df = pd.read_parquet(args.features_path)
    clean_df = select_primary_features(df, include_target=True, include_identifiers=True)
    _, test_df = split_by_student_group(clean_df, test_size=0.20, random_state=42)

    models = {}
    for mname, fname in [
        ("dummy_baseline", "dummy_baseline.joblib"),
        ("logistic_regression", "logistic_regression.joblib"),
        ("random_forest", "random_forest.joblib"),
    ]:
        mpath = args.models_dir / fname
        if mpath.exists():
            models[mname] = load_model(mpath)

    if not models:
        logger.error("No models found in %s", args.models_dir)
        return 1

    generate_evaluation_figures(test_df, models, args.figures_dir)
    print(f"\nEvaluation figures successfully exported to: {args.figures_dir}")
    return 0


def run_audit_fairness(args: argparse.Namespace) -> int:
    """Execute subgroup fairness audit and demographic sensitivity analysis."""
    import pandas as pd

    from student_retention.fairness.subgroup_audit import (
        audit_all_protected_attributes,
        run_demographic_sensitivity_experiment,
    )
    from student_retention.features.feature_selection import select_primary_features
    from student_retention.modeling.predict import (
        load_model,
        load_operational_threshold,
    )
    from student_retention.modeling.splits import split_by_student_group

    logger.info("Executing Phase 7 Subgroup Fairness Audit on %s", args.features_path)
    if not args.features_path.exists():
        logger.error("Features not found at %s", args.features_path)
        return 1

    df = pd.read_parquet(args.features_path)
    model_path = args.models_dir / "random_forest.joblib"
    if not model_path.exists():
        logger.error("Model not found at %s", model_path)
        return 1

    model = load_model(model_path)
    threshold = (
        args.threshold
        if args.threshold is not None
        else load_operational_threshold("random_forest", summary_path=args.output_dir / "model_training_summary.json" if (args.output_dir / "model_training_summary.json").exists() else None)
    )

    _, test_df = split_by_student_group(df, test_size=0.20, random_state=42)
    clean_test_x = select_primary_features(test_df, include_target=False, include_identifiers=False)
    test_probs = model.predict_proba(clean_test_x)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    test_eval_df = test_df.copy()
    test_eval_df["predicted_label"] = test_preds
    test_eval_df["risk_probability"] = test_probs

    audit_df = audit_all_protected_attributes(
        df=test_eval_df,
        target_col="withdrawn_after_cutoff",
        pred_col="predicted_label",
        prob_col="risk_probability",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "subgroup_fairness_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    print("\nSubgroup Fairness Audit Completed Successfully!")
    print(f"  - Decision Threshold: {threshold:.4f} (from training metadata)")
    print(f"  - Evaluated Subgroups: {len(audit_df)}")
    print(f"  - Audit Report CSV:   {audit_path}")

    if args.run_sensitivity:
        print("\nRunning Demographic Sensitivity Experiment...")
        sensitivity = run_demographic_sensitivity_experiment(
            feature_df=df,
            test_size=0.20,
            random_state=42,
            operational_threshold=threshold,
        )
        comp_path = args.output_dir / "demographic_sensitivity_comparison.csv"
        sensitivity["model_comparison"].to_csv(comp_path, index=False)
        print(f"  - Sensitivity Table:  {comp_path}")
        print("\nSensitivity Comparison:")
        print(sensitivity["model_comparison"].to_string(index=False))

    return 0


def run_generate_support_queue(args: argparse.Namespace) -> int:
    """Generate student support triage queue for academic advising."""
    import pandas as pd

    from student_retention.decision_support.prioritize_students import (
        audit_support_queue_capacity,
        build_student_support_queue,
    )
    from student_retention.modeling.predict import load_model
    from student_retention.modeling.splits import split_by_student_group

    logger.info("Generating student support triage queue from %s", args.features_path)
    if not args.features_path.exists():
        logger.error("Features not found at %s", args.features_path)
        return 1

    df = pd.read_parquet(args.features_path)

    # Prefer calibrated model artifact if present
    cal_path = args.models_dir / "random_forest_calibrated.joblib"
    base_path = args.models_dir / "random_forest.joblib"
    if cal_path.exists():
        model_path = cal_path
        logger.info("Using calibrated model artifact from %s", cal_path)
    elif base_path.exists():
        model_path = base_path
        logger.info("Using baseline model artifact from %s", base_path)
    else:
        logger.error("No model found in %s", args.models_dir)
        return 1

    model = load_model(model_path)

    # In accordance with leakage-free evaluation, default to held-out test cohort
    if getattr(args, "full_cohort", False):
        scoring_df = df
        cohort_label = "full cohort retrospective"
    else:
        _, scoring_df = split_by_student_group(df, test_size=0.20, random_state=42)
        cohort_label = "held-out test cohort demonstration"

    logger.info("Scoring %s (N=%d records)...", cohort_label, len(scoring_df))

    queue_df = build_student_support_queue(
        feature_df=scoring_df,
        model=model,
        urgent_threshold=args.urgent_threshold,
        watch_threshold=args.watch_threshold,
        capacity_k=args.capacity,
        save_path=args.output_path,
    )

    cap_audit = audit_support_queue_capacity(queue_df)
    cap_path = args.output_path.parent / "support_queue_capacity_audit.csv"
    cap_audit.to_csv(cap_path, index=False)

    print("\nDecision Support Caseload Queue Generated Successfully!")
    print(f"  - Evaluation Cohort:       {cohort_label} (N={len(scoring_df):,})")
    print(f"  - Total Queue Records:     {len(queue_df):,}")
    print(f"  - Unique Students:         {queue_df['student_key'].nunique():,}")
    print(f"  - Duplicate Slots:         {len(queue_df) - queue_df['student_key'].nunique():,}")
    print(f"  - Urgent (Tier 1) Cases:   {int((queue_df['priority_tier'] == 'Tier 1 (Urgent)').sum()):,}")
    print(f"  - Watch (Tier 2) Cases:    {int((queue_df['priority_tier'] == 'Tier 2 (Watch)').sum()):,}")
    print(f"  - Standard (Tier 3) Cases: {int((queue_df['priority_tier'] == 'Tier 3 (Standard)').sum()):,}")
    print(f"  - Output CSV Destination:  {args.output_path}")
    print(f"  - Capacity Audit CSV:      {cap_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI commands."""
    parser = argparse.ArgumentParser(
        prog="student-retention",
        description="Smart Student Retention & Early Warning System CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: audit
    audit_parser = subparsers.add_parser("audit", help="Audit raw OULAD sources and generate inventory.")
    audit_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help="Path to raw data directory (default: data/raw/oulad)",
    )
    audit_parser.add_argument(
        "--output",
        type=Path,
        default=QUALITY_REPORTS_DIR / "source_inventory.json",
        help="Path to output inventory JSON",
    )

    # Subcommand: clean-and-validate
    cv_parser = subparsers.add_parser(
        "clean-and-validate",
        help="Clean, normalize, and validate datasets.",
    )
    cv_parser.add_argument(
        "--interim-dir",
        type=Path,
        default=INTERIM_NORMALIZED_DIR,
        help="Directory to save normalized interim tables (default: data/interim/normalized)",
    )
    cv_parser.add_argument(
        "--report-path",
        type=Path,
        default=QUALITY_REPORTS_DIR / "data_quality_report.json",
        help="Path for JSON quality report",
    )
    cv_parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Interim table format (default: parquet)",
    )

    # Subcommand: build-cohort
    cohort_parser = subparsers.add_parser(
        "build-cohort",
        help="Construct eligible student cohort and withdrawal target.",
    )
    cohort_parser.add_argument(
        "--policy",
        choices=["policy_c", "policy_b"],
        default="policy_c",
        help="Cohort eligibility policy (default: policy_c)",
    )
    cohort_parser.add_argument(
        "--cutoff",
        type=int,
        default=35,
        help="Prediction cutoff in relative days (default: 35)",
    )
    cohort_parser.add_argument(
        "--data-dir",
        type=Path,
        default=INTERIM_NORMALIZED_DIR,
        help="Directory containing normalized interim tables",
    )
    cohort_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_COHORT_DIR,
        help="Directory to save cohort datasets (default: data/processed/cohort)",
    )
    cohort_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save outputs to disk",
    )

    # Subcommand: run-eda
    eda_parser = subparsers.add_parser(
        "run-eda",
        help="Execute exploratory data analysis and early warning discovery pipeline.",
    )
    eda_parser.add_argument(
        "--cohort-path",
        type=Path,
        default=PROCESSED_COHORT_DIR / "cohort_policy_c.parquet",
        help="Path to cohort dataset (default: data/processed/cohort/cohort_policy_c.parquet)",
    )
    eda_parser.add_argument(
        "--data-dir",
        type=Path,
        default=INTERIM_NORMALIZED_DIR,
        help="Directory containing normalized interim tables (default: data/interim/normalized)",
    )
    eda_parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory to save figures (default: outputs/figures)",
    )
    eda_parser.add_argument(
        "--tables-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory to save tables (default: outputs/tables)",
    )
    eda_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory to save reports (default: outputs/reports)",
    )

    # Subcommand: build-features
    feat_parser = subparsers.add_parser(
        "build-features",
        help="Build Day 35 temporal feature matrices and validation reports.",
    )
    feat_parser.add_argument(
        "--cutoff",
        type=int,
        default=35,
        help="Cutoff relative day for temporal feature snapshot (default: 35)",
    )
    feat_parser.add_argument(
        "--policy",
        choices=["Policy C", "Policy B"],
        default="Policy C",
        help="Cohort policy version to construct features for (default: Policy C)",
    )
    feat_parser.add_argument(
        "--cohort-path",
        type=Path,
        default=None,
        help="Path to cohort dataset (default: data/processed/cohort/cohort_policy_c.parquet)",
    )
    feat_parser.add_argument(
        "--interim-dir",
        type=Path,
        default=INTERIM_NORMALIZED_DIR,
        help="Directory containing normalized interim tables (default: data/interim/normalized)",
    )
    feat_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_FEATURES_DIR,
        help="Directory to save feature matrices (default: data/processed/features)",
    )
    feat_parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory to save quality report (default: outputs/quality_reports or output-dir if custom)",
    )

    # Subcommand: train-models
    train_parser = subparsers.add_parser(
        "train-models",
        help="Train baseline models and evaluate cross-validation performance.",
    )
    train_parser.add_argument(
        "--features-path",
        type=Path,
        default=PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet",
        help="Path to feature matrix Parquet (default: data/processed/features/feature_matrix_policy_c.parquet)",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting and training (default: 42)",
    )

    # Subcommand: evaluate-models
    eval_parser = subparsers.add_parser(
        "evaluate-models",
        help="Evaluate models, generate curves and diagnostic figures.",
    )
    eval_parser.add_argument(
        "--features-path",
        type=Path,
        default=PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet",
        help="Path to feature matrix Parquet (default: data/processed/features/feature_matrix_policy_c.parquet)",
    )
    eval_parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_BASELINE_DIR,
        help="Directory containing trained model artifacts (default: models/baseline)",
    )
    eval_parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory to save generated figures (default: outputs/figures)",
    )

    # Subcommand: audit-fairness
    fairness_parser = subparsers.add_parser(
        "audit-fairness",
        help="Audit model performance and error-rate parity across protected demographic subgroups.",
    )
    fairness_parser.add_argument(
        "--features-path",
        type=Path,
        default=PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet",
        help="Path to feature matrix Parquet (default: data/processed/features/feature_matrix_policy_c.parquet)",
    )
    fairness_parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_BASELINE_DIR,
        help="Directory containing trained model artifacts (default: models/baseline)",
    )
    fairness_parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory to save audit tables (default: outputs/tables)",
    )
    fairness_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Operational decision threshold for binary prediction evaluation (default: resolved from model_training_summary.json)",
    )
    fairness_parser.add_argument(
        "--run-sensitivity",
        action="store_true",
        default=True,
        help="Run secondary demographic sensitivity experiment (default: True)",
    )

    # Subcommand: generate-support-queue
    queue_parser = subparsers.add_parser(
        "generate-support-queue",
        help="Generate student support triage queue for academic advising.",
    )
    queue_parser.add_argument(
        "--features-path",
        type=Path,
        default=PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet",
        help="Path to feature matrix Parquet (default: data/processed/features/feature_matrix_policy_c.parquet)",
    )
    queue_parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_BASELINE_DIR,
        help="Directory containing trained model artifacts (default: models/baseline)",
    )
    queue_parser.add_argument(
        "--output-path",
        type=Path,
        default=PREDICTIONS_DIR / "student_support_queue.csv",
        help="Destination path for output queue CSV (default: outputs/predictions/student_support_queue.csv)",
    )
    queue_parser.add_argument(
        "--urgent-threshold",
        type=float,
        default=0.70,
        help="Cutoff probability for Tier 1 (Urgent) triage (default: 0.70)",
    )
    queue_parser.add_argument(
        "--watch-threshold",
        type=float,
        default=0.40,
        help="Cutoff probability for Tier 2 (Watch) triage (default: 0.40)",
    )
    queue_parser.add_argument(
        "--capacity",
        type=int,
        default=500,
        help="Maximum advisor capacity limit per presentation (default: 500)",
    )
    queue_parser.add_argument(
        "--full-cohort",
        action="store_true",
        default=False,
        help="Generate queue on full cohort retrospective instead of default held-out test cohort (default: False)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "audit":
        return run_audit(args)
    elif args.command == "clean-and-validate":
        return run_clean_and_validate(args)
    elif args.command == "build-cohort":
        return run_build_cohort(args)
    elif args.command == "run-eda":
        return run_eda(args)
    elif args.command == "build-features":
        return run_build_features(args)
    elif args.command == "train-models":
        return run_train_models(args)
    elif args.command == "evaluate-models":
        return run_evaluate_models(args)
    elif args.command == "audit-fairness":
        return run_audit_fairness(args)
    elif args.command == "generate-support-queue":
        return run_generate_support_queue(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
