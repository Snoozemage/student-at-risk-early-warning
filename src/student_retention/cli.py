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
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
