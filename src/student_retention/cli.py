"""Command-line interface dispatcher for student-retention package."""
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from student_retention.cleaning.cleaning_pipeline import CleaningPipeline
from student_retention.ingestion.manifest import SourceManifest
from student_retention.logging_config import get_logger
from student_retention.paths import (
    DATA_RAW_DIR,
    INTERIM_NORMALIZED_DIR,
    QUALITY_REPORTS_DIR,
)
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "audit":
        return run_audit(args)
    elif args.command == "clean-and-validate":
        return run_clean_and_validate(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
