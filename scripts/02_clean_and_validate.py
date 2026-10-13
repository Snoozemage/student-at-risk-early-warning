"""CLI script to run data cleaning, normalization, validation, and quality reporting."""
import argparse
import sys
from pathlib import Path

# Add src to pythonpath if executing as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from student_retention.cleaning.cleaning_pipeline import CleaningPipeline
from student_retention.logging_config import get_logger
from student_retention.paths import INTERIM_NORMALIZED_DIR, QUALITY_REPORTS_DIR
from student_retention.validation.report_generator import generate_quality_report

logger = get_logger("clean_and_validate")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean, normalize, and validate OULAD datasets for Phase 2."
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=INTERIM_NORMALIZED_DIR,
        help="Directory to save normalized interim tables (default: data/interim/normalized)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=QUALITY_REPORTS_DIR / "data_quality_report.json",
        help="Path for JSON quality report",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Storage format for normalized tables (default: parquet)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting Phase 2 Cleaning & Validation Pipeline...")

    # 1. Run cleaning pipeline
    pipeline = CleaningPipeline(output_dir=args.interim_dir)
    cleaning_results = pipeline.run(save_interim=True, save_format=args.format)

    tables = cleaning_results["tables"]
    reports = cleaning_results["reports"]

    missing_reports = {t: r["missing_report"] for t, r in reports.items()}
    id_reports = {t: r["identifier_report"] for t, r in reports.items()}

    # 2. Run quality validation and generate report
    quality_report = generate_quality_report(
        tables=tables,
        missing_reports=missing_reports,
        identifier_reports=id_reports,
        output_path=args.report_path,
    )

    # 3. Print executive summary
    print("\n" + "=" * 80)
    print("PHASE 2 DATA CLEANING & QUALITY VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Overall Status:            {quality_report['status']}")
    print(f"Timestamp:                 {quality_report['timestamp_utc']}")
    print(f"Tables Audited:            {quality_report['summary']['tables_audited']}")
    print(f"Total Rows Audited:        {quality_report['summary']['total_rows_audited']:,}")
    print(f"Schema Conformance:        {'PASSED' if quality_report['summary']['schema_valid'] else 'FAILED'}")
    print(f"Primary Key Uniqueness:    {'PASSED' if quality_report['summary']['primary_keys_valid'] else 'FAILED'}")
    print(f"Referential Integrity:     {'PASSED' if quality_report['summary']['referential_integrity_valid'] else 'FAILED'}")
    print(f"Total Violations:          {quality_report['summary']['total_violations']}")
    print("-" * 80)
    print(f"{'Table':<22} {'Rows':>12} {'Cols':>6} {'PK Valid':>10} {'Schema Valid':>14}")
    print("-" * 80)

    for t_name, t_sum in quality_report["table_summaries"].items():
        pk_valid = quality_report["key_validation"]["tables"].get(t_name, {}).get("is_valid", False)
        sch_valid = quality_report["schema_validation"]["tables"].get(t_name, {}).get("is_valid", False)
        print(
            f"{t_name:<22} {t_sum['row_count']:>12,} {t_sum['col_count']:>6} "
            f"{'YES' if pk_valid else 'NO':>10} {'YES' if sch_valid else 'NO':>14}"
        )

    print("=" * 80)
    print(f"Quality report saved to:   {args.report_path}")
    print(f"Interim tables saved to:   {args.interim_dir}")
    print("=" * 80 + "\n")

    if quality_report["status"] != "PASSED":
        print("Validation failed with violations:")
        for v in quality_report["all_violations"]:
            print(f"  - {v}")
        sys.exit(1)


if __name__ == "__main__":
    main()
