"""CLI script to execute Phase 5 Temporal Feature Engineering."""
import argparse
import sys
from pathlib import Path

# Add src to pythonpath if executing as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from student_retention.logging_config import get_logger
from student_retention.paths import (
    PROCESSED_FEATURES_DIR,
)
from student_retention.transformation.build_feature_table import FeatureTableBuilder

logger = get_logger("build_features")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute Phase 5 Temporal Feature Engineering pipeline strictly bounded by Day 35 cutoff."
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=35,
        help="Cutoff relative day for temporal feature snapshot (default: 35)",
    )
    parser.add_argument(
        "--policy",
        choices=["Policy C", "Policy B"],
        default="Policy C",
        help="Cohort policy version to construct features for (default: Policy C)",
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=None,
        help="Path to cohort dataset (default: data/processed/cohort/cohort_policy_c.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_FEATURES_DIR,
        help="Directory to save feature matrices (default: data/processed/features)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory to save quality report (default: outputs/quality_reports or output-dir if custom)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting Temporal Feature Engineering Pipeline...")
    logger.info("  - Cutoff relative day: %d", args.cutoff)
    logger.info("  - Policy version:      %s", args.policy)

    builder = FeatureTableBuilder(cutoff_day=args.cutoff)
    feature_df, report = builder.run_pipeline(
        cohort_path=args.cohort_path,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        policy_version=args.policy,
    )

    print("\nPhase 5 — Temporal Feature Engineering Pipeline Completed Successfully!")
    print(f"  - Cohort records processed:  {len(feature_df):,}")
    print(f"  - Features extracted:        {len(feature_df.columns):,}")
    print(f"  - Validation status:         {report['metrics']['validation_status']}")
    print(f"  - Feature matrix saved to:   {args.output_dir}")
    print("  - Quality report saved to:   outputs/quality_reports/feature_quality_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
