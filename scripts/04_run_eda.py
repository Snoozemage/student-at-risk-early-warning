"""CLI script to execute Phase 4 Exploratory Data Analysis."""
import argparse
import sys
from pathlib import Path

# Add src to pythonpath if executing as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from student_retention.eda.eda_runner import EDARunner
from student_retention.logging_config import get_logger
from student_retention.paths import (
    FIGURES_DIR,
    INTERIM_NORMALIZED_DIR,
    PROCESSED_COHORT_DIR,
    REPORTS_DIR,
    TABLES_DIR,
)

logger = get_logger("run_eda")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute Phase 4 Exploratory Data Analysis and generate figures, tables, and report."
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=PROCESSED_COHORT_DIR,
        help="Directory containing cohort parquet files (default: data/processed/cohort)",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=INTERIM_NORMALIZED_DIR,
        help="Directory containing normalized interim parquet tables (default: data/interim/normalized)",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory to save summary tables (default: outputs/tables)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory to save figures (default: outputs/figures)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory to save EDA report (default: outputs/reports)",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=35,
        help="Prediction cutoff in relative days (default: 35)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting Phase 4 EDA script execution (Cutoff: Day %d)...", args.cutoff)

    runner = EDARunner(
        cohort_dir=args.cohort_dir,
        interim_dir=args.interim_dir,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        reports_dir=args.reports_dir,
        cutoff_day=args.cutoff,
    )
    results = runner.run()

    cohort_summary = results["cohort_summary"]
    print("\nPhase 4 Exploratory Data Analysis Complete:")
    print(f"  - Cohort analyzed:      {cohort_summary['total_records']:,} students")
    print(f"  - Post-cutoff withdrawn:{cohort_summary['positive_cases']:,} ({cohort_summary['base_rate_pct']}%)")
    print(f"  - Retained completers:  {cohort_summary['negative_cases']:,}")
    print(f"  - Tables exported:      {len(results['saved_tables'])} tables in {args.tables_dir}")
    print(f"  - Figures generated:    {len(results['saved_figures'])} figures in {args.figures_dir}")
    print(f"  - Comprehensive report: {results['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
