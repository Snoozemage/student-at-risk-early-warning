"""CLI script to construct eligible student cohorts and withdrawal target labels."""
import argparse
import sys
from pathlib import Path

# Add src to pythonpath if executing as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from student_retention.logging_config import get_logger
from student_retention.paths import INTERIM_NORMALIZED_DIR, PROCESSED_COHORT_DIR
from student_retention.transformation.build_cohort import CohortBuilder

logger = get_logger("build_cohort")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct eligible student cohorts and withdrawal targets for Phase 3."
    )
    parser.add_argument(
        "--policy",
        choices=["policy_c", "policy_b"],
        default="policy_c",
        help="Cohort eligibility policy (default: policy_c)",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=35,
        help="Prediction cutoff in relative days (default: 35)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=INTERIM_NORMALIZED_DIR,
        help="Directory containing normalized interim tables (default: data/interim/normalized)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_COHORT_DIR,
        help="Directory to save cohort datasets (default: data/processed/cohort)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Executing cohort construction under %s (cutoff: Day %d)...", args.policy.upper(), args.cutoff)

    builder = CohortBuilder(
        cutoff_day=args.cutoff,
        policy=args.policy,
        data_dir=args.data_dir,
    )
    results = builder.run(save_outputs=True, output_dir=args.output_dir)

    audit = results["audit_report"]
    reconciliation = audit["reconciliation"]
    target_dist = audit["target_distribution"]
    saved_paths = results.get("saved_paths", {})

    print("\n" + "=" * 80)
    print(f"OULAD COHORT CONSTRUCTION SUMMARY — {args.policy.upper()} (CUTOFF: DAY {args.cutoff})")
    print("=" * 80)
    print(f"Total Source Records:      {reconciliation['total_source_records']:,}")
    print(f"Total Excluded Records:    {reconciliation['total_excluded_records']:,} ({reconciliation['pct_excluded']}%)")
    print(f"Eligible Cohort Records:   {reconciliation['eligible_cohort_records']:,} ({reconciliation['pct_eligible']}%)")
    print("-" * 80)
    print(f"Positive Cases (Withdrawn): {target_dist['positive_cases']:,} ({target_dist['positive_base_rate_pct']}%)")
    print(f"Negative Cases (Completers):{target_dist['negative_cases']:,}")
    print("Negative Breakdown:")
    for k, v in target_dist.get("negative_class_breakdown", {}).items():
        print(f"  - {k:<15}: {v:,}")
    print("-" * 80)
    print("Mutually Exclusive Exclusion Hierarchy:")
    for step_info in audit.get("exclusion_precedence_hierarchy", {}).values():
        print(f"  Step {step_info['step']} [{step_info['code']}]: {step_info['count']:,} records ({step_info['pct_of_raw']}%)")
        print(f"    -> {step_info['reason']}")
    print("-" * 80)
    print("Saved Artifacts:")
    for path_key, path_val in saved_paths.items():
        print(f"  - {path_key}: {path_val}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
