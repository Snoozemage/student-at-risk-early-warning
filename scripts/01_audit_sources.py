"""CLI script to audit raw OULAD data sources and generate source_inventory.json."""
import argparse
import sys
from pathlib import Path

# Add src to pythonpath if executing as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from student_retention.ingestion.manifest import SourceManifest
from student_retention.logging_config import get_logger
from student_retention.paths import DATA_RAW_DIR, QUALITY_REPORTS_DIR

logger = get_logger("audit_sources")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit raw OULAD sources and generate source inventory.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help="Path to raw data directory (default: data/raw/oulad)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=QUALITY_REPORTS_DIR / "source_inventory.json",
        help="Path to output inventory JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting source inventory audit from %s", args.raw_dir)

    manifest_gen = SourceManifest(raw_data_dir=args.raw_dir)
    manifest = manifest_gen.generate()
    output_path = manifest_gen.save(output_path=args.output, manifest_data=manifest)

    print("\n" + "=" * 80)
    print("OULAD RAW DATA INVENTORY AUDIT SUMMARY")
    print("=" * 80)
    print(f"Dataset: {manifest['dataset_name']}")
    print(f"Generated at: {manifest['generated_at_utc']}")
    print(f"Raw data root: {manifest['raw_data_root']}")
    print(f"Total raw rows: {manifest['total_raw_rows']:,}")
    print(f"Total raw size: {manifest['total_raw_bytes'] / (1024 * 1024):.2f} MB")
    print("-" * 80)
    print(f"{'Table':<22} {'File':<25} {'Rows':>12} {'Size (MB)':>10} {'Status':>8}")
    print("-" * 80)

    for t_name, t_info in manifest["tables"].items():
        if t_info.get("status") == "missing":
            print(f"{t_name:<22} {t_info['filename']:<25} {'N/A':>12} {'N/A':>10} {'MISSING':>8}")
        else:
            print(
                f"{t_name:<22} {t_info['filename']:<25} {t_info['row_count']:>12,} {t_info['file_size_mb']:>10.2f} {'OK':>8}"
            )

    print("=" * 80)
    print(f"Report written to: {output_path}\n")


if __name__ == "__main__":
    main()
