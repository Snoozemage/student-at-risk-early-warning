"""Source data manifest and provenance tracking."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from student_retention.config import get_sources_config
from student_retention.logging_config import get_logger
from student_retention.paths import DATA_RAW_DIR, QUALITY_REPORTS_DIR

logger = get_logger(__name__)


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 checksum of a file efficiently."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class SourceManifest:
    """Manifest generator and tracker for raw source datasets."""

    def __init__(self, raw_data_dir: Path | None = None, sources_config: dict | None = None):
        self.sources_config = sources_config or get_sources_config()
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else Path(self.sources_config.get("raw_data_root", DATA_RAW_DIR))
        self.tables_config = self.sources_config.get("tables", {})
        self._cached_manifest: dict[str, Any] | None = None

    def generate(self, force_refresh: bool = False) -> dict[str, Any]:
        """Generate a complete inventory manifest of raw OULAD source tables."""
        if self._cached_manifest is not None and not force_refresh:
            return self._cached_manifest
        manifest = {
            "dataset_name": "Open University Learning Analytics Dataset (OULAD)",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "raw_data_root": str(self.raw_data_dir.as_posix()),
            "format": self.sources_config.get("format", "csv"),
            "missing_value_marker": self.sources_config.get("missing_value_marker", "?"),
            "tables": {},
            "total_raw_rows": 0,
            "total_raw_bytes": 0,
        }

        for table_name, cfg in self.tables_config.items():
            filename = cfg["filename"]
            file_path = self.raw_data_dir / filename

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                manifest["tables"][table_name] = {
                    "filename": filename,
                    "status": "missing",
                }
                continue

            file_size = file_path.stat().st_size
            sha256 = compute_sha256(file_path)

            # Read CSV to get precise dimensions and schema
            df = pd.read_csv(
                file_path,
                na_values=[self.sources_config.get("missing_value_marker", "?")],
                keep_default_na=False,
                dtype=str,
            )
            row_count, col_count = df.shape

            manifest["total_raw_rows"] += row_count
            manifest["total_raw_bytes"] += file_size

            manifest["tables"][table_name] = {
                "filename": filename,
                "file_path": str(file_path.as_posix()),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "sha256": sha256,
                "row_count": row_count,
                "column_count": col_count,
                "columns": list(df.columns),
                "primary_key": cfg.get("primary_key"),
                "composite_grain": cfg.get("composite_grain") or cfg.get("primary_key", []),
                "is_unique_primary_key": cfg.get("is_unique_primary_key", cfg.get("primary_key") is not None),
                "foreign_keys": cfg.get("foreign_keys", {}),
                "null_counts": df.isna().sum().to_dict(),
            }

            logger.info(f"Inventoried '{table_name}': {row_count:,} rows, {file_size / (1024*1024):.2f} MB, sha256={sha256[:8]}...")

        self._cached_manifest = manifest
        return manifest

    def save(
        self,
        output_path: Path | None = None,
        manifest_data: dict[str, Any] | None = None,
    ) -> Path:
        """Save the manifest to JSON.

        If manifest_data is provided or already computed, saves that exact result
        without rescanning or rehashing source files.
        """
        target_path = output_path or (QUALITY_REPORTS_DIR / "source_inventory.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = manifest_data or self._cached_manifest or self.generate()

        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                existing.update(manifest)
                manifest = existing
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not merge with existing manifest: {e}")

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Source manifest successfully saved to {target_path}")
        return target_path


def generate_source_manifest(raw_dir: Path | None = None, output_path: Path | None = None) -> dict[str, Any]:
    """Convenience function to generate and save source manifest in a single pass."""
    manifest_gen = SourceManifest(raw_data_dir=raw_dir)
    manifest = manifest_gen.generate()
    manifest_gen.save(output_path=output_path, manifest_data=manifest)
    return manifest
