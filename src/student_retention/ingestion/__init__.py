"""Ingestion package for OULAD data sources."""
from student_retention.ingestion.manifest import (
    SourceManifest,
    generate_source_manifest,
)
from student_retention.ingestion.readers import (
    OULADReader,
    load_all_raw_tables,
    load_table,
)

__all__ = [
    "OULADReader",
    "SourceManifest",
    "generate_source_manifest",
    "load_all_raw_tables",
    "load_table",
]
