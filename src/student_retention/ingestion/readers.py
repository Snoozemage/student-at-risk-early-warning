"""Source readers for OULAD tables."""
from pathlib import Path

import pandas as pd

from student_retention.config import get_sources_config
from student_retention.logging_config import get_logger
from student_retention.paths import DATA_RAW_DIR

logger = get_logger(__name__)


class SourceError(Exception):
    """Base exception for source ingestion errors."""


class SourceNotFoundError(SourceError):
    """Raised when a required source file does not exist."""


class SourceSchemaError(SourceError):
    """Raised when source columns do not match expectation."""


class SourceRowMismatchError(SourceError):
    """Raised when source row count mismatches expectation strictly."""


class OULADReader:
    """Reader for Open University Learning Analytics Dataset (OULAD) CSV files."""

    def __init__(self, raw_data_dir: Path | None = None, sources_config: dict | None = None):
        self.sources_config = sources_config or get_sources_config()
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else Path(self.sources_config.get("raw_data_root", DATA_RAW_DIR))
        self.missing_marker = self.sources_config.get("missing_value_marker", "?")
        self.tables_config = self.sources_config.get("tables", {})

    def get_table_path(self, table_name: str) -> Path:
        """Resolve the absolute path to a table file."""
        if table_name not in self.tables_config:
            raise KeyError(f"Unknown table '{table_name}'. Configured tables: {list(self.tables_config.keys())}")
        filename = self.tables_config[table_name]["filename"]
        return self.raw_data_dir / filename

    def read_table(
        self,
        table_name: str,
        validate_counts: bool = True,
        strict_counts: bool = False,
    ) -> pd.DataFrame:
        """Read a single OULAD table from CSV with schema and missing value handling.

        Args:
            table_name: Name of table (e.g. 'student_info')
            validate_counts: Whether to check row and column counts against sources.yaml
            strict_counts: If True, raise exception on row mismatch; if False, log warning.

        Returns:
            pd.DataFrame containing the raw ingested data with '?' converted to pd.NA.
        """
        table_cfg = self.tables_config.get(table_name)
        if not table_cfg:
            raise KeyError(f"Table '{table_name}' not configured in sources.yaml")

        file_path = self.get_table_path(table_name)
        if not file_path.exists():
            raise SourceNotFoundError(f"Source file not found for table '{table_name}': {file_path}")

        logger.info(f"Reading table '{table_name}' from {file_path.name}...")

        # In OULAD, only '?' denotes missing values.
        # We explicitly set na_values to [self.missing_marker] and keep_default_na=False to avoid
        # accidentally converting valid strings like 'NA' or empty values without auditing.
        df = pd.read_csv(
            file_path,
            na_values=[self.missing_marker],
            keep_default_na=False,
            dtype=str,  # Read raw as strings first to prevent loss of precision or silent coercion
        )

        row_count, col_count = df.shape
        expected_rows = table_cfg.get("expected_rows")
        expected_cols = table_cfg.get("expected_cols")

        if validate_counts:
            if expected_cols is not None and col_count != expected_cols:
                raise SourceSchemaError(
                    f"Table '{table_name}' column count mismatch: expected {expected_cols}, got {col_count}"
                )
            if expected_rows is not None and row_count != expected_rows:
                msg = f"Table '{table_name}' row count mismatch: expected {expected_rows}, got {row_count}"
                if strict_counts:
                    raise SourceRowMismatchError(msg)
                else:
                    logger.warning(msg)

        logger.info(f"Successfully loaded '{table_name}': {row_count:,} rows, {col_count} columns.")
        return df

    def read_all(self, validate_counts: bool = True, strict_counts: bool = False) -> dict[str, pd.DataFrame]:
        """Read all configured OULAD tables."""
        tables = {}
        for table_name in self.tables_config:
            tables[table_name] = self.read_table(
                table_name,
                validate_counts=validate_counts,
                strict_counts=strict_counts,
            )
        return tables

    # Alias for convenience
    read_all_tables = read_all


def load_table(table_name: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Convenience helper to read a single table."""
    reader = OULADReader(raw_data_dir=raw_dir)
    return reader.read_table(table_name)


def load_all_raw_tables(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Convenience helper to read all raw tables."""
    reader = OULADReader(raw_data_dir=raw_dir)
    return reader.read_all()
