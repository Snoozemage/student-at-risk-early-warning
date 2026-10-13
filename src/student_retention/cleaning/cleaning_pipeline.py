"""Cleaning pipeline coordinating missingness standardization, identifier validation, and type normalization."""
from pathlib import Path
from typing import Any

import pandas as pd

from student_retention.cleaning.identifiers import validate_identifiers
from student_retention.cleaning.missing_values import (
    audit_missing_values,
    clean_missing_values,
)
from student_retention.cleaning.normalization import normalize_table
from student_retention.config import get_schema_config, get_sources_config
from student_retention.ingestion.readers import OULADReader
from student_retention.logging_config import get_logger
from student_retention.paths import INTERIM_NORMALIZED_DIR, ensure_directories

logger = get_logger(__name__)


class CleaningPipeline:
    """Coordinates ingestion, cleaning, normalization, and persistence of OULAD tables."""

    def __init__(
        self,
        output_dir: Path | None = None,
        sources_config: dict[str, Any] | None = None,
        schema_config: dict[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else INTERIM_NORMALIZED_DIR
        self.sources_cfg = sources_config or get_sources_config()
        self.schema_cfg = schema_config or get_schema_config()
        self.reader = OULADReader(sources_config=self.sources_cfg)

    def clean_table(self, table_name: str, df: pd.DataFrame) -> dict[str, Any]:
        """Clean and normalize a single table DataFrame in-memory.

        Steps:
        1. Replace literal missing markers with pd.NA.
        2. Validate identifier integrity and ranges.
        3. Audit missing value constraints.
        4. Normalize schema data types (Int64, Float64, string).

        Returns:
            Dictionary with:
            - 'normalized_df': cleaned & typed DataFrame
            - 'identifier_report': dict
            - 'missing_report': dict
        """
        logger.info("Cleaning and normalizing table: %s (%d rows)", table_name, len(df))

        # 1. Clean missing markers
        cleaned_df = clean_missing_values(table_name, df)

        # 2. Validate identifiers
        id_report = validate_identifiers(table_name, cleaned_df)
        if not id_report.get("is_valid", True):
            logger.warning(
                "Table '%s' had identifier warnings/errors: %s",
                table_name,
                id_report.get("errors", []),
            )

        # 3. Audit missing values
        missing_report = audit_missing_values(table_name, cleaned_df, schema_cfg=self.schema_cfg)
        if not missing_report.get("is_valid", True):
            logger.warning(
                "Table '%s' had missing value violations: %s",
                table_name,
                missing_report.get("violations", []),
            )

        # 4. Normalize data types
        normalized_df = normalize_table(table_name, cleaned_df, schema_cfg=self.schema_cfg)

        return {
            "table_name": table_name,
            "normalized_df": normalized_df,
            "identifier_report": id_report,
            "missing_report": missing_report,
            "row_count": len(normalized_df),
            "col_count": len(normalized_df.columns),
        }

    def run(
        self,
        tables: dict[str, pd.DataFrame] | None = None,
        save_interim: bool = True,
        save_format: str = "parquet",
    ) -> dict[str, Any]:
        """Execute the cleaning pipeline across all tables.

        Args:
            tables: Optional pre-loaded tables dict. If None, loaded via OULADReader.
            save_interim: Whether to persist normalized tables to disk.
            save_format: 'parquet' or 'csv'.

        Returns:
            Execution summary dict with normalized DataFrames and validation reports.
        """
        ensure_directories()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if tables is None:
            logger.info("Loading raw tables from storage...")
            tables = self.reader.read_all_tables()

        normalized_tables: dict[str, pd.DataFrame] = {}
        reports: dict[str, Any] = {}
        saved_files: dict[str, str] = {}

        for table_name, df in tables.items():
            result = self.clean_table(table_name, df)
            norm_df = result["normalized_df"]
            normalized_tables[table_name] = norm_df
            reports[table_name] = {
                "identifier_report": result["identifier_report"],
                "missing_report": result["missing_report"],
                "row_count": result["row_count"],
                "col_count": result["col_count"],
            }

            if save_interim:
                if save_format == "parquet":
                    out_path = self.output_dir / f"{table_name}.parquet"
                    norm_df.to_parquet(out_path, index=False, engine="pyarrow")
                elif save_format == "csv":
                    out_path = self.output_dir / f"{table_name}.csv"
                    norm_df.to_csv(out_path, index=False)
                else:
                    raise ValueError(f"Unsupported save_format: {save_format}")

                saved_files[table_name] = str(out_path)
                logger.info("Persisted normalized table '%s' to %s", table_name, out_path)

        return {
            "tables": normalized_tables,
            "reports": reports,
            "saved_files": saved_files,
            "status": "success",
        }


def run_cleaning_pipeline(
    output_dir: Path | None = None,
    save_interim: bool = True,
    save_format: str = "parquet",
) -> dict[str, Any]:
    """Convenience function to run cleaning pipeline."""
    pipeline = CleaningPipeline(output_dir=output_dir)
    return pipeline.run(save_interim=save_interim, save_format=save_format)
