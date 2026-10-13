"""Data type normalization and schema conformance for OULAD tables."""
from typing import Any

import pandas as pd

from student_retention.config import get_schema_config
from student_retention.logging_config import get_logger

logger = get_logger(__name__)

TYPE_MAP = {
    "string": "string",
    "integer": "Int64",
    "float": "Float64",
    "boolean": "boolean",
}


class DataCoercionError(TypeError):
    """Raised when a non-null source value cannot be coerced to the declared schema type."""

    def __init__(
        self,
        table_name: str,
        column: str,
        expected_type: str,
        malformed_count: int,
        sample_values: list[Any],
    ):
        self.table_name = table_name
        self.column = column
        self.expected_type = expected_type
        self.malformed_count = malformed_count
        self.sample_values = sample_values
        super().__init__(
            f"Table '{table_name}' column '{column}' contains {malformed_count} malformed non-null value(s) "
            f"that cannot be coerced to '{expected_type}'. Example values: {sample_values}"
        )


def normalize_table(
    table_name: str,
    df: pd.DataFrame,
    schema_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Normalize a table's columns and data types according to schema.yaml.

    Distinguishes legitimate missing values (e.g. pd.NA, '?') from malformed
    non-null values that cannot be parsed into the declared type.

    Args:
        table_name: Name of table (e.g. 'student_info')
        df: Raw or pre-cleaned DataFrame
        schema_cfg: Optional parsed schema.yaml

    Returns:
        DataFrame with standardized types (Int64, Float64, string) and pd.NA nulls.

    Raises:
        DataCoercionError: If non-null values cannot be coerced to the declared type.
        ValueError: If required columns defined in schema are missing from df.
    """
    schema = schema_cfg or get_schema_config()
    raw_schemas = schema.get("raw_schemas", {})

    if table_name in raw_schemas:
        table_schema = raw_schemas[table_name]
    else:
        table_schema = schema.get("tables", {}).get(table_name, {}).get("columns", {})

    if not table_schema:
        logger.warning("No schema specification found for table '%s'. Returning as-is.", table_name)
        return df.copy()

    normalized_df = df.copy()

    # Verify column presence
    missing_cols = [col for col in table_schema if col not in normalized_df.columns]
    if missing_cols:
        raise ValueError(
            f"Table '{table_name}' is missing expected schema columns: {missing_cols}"
        )

    for col, spec in table_schema.items():
        col_type = spec.get("type", "string")
        is_nullable = spec.get("nullable", True)

        raw_series = normalized_df[col]

        # Standardize any surviving literal '?' missing markers to pd.NA
        if pd.api.types.is_string_dtype(raw_series.dtype) or pd.api.types.is_object_dtype(raw_series.dtype):
            raw_series = raw_series.mask(raw_series == "?", pd.NA)

        # Identify which values were non-null before type coercion
        was_notna = raw_series.notna()
        # For string columns, empty strings are preserved for schema/missingness audit
        if col_type == "string" and (
            pd.api.types.is_string_dtype(raw_series.dtype) or pd.api.types.is_object_dtype(raw_series.dtype)
        ):
            was_notna = was_notna & (raw_series != "")

        if col_type == "string":
            series = raw_series.astype("string").str.strip()
            normalized_df[col] = series

        elif col_type == "integer":
            coerced_numeric = pd.to_numeric(raw_series, errors="coerce")
            conversion_failed = was_notna & coerced_numeric.isna()

            # Check for non-integer fractional values in integer column
            non_null_numeric = coerced_numeric[coerced_numeric.notna()]
            has_fractions = (non_null_numeric % 1 != 0)

            malformed_mask = conversion_failed.copy()
            if has_fractions.any():
                fraction_indices = non_null_numeric[has_fractions].index
                malformed_mask.loc[fraction_indices] = True

            if malformed_mask.any():
                bad_count = int(malformed_mask.sum())
                bad_examples = list(raw_series[malformed_mask].unique()[:5])
                raise DataCoercionError(
                    table_name=table_name,
                    column=col,
                    expected_type="integer",
                    malformed_count=bad_count,
                    sample_values=bad_examples,
                )

            normalized_df[col] = coerced_numeric.astype("Int64")

        elif col_type == "float":
            coerced_numeric = pd.to_numeric(raw_series, errors="coerce")
            conversion_failed = was_notna & coerced_numeric.isna()

            if conversion_failed.any():
                bad_count = int(conversion_failed.sum())
                bad_examples = list(raw_series[conversion_failed].unique()[:5])
                raise DataCoercionError(
                    table_name=table_name,
                    column=col,
                    expected_type="float",
                    malformed_count=bad_count,
                    sample_values=bad_examples,
                )

            normalized_df[col] = coerced_numeric.astype("Float64")

        elif col_type == "boolean":
            try:
                normalized_df[col] = raw_series.astype("boolean")
            except Exception as e:
                bad_count = int(was_notna.sum())
                bad_examples = list(raw_series[was_notna].unique()[:5])
                raise DataCoercionError(
                    table_name=table_name,
                    column=col,
                    expected_type="boolean",
                    malformed_count=bad_count,
                    sample_values=bad_examples,
                ) from e

        else:
            target_dtype = TYPE_MAP.get(col_type, "string")
            normalized_df[col] = raw_series.astype(target_dtype)

        # Check non-null constraint after coercion
        if not is_nullable:
            null_count = int(normalized_df[col].isna().sum())
            if null_count > 0:
                logger.warning(
                    "Column '%s' in table '%s' has %d nulls after type coercion despite nullable: false",
                    col,
                    table_name,
                    null_count,
                )

    # Reorder columns to match schema specification order, preserving any additional columns at the end
    ordered_cols = [c for c in table_schema if c in normalized_df.columns]
    extra_cols = [c for c in normalized_df.columns if c not in table_schema]
    normalized_df = normalized_df[ordered_cols + extra_cols]

    logger.debug("Successfully normalized %d rows in '%s'.", len(normalized_df), table_name)
    return normalized_df


def normalize_all_tables(
    tables: dict[str, pd.DataFrame],
    schema_cfg: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Normalize all loaded tables in a dictionary.

    Args:
        tables: Dictionary mapping table_name to pd.DataFrame
        schema_cfg: Optional parsed schema.yaml

    Returns:
        Dictionary mapping table_name to normalized pd.DataFrame
    """
    normalized_tables = {}
    for table_name, df in tables.items():
        normalized_tables[table_name] = normalize_table(table_name, df, schema_cfg=schema_cfg)
    return normalized_tables
