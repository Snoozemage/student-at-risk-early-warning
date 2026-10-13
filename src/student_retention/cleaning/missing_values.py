"""Missing values auditing and cleaning according to schema contracts."""
from typing import Any

import pandas as pd

from student_retention.config import get_quality_rules_config, get_schema_config
from student_retention.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_FORBIDDEN_MARKERS = ["", "null", "NULL", "na", "NA", "None"]


def audit_missing_values(
    table_name: str,
    df: pd.DataFrame,
    schema_cfg: dict | None = None,
    quality_rules_cfg: dict | None = None,
) -> dict[str, Any]:
    """Audit missing values against schema and quality rules contracts.

    Checks:
    - Non-null constraints defined in schema.yaml.
    - Forbidden missing markers defined in quality_rules.yaml (e.g. '', 'null', 'NA').

    Args:
        table_name: Name of table
        df: DataFrame to audit
        schema_cfg: Parsed schema.yaml data (optional)
        quality_rules_cfg: Parsed quality_rules.yaml data (optional)

    Returns:
        Dictionary with audit metrics, violations, and null counts.
    """
    schema = schema_cfg or get_schema_config()
    quality_rules = quality_rules_cfg or get_quality_rules_config()

    raw_schemas = schema.get("raw_schemas", {})
    if table_name in raw_schemas:
        table_schema = raw_schemas[table_name]
    else:
        table_schema = schema.get("tables", {}).get(table_name, {}).get("columns", {})

    forbidden_markers = (
        quality_rules.get("source_quality_rules", {})
        .get("missing_value_marker", {})
        .get("forbidden_markers", DEFAULT_FORBIDDEN_MARKERS)
    )

    null_counts = df.isna().sum().to_dict()
    null_percentages = (
        {col: round(count / len(df) * 100, 2) for col, count in null_counts.items()}
        if len(df) > 0
        else {}
    )

    violations: list[str] = []
    forbidden_marker_violations: list[dict[str, Any]] = []

    # 1. Non-nullable columns check
    for col, count in null_counts.items():
        col_spec = table_schema.get(col, {})
        is_nullable = col_spec.get("nullable", True)

        if count > 0 and not is_nullable:
            violations.append(
                f"Column '{col}' in '{table_name}' has {count} nulls but is defined as non-nullable"
            )
        elif count > 0:
            logger.debug("Column '%s' has %d nulls (%s%%) - permitted by schema.", col, count, null_percentages[col])

    # 2. Check for configured forbidden markers across string and object columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        col_s = df[col]
        for marker in forbidden_markers:
            if marker == "":
                marker_mask = (col_s == "") | (col_s.notna() & (col_s.astype(str).str.strip() == ""))
            else:
                marker_mask = (col_s == marker) | (col_s.notna() & (col_s.astype(str).str.strip() == marker))
            marker_count = int(marker_mask.sum())
            if marker_count > 0:
                marker_repr = "empty string" if marker == "" else f"'{marker}'"
                violations.append(
                    f"Column '{col}' in '{table_name}' contains {marker_count} forbidden marker {marker_repr} values"
                )
                forbidden_marker_violations.append({
                    "table": table_name,
                    "column": col,
                    "marker": marker,
                    "count": marker_count,
                })

    return {
        "table_name": table_name,
        "total_rows": len(df),
        "null_counts": null_counts,
        "null_percentages": null_percentages,
        "forbidden_marker_violations": forbidden_marker_violations,
        "violations": violations,
        "is_valid": len(violations) == 0,
    }


def clean_missing_values(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Standardize missing values according to documented rules.

    Never replaces missing values with 0 by default.
    Ensures missing values are consistently represented as pd.NA.
    Converts only the configured OULAD missing-value marker ('?') to pd.NA.
    Preserves other literal strings (e.g. '\\?') for inspection and auditing.
    """
    cleaned_df = df.copy()

    # Convert only the configured '?' string to pd.NA
    cleaned_df = cleaned_df.replace({"?": pd.NA})

    return cleaned_df
