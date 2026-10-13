"""Schema and column constraint validation for normalized OULAD tables."""
from typing import Any

import pandas as pd

from student_retention.config import get_schema_config
from student_retention.logging_config import get_logger

logger = get_logger(__name__)


def validate_table_schema(
    table_name: str,
    df: pd.DataFrame,
    schema_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a table DataFrame against schema.yaml contracts.

    Checks:
    - Expected columns existence.
    - Column data types and actual data-type conformance (detects malformed tokens).
    - Non-null constraints.
    - Allowed categorical values.
    - Min/max numeric range boundaries.

    Args:
        table_name: Name of table
        df: DataFrame to validate
        schema_cfg: Parsed schema.yaml (optional)

    Returns:
        Structured validation result dict.
    """
    schema = schema_cfg or get_schema_config()
    raw_schemas = schema.get("raw_schemas", {})

    if table_name in raw_schemas:
        table_schema = raw_schemas[table_name]
    else:
        table_schema = schema.get("tables", {}).get(table_name, {}).get("columns", {})

    violations: list[str] = []
    warnings: list[str] = []
    column_results: dict[str, dict[str, Any]] = {}

    if not table_schema:
        warnings.append(f"No schema specification found for table '{table_name}'")
        return {
            "table_name": table_name,
            "is_valid": True,
            "violations": violations,
            "warnings": warnings,
            "column_checks": column_results,
        }

    for expected_col, col_def in table_schema.items():
        if expected_col not in df.columns:
            violations.append(f"Missing required column '{expected_col}' in '{table_name}'")
            column_results[expected_col] = {
                "present": False,
                "is_valid": False,
                "violations": ["Missing column"],
            }
            continue

        col_violations: list[str] = []
        col_series = df[expected_col]
        col_type = col_def.get("type", "string")
        is_nullable = col_def.get("nullable", True)
        null_count = int(col_series.isna().sum())

        # 1. Nullability check
        if not is_nullable and null_count > 0:
            col_violations.append(
                f"Contains {null_count} nulls but nullable: false"
            )

        # 2. Actual data-type conformance check
        was_notna = col_series.notna() & (col_series != "") & (col_series != "?")
        coerced_numeric = None

        if col_type == "integer":
            if not pd.api.types.is_integer_dtype(col_series.dtype):
                col_violations.append(
                    f"Column '{expected_col}' has incompatible dtype '{col_series.dtype}', expected integer dtype"
                )

            coerced_numeric = pd.to_numeric(col_series, errors="coerce")
            non_numeric = was_notna & coerced_numeric.isna()
            non_null_numeric = coerced_numeric[coerced_numeric.notna()]
            has_fraction = (non_null_numeric % 1 != 0)

            malformed_mask = non_numeric.copy()
            if has_fraction.any():
                fraction_indices = non_null_numeric[has_fraction].index
                malformed_mask.loc[fraction_indices] = True

            if malformed_mask.any():
                bad_count = int(malformed_mask.sum())
                bad_examples = list(col_series[malformed_mask].unique()[:5])
                col_violations.append(
                    f"Contains {bad_count} malformed non-integer values (examples: {bad_examples})"
                )

        elif col_type == "float":
            if not (pd.api.types.is_float_dtype(col_series.dtype) or pd.api.types.is_integer_dtype(col_series.dtype)):
                col_violations.append(
                    f"Column '{expected_col}' has incompatible dtype '{col_series.dtype}', expected float dtype"
                )

            coerced_numeric = pd.to_numeric(col_series, errors="coerce")
            non_numeric = was_notna & coerced_numeric.isna()
            if non_numeric.any():
                bad_count = int(non_numeric.sum())
                bad_examples = list(col_series[non_numeric].unique()[:5])
                col_violations.append(
                    f"Contains {bad_count} malformed non-float values (examples: {bad_examples})"
                )

        elif col_type == "string":
            if not (pd.api.types.is_string_dtype(col_series.dtype) or pd.api.types.is_object_dtype(col_series.dtype)):
                col_violations.append(
                    f"Column '{expected_col}' has incompatible dtype '{col_series.dtype}', expected string dtype"
                )

        # 3. Allowed values check
        allowed = col_def.get("allowed_values")
        if allowed is not None:
            non_null_vals = col_series.dropna()
            # Exclude legitimate missing markers if raw string
            non_null_vals = non_null_vals[non_null_vals != "?"]
            allowed_set = set(allowed)
            if col_type == "string":
                unique_vals = set(non_null_vals.astype(str).unique())
                invalid_vals = unique_vals - {str(a) for a in allowed_set}
            else:
                unique_vals = set(non_null_vals.unique())
                invalid_vals = unique_vals - allowed_set

            if invalid_vals:
                col_violations.append(
                    f"Contains values not in allowed_values: {list(invalid_vals)[:5]}"
                )

        # 4. Numeric range bounds check
        min_bound = col_def.get("min")
        max_bound = col_def.get("max")
        if min_bound is not None or max_bound is not None:
            if coerced_numeric is None:
                coerced_numeric = pd.to_numeric(col_series, errors="coerce")
            numeric_vals = coerced_numeric.dropna()

            if min_bound is not None and len(numeric_vals) > 0:
                below_min = int((numeric_vals < min_bound).sum())
                if below_min > 0:
                    col_violations.append(
                        f"Contains {below_min} values below min={min_bound} (lowest: {numeric_vals.min()})"
                    )
            if max_bound is not None and len(numeric_vals) > 0:
                above_max = int((numeric_vals > max_bound).sum())
                if above_max > 0:
                    col_violations.append(
                        f"Contains {above_max} values above max={max_bound} (highest: {numeric_vals.max()})"
                    )

        if col_violations:
            violations.extend([f"[{table_name}.{expected_col}] {v}" for v in col_violations])

        column_results[expected_col] = {
            "present": True,
            "dtype": str(col_series.dtype),
            "expected_type": col_type,
            "null_count": null_count,
            "is_valid": len(col_violations) == 0,
            "violations": col_violations,
        }

    # Unexpected columns
    unexpected_cols = [c for c in df.columns if c not in table_schema]
    if unexpected_cols:
        warnings.append(f"Unexpected columns found in '{table_name}': {unexpected_cols}")

    is_valid = len(violations) == 0

    return {
        "table_name": table_name,
        "is_valid": is_valid,
        "total_rows": len(df),
        "violations": violations,
        "warnings": warnings,
        "column_checks": column_results,
    }


def validate_all_schemas(
    tables: dict[str, pd.DataFrame],
    schema_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all tables against their schema specifications."""
    results = {}
    all_valid = True
    total_violations = 0

    for table_name, df in tables.items():
        res = validate_table_schema(table_name, df, schema_cfg=schema_cfg)
        results[table_name] = res
        if not res["is_valid"]:
            all_valid = False
            total_violations += len(res["violations"])

    return {
        "is_valid": all_valid,
        "total_violations": total_violations,
        "tables": results,
    }
