"""Primary key uniqueness and grain validation for OULAD tables."""
from typing import Any

import pandas as pd

from student_retention.config import get_quality_rules_config
from student_retention.logging_config import get_logger

logger = get_logger(__name__)


def validate_table_primary_key(
    table_name: str,
    df: pd.DataFrame,
    rules_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate primary key uniqueness and null-freedom for a table.

    Distinguishes strict unique primary keys from permitted repeated grains
    in event log tables (such as student_vle).

    Args:
        table_name: Name of table
        df: Table DataFrame
        rules_cfg: Parsed quality_rules.yaml (optional)

    Returns:
        Structured result dict with key uniqueness and grain repetition metrics.
    """
    quality_rules = rules_cfg or get_quality_rules_config()
    source_rules = quality_rules.get("source_quality_rules", {})
    key_uniqueness = source_rules.get("key_uniqueness", {})

    rule = key_uniqueness.get(table_name)
    if not rule:
        logger.debug("No key uniqueness rule configured for '%s'", table_name)
        return {
            "table_name": table_name,
            "grain": [],
            "is_primary_key": True,
            "is_valid": True,
            "duplicate_count": 0,
            "null_key_count": 0,
            "violations": [],
        }

    grain: list[str] = rule.get("grain", [])
    allow_duplicates: bool = rule.get("allow_duplicates", False)
    is_primary_key: bool = rule.get("is_primary_key", not allow_duplicates)
    semantic_type: str = rule.get("semantic_type", "entity_table" if is_primary_key else "event_log")

    violations: list[str] = []

    # Check key column presence
    missing_keys = [k for k in grain if k not in df.columns]
    if missing_keys:
        violations.append(f"Table '{table_name}' missing key columns: {missing_keys}")
        return {
            "table_name": table_name,
            "grain": grain,
            "is_primary_key": is_primary_key,
            "semantic_type": semantic_type,
            "is_valid": False,
            "duplicate_count": 0,
            "null_key_count": 0,
            "violations": violations,
        }

    # Check for nulls in primary key columns
    null_keys = 0
    for col in grain:
        col_nulls = int(df[col].isna().sum())
        if col_nulls > 0:
            null_keys += col_nulls
            violations.append(f"Key column '{col}' contains {col_nulls} null values in '{table_name}'")

    total_rows = len(df)
    dup_mask = df.duplicated(subset=grain, keep=False)
    rows_in_repeated_groups = int(dup_mask.sum())
    distinct_keys = int(df[grain].drop_duplicates().shape[0])
    excess_duplicate_rows = total_rows - distinct_keys
    repeated_keys_count = distinct_keys - int((~dup_mask).sum()) if rows_in_repeated_groups > 0 else 0

    if is_primary_key and not allow_duplicates and rows_in_repeated_groups > 0:
        violations.append(
            f"Table '{table_name}' violates primary key uniqueness: {rows_in_repeated_groups} rows "
            f"belong to repeated key groups across grain {grain} "
            f"({total_rows} total rows, {distinct_keys} unique keys, {excess_duplicate_rows} excess duplicates)"
        )

    is_valid = len(violations) == 0

    return {
        "table_name": table_name,
        "grain": grain,
        "is_primary_key": is_primary_key,
        "allow_duplicates": allow_duplicates,
        "semantic_type": semantic_type,
        "is_valid": is_valid,
        "total_rows": total_rows,
        "distinct_keys": distinct_keys,
        "rows_in_repeated_groups": rows_in_repeated_groups,
        "excess_duplicate_rows": excess_duplicate_rows,
        "repeated_keys_count": repeated_keys_count,
        "null_key_count": null_keys,
        "violations": violations,
    }


def validate_all_primary_keys(
    tables: dict[str, pd.DataFrame],
    rules_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate primary key uniqueness and audit grain repetition for all tables."""
    results = {}
    all_valid = True
    total_violations = 0

    for table_name, df in tables.items():
        res = validate_table_primary_key(table_name, df, rules_cfg=rules_cfg)
        results[table_name] = res
        if not res["is_valid"]:
            all_valid = False
            total_violations += len(res["violations"])

    return {
        "is_valid": all_valid,
        "total_violations": total_violations,
        "tables": results,
    }
