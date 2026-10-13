"""Quality report generation and JSON persistence for Phase 2 validation."""
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from student_retention.logging_config import get_logger
from student_retention.paths import QUALITY_REPORTS_DIR
from student_retention.validation.key_checks import validate_all_primary_keys
from student_retention.validation.label_checks import validate_source_labels
from student_retention.validation.relationship_checks import validate_all_relationships
from student_retention.validation.schema_checks import validate_all_schemas
from student_retention.validation.temporal_checks import (
    validate_source_temporal_properties,
)

logger = get_logger(__name__)


def generate_quality_report(
    tables: dict[str, pd.DataFrame],
    missing_reports: dict[str, Any] | None = None,
    identifier_reports: dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive data quality report for all normalized tables.

    Args:
        tables: Dictionary of normalized table DataFrames.
        missing_reports: Optional per-table missing values audit reports.
        identifier_reports: Optional per-table identifier validation reports.
        output_path: Optional path to save JSON report.

    Returns:
        Consolidated dictionary with overall status, metrics, and detailed checks.
    """
    logger.info("Executing full suite of data quality checks...")

    # 1. Schema checks
    schema_results = validate_all_schemas(tables)

    # 2. Key uniqueness checks
    key_results = validate_all_primary_keys(tables)

    # 3. Referential integrity checks
    rel_results = validate_all_relationships(tables)

    # 4. Label and outcome consistency checks
    label_results: dict[str, Any] = {}
    if "student_info" in tables:
        label_results = validate_source_labels(
            student_info_df=tables["student_info"],
            student_registration_df=tables.get("student_registration"),
        )

    # 5. Temporal properties and chronology checks
    temporal_results = validate_source_temporal_properties(tables)

    # Aggregate fatal violations
    all_violations: list[str] = []
    all_warnings: list[str] = []

    for s_res in schema_results.get("tables", {}).values():
        all_violations.extend(s_res.get("violations", []))

    for k_res in key_results.get("tables", {}).values():
        all_violations.extend(k_res.get("violations", []))

    for r_res in rel_results.get("relationships", []):
        all_violations.extend(r_res.get("violations", []))

    if missing_reports:
        for m_res in missing_reports.values():
            all_violations.extend(m_res.get("violations", []))

    if identifier_reports:
        for i_res in identifier_reports.values():
            all_violations.extend(i_res.get("errors", []))

    if label_results:
        all_violations.extend(label_results.get("violations", []))
        all_warnings.extend(label_results.get("warnings", []))

    if temporal_results:
        all_violations.extend(temporal_results.get("violations", []))
        all_warnings.extend(temporal_results.get("warnings", []))

    is_overall_valid = len(all_violations) == 0

    table_summaries = {}
    for name, df in tables.items():
        table_summaries[name] = {
            "row_count": len(df),
            "col_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        }

    # Separate strict primary key validations from permitted grain repetitions
    pk_validation_tables = {}
    grain_repetition_audit = {}
    for t_name, k_res in key_results.get("tables", {}).items():
        if k_res.get("is_primary_key", True):
            pk_validation_tables[t_name] = k_res
        else:
            grain_repetition_audit[t_name] = k_res

    configured_rel_count = rel_results.get("configured_rules_count", len(rel_results.get("relationships", [])))
    executed_rel_count = rel_results.get("executed_rules_count", len(rel_results.get("relationships", [])))

    report = {
        "report_type": "data_quality_report",
        "phase": "Phase 2 - Ingestion, Cleaning and Data Quality",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": "PASSED" if is_overall_valid else "FAILED",
        "summary": {
            "tables_audited": len(tables),
            "total_rows_audited": sum(len(df) for df in tables.values()),
            "total_violations": len(all_violations),
            "total_warnings": len(all_warnings),
            "schema_valid": schema_results["is_valid"],
            "primary_keys_valid": key_results["is_valid"],
            "referential_integrity_valid": rel_results["is_valid"],
            "label_valid": label_results.get("is_valid", True),
            "temporal_valid": temporal_results.get("is_valid", True),
            "relationship_rules_configured": configured_rel_count,
            "relationship_rules_executed": executed_rel_count,
        },
        "all_violations": all_violations,
        "all_warnings": all_warnings,
        "table_summaries": table_summaries,
        "schema_validation": schema_results,
        "key_validation": {
            "is_valid": key_results["is_valid"],
            "total_violations": key_results["total_violations"],
            "tables": key_results["tables"],
            "unique_primary_key_checks": pk_validation_tables,
            "permitted_grain_repetition_audit": grain_repetition_audit,
        },
        "relationship_validation": rel_results,
        "label_validation": label_results,
        "temporal_validation": temporal_results,
        "missing_values_audit": missing_reports or {},
        "identifier_validation": identifier_reports or {},
    }

    report_dest = output_path or (QUALITY_REPORTS_DIR / "data_quality_report.json")
    report_dest.parent.mkdir(parents=True, exist_ok=True)

    with open(report_dest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved data quality report to %s (Status: %s)", report_dest, report["status"])
    return report
