"""Label and outcome category validation for OULAD tables."""
from typing import Any

import pandas as pd

from student_retention.logging_config import get_logger

logger = get_logger(__name__)

VALID_FINAL_RESULTS = {"Pass", "Distinction", "Fail", "Withdrawn"}


def validate_source_labels(
    student_info_df: pd.DataFrame,
    student_registration_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Validate source outcome labels and cross-field outcome consistency.

    Checks:
    - final_result domain conformance against documented OULAD categories.
    - Contradictions between final_result and date_unregistration (e.g. Fail with date_unregistration).
    - Known source anomalies (specifically BBB / 2013J / 554243).
    - Withdrawn students with missing date_unregistration.
    - Unregistration before registration chronological contradiction.

    Args:
        student_info_df: DataFrame of student_info table.
        student_registration_df: Optional DataFrame of student_registration table.

    Returns:
        Structured dictionary of label check findings with severity, counts, and examples.
    """
    fatal_violations: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    # 1. Outcome category domain check
    if "final_result" not in student_info_df.columns:
        fatal_violations.append("Column 'final_result' missing from student_info")
        return {
            "is_valid": False,
            "total_fatal_errors": 1,
            "total_warnings": 0,
            "checks": [],
            "violations": fatal_violations,
            "warnings": warnings,
        }

    unknown_mask = ~student_info_df["final_result"].isin(VALID_FINAL_RESULTS)
    unknown_count = int(unknown_mask.sum())

    unknown_examples = []
    if unknown_count > 0:
        bad_rows = student_info_df[unknown_mask].head(5)
        for _, r in bad_rows.iterrows():
            unknown_examples.append({
                "code_module": str(r.get("code_module", "")),
                "code_presentation": str(r.get("code_presentation", "")),
                "id_student": int(r["id_student"]) if pd.notna(r.get("id_student")) else None,
                "final_result": str(r.get("final_result", "")),
            })
        fatal_violations.append(
            f"Found {unknown_count} records with unknown/missing final_result categories (examples: {unknown_examples})"
        )

    checks.append({
        "check_name": "outcome_categories_valid",
        "severity": "ERROR",
        "count": unknown_count,
        "is_valid": unknown_count == 0,
        "examples": unknown_examples,
        "rationale": "Final outcome category must belong to documented OULAD set: Pass, Distinction, Fail, Withdrawn.",
    })

    # Cross-table checks with student_registration
    if student_registration_df is not None:
        join_keys = ["code_module", "code_presentation", "id_student"]
        if all(k in student_info_df.columns for k in join_keys) and all(k in student_registration_df.columns for k in join_keys):
            merged = pd.merge(
                student_info_df[join_keys + ["final_result"]],
                student_registration_df[join_keys + ["date_registration", "date_unregistration"]],
                on=join_keys,
                how="inner",
            )

            # 2. Non-withdrawn with unregistration date (Contradiction)
            non_withdrawn_mask = (
                merged["final_result"].isin(["Pass", "Distinction", "Fail"])
                & merged["date_unregistration"].notna()
            )
            non_withdrawn_unreg_count = int(non_withdrawn_mask.sum())

            non_withdrawn_examples = []
            has_record_554243 = False

            if non_withdrawn_unreg_count > 0:
                for _, r in merged[non_withdrawn_mask].head(10).iterrows():
                    ex = {
                        "code_module": str(r["code_module"]),
                        "code_presentation": str(r["code_presentation"]),
                        "id_student": int(r["id_student"]),
                        "final_result": str(r["final_result"]),
                        "date_unregistration": float(r["date_unregistration"]),
                    }
                    if ex["code_module"] == "BBB" and ex["code_presentation"] == "2013J" and ex["id_student"] == 554243:
                        has_record_554243 = True
                    non_withdrawn_examples.append(ex)

                # Check specifically for BBB / 2013J / 554243 across all matches
                if not has_record_554243:
                    rec_554243 = merged[
                        (merged["code_module"] == "BBB")
                        & (merged["code_presentation"] == "2013J")
                        & (merged["id_student"] == 554243)
                        & non_withdrawn_mask
                    ]
                    if len(rec_554243) > 0:
                        has_record_554243 = True

                warn_msg = (
                    f"Found {non_withdrawn_unreg_count} record(s) where student outcome is non-Withdrawn ('Fail', 'Pass', 'Distinction') "
                    f"but date_unregistration is populated (includes documented anomaly BBB/2013J/554243: {has_record_554243})."
                )
                warnings.append(warn_msg)

            checks.append({
                "check_name": "non_withdrawn_with_unregistration_date",
                "severity": "WARNING",
                "count": non_withdrawn_unreg_count,
                "is_valid": True,  # Non-fatal source anomaly
                "known_record_554243_detected": has_record_554243,
                "examples": non_withdrawn_examples,
                "rationale": (
                    "OULAD documentation specifies that only Withdrawn students have date_unregistration recorded. "
                    "Records such as BBB/2013J/554243 (Fail with date_unregistration=166.0) represent documented source contradictions."
                ),
            })

            # 3. Withdrawn with missing unregistration date
            withdrawn_missing_unreg_mask = (
                (merged["final_result"] == "Withdrawn")
                & merged["date_unregistration"].isna()
            )
            withdrawn_missing_unreg_count = int(withdrawn_missing_unreg_mask.sum())

            withdrawn_missing_unreg_examples = []
            if withdrawn_missing_unreg_count > 0:
                for _, r in merged[withdrawn_missing_unreg_mask].head(5).iterrows():
                    withdrawn_missing_unreg_examples.append({
                        "code_module": str(r["code_module"]),
                        "code_presentation": str(r["code_presentation"]),
                        "id_student": int(r["id_student"]),
                        "final_result": "Withdrawn",
                    })
                warnings.append(
                    f"Found {withdrawn_missing_unreg_count} Withdrawn record(s) with missing date_unregistration timestamp."
                )

            checks.append({
                "check_name": "withdrawn_with_missing_unregistration_date",
                "severity": "WARNING",
                "count": withdrawn_missing_unreg_count,
                "is_valid": True,  # Non-fatal source anomaly
                "examples": withdrawn_missing_unreg_examples,
                "rationale": "Students recorded as Withdrawn but lacking date_unregistration cannot have exact withdrawal timing confirmed.",
            })

            # 4. Unregistration before registration
            both_dates = merged[merged["date_registration"].notna() & merged["date_unregistration"].notna()].copy()
            both_dates["reg_num"] = pd.to_numeric(both_dates["date_registration"], errors="coerce")
            both_dates["unreg_num"] = pd.to_numeric(both_dates["date_unregistration"], errors="coerce")

            unreg_before_reg_mask = both_dates["unreg_num"] < both_dates["reg_num"]
            unreg_before_reg_count = int(unreg_before_reg_mask.sum())

            unreg_before_reg_examples = []
            if unreg_before_reg_count > 0:
                for _, r in both_dates[unreg_before_reg_mask].head(5).iterrows():
                    unreg_before_reg_examples.append({
                        "code_module": str(r["code_module"]),
                        "code_presentation": str(r["code_presentation"]),
                        "id_student": int(r["id_student"]),
                        "date_registration": float(r["reg_num"]),
                        "date_unregistration": float(r["unreg_num"]),
                    })
                fatal_violations.append(
                    f"Found {unreg_before_reg_count} record(s) where date_unregistration < date_registration (chronological contradiction)."
                )

            checks.append({
                "check_name": "unregistration_before_registration",
                "severity": "ERROR",
                "count": unreg_before_reg_count,
                "is_valid": unreg_before_reg_count == 0,
                "examples": unreg_before_reg_examples,
                "rationale": "A student cannot unregister before their registration date.",
            })

    total_fatal_errors = len(fatal_violations)
    total_warnings = len(warnings)

    return {
        "is_valid": total_fatal_errors == 0,
        "total_fatal_errors": total_fatal_errors,
        "total_warnings": total_warnings,
        "checks": checks,
        "violations": fatal_violations,
        "warnings": warnings,
    }
