"""Comprehensive schema, domain, and invariant validation for Day 35 feature tables."""
from typing import Any

import pandas as pd

from student_retention.features.feature_definitions import (
    CANONICAL_GRAIN,
    CANONICAL_TARGET,
    FORBIDDEN_FEATURES,
    PRIMARY_FEATURE_ALLOWLIST,
)


def validate_feature_table(
    df: pd.DataFrame,
    expected_row_count: int | None = None,
    policy_version: str = "Policy C",
) -> dict[str, Any]:
    """Validate feature table against schema, domain bounds, and mathematical invariants.

    Args:
        df: Feature table DataFrame to validate.
        expected_row_count: Optional expected row count (e.g. 27,128 for Policy C).
        policy_version: Policy identifier ("Policy C" or "Policy B").

    Returns:
        dict: Validation report containing check results, error logs, and dataset statistics.

    Raises:
        ValueError: If any critical validation check fails.
    """
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "policy_version": policy_version,
    }

    # 1. Expected row count check
    if expected_row_count is not None and len(df) != expected_row_count:
        errors.append(
            f"Row count mismatch: expected {expected_row_count}, got {len(df)}"
        )

    # 2. Canonical grain uniqueness check
    for key in CANONICAL_GRAIN:
        if key not in df.columns:
            errors.append(f"Canonical grain key missing: '{key}'")

    if all(k in df.columns for k in CANONICAL_GRAIN):
        duplicates = df.duplicated(subset=CANONICAL_GRAIN).sum()
        metrics["duplicate_grain_count"] = int(duplicates)
        if duplicates > 0:
            errors.append(
                f"Canonical grain is not unique: found {duplicates} duplicate rows"
            )

    # 3. Allowlist presence check
    missing_features = [f for f in PRIMARY_FEATURE_ALLOWLIST if f not in df.columns]
    metrics["missing_allowlist_features"] = missing_features
    if missing_features:
        errors.append(
            f"Missing required primary allowlist features ({len(missing_features)}): {missing_features}"
        )

    # 4. Anti-leakage check
    # Check that outcome variables or unrecorded scores are not present.
    # Note: CANONICAL_GRAIN identifiers (like id_student) are permitted as table keys,
    # but forbidden from being predictive features in PRIMARY_FEATURE_ALLOWLIST.
    forbidden_features_in_allowlist = [f for f in FORBIDDEN_FEATURES if f in PRIMARY_FEATURE_ALLOWLIST]
    if forbidden_features_in_allowlist:
        errors.append(
            f"Forbidden leakage features present in PRIMARY_FEATURE_ALLOWLIST: {forbidden_features_in_allowlist}"
        )

    forbidden_present_in_table = [
        f for f in FORBIDDEN_FEATURES
        if f in df.columns and f != CANONICAL_TARGET and f not in CANONICAL_GRAIN
    ]
    if forbidden_present_in_table:
        errors.append(
            f"Prohibited leakage columns present in feature table: {forbidden_present_in_table}"
        )

    # If critical structural columns are missing, abort early before domain checks
    if errors:
        report = {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        }
        raise ValueError("Feature table validation failed:\n" + "\n".join(errors))

    # 5. Nullability check
    null_counts = df[PRIMARY_FEATURE_ALLOWLIST].isna().sum().to_dict()
    metrics["null_counts"] = null_counts

    for feat in PRIMARY_FEATURE_ALLOWLIST:
        null_n = null_counts[feat]
        if feat in ("registration_lead_time_days", "is_late_registration"):
            if policy_version == "Policy C" and null_n > 0:
                errors.append(
                    f"Unexpected nulls in '{feat}' under Policy C: found {null_n} nulls (expected 0)"
                )
            elif policy_version == "Policy B" and null_n > 0:
                metrics[f"{feat}_permitted_nulls"] = int(null_n)
        else:
            if null_n > 0:
                errors.append(
                    f"Feature '{feat}' contains {null_n} unexpected null values"
                )

    # 6. Domain and Value Range Checks
    # Course Context
    invalid_modules = set(df["code_module"].unique()) - {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"}
    if invalid_modules:
        errors.append(f"Invalid code_module values detected: {invalid_modules}")

    invalid_seasons = set(df["presentation_season"].unique()) - {"B", "J"}
    if invalid_seasons:
        errors.append(f"Invalid presentation_season values detected: {invalid_seasons}")

    if (df["studied_credits"] <= 0).any():
        errors.append("studied_credits contains non-positive values")

    if (df["num_of_prev_attempts"] < 0).any():
        errors.append("num_of_prev_attempts contains negative values")

    # Registration
    valid_late_flags = set(df["is_late_registration"].dropna().unique()) - {0, 1}
    if valid_late_flags:
        errors.append(f"Invalid is_late_registration values detected: {valid_late_flags}")

    # VLE Digital Engagement
    vle_non_negative = [
        "vle_total_clicks_window",
        "vle_unique_sites_accessed",
        "vle_pre_course_clicks",
        "vle_recent_clicks_14d",
        "vle_velocity_ratio",
        "vle_clicks_forumng",
        "vle_clicks_oucontent",
        "vle_clicks_quiz",
    ]
    for col in vle_non_negative:
        if (df[col] < 0).any():
            errors.append(f"VLE feature '{col}' contains negative values")

    if ((df["vle_active_days_window"] < 0) | (df["vle_active_days_window"] > 61)).any():
        errors.append("vle_active_days_window out of valid range [0, 61]")

    if ((df["days_since_last_vle_interaction"] < 0) | (df["days_since_last_vle_interaction"] > 65)).any():
        errors.append("days_since_last_vle_interaction out of valid range [0, 65]")

    invalid_inactivity_flags = set(df["has_no_vle_activity_by_cutoff"].unique()) - {0, 1}
    if invalid_inactivity_flags:
        errors.append(f"has_no_vle_activity_by_cutoff contains invalid values: {invalid_inactivity_flags}")

    # Assessment Progress
    assess_counts = [
        "assessments_due_count",
        "submissions_count_lte_cutoff",
        "submitted_on_time_count",
        "late_submissions_count_lte_cutoff",
        "missing_due_submissions_count",
    ]
    for col in assess_counts:
        if (df[col] < 0).any():
            errors.append(f"Assessment feature '{col}' contains negative values")

    # 7. Mathematical Invariant Checks
    # Assessment Counter Partition: submissions = on_time + late
    sub_partition_diff = (
        df["submissions_count_lte_cutoff"]
        - (df["submitted_on_time_count"] + df["late_submissions_count_lte_cutoff"])
    )
    if (sub_partition_diff != 0).any():
        violators = int((sub_partition_diff != 0).sum())
        errors.append(
            f"Assessment partition invariant violated in {violators} rows: "
            f"submissions_count_lte_cutoff != submitted_on_time_count + late_submissions_count_lte_cutoff"
        )

    # Assessment Upper Bounds: submissions <= due and missing <= due
    if (df["submissions_count_lte_cutoff"] > df["assessments_due_count"]).any():
        violators = int((df["submissions_count_lte_cutoff"] > df["assessments_due_count"]).sum())
        errors.append(
            f"Assessment submission upper bound violated in {violators} rows: "
            f"submissions_count_lte_cutoff > assessments_due_count"
        )

    if (df["missing_due_submissions_count"] > df["assessments_due_count"]).any():
        violators = int((df["missing_due_submissions_count"] > df["assessments_due_count"]).sum())
        errors.append(
            f"Assessment missing upper bound violated in {violators} rows: "
            f"missing_due_submissions_count > assessments_due_count"
        )

    # VLE Inactivity Synchronization: has_no_vle_activity == 1 <=> total_clicks == 0
    inactivity_sync_violations = (
        (df["has_no_vle_activity_by_cutoff"] == 1) != (df["vle_total_clicks_window"] == 0)
    ).sum()
    if inactivity_sync_violations > 0:
        errors.append(
            f"VLE inactivity indicator mismatch in {inactivity_sync_violations} rows: "
            f"has_no_vle_activity_by_cutoff == 1 does not match vle_total_clicks_window == 0"
        )

    # VLE Inactivity Sentinel: has_no_vle_activity == 1 => days_since == 65
    sentinel_violations = (
        (df["has_no_vle_activity_by_cutoff"] == 1)
        & (df["days_since_last_vle_interaction"] != 65)
    ).sum()
    if sentinel_violations > 0:
        errors.append(
            f"VLE inactivity sentinel value violated in {sentinel_violations} rows: "
            f"expected days_since_last_vle_interaction == 65 for inactive students"
        )

    # 8. Target Variable Checks (if target is present)
    if CANONICAL_TARGET in df.columns:
        target_vals = set(df[CANONICAL_TARGET].unique()) - {0, 1}
        if target_vals:
            errors.append(f"Canonical target contains non-binary values: {target_vals}")
        else:
            base_rate = float(df[CANONICAL_TARGET].mean())
            metrics["target_positive_base_rate"] = base_rate
            if not (0.10 <= base_rate <= 0.25):
                warnings.append(
                    f"Target base rate {base_rate:.4f} is outside expected range [0.10, 0.25]"
                )

    metrics["validation_status"] = "PASSED" if not errors else "FAILED"
    metrics["error_count"] = len(errors)
    metrics["warning_count"] = len(warnings)

    report = {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }

    if errors:
        raise ValueError(
            f"Feature table validation failed with {len(errors)} errors:\n"
            + "\n".join(errors)
        )

    return report
