"""Target definition, extraction, and outcome leakage assertions."""
from typing import Any

import numpy as np
import pandas as pd

from student_retention.logging_config import get_logger
from student_retention.validation.relationship_checks import validate_relationship
from student_retention.validation.schema_checks import validate_courses_reference_table

logger = get_logger(__name__)

FORBIDDEN_PREDICTOR_COLUMNS = {
    "final_result",
    "date_unregistration",
    "withdrawn_after_cutoff",
    "target_withdrawn",
    "withdrawal_target",
}

ALLOWED_FINAL_RESULTS = {"Pass", "Fail", "Withdrawn", "Distinction"}


def validate_canonical_keys(
    df: pd.DataFrame,
    table_name: str,
    error_cls: type[Exception] = ValueError,
) -> None:
    """Validate canonical key columns for presence, missingness, blanks, and valid positive integer id_student.

    Args:
        df: Input DataFrame to validate.
        table_name: Name of the table for descriptive error reporting.
        error_cls: Exception class to raise on validation failure.

    Raises:
        error_cls: If any canonical key column is missing, contains nulls/blanks,
            or if id_student contains non-integer, non-positive, or malformed values.
    """
    grain = ["code_module", "code_presentation", "id_student"]

    # 1. Verify presence of canonical columns
    missing_cols = [col for col in grain if col not in df.columns]
    if missing_cols:
        raise error_cls(f"{table_name} is missing canonical key columns: {missing_cols}")

    # 2. Check string keys: code_module and code_presentation must be string-typed
    for col in ["code_module", "code_presentation"]:
        series = df[col]
        # Enforce string type: reject numeric, boolean, or other non-string dtypes
        if not (pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype)):
            raise error_cls(
                f"{table_name} column '{col}' has dtype '{series.dtype}' but must be string-typed "
                f"(expected str/object for canonical module/presentation codes)."
            )

        # For object-dtype columns, validate individual non-null elements
        null_mask = series.isna()
        if pd.api.types.is_object_dtype(series.dtype):
            non_null_vals = series[~null_mask]
            if len(non_null_vals) > 0:
                non_string_mask = pd.Series(
                    [not isinstance(v, str) for v in non_null_vals],
                    index=non_null_vals.index,
                )
                if non_string_mask.any():
                    bad_count = int(non_string_mask.sum())
                    bad_examples = list(non_null_vals[non_string_mask].unique()[:3])
                    raise error_cls(
                        f"{table_name} column '{col}' contains {bad_count} non-string value(s) "
                        f"(sample: {bad_examples}). Expected valid string module/presentation codes."
                    )

        # Identify nulls or whitespace-only/blank strings
        null_mask = null_mask | (series.astype(str).str.strip() == "")
        bad_count = int(null_mask.sum())
        if bad_count > 0:
            raise error_cls(
                f"{table_name} contains {bad_count} null/missing values in canonical key column '{col}' "
                f"(empty or whitespace-only keys are strictly rejected)."
            )

    # 3. Validate id_student: must be a valid positive integer
    id_series = df["id_student"]

    # Reject boolean dtype (bool is a subclass of int in Python; True/False are not valid student IDs)
    if pd.api.types.is_bool_dtype(id_series.dtype):
        bad_count = int(id_series.notna().sum())
        bad_examples = list(id_series.dropna().unique()[:3])
        raise error_cls(
            f"{table_name} contains {bad_count} boolean value(s) in canonical key 'id_student' (sample: {bad_examples}). "
            f"Expected valid positive integers."
        )

    # Reject float dtypes explicitly (e.g. float64, Float64)
    if pd.api.types.is_float_dtype(id_series.dtype):
        bad_count = int(id_series.notna().sum())
        bad_examples = list(id_series.dropna().unique()[:3])
        raise error_cls(
            f"{table_name} column 'id_student' has float dtype '{id_series.dtype}' with {bad_count} "
            f"float/fractional value(s) in canonical key 'id_student' (sample: {bad_examples}). "
            f"Expected valid integer student IDs."
        )

    # Reject string dtype (StringDtype)
    if isinstance(id_series.dtype, pd.StringDtype):
        bad_count = int(id_series.notna().sum())
        bad_examples = list(id_series.dropna().unique()[:3])
        raise error_cls(
            f"{table_name} contains {bad_count} non-numeric/malformed value(s) (strings where integer required) "
            f"in canonical key 'id_student' (sample: {bad_examples}). "
            f"Expected valid positive integers."
        )

    # Reject other unsupported non-integer dtypes
    if not (pd.api.types.is_integer_dtype(id_series.dtype) or pd.api.types.is_object_dtype(id_series.dtype)):
        bad_count = int(id_series.notna().sum())
        bad_examples = list(id_series.dropna().unique()[:3])
        raise error_cls(
            f"{table_name} contains {bad_count} non-numeric/malformed value(s) (strings where integer required) "
            f"in canonical key 'id_student' (sample: {bad_examples}). "
            f"Expected valid positive integers."
        )

    # Identify nulls or blank strings in id_student
    id_null_mask = id_series.isna()
    if pd.api.types.is_object_dtype(id_series.dtype) or isinstance(id_series.dtype, pd.StringDtype):
        id_null_mask = id_null_mask | (id_series.astype(str).str.strip() == "")
    id_null_count = int(id_null_mask.sum())
    if id_null_count > 0:
        raise error_cls(
            f"{table_name} contains {id_null_count} null/missing values in canonical key column 'id_student' "
            f"(empty or whitespace-only keys are strictly rejected)."
        )

    non_null_ids = id_series[~id_null_mask]

    # For object-dtype series, inspect individual non-null elements
    if pd.api.types.is_object_dtype(id_series.dtype):
        # Catch booleans: both Python bool and NumPy np.bool_
        bool_mask = pd.Series([isinstance(v, (bool, np.bool_)) for v in non_null_ids], index=non_null_ids.index)
        if bool_mask.any():
            bad_count = int(bool_mask.sum())
            bad_examples = list(non_null_ids[bool_mask].unique()[:3])
            raise error_cls(
                f"{table_name} contains {bad_count} boolean value(s) in canonical key 'id_student' (sample: {bad_examples}). "
                f"Expected valid positive integers."
            )

        # Catch float values: float, np.floating (e.g. 123.0)
        float_mask = pd.Series([isinstance(v, (float, np.floating)) for v in non_null_ids], index=non_null_ids.index)
        if float_mask.any():
            bad_count = int(float_mask.sum())
            bad_examples = list(non_null_ids[float_mask].unique()[:3])
            raise error_cls(
                f"{table_name} contains {bad_count} float/fractional value(s) in canonical key 'id_student' (sample: {bad_examples}). "
                f"Expected valid positive integers."
            )

        # Catch string values (both non-numeric strings and numeric strings where an integer is required)
        str_mask = pd.Series([isinstance(v, str) for v in non_null_ids], index=non_null_ids.index)
        if str_mask.any():
            bad_count = int(str_mask.sum())
            bad_examples = list(non_null_ids[str_mask].unique()[:3])
            raise error_cls(
                f"{table_name} contains {bad_count} non-numeric/malformed value(s) (strings where integer required) "
                f"in canonical key 'id_student' (sample: {bad_examples}). "
                f"Expected valid positive integers."
            )

        # Catch any other non-integer types (e.g. Decimal, complex, etc.)
        non_int_mask = pd.Series([not isinstance(v, (int, np.integer)) for v in non_null_ids], index=non_null_ids.index)
        if non_int_mask.any():
            bad_count = int(non_int_mask.sum())
            bad_examples = list(non_null_ids[non_int_mask].unique()[:3])
            raise error_cls(
                f"{table_name} contains {bad_count} non-integer numeric/object value(s) in canonical key 'id_student' (sample: {bad_examples}). "
                f"Expected valid positive integers."
            )

    # Reject string dtype (StringDtype)
    if isinstance(id_series.dtype, pd.StringDtype):
        bad_count = len(non_null_ids)
        bad_examples = list(non_null_ids.unique()[:3])
        raise error_cls(
            f"{table_name} contains {bad_count} non-numeric/malformed value(s) (strings where integer required) "
            f"in canonical key 'id_student' (sample: {bad_examples}). "
            f"Expected valid positive integers."
        )

    # Check non-null values for numeric validity
    coerced = pd.to_numeric(non_null_ids, errors="coerce")
    malformed_mask = coerced.isna() | np.isinf(coerced)
    if malformed_mask.any():
        bad_count = int(malformed_mask.sum())
        bad_examples = list(non_null_ids[malformed_mask].unique()[:3])
        raise error_cls(
            f"{table_name} contains {bad_count} non-numeric/malformed value(s) in canonical key 'id_student' (sample: {bad_examples}). "
            f"Expected valid positive integers."
        )

    # Check for fractional numbers (e.g. 1.5)
    fractional_mask = (coerced % 1 != 0)
    if fractional_mask.any():
        bad_count = int(fractional_mask.sum())
        bad_examples = list(non_null_ids[fractional_mask].unique()[:3])
        raise error_cls(
            f"{table_name} contains {bad_count} fractional value(s) in canonical key 'id_student' (sample: {bad_examples}). "
            f"Expected valid positive integers."
        )

    # Check strictly positive (id_student > 0)
    non_positive_mask = (coerced <= 0)
    if non_positive_mask.any():
        bad_count = int(non_positive_mask.sum())
        bad_examples = list(non_null_ids[non_positive_mask].unique()[:3])
        raise error_cls(
            f"{table_name} contains {bad_count} non-positive value(s) (<= 0) in canonical key 'id_student' (sample: {bad_examples}). "
            f"Expected valid positive integers."
        )


def validate_final_result(
    df: pd.DataFrame,
    table_name: str,
    error_cls: type[Exception] = ValueError,
) -> None:
    """Validate presence, string-typing, non-nullability, and domain of final_result column.

    Args:
        df: Input DataFrame to validate.
        table_name: Name of the table for descriptive reporting.
        error_cls: Exception class to raise on validation failure.

    Raises:
        error_cls: If 'final_result' is missing, contains non-string types, nulls/blanks,
            or values outside the permitted domain {'Pass', 'Fail', 'Withdrawn', 'Distinction'}.
    """
    if "final_result" not in df.columns:
        raise error_cls(f"{table_name} is missing required column 'final_result'.")

    series = df["final_result"]

    # Dtype check
    if not (pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype)):
        raise error_cls(
            f"{table_name} column 'final_result' has dtype '{series.dtype}', expected string dtype."
        )

    # Check individual elements in object dtype
    null_mask = series.isna()
    if pd.api.types.is_object_dtype(series.dtype):
        non_null_vals = series[~null_mask]
        if len(non_null_vals) > 0:
            non_str = pd.Series([not isinstance(v, str) for v in non_null_vals], index=non_null_vals.index)
            if non_str.any():
                bad_count = int(non_str.sum())
                bad_examples = list(non_null_vals[non_str].unique()[:3])
                raise error_cls(
                    f"{table_name} column 'final_result' contains {bad_count} non-string value(s) "
                    f"(sample: {bad_examples}). Expected valid string outcome categories."
                )

    # Check nulls or whitespace-only strings
    null_mask = null_mask | (series.astype(str).str.strip() == "")
    if null_mask.any():
        bad_count = int(null_mask.sum())
        raise error_cls(
            f"{table_name} column 'final_result' contains {bad_count} null or blank value(s). "
            f"Expected non-null outcomes from {sorted(ALLOWED_FINAL_RESULTS)}."
        )

    # Check permitted outcome domain
    invalid_mask = ~series.isin(ALLOWED_FINAL_RESULTS)
    if invalid_mask.any():
        bad_count = int(invalid_mask.sum())
        bad_examples = list(series[invalid_mask].unique()[:3])
        raise error_cls(
            f"{table_name} column 'final_result' contains {bad_count} invalid outcome value(s) "
            f"(sample: {bad_examples}). Allowed values: {sorted(ALLOWED_FINAL_RESULTS)}."
        )


def parse_and_validate_temporal_date(
    series: pd.Series,
    col_name: str,
    table_name: str = "student_registration",
    error_cls: type[Exception] = ValueError,
) -> pd.Series:
    """Parse integer temporal day offsets, distinguishing legitimate missing values from malformed tokens.

    Per the OULAD source-data contract (docs/data_sources.md §6) and quality_rules.yaml, the only
    string-based legitimate missing marker in the raw source is '?'.  Genuine pandas-level nulls
    (pd.NA, np.nan, None) are also accepted as missing.  All other non-numeric strings — including
    the explicit forbidden markers '' (empty string), 'NA', 'null', 'NULL', 'None' — are rejected
    with a descriptive error rather than silently converted to pd.NA.

    Valid non-missing integer tokens are converted to nullable Int64.
    Malformed non-null values (non-numeric strings, fractional numbers, booleans) raise error_cls.

    Args:
        series: Input temporal Series.
        col_name: Column name (e.g. 'date_registration' or 'date_unregistration').
        table_name: Table name for error messages.
        error_cls: Exception class to raise on malformed values.

    Returns:
        pd.Series with dtype 'Int64' containing valid integer offsets and pd.NA.

    Raises:
        error_cls: If any non-null value is malformed (unparseable, fractional, boolean, forbidden marker).
    """
    if pd.api.types.is_bool_dtype(series.dtype):
        bad_count = int(series.notna().sum())
        bad_examples = list(series.dropna().unique()[:3])
        raise error_cls(
            f"{table_name} column '{col_name}' contains {bad_count} boolean value(s) (sample: {bad_examples}). "
            f"Expected integer relative day offsets or legitimate missing values."
        )

    # Forbidden markers per quality_rules.yaml forbidden_markers and OULAD source-data contract.
    FORBIDDEN_STRING_MARKERS = {"", "NA", "na", "null", "NULL", "None"}
    # Legitimate string-based missing marker per OULAD source-data contract (data_sources.md §6).
    ALLOWED_STRING_MARKERS = {"?"}

    # Identify pandas-level nulls (pd.NA, np.nan, None)
    missing_mask = series.isna()

    if pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype):
        stripped = series.astype(str).str.strip()

        # Detect and reject forbidden string markers before treating anything as missing
        candidate_mask = ~missing_mask
        if candidate_mask.any():
            forbidden_mask = candidate_mask & stripped.isin(FORBIDDEN_STRING_MARKERS)
            if forbidden_mask.any():
                bad_count = int(forbidden_mask.sum())
                bad_examples = list(series[forbidden_mask].unique()[:3])
                raise error_cls(
                    f"{table_name} column '{col_name}' contains {bad_count} forbidden missing-value marker(s) "
                    f"(sample: {bad_examples!r}). These strings are prohibited by the source-data quality contract "
                    f"(quality_rules.yaml forbidden_markers). Use '?' for legitimate missing values."
                )

        # Accept only documented OULAD string-based missing markers
        missing_mask = missing_mask | (candidate_mask & stripped.isin(ALLOWED_STRING_MARKERS))

    result = pd.Series(index=series.index, dtype="Int64", data=pd.NA)
    non_missing = series[~missing_mask]
    if len(non_missing) == 0:
        return result

    # Check for object-dtype booleans (both Python bool and NumPy np.bool_)
    if pd.api.types.is_object_dtype(series.dtype):
        bool_mask = pd.Series([isinstance(v, (bool, np.bool_)) for v in non_missing], index=non_missing.index)
        if bool_mask.any():
            bad_count = int(bool_mask.sum())
            bad_examples = list(non_missing[bool_mask].unique()[:3])
            raise error_cls(
                f"{table_name} column '{col_name}' contains {bad_count} boolean value(s) (sample: {bad_examples}). "
                f"Expected integer relative day offsets or legitimate missing values."
            )

    # Check unparseable strings and non-finite values
    coerced = pd.to_numeric(non_missing, errors="coerce")
    unparseable = coerced.isna() | np.isinf(coerced)
    if unparseable.any():
        bad_count = int(unparseable.sum())
        bad_examples = list(non_missing[unparseable].unique()[:3])
        raise error_cls(
            f"{table_name} column '{col_name}' contains {bad_count} malformed non-null value(s) "
            f"that cannot be parsed as integer day offsets (sample: {bad_examples})."
        )

    # Check fractional values
    fractional = (coerced % 1 != 0)
    if fractional.any():
        bad_count = int(fractional.sum())
        bad_examples = list(non_missing[fractional].unique()[:3])
        raise error_cls(
            f"{table_name} column '{col_name}' contains {bad_count} fractional value(s) in integer day offset column '{col_name}' (sample: {bad_examples})."
        )

    result.loc[~missing_mask] = coerced.astype("Int64")
    return result


def create_withdrawal_target(
    student_info_df: pd.DataFrame,
    student_registration_df: pd.DataFrame,
    cutoff_day: int = 35,
    courses_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the binary withdrawal target variable strictly after prediction cutoff.

    Formula:
        y_i = 1 if final_result == 'Withdrawn' and date_unregistration > cutoff_day
        y_i = 0 if final_result in ['Pass', 'Distinction', 'Fail'] and (date_unregistration is null or date_unregistration > cutoff_day)
        y_i = <NA> if date_unregistration <= cutoff_day or (final_result == 'Withdrawn' and date_unregistration is null)

    Args:
        student_info_df: student_info DataFrame containing final_result.
        student_registration_df: student_registration DataFrame containing date_unregistration.
        cutoff_day: Relative day cutoff (default: 35).
        courses_df: Optional courses DataFrame to validate course referential integrity.

    Returns:
        DataFrame containing grain ['code_module', 'code_presentation', 'id_student'] and 'withdrawn_after_cutoff'.

    Raises:
        ValueError: If input tables contain missing canonical keys, duplicate keys, mismatched key sets,
            missing/invalid final_result, malformed temporal values, or invalid courses.
        RuntimeError: If row count mismatch is detected during join.
    """
    grain = ["code_module", "code_presentation", "id_student"]

    # Reject missing/null/blank canonical keys and validate id_student as positive integer
    validate_canonical_keys(student_info_df, "student_info_df", ValueError)
    validate_canonical_keys(student_registration_df, "student_registration_df", ValueError)

    # Validate referential integrity against courses if provided
    if courses_df is not None:
        validate_courses_reference_table(courses_df, error_cls=ValueError)
        rel_info = validate_relationship("courses", courses_df, "student_info", student_info_df, ["code_module", "code_presentation"], allow_orphans=False)
        if not rel_info["is_valid"]:
            raise ValueError(f"Referential integrity violation: student_info references invalid courses: {rel_info['violations']}")
        rel_reg = validate_relationship("courses", courses_df, "student_registration", student_registration_df, ["code_module", "code_presentation"], allow_orphans=False)
        if not rel_reg["is_valid"]:
            raise ValueError(f"Referential integrity violation: student_registration references invalid courses: {rel_reg['violations']}")

    # Validate final_result presence, string-typing, and permitted domain
    validate_final_result(student_info_df, "student_info_df", ValueError)

    # Verify date_unregistration presence in student_registration_df
    if "date_unregistration" not in student_registration_df.columns:
        raise ValueError("student_registration_df is missing required column 'date_unregistration'.")

    # Validate date_registration if present in student_registration_df
    if "date_registration" in student_registration_df.columns:
        parse_and_validate_temporal_date(
            student_registration_df["date_registration"],
            col_name="date_registration",
            table_name="student_registration_df",
            error_cls=ValueError,
        )

    # Guard against duplicate canonical keys before joining
    info_dups = int(student_info_df.duplicated(subset=grain).sum())
    if info_dups > 0:
        raise ValueError(
            f"student_info_df contains {info_dups} duplicate canonical keys on {grain}."
        )

    reg_dups = int(student_registration_df.duplicated(subset=grain).sum())
    if reg_dups > 0:
        raise ValueError(
            f"student_registration_df contains {reg_dups} duplicate canonical keys on {grain}."
        )

    # Verify that canonical key sets match exactly between tables before joining
    info_keys = set(zip(student_info_df["code_module"], student_info_df["code_presentation"], student_info_df["id_student"]))
    reg_keys = set(zip(student_registration_df["code_module"], student_registration_df["code_presentation"], student_registration_df["id_student"]))

    only_in_info = info_keys - reg_keys
    only_in_reg = reg_keys - info_keys
    if only_in_info or only_in_reg:
        err_parts = []
        if only_in_info:
            sample_info = sorted(only_in_info)[:3]
            err_parts.append(f"{len(only_in_info)} keys present in student_info but missing from student_registration (sample: {sample_info})")
        if only_in_reg:
            sample_reg = sorted(only_in_reg)[:3]
            err_parts.append(f"{len(only_in_reg)} keys present in student_registration but missing from student_info (sample: {sample_reg})")
        raise ValueError(
            f"Canonical key mismatch between student_info and student_registration on {grain}: {'; '.join(err_parts)}. "
            f"Tables must possess identical canonical key sets."
        )

    info_subset = student_info_df[grain + ["final_result"]].copy()
    reg_subset = student_registration_df[grain + ["date_unregistration"]].copy()

    merged = pd.merge(info_subset, reg_subset, on=grain, how="inner", validate="one_to_one")
    if len(merged) != len(info_subset) or len(merged) != len(reg_subset):
        raise RuntimeError(
            f"Row count mismatch during target merge: merged {len(merged)} rows, "
            f"expected {len(info_subset)} from info and {len(reg_subset)} from reg."
        )

    # Strict temporal parsing and validation for unregistration date
    unreg_dates = parse_and_validate_temporal_date(
        merged["date_unregistration"],
        col_name="date_unregistration",
        table_name="student_registration_df",
        error_cls=ValueError,
    )
    final_results = merged["final_result"].astype(str)

    # Initialize target series with nullable Int64
    targets = pd.Series(index=merged.index, dtype="Int64", data=pd.NA)

    # 1. Negative class: Successful completion or academic failure (Pass, Distinction, Fail)
    # strictly without early unregistration on or before cutoff date
    completer_mask = final_results.isin(["Pass", "Distinction", "Fail"]) & (unreg_dates.isna() | (unreg_dates > cutoff_day))
    targets.loc[completer_mask] = 0

    # 2. Positive class: Withdrew strictly after cutoff date
    withdrawn_post_cutoff_mask = (final_results == "Withdrawn") & unreg_dates.notna() & (unreg_dates > cutoff_day)
    targets.loc[withdrawn_post_cutoff_mask] = 1

    # Pre-cutoff departures (date_unregistration <= cutoff_day) and unverified withdrawals remain pd.NA (ineligible records)
    result_df = merged[grain].copy()
    result_df["withdrawn_after_cutoff"] = targets

    pos_count = int((targets == 1).sum())
    neg_count = int((targets == 0).sum())
    na_count = int(targets.isna().sum())
    logger.info(
        "Created withdrawal target: %d positive (1), %d negative (0), %d unassigned/ineligible (NA)",
        pos_count,
        neg_count,
        na_count,
    )
    return result_df


def validate_withdrawal_target(
    target_df: pd.DataFrame,
    raise_on_invalid: bool = False,
) -> dict[str, Any]:
    """Validate target distribution, value domains, and key uniqueness.

    Enforces strict integer target type, rejecting float-typed Series (e.g. [0.0, 1.0])
    and non-integer values even if they compare numerically equal to {0, 1}.

    Args:
        target_df: DataFrame containing target column.
        raise_on_invalid: If True, raise ValueError on validation failure.

    Returns:
        Structured audit dictionary of target metrics.

    Raises:
        ValueError: If target column is missing, or if raise_on_invalid is True and validation fails.
    """
    grain = ["code_module", "code_presentation", "id_student"]
    target_col = "withdrawn_after_cutoff"

    if target_col not in target_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    total_rows = len(target_df)
    unique_grain_count = len(target_df.drop_duplicates(subset=grain))
    is_grain_unique = total_rows == unique_grain_count

    target_series = target_df[target_col]

    # Target must not be boolean-typed
    is_bool_dtype = pd.api.types.is_bool_dtype(target_series.dtype)
    # Target must not be float-typed (must be integer dtype)
    is_float_dtype = pd.api.types.is_float_dtype(target_series.dtype)

    valid_targets = target_series.dropna()

    # In object-dtype Series, verify no boolean, float, or non-integer instances exist
    has_bool_values = False
    has_float_values = False
    has_non_int_values = False
    if pd.api.types.is_object_dtype(target_series.dtype):
        has_bool_values = any(isinstance(v, (bool, np.bool_)) for v in valid_targets)
        has_float_values = any(isinstance(v, (float, np.floating)) for v in valid_targets)
        has_non_int_values = any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in valid_targets)

    # Check numeric domain {0, 1} and no fractional values
    distinct_values = set(valid_targets.unique())
    numeric_coerced = pd.to_numeric(valid_targets, errors="coerce")
    has_fractions = bool((numeric_coerced % 1 != 0).any())

    is_strict_int_type = (
        pd.api.types.is_integer_dtype(target_series.dtype)
        or (pd.api.types.is_object_dtype(target_series.dtype) and not has_float_values and not has_bool_values and not has_non_int_values)
    )

    is_domain_valid = (
        not is_bool_dtype
        and not is_float_dtype
        and not has_bool_values
        and not has_float_values
        and not has_non_int_values
        and is_strict_int_type
        and not has_fractions
        and distinct_values.issubset({0, 1})
    )

    pos_count = int((valid_targets == 1).sum())
    neg_count = int((valid_targets == 0).sum())
    na_count = int(target_series.isna().sum())
    base_rate = round(pos_count / len(valid_targets) * 100, 4) if len(valid_targets) > 0 else 0.0

    is_valid = is_grain_unique and is_domain_valid
    if raise_on_invalid and not is_valid:
        reasons = []
        if not is_grain_unique:
            reasons.append(f"non-unique grain (total rows: {total_rows}, unique: {unique_grain_count})")
        if not is_domain_valid:
            if is_bool_dtype or has_bool_values:
                reasons.append("boolean values detected in target column (expected integer 0/1)")
            elif has_fractions:
                reasons.append("fractional values detected in target column")
            elif is_float_dtype or has_float_values:
                reasons.append("float values detected in target column (expected integer 0/1)")
            elif has_non_int_values:
                reasons.append("non-integer values detected in target column (expected integer 0/1)")
            else:
                reasons.append(f"invalid domain values: {distinct_values - {0, 1}}")
        raise ValueError(f"Target validation failed: {'; '.join(reasons)}")

    return {
        "total_rows": total_rows,
        "is_grain_unique": is_grain_unique,
        "is_domain_valid": is_domain_valid,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "na_count": na_count,
        "positive_base_rate_pct": base_rate,
        "is_valid": is_valid,
    }


def assert_no_target_leakage(columns_or_df: list[str] | pd.DataFrame) -> None:
    """Assert that outcome labels and future administrative variables do not leak into predictors.

    Raises:
        ValueError: If any forbidden outcome column is detected.
    """
    if isinstance(columns_or_df, pd.DataFrame):
        cols = set(columns_or_df.columns)
    else:
        cols = set(columns_or_df)

    leaked = cols.intersection(FORBIDDEN_PREDICTOR_COLUMNS)
    if leaked:
        raise ValueError(
            f"Target leakage detected! The following outcome columns must never enter feature predictors: {sorted(leaked)}"
        )
