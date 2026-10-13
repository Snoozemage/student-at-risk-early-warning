"""Unit tests for withdrawal target variable creation and leakage assertions."""
import numpy as np
import pandas as pd
import pytest

from student_retention.labels.withdrawal_target import (
    assert_no_target_leakage,
    create_withdrawal_target,
    validate_canonical_keys,
    validate_withdrawal_target,
)


@pytest.fixture
def target_sample_dfs():
    info_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3, 4, 5],
        "final_result": ["Withdrawn", "Pass", "Fail", "Distinction", "Withdrawn"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3, 4, 5],
        "date_unregistration": [100, pd.NA, pd.NA, pd.NA, 20],  # student 1 withdrew day 100 (>35), student 5 withdrew day 20 (<=35)
    })
    return info_df, reg_df


def test_create_withdrawal_target(target_sample_dfs):
    info_df, reg_df = target_sample_dfs
    target_df = create_withdrawal_target(info_df, reg_df, cutoff_day=35)

    assert "withdrawn_after_cutoff" in target_df.columns
    # Student 1: Withdrawn day 100 > 35 -> 1
    assert target_df.loc[target_df["id_student"] == 1, "withdrawn_after_cutoff"].iloc[0] == 1
    # Student 2: Pass -> 0
    assert target_df.loc[target_df["id_student"] == 2, "withdrawn_after_cutoff"].iloc[0] == 0
    # Student 3: Fail -> 0
    assert target_df.loc[target_df["id_student"] == 3, "withdrawn_after_cutoff"].iloc[0] == 0
    # Student 4: Distinction -> 0
    assert target_df.loc[target_df["id_student"] == 4, "withdrawn_after_cutoff"].iloc[0] == 0
    # Student 5: Withdrawn day 20 <= 35 -> pd.NA (ineligible pre-cutoff departure)
    assert pd.isna(target_df.loc[target_df["id_student"] == 5, "withdrawn_after_cutoff"].iloc[0])


def test_validate_withdrawal_target(target_sample_dfs):
    info_df, reg_df = target_sample_dfs
    target_df = create_withdrawal_target(info_df, reg_df, cutoff_day=35)
    audit = validate_withdrawal_target(target_df)

    assert audit["is_valid"] is True
    assert audit["is_grain_unique"] is True
    assert audit["is_domain_valid"] is True
    assert audit["positive_count"] == 1
    assert audit["negative_count"] == 3
    assert audit["na_count"] == 1
    assert audit["positive_base_rate_pct"] == 25.0


def test_assert_no_target_leakage_clean():
    clean_cols = ["code_module", "code_presentation", "id_student", "vle_total_clicks_obs", "assessments_due_count"]
    # Should not raise
    assert_no_target_leakage(clean_cols)
    assert_no_target_leakage(pd.DataFrame(columns=clean_cols))


def test_assert_no_target_leakage_detected():
    bad_cols_cases = [
        ["code_module", "final_result"],
        ["id_student", "date_unregistration"],
        ["id_student", "withdrawn_after_cutoff"],
        ["id_student", "target_withdrawn"],
    ]
    for cols in bad_cols_cases:
        with pytest.raises(ValueError, match="Target leakage detected"):
            assert_no_target_leakage(cols)
        with pytest.raises(ValueError, match="Target leakage detected"):
            assert_no_target_leakage(pd.DataFrame(columns=cols))


def test_create_withdrawal_target_duplicate_info_keys_raises_error():
    info_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 1],  # Duplicate key!
        "final_result": ["Pass", "Fail"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="student_info_df contains 1 duplicate canonical keys"):
        create_withdrawal_target(info_df, reg_df)


def test_create_withdrawal_target_duplicate_reg_keys_raises_error():
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 1],  # Duplicate key!
        "date_unregistration": [pd.NA, 100],
    })
    with pytest.raises(ValueError, match="student_registration_df contains 1 duplicate canonical keys"):
        create_withdrawal_target(info_df, reg_df)


def test_validate_withdrawal_target_raise_on_invalid_duplicate_keys():
    dup_target_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 1],  # Duplicate grain
        "withdrawn_after_cutoff": [0, 1],
    })
    with pytest.raises(ValueError, match="Target validation failed: non-unique grain"):
        validate_withdrawal_target(dup_target_df, raise_on_invalid=True)


def test_validate_withdrawal_target_raise_on_invalid_domain():
    bad_domain_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "withdrawn_after_cutoff": [2],  # Invalid domain value
    })
    with pytest.raises(ValueError, match="Target validation failed: invalid domain values"):
        validate_withdrawal_target(bad_domain_df, raise_on_invalid=True)


def test_create_withdrawal_target_unique_keys_different_key_sets_raises_error():
    # Both tables have strictly unique keys, but key sets differ (info has student 2, reg has student 3)
    info_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "final_result": ["Pass", "Fail"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 3],
        "date_unregistration": [pd.NA, 100],
    })
    with pytest.raises(ValueError, match="Canonical key mismatch between student_info and student_registration"):
        create_withdrawal_target(info_df, reg_df)


def test_create_withdrawal_target_info_has_unmatched_key_raises_error():
    info_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "final_result": ["Pass", "Fail"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="keys present in student_info but missing from student_registration"):
        create_withdrawal_target(info_df, reg_df)


def test_create_withdrawal_target_reg_has_unmatched_key_raises_error():
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "date_unregistration": [pd.NA, 100],
    })
    with pytest.raises(ValueError, match="keys present in student_registration but missing from student_info"):
        create_withdrawal_target(info_df, reg_df)


def test_create_withdrawal_target_null_canonical_keys_raises_error():
    # Both tables have identical missing keys (e.g. id_student is NA)
    info_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, pd.NA],
        "final_result": ["Pass", "Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, pd.NA],
        "date_unregistration": [pd.NA, pd.NA],
    })
    with pytest.raises(ValueError, match="student_info_df contains 1 null/missing values in canonical key column 'id_student'"):
        create_withdrawal_target(info_df, reg_df)

    # When reg_df has null canonical key
    info_clean = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_null = pd.DataFrame({
        "code_module": [pd.NA],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="student_registration_df contains 1 null/missing values in canonical key column 'code_module'"):
        create_withdrawal_target(info_clean, reg_null)


def test_create_withdrawal_target_completer_early_unregistration_remains_na():
    """Records with date_unregistration <= cutoff_day must remain pd.NA (unassigned),
    even if final_result is Pass, Distinction, or Fail."""
    info_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3, 4, 5],
        "final_result": ["Pass", "Distinction", "Fail", "Pass", "Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3, 4, 5],
        "date_unregistration": [20, 35, 10, 40, pd.NA],  # 1, 2, 3 are <= 35; 4 is > 35; 5 is NA
    })
    target_df = create_withdrawal_target(info_df, reg_df, cutoff_day=35)

    # Students 1, 2, 3 unregistered on or before cutoff: must remain pd.NA (not 0)
    assert pd.isna(target_df.loc[target_df["id_student"] == 1, "withdrawn_after_cutoff"].iloc[0])
    assert pd.isna(target_df.loc[target_df["id_student"] == 2, "withdrawn_after_cutoff"].iloc[0])
    assert pd.isna(target_df.loc[target_df["id_student"] == 3, "withdrawn_after_cutoff"].iloc[0])

    # Students 4 and 5 unregistered strictly after cutoff or never: must be assigned 0
    assert target_df.loc[target_df["id_student"] == 4, "withdrawn_after_cutoff"].iloc[0] == 0
    assert target_df.loc[target_df["id_student"] == 5, "withdrawn_after_cutoff"].iloc[0] == 0


def test_create_withdrawal_target_blank_canonical_keys_raises_error():
    """Identical blank or whitespace-only keys in both tables must raise descriptive errors."""
    # Whitespace in code_module
    info_blank_mod = pd.DataFrame({
        "code_module": ["AAA", "   "],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "final_result": ["Pass", "Pass"],
    })
    reg_blank_mod = pd.DataFrame({
        "code_module": ["AAA", "   "],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "date_unregistration": [pd.NA, pd.NA],
    })
    with pytest.raises(ValueError, match="canonical key column 'code_module'"):
        create_withdrawal_target(info_blank_mod, reg_blank_mod)

    # Empty string in code_presentation
    info_empty_pres = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", ""],
        "id_student": [1, 2],
        "final_result": ["Pass", "Pass"],
    })
    reg_empty_pres = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", ""],
        "id_student": [1, 2],
        "date_unregistration": [pd.NA, pd.NA],
    })
    with pytest.raises(ValueError, match="canonical key column 'code_presentation'"):
        create_withdrawal_target(info_empty_pres, reg_empty_pres)


def test_create_withdrawal_target_invalid_id_student_schema_raises_error():
    """id_student must be a valid positive integer; non-positive, fractional, and malformed values are rejected."""
    # Zero id_student
    info_zero = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [0],
        "final_result": ["Pass"],
    })
    reg_zero = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [0],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="non-positive value.*in canonical key 'id_student'"):
        create_withdrawal_target(info_zero, reg_zero)

    # Negative id_student
    info_neg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [-99],
        "final_result": ["Pass"],
    })
    reg_neg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [-99],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="non-positive value.*in canonical key 'id_student'"):
        create_withdrawal_target(info_neg, reg_neg)

    # Non-numeric string id_student
    info_str = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": ["invalid_id"],
        "final_result": ["Pass"],
    })
    reg_str = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": ["invalid_id"],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="non-numeric/malformed value.*in canonical key 'id_student'"):
        create_withdrawal_target(info_str, reg_str)

    # Fractional id_student
    info_frac = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [10.5],
        "final_result": ["Pass"],
    })
    reg_frac = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [10.5],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="fractional value.*in canonical key 'id_student'"):
        create_withdrawal_target(info_frac, reg_frac)


def test_create_withdrawal_target_malformed_temporal_dates_raises_error():
    """Malformed non-null date tokens cannot silently receive target 0 or be treated as legitimate missing."""
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    # Malformed date_unregistration string
    reg_bad_unreg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": ["invalid_date_token"],
    })
    with pytest.raises(ValueError, match="date_unregistration.*malformed non-null value"):
        create_withdrawal_target(info_df, reg_bad_unreg)

    # Fractional date_unregistration
    reg_frac_unreg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [25.5],
    })
    with pytest.raises(ValueError, match="date_unregistration.*fractional value"):
        create_withdrawal_target(info_df, reg_frac_unreg)

    # Malformed date_registration
    reg_bad_reg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_registration": ["not_a_day"],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="date_registration.*malformed non-null value"):
        create_withdrawal_target(info_df, reg_bad_reg)


def test_create_withdrawal_target_forbidden_temporal_markers_raise_error():
    """Forbidden missing-value markers per quality_rules.yaml must raise, not silently become pd.NA.

    Per data_sources.md \u00a76 and quality_rules.yaml forbidden_markers: '', 'NA', 'null', 'NULL', 'None'
    are not legitimate OULAD missing representations. They must never silently produce target=0.
    """
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    for forbidden_token in ["", "NA", "null", "NULL", "None"]:
        reg_df = pd.DataFrame({
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "id_student": [1],
            "date_unregistration": [forbidden_token],
        })
        with pytest.raises(ValueError, match="forbidden missing-value marker"):
            create_withdrawal_target(info_df, reg_df), f"Expected error for token {forbidden_token!r}"


def test_create_withdrawal_target_legitimate_missing_dates_preserved():
    """Documented legitimate missing values ('?', pd.NA, None) are properly mapped to pd.NA.

    Per the OULAD source-data contract (data_sources.md \u00a76), the only string-based legitimate missing
    marker in raw CSV files is '?'. Pandas-level nulls (pd.NA, None) are also legitimate.
    Empty strings ('') are forbidden per quality_rules.yaml and must NOT appear here.
    """
    info_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3],
        "final_result": ["Pass", "Pass", "Withdrawn"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3],
        "date_unregistration": ["?", pd.NA, None],
    })
    target_df = create_withdrawal_target(info_df, reg_df, cutoff_day=35)
    # Completer with '?' unregistration date receives target 0
    assert target_df.loc[target_df["id_student"] == 1, "withdrawn_after_cutoff"].iloc[0] == 0
    # Completer with pd.NA unregistration date receives target 0
    assert target_df.loc[target_df["id_student"] == 2, "withdrawn_after_cutoff"].iloc[0] == 0
    # Withdrawn student with missing unregistration date remains pd.NA (ineligible)
    assert pd.isna(target_df.loc[target_df["id_student"] == 3, "withdrawn_after_cutoff"].iloc[0])


def test_create_withdrawal_target_non_string_code_module_raises_error():
    """code_module and code_presentation must be string-typed; integer dtype is rejected."""
    info_int_module = pd.DataFrame({
        "code_module": pd.array([1], dtype="Int64"),
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_int_module = pd.DataFrame({
        "code_module": pd.array([1], dtype="Int64"),
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="must be string-typed"):
        create_withdrawal_target(info_int_module, reg_int_module)


def test_create_withdrawal_target_boolean_id_in_object_dtype_raises_error():
    """Boolean values mixed into an object-dtype id_student column must be rejected.

    pd.to_numeric(True) == 1 and pd.to_numeric(False) == 0, so booleans would silently
    pass numeric checks without explicit isinstance detection.
    """
    info_bool_mixed = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": pd.array([True, 268], dtype=object),  # object-dtype, mixed bool + int
        "final_result": ["Pass", "Pass"],
    })
    reg_bool_mixed = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": pd.array([True, 268], dtype=object),
        "date_unregistration": [pd.NA, pd.NA],
    })
    with pytest.raises(ValueError, match="boolean value.*in canonical key 'id_student'"):
        create_withdrawal_target(info_bool_mixed, reg_bool_mixed)


def test_create_withdrawal_target_numpy_boolean_id_in_object_dtype_raises_error():
    """NumPy booleans (np.bool_) mixed into object-dtype id_student must also be rejected."""
    info_np_bool = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": pd.array([np.bool_(True), 268], dtype=object),
        "final_result": ["Pass", "Pass"],
    })
    reg_np_bool = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": pd.array([np.bool_(True), 268], dtype=object),
        "date_unregistration": [pd.NA, pd.NA],
    })
    with pytest.raises(ValueError, match="boolean value.*in canonical key 'id_student'"):
        create_withdrawal_target(info_np_bool, reg_np_bool)


def test_create_withdrawal_target_numeric_string_id_student_raises_error():
    """Numeric strings where an integer is required must be rejected in id_student."""
    info_str_ids = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": ["100", "200"],
        "final_result": ["Pass", "Pass"],
    })
    reg_str_ids = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": ["100", "200"],
        "date_unregistration": [pd.NA, pd.NA],
    })
    with pytest.raises(ValueError, match="non-numeric/malformed value.*in canonical key 'id_student'"):
        create_withdrawal_target(info_str_ids, reg_str_ids)


def test_create_withdrawal_target_object_dtype_non_string_code_module_raises_error():
    """Non-string elements (ints, floats, booleans) in object-dtype code_module must be rejected."""
    info_obj_int = pd.DataFrame({
        "code_module": pd.array([123, "BBB"], dtype=object),
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "final_result": ["Pass", "Pass"],
    })
    reg_obj_int = pd.DataFrame({
        "code_module": pd.array([123, "BBB"], dtype=object),
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "date_unregistration": [pd.NA, pd.NA],
    })
    with pytest.raises(ValueError, match="contains 1 non-string value.*Expected valid string"):
        create_withdrawal_target(info_obj_int, reg_obj_int)


def test_create_withdrawal_target_numpy_boolean_in_temporal_dates_raises_error():
    """NumPy booleans in temporal date columns must raise, not silently become day 1."""
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": pd.Series([np.bool_(True)], dtype=object),
    })
    with pytest.raises(ValueError, match="contains 1 boolean value"):
        create_withdrawal_target(info_df, reg_df)


def test_create_withdrawal_target_invalid_final_result_domain_raises_error():
    """final_result values outside permitted domain {'Pass', 'Fail', 'Withdrawn', 'Distinction'} must raise."""
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Dropout"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="invalid outcome value.*Allowed values"):
        create_withdrawal_target(info_df, reg_df)


def test_create_withdrawal_target_missing_final_result_column_raises_error():
    """student_info_df without 'final_result' column must raise a descriptive ValueError."""
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="missing required column 'final_result'"):
        create_withdrawal_target(info_df, reg_df)


def test_create_withdrawal_target_missing_date_unregistration_column_raises_error():
    """student_registration_df without 'date_unregistration' column must raise a descriptive ValueError."""
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
    })
    with pytest.raises(ValueError, match="missing required column 'date_unregistration'"):
        create_withdrawal_target(info_df, reg_df)


def test_validate_withdrawal_target_rejects_boolean_and_fractional_targets():
    """validate_withdrawal_target must reject boolean and fractional target values."""
    # Boolean dtype
    bool_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "withdrawn_after_cutoff": pd.Series([True, False], dtype=bool),
    })
    with pytest.raises(ValueError, match="boolean values detected in target column"):
        validate_withdrawal_target(bool_df, raise_on_invalid=True)

    # Object dtype with boolean instances
    obj_bool_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "withdrawn_after_cutoff": pd.array([True, 0], dtype=object),
    })
    with pytest.raises(ValueError, match="boolean values detected in target column"):
        validate_withdrawal_target(obj_bool_df, raise_on_invalid=True)

    # Fractional values
    frac_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "withdrawn_after_cutoff": [0.5],
    })
    with pytest.raises(ValueError, match="fractional values detected in target column"):
        validate_withdrawal_target(frac_df, raise_on_invalid=True)


def test_validate_canonical_keys_rejects_whole_floats_and_decimals():
    """validate_canonical_keys must reject whole floats (123.0) and Decimal objects in id_student."""
    from decimal import Decimal

    # Float dtype with whole numbers (123.0)
    df_float = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [123.0],
    })
    with pytest.raises(ValueError, match="float/fractional value.*in canonical key 'id_student'"):
        validate_canonical_keys(df_float, "test_table", ValueError)

    # Object dtype with whole float (123.0)
    df_obj_float = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": pd.array([123.0], dtype=object),
    })
    with pytest.raises(ValueError, match="float/fractional value.*in canonical key 'id_student'"):
        validate_canonical_keys(df_obj_float, "test_table", ValueError)

    # Object dtype with Decimal(123)
    df_decimal = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": pd.array([Decimal(123)], dtype=object),
    })
    with pytest.raises(ValueError, match="non-integer numeric/object value.*in canonical key 'id_student'"):
        validate_canonical_keys(df_decimal, "test_table", ValueError)

    # Valid positive integers should pass
    df_valid = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": pd.Series([123, 456], dtype="Int64"),
    })
    validate_canonical_keys(df_valid, "test_table", ValueError)


def test_validate_withdrawal_target_rejects_float_and_non_integer_types():
    """validate_withdrawal_target must reject float-typed [0.0, 1.0] and non-integer target series."""
    from decimal import Decimal

    # Float-typed target [0.0, 1.0]
    float_target_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "withdrawn_after_cutoff": [0.0, 1.0],
    })
    assert float_target_df["withdrawn_after_cutoff"].dtype == "float64"
    with pytest.raises(ValueError, match="float values detected in target column"):
        validate_withdrawal_target(float_target_df, raise_on_invalid=True)

    # Object dtype with whole floats [0.0, 1.0]
    obj_float_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "withdrawn_after_cutoff": pd.array([0.0, 1.0], dtype=object),
    })
    with pytest.raises(ValueError, match="float values detected in target column"):
        validate_withdrawal_target(obj_float_df, raise_on_invalid=True)

    # Object dtype with Decimal
    obj_dec_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "withdrawn_after_cutoff": pd.array([Decimal(0), Decimal(1)], dtype=object),
    })
    with pytest.raises(ValueError, match="non-integer values detected in target column"):
        validate_withdrawal_target(obj_dec_df, raise_on_invalid=True)

    # Valid integer targets pass
    valid_target_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "withdrawn_after_cutoff": pd.Series([0, 1], dtype="Int64"),
    })
    res = validate_withdrawal_target(valid_target_df, raise_on_invalid=True)
    assert res["is_valid"] is True


def test_create_withdrawal_target_courses_referential_integrity():
    """create_withdrawal_target rejects inputs that violate referential integrity to courses."""
    courses_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "module_presentation_length": [268],
    })
    info_bad = pd.DataFrame({
        "code_module": ["INVALID"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_bad = pd.DataFrame({
        "code_module": ["INVALID"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="Referential integrity violation"):
        create_withdrawal_target(info_bad, reg_bad, courses_df=courses_df)


def test_create_withdrawal_target_rejects_invalid_course_code_in_courses():
    """create_withdrawal_target must reject invalid course code ('XXX') in courses_df."""
    courses_bad = pd.DataFrame({
        "code_module": ["XXX"],
        "code_presentation": ["2013J"],
        "module_presentation_length": [268],
    })
    info_df = pd.DataFrame({
        "code_module": ["XXX"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["XXX"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="courses reference table schema"):
        create_withdrawal_target(info_df, reg_df, courses_df=courses_bad)


def test_create_withdrawal_target_rejects_invalid_presentation_code_in_courses():
    """create_withdrawal_target must reject invalid presentation code ('2099X') in courses_df."""
    courses_bad = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2099X"],
        "module_presentation_length": [268],
    })
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2099X"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2099X"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="courses reference table schema"):
        create_withdrawal_target(info_df, reg_df, courses_df=courses_bad)


def test_create_withdrawal_target_rejects_duplicate_primary_key_in_courses():
    """create_withdrawal_target must reject duplicate primary key in courses_df."""
    courses_bad = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "module_presentation_length": [268, 268],
    })
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="courses reference table primary key"):
        create_withdrawal_target(info_df, reg_df, courses_df=courses_bad)


def test_create_withdrawal_target_rejects_invalid_length_in_courses():
    """create_withdrawal_target must reject invalid module_presentation_length in courses_df."""
    courses_bad = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "module_presentation_length": [500],
    })
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(ValueError, match="courses reference table schema"):
        create_withdrawal_target(info_df, reg_df, courses_df=courses_bad)
