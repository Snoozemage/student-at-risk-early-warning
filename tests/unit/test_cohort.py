"""Unit tests for CohortBuilder deterministic precedence hierarchy and policies."""
import numpy as np
import pandas as pd
import pytest

from student_retention.transformation.build_cohort import (
    CohortBuilder,
    CohortInvariantError,
)


@pytest.fixture
def synthetic_courses_df():
    """Valid synthetic courses reference table covering modules AAA and BBB in 2013J."""
    return pd.DataFrame({
        "code_module": ["AAA", "BBB"],
        "code_presentation": ["2013J", "2013J"],
        "module_presentation_length": [268, 268],
    })


@pytest.fixture(autouse=True)
def inject_synthetic_courses_df(monkeypatch, synthetic_courses_df):
    """Ensure synthetic unit tests inject a valid courses reference table when neither courses_df nor custom data_dir is provided."""
    original_init = CohortBuilder.__init__

    def patched_init(self, *args, **kwargs):
        data_dir = kwargs.get("data_dir")
        if len(args) >= 3 and data_dir is None:
            data_dir = args[2]

        if ("courses_df" not in kwargs or kwargs["courses_df"] is None) and data_dir is None:
            kwargs["courses_df"] = synthetic_courses_df
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(CohortBuilder, "__init__", patched_init)


@pytest.fixture
def cohort_synthetic_dfs():
    info_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2013J", "2013J", "2013J"],
        "id_student": [101, 102, 103, 104, 105, 554243, 107],
        "final_result": ["Withdrawn", "Pass", "Pass", "Withdrawn", "Pass", "Fail", "Withdrawn"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2013J", "2013J", "2013J"],
        "id_student": [101, 102, 103, 104, 105, 554243, 107],
        "date_registration": [-30, 45, pd.NA, -20, -10, -35, -50],
        # 101: unreg Day 20 <= 35 -> Precedence 1
        # 102: reg Day 45 > 35 -> Precedence 2
        # 103: reg is null, no unreg -> Precedence 3 (excluded Policy C, kept Policy B)
        # 104: unreg is null, final_result == Withdrawn -> Precedence 4
        # 105: eligible pass student (reg -10, unreg NA) -> Eligible
        # 554243: contradictory record (reg -35, unreg 166, final_result Fail) -> Precedence 5 (excluded C, kept B)
        # 107: eligible withdrawn student (reg -50, unreg 120 > 35) -> Eligible (Positive)
        "date_unregistration": [20, pd.NA, pd.NA, pd.NA, pd.NA, 166, 120],
    })
    return info_df, reg_df


def test_cohort_precedence_policy_c(cohort_synthetic_dfs):
    info_df, reg_df = cohort_synthetic_dfs
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    results = builder.build_cohort(info_df, reg_df)

    eligible_df = results["cohort_df"]
    exclusions_df = results["exclusions_df"]
    audit = results["audit_report"]

    # Exactly 2 eligible records: 105 (negative, Pass) and 107 (positive, Withdrawn Day 120)
    assert len(eligible_df) == 2
    assert set(eligible_df["id_student"]) == {105, 107}
    assert eligible_df.loc[eligible_df["id_student"] == 107, "withdrawn_after_cutoff"].iloc[0] == 1
    assert eligible_df.loc[eligible_df["id_student"] == 105, "withdrawn_after_cutoff"].iloc[0] == 0

    # Exactly 5 excluded records: 101, 102, 103, 104, 554243
    assert len(exclusions_df) == 5
    excl_by_student = exclusions_df.set_index("id_student")["exclusion_code"].to_dict()
    assert excl_by_student[101] == "EXCL_PRE_CUTOFF_UNREG"
    assert excl_by_student[102] == "EXCL_LATE_REGISTRATION"
    assert excl_by_student[103] == "EXCL_MISSING_REGISTRATION_DATE"
    assert excl_by_student[104] == "EXCL_WITHDRAWN_MISSING_UNREG_DATE"
    assert excl_by_student[554243] == "EXCL_CONTRADICTORY_RECORD"

    # Reconciliation
    reconcil = audit["reconciliation"]
    assert reconcil["total_source_records"] == 7
    assert reconcil["total_excluded_records"] == 5
    assert reconcil["eligible_cohort_records"] == 2
    assert reconcil["is_reconciled"] is True


def test_cohort_precedence_policy_b(cohort_synthetic_dfs):
    info_df, reg_df = cohort_synthetic_dfs
    builder = CohortBuilder(cutoff_day=35, policy="policy_b")
    results = builder.build_cohort(info_df, reg_df)

    eligible_df = results["cohort_df"]
    exclusions_df = results["exclusions_df"]

    # Under Policy B:
    # 103 (missing registration date) is RETAINED as eligible
    # 554243 (contradictory record) is RETAINED as eligible negative (Fail)
    # Total eligible: 105, 107, 103, 554243 -> 4 eligible
    assert len(eligible_df) == 4
    assert set(eligible_df["id_student"]) == {105, 107, 103, 554243}
    assert eligible_df.loc[eligible_df["id_student"] == 554243, "withdrawn_after_cutoff"].iloc[0] == 0

    # Exactly 3 excluded records: 101, 102, 104
    assert len(exclusions_df) == 3
    assert set(exclusions_df["id_student"]) == {101, 102, 104}


def test_precedence_order_overlap_handling():
    # If a record has both date_registration is null AND date_unregistration <= 35,
    # Precedence Step 1 (EXCL_PRE_CUTOFF_UNREG) must claim it first before Step 3.
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [999],
        "final_result": ["Withdrawn"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [999],
        "date_registration": [pd.NA],  # Would trigger Step 3
        "date_unregistration": [-10],   # Triggers Step 1
    })
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    results = builder.build_cohort(info_df, reg_df)

    exclusions = results["exclusions_df"]
    assert len(exclusions) == 1
    assert exclusions["exclusion_step"].iloc[0] == 1
    assert exclusions["exclusion_code"].iloc[0] == "EXCL_PRE_CUTOFF_UNREG"


def test_cohort_builder_duplicate_key_in_info_raises_error(cohort_synthetic_dfs):
    info_df, reg_df = cohort_synthetic_dfs
    # Duplicate student 105 in info_df
    dup_row = info_df.loc[info_df["id_student"] == 105]
    bad_info_df = pd.concat([info_df, dup_row], ignore_index=True)

    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    with pytest.raises(CohortInvariantError, match="student_info_df contains 1 duplicate canonical keys"):
        builder.build_cohort(bad_info_df, reg_df)


def test_cohort_builder_duplicate_key_in_reg_raises_error(cohort_synthetic_dfs):
    info_df, reg_df = cohort_synthetic_dfs
    # Duplicate student 105 in reg_df
    dup_row = reg_df.loc[reg_df["id_student"] == 105]
    bad_reg_df = pd.concat([reg_df, dup_row], ignore_index=True)

    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    with pytest.raises(CohortInvariantError, match="student_registration_df contains 1 duplicate canonical keys"):
        builder.build_cohort(info_df, bad_reg_df)


def test_cohort_builder_target_completeness_invariant_violation(cohort_synthetic_dfs, monkeypatch):
    info_df, reg_df = cohort_synthetic_dfs
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")

    # Monkeypatch create_withdrawal_target to return an NA target for an eligible student
    def mock_bad_target(student_info_df, student_registration_df, cutoff_day):
        grain = ["code_module", "code_presentation", "id_student"]
        df = student_info_df[grain].copy()
        df["withdrawn_after_cutoff"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        return df

    monkeypatch.setattr(
        "student_retention.transformation.build_cohort.create_withdrawal_target",
        mock_bad_target,
    )

    with pytest.raises(CohortInvariantError, match="Target completeness invariant violated"):
        builder.build_cohort(info_df, reg_df)


def test_cohort_builder_class_count_reconciliation_invariant_violation():
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    grain = ["code_module", "code_presentation", "id_student"]
    eligible_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "date_unregistration": [pd.NA, 100],
        "withdrawn_after_cutoff": [0, 1],
        "final_result": ["Pass", "Withdrawn"],
    })
    exclusions_df = pd.DataFrame(columns=grain + ["exclusion_step", "exclusion_code", "exclusion_reason"])

    # Pass pos_count=1, neg_count=0 (sum is 1, but eligible_df has 2 rows)
    with pytest.raises(CohortInvariantError, match="Class-count reconciliation invariant violated"):
        builder._assert_cohort_invariants(
            eligible_df=eligible_df,
            exclusions_df=exclusions_df,
            total_source_records=2,
            exclusion_steps_summary={},
            pos_count=1,
            neg_count=0,
            neg_breakdown={"Pass": 1},
        )


def test_cohort_builder_source_reconciliation_invariant_violation():
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    grain = ["code_module", "code_presentation", "id_student"]
    eligible_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [pd.NA],
        "withdrawn_after_cutoff": [0],
        "final_result": ["Pass"],
    })
    exclusions_df = pd.DataFrame(columns=grain + ["exclusion_step", "exclusion_code", "exclusion_reason"])

    # total_source_records is 5, but eligible + excluded is 1
    with pytest.raises(CohortInvariantError, match="Source-to-output reconciliation invariant violated"):
        builder._assert_cohort_invariants(
            eligible_df=eligible_df,
            exclusions_df=exclusions_df,
            total_source_records=5,
            exclusion_steps_summary={},
            pos_count=0,
            neg_count=1,
            neg_breakdown={"Pass": 1},
        )


def test_cohort_builder_anti_leakage_invariant_violation():
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    grain = ["code_module", "code_presentation", "id_student"]
    # Student has unregistration date 20 <= 35, which violates anti-leakage if in eligible_df
    eligible_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_unregistration": [20],
        "withdrawn_after_cutoff": [1],
        "final_result": ["Withdrawn"],
    })
    exclusions_df = pd.DataFrame(columns=grain + ["exclusion_step", "exclusion_code", "exclusion_reason"])

    with pytest.raises(CohortInvariantError, match="Anti-leakage invariant violated"):
        builder._assert_cohort_invariants(
            eligible_df=eligible_df,
            exclusions_df=exclusions_df,
            total_source_records=1,
            exclusion_steps_summary={},
            pos_count=1,
            neg_count=0,
            neg_breakdown={},
        )


def test_cohort_builder_duplicate_grain_invariant_violation():
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    grain = ["code_module", "code_presentation", "id_student"]
    # Duplicate grain in eligible_df
    eligible_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 1],
        "date_unregistration": [pd.NA, pd.NA],
        "withdrawn_after_cutoff": [0, 0],
        "final_result": ["Pass", "Pass"],
    })
    exclusions_df = pd.DataFrame(columns=grain + ["exclusion_step", "exclusion_code", "exclusion_reason"])

    with pytest.raises(CohortInvariantError, match="Cohort grain invariant violated"):
        builder._assert_cohort_invariants(
            eligible_df=eligible_df,
            exclusions_df=exclusions_df,
            total_source_records=2,
            exclusion_steps_summary={},
            pos_count=0,
            neg_count=2,
            neg_breakdown={"Pass": 2},
        )


def test_cohort_builder_key_set_mismatch_raises_error(cohort_synthetic_dfs):
    info_df, reg_df = cohort_synthetic_dfs
    # Replace student 105 with 999 in reg_df, creating unique keys but mismatched key sets
    bad_reg_df = reg_df.copy()
    bad_reg_df.loc[bad_reg_df["id_student"] == 105, "id_student"] = 999

    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    with pytest.raises(CohortInvariantError, match="Canonical key mismatch between student_info and student_registration"):
        builder.build_cohort(info_df, bad_reg_df)


def test_cohort_builder_unmatched_info_keys_raises_error(cohort_synthetic_dfs):
    info_df, reg_df = cohort_synthetic_dfs
    # Remove one student from reg_df
    bad_reg_df = reg_df.loc[reg_df["id_student"] != 105].copy()

    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    with pytest.raises(CohortInvariantError, match="keys in student_info missing from student_registration"):
        builder.build_cohort(info_df, bad_reg_df)


def test_cohort_builder_unmatched_reg_keys_raises_error(cohort_synthetic_dfs):
    info_df, reg_df = cohort_synthetic_dfs
    # Remove one student from info_df
    bad_info_df = info_df.loc[info_df["id_student"] != 105].copy()

    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    with pytest.raises(CohortInvariantError, match="keys in student_registration missing from student_info"):
        builder.build_cohort(bad_info_df, reg_df)


def test_cohort_builder_source_reconciliation_independent_counts_recorded(cohort_synthetic_dfs):
    info_df, reg_df = cohort_synthetic_dfs
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    results = builder.build_cohort(info_df, reg_df)

    reconciliation = results["audit_report"]["reconciliation"]
    assert reconciliation["raw_student_info_records"] == 7
    assert reconciliation["raw_student_registration_records"] == 7
    assert reconciliation["total_source_records"] == 7
    assert reconciliation["is_reconciled"] is True


def test_cohort_builder_null_canonical_keys_raises_error():
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
        "date_registration": [-30, -30],
        "date_unregistration": [pd.NA, pd.NA],
    })
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    with pytest.raises(CohortInvariantError, match="student_info_df contains 1 null/missing values in canonical key column 'id_student'"):
        builder.build_cohort(info_df, reg_df)

    # When reg_df has null canonical key
    info_clean = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_null = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": [pd.NA],
        "id_student": [1],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="student_registration_df contains 1 null/missing values in canonical key column 'code_presentation'"):
        builder.build_cohort(info_clean, reg_null)


def test_cohort_builder_blank_canonical_keys_raises_error():
    """Identical blank or whitespace-only keys in both tables must raise CohortInvariantError."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")

    # Whitespace in code_module in both tables
    info_blank = pd.DataFrame({
        "code_module": ["   "],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_blank = pd.DataFrame({
        "code_module": ["   "],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="canonical key column 'code_module'"):
        builder.build_cohort(info_blank, reg_blank)

    # Empty string in code_presentation in both tables
    info_empty = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": [""],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_empty = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": [""],
        "id_student": [1],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="canonical key column 'code_presentation'"):
        builder.build_cohort(info_empty, reg_empty)


def test_cohort_builder_invalid_id_student_schema_raises_error():
    """id_student must be a valid positive integer; non-positive, fractional, and malformed values are rejected."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")

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
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="non-positive value.*in canonical key 'id_student'"):
        builder.build_cohort(info_zero, reg_zero)

    # Negative id_student
    info_neg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [-1],
        "final_result": ["Pass"],
    })
    reg_neg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [-1],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="non-positive value.*in canonical key 'id_student'"):
        builder.build_cohort(info_neg, reg_neg)

    # Malformed non-numeric string id_student
    info_str = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": ["bad_id"],
        "final_result": ["Pass"],
    })
    reg_str = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": ["bad_id"],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="non-numeric/malformed value.*in canonical key 'id_student'"):
        builder.build_cohort(info_str, reg_str)

    # Fractional id_student
    info_frac = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1.5],
        "final_result": ["Pass"],
    })
    reg_frac = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1.5],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="fractional value.*in canonical key 'id_student'"):
        builder.build_cohort(info_frac, reg_frac)


def test_cohort_builder_malformed_temporal_dates_raises_error():
    """Malformed non-null date values cannot silently become missing or receive targets."""
    # Invalid date_registration cannot be treated as missing under Policy C
    builder_c = CohortBuilder(cutoff_day=35, policy="policy_c")
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "final_result": ["Pass"],
    })
    reg_bad_reg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_registration": ["corrupted_date"],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="date_registration.*malformed non-null value"):
        builder_c.build_cohort(info_df, reg_bad_reg)

    # Invalid date_registration cannot bypass under Policy B
    builder_b = CohortBuilder(cutoff_day=35, policy="policy_b")
    with pytest.raises(CohortInvariantError, match="date_registration.*malformed non-null value"):
        builder_b.build_cohort(info_df, reg_bad_reg)

    # Invalid date_unregistration cannot be silently coerced
    reg_bad_unreg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_registration": [-30],
        "date_unregistration": ["invalid_unreg"],
    })
    with pytest.raises(CohortInvariantError, match="date_unregistration.*malformed non-null value"):
        builder_c.build_cohort(info_df, reg_bad_unreg)

    # Fractional date_registration
    reg_frac_reg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_registration": [12.4],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="date_registration.*fractional value"):
        builder_c.build_cohort(info_df, reg_frac_reg)


def test_cohort_builder_raw_csv_legitimate_missing_dates_handled():
    """Raw-CSV '?' and pandas-level nulls for missing dates are correctly processed.

    Per the OULAD source-data contract (data_sources.md \u00a76) the only documented raw-CSV
    missing marker is '?'. Empty strings ('') are forbidden and must not appear.
    """
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    info_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3],
        "final_result": ["Pass", "Pass", "Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3],
        "date_registration": [-30, "-20", -10],  # valid integers / integer strings
        "date_unregistration": ["?", pd.NA, None],  # documented OULAD-legitimate missing markers
    })
    results = builder.build_cohort(info_df, reg_df)
    assert len(results["cohort_df"]) == 3
    assert results["cohort_df"]["withdrawn_after_cutoff"].tolist() == [0, 0, 0]


def test_cohort_builder_forbidden_temporal_markers_raise_error():
    """Forbidden temporal missing-value markers must raise CohortInvariantError, not silently produce target=0.

    Per data_sources.md \u00a76 and quality_rules.yaml forbidden_markers:
    '', 'NA', 'null', 'NULL', 'None' are not valid OULAD missing representations.
    """
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
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
            "date_registration": [-30],
            "date_unregistration": [forbidden_token],
        })
        with pytest.raises(CohortInvariantError, match="forbidden missing-value marker"):
            builder.build_cohort(info_df, reg_df), f"Expected CohortInvariantError for token {forbidden_token!r}"


def test_cohort_builder_non_string_code_module_raises_error():
    """code_module and code_presentation must be string-typed; integer dtype is rejected."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
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
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="must be string-typed"):
        builder.build_cohort(info_int_module, reg_int_module)


def test_cohort_builder_boolean_id_in_object_dtype_raises_error():
    """Boolean values mixed into an object-dtype id_student column must be rejected.

    pd.to_numeric(True) == 1, so booleans would silently pass numeric checks without
    explicit isinstance detection.
    """
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
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
        "date_registration": [-30, -30],
        "date_unregistration": [pd.NA, pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="boolean value.*in canonical key 'id_student'"):
        builder.build_cohort(info_bool_mixed, reg_bool_mixed)


def test_cohort_builder_numpy_boolean_id_student_raises_error():
    """NumPy booleans (np.bool_) in id_student must be rejected, not coerced to 1/0."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": pd.array([np.bool_(True)], dtype=object),
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": pd.array([np.bool_(True)], dtype=object),
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="boolean value.*in canonical key 'id_student'"):
        builder.build_cohort(info_df, reg_df)


def test_cohort_builder_numeric_string_id_student_raises_error():
    """Numeric string in id_student ('268') must be rejected even if parseable as int."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": ["268"],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": ["268"],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="strings where integer required"):
        builder.build_cohort(info_df, reg_df)


def test_cohort_builder_object_dtype_non_string_code_module_raises_error():
    """Object-dtype code_module containing non-string objects (int, float) must be rejected."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    info_df = pd.DataFrame({
        "code_module": pd.array([123], dtype=object),
        "code_presentation": ["2013J"],
        "id_student": [101],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": pd.array([123], dtype=object),
        "code_presentation": ["2013J"],
        "id_student": [101],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="contains 1 non-string value"):
        builder.build_cohort(info_df, reg_df)


def test_cohort_builder_numpy_boolean_temporal_date_raises_error():
    """NumPy booleans in date_registration or date_unregistration must be rejected."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "final_result": ["Pass"],
    })
    reg_bad_reg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "date_registration": pd.array([np.bool_(False)], dtype=object),
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="date_registration.*boolean value"):
        builder.build_cohort(info_df, reg_bad_reg)

    reg_bad_unreg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "date_registration": [-30],
        "date_unregistration": pd.array([np.bool_(True)], dtype=object),
    })
    with pytest.raises(CohortInvariantError, match="date_unregistration.*boolean value"):
        builder.build_cohort(info_df, reg_bad_unreg)


def test_cohort_builder_invalid_final_result_domain_or_type_raises_error():
    """final_result must be string-typed and restricted to allowed OULAD outcomes."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })

    # Non-string integer in final_result
    info_int_result = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "final_result": [1],
    })
    with pytest.raises(CohortInvariantError, match="expected string dtype"):
        builder.build_cohort(info_int_result, reg_df)

    # Invalid outcome domain value
    info_bad_domain = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "final_result": ["Graduated"],
    })
    with pytest.raises(CohortInvariantError, match="invalid outcome value.*Graduated"):
        builder.build_cohort(info_bad_domain, reg_df)

    # Missing/null final_result
    info_null_result = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "final_result": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="null or blank value"):
        builder.build_cohort(info_null_result, reg_df)


def test_cohort_builder_missing_registration_columns_raises_error():
    """Missing required date_registration or date_unregistration column raises CohortInvariantError."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "final_result": ["Pass"],
    })
    reg_missing_unreg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "date_registration": [-30],
    })
    with pytest.raises(CohortInvariantError, match="missing required column 'date_unregistration'"):
        builder.build_cohort(info_df, reg_missing_unreg)

    reg_missing_reg = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="missing required column 'date_registration'"):
        builder.build_cohort(info_df, reg_missing_reg)


def test_cohort_builder_load_inputs_raw_fallback(tmp_path):
    """load_inputs falls back to raw CSV when normalized parquet is missing, normalizing schema."""
    data_dir = tmp_path / "raw_fallback_dir"
    data_dir.mkdir()

    # Create valid minimal courses.csv
    courses_csv = data_dir / "courses.csv"
    courses_csv.write_text(
        "code_module,code_presentation,module_presentation_length\n"
        "AAA,2013J,268\n"
        "BBB,2013J,268\n",
        encoding="utf-8",
    )

    # Create valid minimal studentInfo.csv
    info_csv = data_dir / "studentInfo.csv"
    info_csv.write_text(
        "code_module,code_presentation,id_student,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result\n"
        "AAA,2013J,101,M,East Anglian Region,HE Qualification,90-100%,55<=,0,240,N,Pass\n"
        "BBB,2013J,102,F,Scotland,HE Qualification,20-30%,35-55,0,60,N,Withdrawn\n",
        encoding="utf-8",
    )

    # Create valid minimal studentRegistration.csv
    reg_csv = data_dir / "studentRegistration.csv"
    reg_csv.write_text(
        "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
        "AAA,2013J,101,-30,?\n"
        "BBB,2013J,102,-20,100\n",
        encoding="utf-8",
    )

    builder = CohortBuilder(cutoff_day=35, policy="policy_c", data_dir=data_dir)
    info_df, reg_df = builder.load_inputs()

    assert len(info_df) == 2
    assert len(reg_df) == 2
    # Check nullable types enforced by normalization
    assert pd.api.types.is_string_dtype(info_df["code_module"])
    assert pd.api.types.is_string_dtype(info_df["code_presentation"])
    assert pd.api.types.is_integer_dtype(info_df["id_student"])
    assert pd.api.types.is_integer_dtype(reg_df["date_registration"])
    assert pd.api.types.is_integer_dtype(reg_df["date_unregistration"])
    assert pd.isna(reg_df.loc[reg_df["id_student"] == 101, "date_unregistration"].iloc[0])
    assert reg_df.loc[reg_df["id_student"] == 102, "date_unregistration"].iloc[0] == 100


def test_cohort_builder_raw_fallback_rejects_forbidden_missing_markers(tmp_path):
    """Raw CSV fallback must reject configured forbidden markers ('', 'NA', 'null', etc.) in dates."""
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()

    # Create valid courses.csv
    courses_csv = data_dir / "courses.csv"
    courses_csv.write_text("code_module,code_presentation,module_presentation_length\nAAA,2013J,268\n", encoding="utf-8")

    # Create valid studentInfo.csv
    info_csv = data_dir / "studentInfo.csv"
    info_csv.write_text(
        "code_module,code_presentation,id_student,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result\n"
        "AAA,2013J,101,M,East Anglian Region,HE Qualification,90-100%,55<=,0,240,N,Pass\n",
        encoding="utf-8",
    )

    forbidden_markers = ["", "NA", "na", "null", "NULL", "None"]
    builder = CohortBuilder(cutoff_day=35, policy="policy_c", data_dir=data_dir)

    for marker in forbidden_markers:
        # Test forbidden marker in date_registration
        reg_csv = data_dir / "studentRegistration.csv"
        reg_csv.write_text(
            f"code_module,code_presentation,id_student,date_registration,date_unregistration\n"
            f"AAA,2013J,101,{marker},?\n",
            encoding="utf-8",
        )
        with pytest.raises(CohortInvariantError, match="missing-value/forbidden-marker violations|DataCoercionError"):
            builder.load_inputs()

        # Test forbidden marker in date_unregistration
        reg_csv.write_text(
            f"code_module,code_presentation,id_student,date_registration,date_unregistration\n"
            f"AAA,2013J,101,-30,{marker}\n",
            encoding="utf-8",
        )
        with pytest.raises(CohortInvariantError, match="missing-value/forbidden-marker violations|DataCoercionError"):
            builder.load_inputs()

    # Documented legitimate missing marker '?' must succeed and produce pd.NA
    reg_csv.write_text(
        "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
        "AAA,2013J,101,?,?\n",
        encoding="utf-8",
    )
    loaded_info, loaded_reg = builder.load_inputs()
    assert len(loaded_info) == 1
    assert len(loaded_reg) == 1
    assert pd.isna(loaded_reg["date_registration"].iloc[0])
    assert pd.isna(loaded_reg["date_unregistration"].iloc[0])
    assert pd.api.types.is_integer_dtype(loaded_reg["date_registration"].dtype)
    assert pd.api.types.is_integer_dtype(loaded_reg["date_unregistration"].dtype)


def test_cohort_builder_raw_fallback_rejects_invalid_courses_foreign_key(tmp_path):
    """Raw fallback must reject student tables referencing invalid courses even if both agree."""
    data_dir = tmp_path / "raw_data_bad_course"
    data_dir.mkdir()

    courses_csv = data_dir / "courses.csv"
    courses_csv.write_text("code_module,code_presentation,module_presentation_length\nAAA,2013J,268\n", encoding="utf-8")

    info_csv = data_dir / "studentInfo.csv"
    info_csv.write_text(
        "code_module,code_presentation,id_student,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result\n"
        "INVALID,2013J,101,M,East Anglian Region,HE Qualification,90-100%,55<=,0,240,N,Pass\n",
        encoding="utf-8",
    )

    reg_csv = data_dir / "studentRegistration.csv"
    reg_csv.write_text(
        "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
        "INVALID,2013J,101,-30,?\n",
        encoding="utf-8",
    )

    builder = CohortBuilder(cutoff_day=35, policy="policy_c", data_dir=data_dir)
    with pytest.raises(CohortInvariantError, match="referential integrity violations"):
        builder.load_inputs()


def test_cohort_builder_build_cohort_rejects_invalid_courses_foreign_key():
    """build_cohort must enforce referential integrity to courses when both student tables agree on invalid key."""
    courses_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "module_presentation_length": [268],
    })
    info_df = pd.DataFrame({
        "code_module": ["INVALID"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["INVALID"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })

    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    with pytest.raises(CohortInvariantError, match="Referential integrity invariant violated.*invalid courses"):
        builder.build_cohort(info_df, reg_df, courses_df=courses_df)


def test_cohort_builder_rejects_whole_floats_and_decimals_in_id_student():
    """build_cohort must reject float-typed (101.0) and Decimal id_student values."""
    from decimal import Decimal

    builder = CohortBuilder(cutoff_day=35, policy="policy_c")

    # Float dtype with whole numbers (101.0)
    info_float = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101.0],
        "final_result": ["Pass"],
    })
    reg_float = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101.0],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="float/fractional value.*in canonical key 'id_student'"):
        builder.build_cohort(info_float, reg_float)

    # Object dtype with whole float (101.0)
    info_obj_float = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": pd.array([101.0], dtype=object),
        "final_result": ["Pass"],
    })
    reg_obj_float = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": pd.array([101.0], dtype=object),
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="float/fractional value.*in canonical key 'id_student'"):
        builder.build_cohort(info_obj_float, reg_obj_float)

    # Object dtype with Decimal
    info_dec = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": pd.array([Decimal(101)], dtype=object),
        "final_result": ["Pass"],
    })
    reg_dec = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": pd.array([Decimal(101)], dtype=object),
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="non-integer numeric/object value.*in canonical key 'id_student'"):
        builder.build_cohort(info_dec, reg_dec)


def test_cohort_builder_rejects_invalid_course_code_in_parent_courses():
    """build_cohort must reject invalid course code ('XXX') in parent courses even when both student tables match."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    courses_bad = pd.DataFrame({
        "code_module": ["XXX"],
        "code_presentation": ["2013J"],
        "module_presentation_length": [268],
    })
    info_df = pd.DataFrame({
        "code_module": ["XXX"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["XXX"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="courses reference table schema"):
        builder.build_cohort(info_df, reg_df, courses_df=courses_bad)


def test_cohort_builder_rejects_invalid_presentation_code_in_parent_courses():
    """build_cohort must reject invalid presentation code ('2099X') in parent courses even when both student tables match."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    courses_bad = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2099X"],
        "module_presentation_length": [268],
    })
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2099X"],
        "id_student": [101],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2099X"],
        "id_student": [101],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="courses reference table schema"):
        builder.build_cohort(info_df, reg_df, courses_df=courses_bad)


def test_cohort_builder_rejects_duplicate_primary_key_in_parent_courses():
    """build_cohort must enforce primary key uniqueness on (code_module, code_presentation) in courses."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    courses_bad = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "module_presentation_length": [268, 268],
    })
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="courses reference table primary key"):
        builder.build_cohort(info_df, reg_df, courses_df=courses_bad)


def test_cohort_builder_rejects_invalid_length_in_parent_courses():
    """build_cohort must validate module_presentation_length against schema bounds (200-300)."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    courses_bad = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "module_presentation_length": [500],
    })
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "final_result": ["Pass"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [101],
        "date_registration": [-30],
        "date_unregistration": [pd.NA],
    })
    with pytest.raises(CohortInvariantError, match="courses reference table schema"):
        builder.build_cohort(info_df, reg_df, courses_df=courses_bad)


def test_cohort_builder_init_rejects_invalid_courses_df():
    """CohortBuilder.__init__ must validate courses_df if passed directly."""
    courses_bad = pd.DataFrame({
        "code_module": ["XXX"],
        "code_presentation": ["2013J"],
        "module_presentation_length": [268],
    })
    with pytest.raises(CohortInvariantError, match="courses reference table schema"):
        CohortBuilder(cutoff_day=35, policy="policy_c", courses_df=courses_bad)


def test_cohort_builder_raw_fallback_rejects_invalid_courses_schema_or_duplicate_pk(tmp_path):
    """Raw fallback _load_courses must reject invalid schema or duplicate PKs in courses.csv."""
    data_dir = tmp_path / "raw_data_bad_courses"
    data_dir.mkdir()

    # 1. Invalid course code in courses.csv
    (data_dir / "courses.csv").write_text(
        "code_module,code_presentation,module_presentation_length\n"
        "XXX,2013J,268\n",
        encoding="utf-8",
    )
    (data_dir / "studentInfo.csv").write_text(
        "code_module,code_presentation,id_student,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result\n"
        "XXX,2013J,101,M,East Anglian Region,HE Qualification,90-100%,55<=,0,240,N,Pass\n",
        encoding="utf-8",
    )
    (data_dir / "studentRegistration.csv").write_text(
        "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
        "XXX,2013J,101,-30,?\n",
        encoding="utf-8",
    )
    builder = CohortBuilder(cutoff_day=35, policy="policy_c", data_dir=data_dir)
    with pytest.raises(CohortInvariantError, match="courses reference table schema"):
        builder.load_inputs()

    # 2. Duplicate PK in courses.csv
    (data_dir / "courses.csv").write_text(
        "code_module,code_presentation,module_presentation_length\n"
        "AAA,2013J,268\n"
        "AAA,2013J,268\n",
        encoding="utf-8",
    )
    (data_dir / "studentInfo.csv").write_text(
        "code_module,code_presentation,id_student,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result\n"
        "AAA,2013J,101,M,East Anglian Region,HE Qualification,90-100%,55<=,0,240,N,Pass\n",
        encoding="utf-8",
    )
    (data_dir / "studentRegistration.csv").write_text(
        "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
        "AAA,2013J,101,-30,?\n",
        encoding="utf-8",
    )
    builder = CohortBuilder(cutoff_day=35, policy="policy_c", data_dir=data_dir)
    with pytest.raises(CohortInvariantError, match="courses reference table primary key"):
        builder.load_inputs()
