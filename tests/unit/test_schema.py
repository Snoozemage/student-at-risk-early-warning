"""Unit tests for schema constraint validation."""
import pandas as pd

from student_retention.validation.schema_checks import (
    validate_all_schemas,
    validate_table_schema,
)


def test_schema_valid_table(sample_courses_df):
    res = validate_table_schema("courses", sample_courses_df)
    assert res["is_valid"] is True
    assert len(res["violations"]) == 0


def test_schema_missing_column(sample_courses_df):
    bad_df = sample_courses_df.drop(columns=["module_presentation_length"])
    res = validate_table_schema("courses", bad_df)
    assert res["is_valid"] is False
    assert any("Missing required column 'module_presentation_length'" in v for v in res["violations"])


def test_schema_invalid_allowed_value(sample_student_info_df):
    bad_df = sample_student_info_df.copy()
    bad_df.loc[0, "final_result"] = "UnknownOutcome"
    res = validate_table_schema("student_info", bad_df)
    assert res["is_valid"] is False
    assert any("allowed_values" in v for v in res["violations"])


def test_schema_numeric_bounds_violation(sample_courses_df):
    bad_df = sample_courses_df.copy()
    bad_df.loc[0, "module_presentation_length"] = 150  # below min of 200
    res = validate_table_schema("courses", bad_df)
    assert res["is_valid"] is False
    assert any("below min=200" in v for v in res["violations"])


def test_schema_non_nullable_violation(sample_assessments_df):
    bad_df = sample_assessments_df.copy()
    bad_df.loc[0, "weight"] = pd.NA
    res = validate_table_schema("assessments", bad_df)
    assert res["is_valid"] is False
    assert any("nullable: false" in v for v in res["violations"])


def test_validate_all_schemas(sample_tables):
    res = validate_all_schemas(sample_tables)
    assert res["is_valid"] is True
    assert res["total_violations"] == 0


def test_schema_malformed_integer_value(sample_courses_df):
    bad_df = sample_courses_df.copy()
    bad_df["module_presentation_length"] = ["invalid_str", "268"]
    res = validate_table_schema("courses", bad_df)
    assert res["is_valid"] is False
    assert any("malformed non-integer" in v for v in res["violations"])


def test_schema_fractional_in_integer(sample_courses_df):
    bad_df = sample_courses_df.copy()
    bad_df["module_presentation_length"] = [268.7, 268.0]
    res = validate_table_schema("courses", bad_df)
    assert res["is_valid"] is False
    assert any("malformed non-integer" in v for v in res["violations"])


def test_schema_malformed_float_value(sample_assessments_df):
    bad_df = sample_assessments_df.copy()
    bad_df["weight"] = bad_df["weight"].astype(object)
    bad_df.loc[0, "weight"] = "not_a_float"
    res = validate_table_schema("assessments", bad_df)
    assert res["is_valid"] is False
    assert any("malformed non-float" in v for v in res["violations"])


def test_schema_legitimate_missing_marker_in_nullable(sample_assessments_df):
    valid_df = sample_assessments_df.copy()
    valid_df["date"] = valid_df["date"].astype("Int64")
    valid_df.loc[0, "date"] = pd.NA
    res = validate_table_schema("assessments", valid_df)
    assert res["is_valid"] is True


def test_schema_string_numbers_rejected_for_integer(sample_courses_df):
    # A column containing "268" and "269" as strings must NOT pass the integer schema contract
    bad_df = sample_courses_df.copy()
    bad_df["module_presentation_length"] = ["268", "269"]
    res = validate_table_schema("courses", bad_df)
    assert res["is_valid"] is False
    assert any("incompatible dtype" in v for v in res["violations"])


def test_schema_normalized_integer_dtype_accepted(sample_courses_df):
    valid_df = sample_courses_df.copy()
    valid_df["module_presentation_length"] = valid_df["module_presentation_length"].astype("Int64")
    res = validate_table_schema("courses", valid_df)
    assert res["is_valid"] is True
    assert len(res["violations"]) == 0


def test_schema_nullable_float_representation_accepted(sample_assessments_df):
    valid_df = sample_assessments_df.copy()
    valid_df["weight"] = valid_df["weight"].astype("Float64")
    res = validate_table_schema("assessments", valid_df)
    assert res["is_valid"] is True
    assert len(res["violations"]) == 0
