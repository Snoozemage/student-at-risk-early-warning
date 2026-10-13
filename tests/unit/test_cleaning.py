"""Unit tests for identifier validation, missingness handling, and normalization."""
import pandas as pd
import pytest

from student_retention.cleaning.identifiers import validate_identifiers
from student_retention.cleaning.missing_values import (
    audit_missing_values,
    clean_missing_values,
)
from student_retention.cleaning.normalization import DataCoercionError, normalize_table


def test_validate_identifiers_valid(sample_student_info_df):
    report = validate_identifiers("student_info", sample_student_info_df)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0


def test_validate_identifiers_invalid_module(sample_student_info_df):
    bad_df = sample_student_info_df.copy()
    bad_df.loc[0, "code_module"] = "ZZZ"
    report = validate_identifiers("student_info", bad_df)
    assert report["is_valid"] is False
    assert any("invalid code_module" in err for err in report["errors"])


def test_validate_identifiers_invalid_presentation(sample_student_info_df):
    bad_df = sample_student_info_df.copy()
    bad_df.loc[0, "code_presentation"] = "2020X"
    report = validate_identifiers("student_info", bad_df)
    assert report["is_valid"] is False
    assert any("invalid code_presentation" in err for err in report["errors"])


def test_validate_identifiers_non_positive_student_id(sample_student_info_df):
    bad_df = sample_student_info_df.copy()
    bad_df.loc[0, "id_student"] = -5
    report = validate_identifiers("student_info", bad_df)
    assert report["is_valid"] is False
    assert any("non-positive id_student" in err for err in report["errors"])


def test_clean_missing_values_marker_replacement():
    df = pd.DataFrame({"col1": ["val", "?", "other"], "col2": [1, "?", 3]})
    cleaned = clean_missing_values("test_table", df)
    assert pd.isna(cleaned.loc[1, "col1"])
    assert pd.isna(cleaned.loc[1, "col2"])


def test_audit_missing_values_forbidden_marker(sample_courses_df):
    bad_df = sample_courses_df.copy()
    bad_df.loc[0, "code_module"] = ""  # empty string forbidden
    report = audit_missing_values("courses", bad_df)
    assert report["is_valid"] is False
    assert any("empty string" in v for v in report["violations"])


def test_audit_missing_values_all_forbidden_markers():
    for marker in ["", "null", "NULL", "na", "NA", "None"]:
        df = pd.DataFrame({
            "code_module": [marker, "BBB"],
            "code_presentation": ["2013J", "2013J"],
            "module_presentation_length": [268, 268],
        })
        report = audit_missing_values("courses", df)
        assert report["is_valid"] is False
        assert len(report["forbidden_marker_violations"]) >= 1
        violation = report["forbidden_marker_violations"][0]
        assert violation["table"] == "courses"
        assert violation["column"] == "code_module"
        assert violation["marker"] == marker
        assert violation["count"] == 1


def test_clean_missing_values_preserves_legitimate_strings():
    # Only '?' is converted to pd.NA. Forbidden markers like 'NA', 'null' are NOT silently turned into pd.NA.
    df = pd.DataFrame({
        "code_module": ["AAA", "?", "NA", "null"],
    })
    cleaned = clean_missing_values("courses", df)
    assert cleaned.loc[0, "code_module"] == "AAA"
    assert pd.isna(cleaned.loc[1, "code_module"])
    assert cleaned.loc[2, "code_module"] == "NA"
    assert cleaned.loc[3, "code_module"] == "null"


def test_clean_missing_values_strict_marker_preserves_escaped_question():
    """Verify strictly configured marker '?' becomes pd.NA while '\\?' is preserved."""
    df = pd.DataFrame({
        "code_module": ["AAA", "?", r"\?", "NA"],
    })
    cleaned = clean_missing_values("courses", df)
    assert cleaned.loc[0, "code_module"] == "AAA"
    assert pd.isna(cleaned.loc[1, "code_module"])
    assert cleaned.loc[2, "code_module"] == r"\?"
    assert cleaned.loc[3, "code_module"] == "NA"


def test_normalize_preserves_escaped_question_marker():
    """Verify normalize_table converts '?' to pd.NA but preserves '\\?' as a string without silent coercion."""
    df = pd.DataFrame({
        "code_module": ["AAA", "?", r"\?"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "module_presentation_length": [268, 268, 268],
    })
    normalized = normalize_table("courses", df)
    assert normalized.loc[0, "code_module"] == "AAA"
    assert pd.isna(normalized.loc[1, "code_module"])
    assert normalized.loc[2, "code_module"] == r"\?"

    # In integer column, '\?' is not a missing marker and must raise DataCoercionError
    bad_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "module_presentation_length": [r"\?"],
    })
    with pytest.raises(DataCoercionError):
        normalize_table("courses", bad_df)



def test_audit_missing_values_non_nullable_violation(sample_courses_df):
    bad_df = sample_courses_df.copy()
    bad_df.loc[0, "code_module"] = pd.NA
    report = audit_missing_values("courses", bad_df)
    assert report["is_valid"] is False
    assert any("non-nullable" in v for v in report["violations"])


def test_normalize_table_type_coercion(sample_courses_df):
    # Pass strings
    str_courses = pd.DataFrame({
        "code_module": ["AAA ", "BBB"],
        "code_presentation": ["2013J", "2013J"],
        "module_presentation_length": ["268", "268"],
    })
    normalized = normalize_table("courses", str_courses)

    # String should be stripped and string dtype
    assert normalized["code_module"].iloc[0] == "AAA"
    assert str(normalized["code_module"].dtype) == "string"
    assert str(normalized["module_presentation_length"].dtype) == "Int64"
    assert normalized["module_presentation_length"].iloc[0] == 268


def test_normalize_table_missing_columns():
    df = pd.DataFrame({"code_module": ["AAA"]})
    with pytest.raises(ValueError, match="missing expected schema columns"):
        normalize_table("courses", df)


def test_normalize_malformed_integer_non_nullable():
    df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "module_presentation_length": ["not_a_number"],
    })
    with pytest.raises(DataCoercionError) as exc_info:
        normalize_table("courses", df)
    assert exc_info.value.table_name == "courses"
    assert exc_info.value.column == "module_presentation_length"
    assert exc_info.value.expected_type == "integer"
    assert exc_info.value.malformed_count == 1
    assert "not_a_number" in exc_info.value.sample_values


def test_normalize_malformed_integer_nullable():
    # assessments.date is nullable: true, but "bad_date" must NOT silently turn into pd.NA
    df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_assessment": [1757],
        "assessment_type": ["TMA"],
        "date": ["bad_date"],
        "weight": [10.0],
    })
    with pytest.raises(DataCoercionError) as exc_info:
        normalize_table("assessments", df)
    assert exc_info.value.column == "date"
    assert "bad_date" in exc_info.value.sample_values


def test_normalize_fractional_in_integer():
    # Integer column with fractional float
    df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "module_presentation_length": [268.5],
    })
    with pytest.raises(DataCoercionError) as exc_info:
        normalize_table("courses", df)
    assert exc_info.value.column == "module_presentation_length"


def test_normalize_malformed_float():
    df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_assessment": [1757],
        "assessment_type": ["TMA"],
        "date": [19],
        "weight": ["not_a_float"],
    })
    with pytest.raises(DataCoercionError) as exc_info:
        normalize_table("assessments", df)
    assert exc_info.value.column == "weight"
    assert exc_info.value.expected_type == "float"


def test_normalize_legitimate_missing_marker_preserved():
    # In assessments, date is nullable. '?' or pd.NA should coerce to pd.NA cleanly without error
    df = pd.DataFrame({
        "code_module": ["AAA", "BBB"],
        "code_presentation": ["2013J", "2013J"],
        "id_assessment": [1757, 1758],
        "assessment_type": ["TMA", "TMA"],
        "date": ["?", pd.NA],
        "weight": [10.0, 20.0],
    })
    normalized = normalize_table("assessments", df)
    assert pd.isna(normalized.loc[0, "date"])
    assert pd.isna(normalized.loc[1, "date"])
    assert str(normalized["date"].dtype) == "Int64"
