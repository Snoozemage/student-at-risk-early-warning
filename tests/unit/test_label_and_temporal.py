"""Unit tests for source-level label and temporal property checks."""
import pandas as pd

from student_retention.validation.label_checks import validate_source_labels
from student_retention.validation.temporal_checks import (
    validate_source_temporal_properties,
)


def test_label_checks_valid(sample_student_info_df, sample_student_registration_df):
    res = validate_source_labels(sample_student_info_df, sample_student_registration_df)
    assert res["is_valid"] is True
    assert res["total_fatal_errors"] == 0
    assert len(res["violations"]) == 0


def test_label_checks_unknown_outcome_error(sample_student_info_df, sample_student_registration_df):
    bad_info = sample_student_info_df.copy()
    bad_info.loc[0, "final_result"] = "UnknownCategory"
    res = validate_source_labels(bad_info, sample_student_registration_df)
    assert res["is_valid"] is False
    assert res["total_fatal_errors"] == 1
    assert any("unknown/missing final_result" in v for v in res["violations"])


def test_label_checks_record_554243_anomaly_warning(sample_student_info_df, sample_student_registration_df):
    # Construct record BBB / 2013J / 554243 with Fail and date_unregistration 166.0
    info_df = pd.DataFrame({
        "code_module": ["BBB"],
        "code_presentation": ["2013J"],
        "id_student": [554243],
        "final_result": ["Fail"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["BBB"],
        "code_presentation": ["2013J"],
        "id_student": [554243],
        "date_registration": [-29],
        "date_unregistration": [166.0],
    })
    res = validate_source_labels(info_df, reg_df)
    # Must NOT fail the pipeline (classified as warning, not fatal error)
    assert res["is_valid"] is True
    assert res["total_fatal_errors"] == 0
    assert res["total_warnings"] == 1

    check_res = next(c for c in res["checks"] if c["check_name"] == "non_withdrawn_with_unregistration_date")
    assert check_res["severity"] == "WARNING"
    assert check_res["count"] == 1
    assert check_res["known_record_554243_detected"] is True


def test_label_checks_withdrawn_missing_unregistration_warning():
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [12345],
        "final_result": ["Withdrawn"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [12345],
        "date_registration": [-50],
        "date_unregistration": [pd.NA],
    })
    res = validate_source_labels(info_df, reg_df)
    assert res["is_valid"] is True
    assert res["total_fatal_errors"] == 0
    assert res["total_warnings"] == 1
    check_res = next(c for c in res["checks"] if c["check_name"] == "withdrawn_with_missing_unregistration_date")
    assert check_res["count"] == 1


def test_label_checks_unregistration_before_registration_error():
    info_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [12345],
        "final_result": ["Withdrawn"],
    })
    reg_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [12345],
        "date_registration": [-10],
        "date_unregistration": [-20],  # unreg before reg
    })
    res = validate_source_labels(info_df, reg_df)
    assert res["is_valid"] is False
    assert res["total_fatal_errors"] >= 1
    assert any("date_unregistration < date_registration" in v for v in res["violations"])


def test_temporal_checks_valid(sample_tables):
    res = validate_source_temporal_properties(sample_tables)
    assert res["is_valid"] is True
    assert res["total_fatal_errors"] == 0


def test_temporal_checks_course_length_error(sample_tables):
    bad_tables = dict(sample_tables)
    bad_courses = sample_tables["courses"].copy()
    bad_courses.loc[0, "module_presentation_length"] = 0
    bad_tables["courses"] = bad_courses

    res = validate_source_temporal_properties(bad_tables)
    assert res["is_valid"] is False
    assert any("non-positive" in v for v in res["violations"])


def test_temporal_checks_inverted_vle_week_error(sample_tables):
    bad_tables = dict(sample_tables)
    bad_vle = sample_tables["vle"].copy()
    bad_vle.loc[0, "week_from"] = 10
    bad_vle.loc[0, "week_to"] = 5  # week_to < week_from
    bad_tables["vle"] = bad_vle

    res = validate_source_temporal_properties(bad_tables)
    assert res["is_valid"] is False
    assert any("week_to < week_from" in v for v in res["violations"])


def test_temporal_checks_missing_dates_are_warnings(sample_tables):
    # In OULAD, missing registration date (45 records) and missing assessment date (11 exams) are warnings, not fatal errors
    test_tables = dict(sample_tables)
    reg_df = sample_tables["student_registration"].copy()
    reg_df.loc[0, "date_registration"] = pd.NA
    test_tables["student_registration"] = reg_df

    ass_df = sample_tables["assessments"].copy()
    ass_df.loc[0, "date"] = pd.NA
    test_tables["assessments"] = ass_df

    res = validate_source_temporal_properties(test_tables)
    assert res["is_valid"] is True
    assert res["total_fatal_errors"] == 0
    assert res["total_warnings"] >= 2


def test_temporal_checks_student_vle_valid(sample_tables):
    res = validate_source_temporal_properties(sample_tables)
    assert res["is_valid"] is True
    assert res["total_fatal_errors"] == 0

    vle_checks = {c["check_name"]: c for c in res["checks"] if c["table"] == "student_vle"}
    assert "interaction_date_present" in vle_checks
    assert vle_checks["interaction_date_present"]["is_valid"] is True
    assert vle_checks["interaction_date_present"]["count"] == 0

    assert "precourse_interaction_dates_plausible" in vle_checks
    assert vle_checks["precourse_interaction_dates_plausible"]["count"] == 0

    assert "postcourse_interaction_dates_plausible" in vle_checks
    assert vle_checks["postcourse_interaction_dates_plausible"]["count"] == 0


def test_temporal_checks_student_vle_missing_date_error(sample_tables):
    bad_tables = dict(sample_tables)
    bad_vle = sample_tables["student_vle"].copy()
    bad_vle.loc[0, "date"] = pd.NA
    bad_tables["student_vle"] = bad_vle

    res = validate_source_temporal_properties(bad_tables)
    assert res["is_valid"] is False
    assert any("missing interaction date" in v for v in res["violations"])

    vle_check = next(c for c in res["checks"] if c["check_name"] == "interaction_date_present")
    assert vle_check["is_valid"] is False
    assert vle_check["count"] == 1
    assert len(vle_check["examples"]) == 1


def test_temporal_checks_student_vle_early_date_warning(sample_tables):
    test_tables = dict(sample_tables)
    test_vle = sample_tables["student_vle"].copy()
    test_vle.loc[0, "date"] = -30  # earlier than Day -25
    test_tables["student_vle"] = test_vle

    res = validate_source_temporal_properties(test_tables)
    assert res["is_valid"] is True  # non-fatal warning
    assert any("before Day -25" in w for w in res["warnings"])

    early_check = next(c for c in res["checks"] if c["check_name"] == "precourse_interaction_dates_plausible")
    assert early_check["count"] == 1
    assert len(early_check["examples"]) == 1
    assert early_check["examples"][0]["date"] == -30


def test_temporal_checks_student_vle_late_date_warning(sample_tables):
    test_tables = dict(sample_tables)
    test_vle = sample_tables["student_vle"].copy()
    # AAA 2013J presentation length is 268 days
    test_vle.loc[0, "date"] = 280  # exceeds presentation length
    test_tables["student_vle"] = test_vle

    res = validate_source_temporal_properties(test_tables)
    assert res["is_valid"] is True  # non-fatal warning
    assert any("after module presentation completion" in w for w in res["warnings"])

    late_check = next(c for c in res["checks"] if c["check_name"] == "postcourse_interaction_dates_plausible")
    assert late_check["count"] == 1
    assert len(late_check["examples"]) == 1
    assert late_check["examples"][0]["date"] == 280


def test_temporal_checks_student_vle_boundary_values(sample_tables):
    # Boundary values: Day -25 and Day 268 (exact course presentation length) must pass with 0 warnings
    test_tables = dict(sample_tables)
    test_vle = sample_tables["student_vle"].copy()
    test_vle.loc[0, "date"] = -25  # exact lower boundary
    test_vle.loc[1, "date"] = 268  # exact presentation length boundary
    test_tables["student_vle"] = test_vle

    res = validate_source_temporal_properties(test_tables)
    assert res["is_valid"] is True
    assert res["total_fatal_errors"] == 0
    assert not any("before Day -25" in w for w in res["warnings"])
    assert not any("after module presentation completion" in w for w in res["warnings"])
