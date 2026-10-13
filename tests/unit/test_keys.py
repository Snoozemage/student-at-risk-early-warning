"""Unit tests for primary key, grain repetition, and referential integrity validation."""
import pandas as pd

from student_retention.validation.key_checks import (
    validate_all_primary_keys,
    validate_table_primary_key,
)
from student_retention.validation.relationship_checks import (
    validate_all_relationships,
    validate_relationship,
    validate_student_assessment_enrollment,
)


def test_primary_key_valid(sample_courses_df):
    res = validate_table_primary_key("courses", sample_courses_df)
    assert res["is_valid"] is True
    assert res["is_primary_key"] is True
    assert res["rows_in_repeated_groups"] == 0
    assert res["null_key_count"] == 0


def test_primary_key_duplicate(sample_courses_df):
    dup_df = pd.concat([sample_courses_df, sample_courses_df.iloc[[0]]], ignore_index=True)
    res = validate_table_primary_key("courses", dup_df)
    assert res["is_valid"] is False
    assert res["rows_in_repeated_groups"] == 2
    assert res["excess_duplicate_rows"] == 1
    assert any("violates primary key uniqueness" in v for v in res["violations"])


def test_student_vle_permitted_repeated_grain(sample_student_vle_df):
    # Add duplicate interaction on same grain (student 11391 visiting site 546943 again on day -10)
    dup_row = sample_student_vle_df.iloc[[0]].copy()
    dup_row["sum_click"] = 8
    event_df = pd.concat([sample_student_vle_df, dup_row], ignore_index=True)

    res = validate_table_primary_key("student_vle", event_df)
    # Permitted by event log semantics: is_valid remains True
    assert res["is_valid"] is True
    assert res["is_primary_key"] is False
    assert res["allow_duplicates"] is True
    assert res["semantic_type"] == "interaction_event_log"
    assert res["rows_in_repeated_groups"] == 2
    assert res["excess_duplicate_rows"] == 1
    assert len(res["violations"]) == 0


def test_primary_key_null_value(sample_courses_df):
    null_df = sample_courses_df.copy()
    null_df.loc[0, "code_module"] = pd.NA
    res = validate_table_primary_key("courses", null_df)
    assert res["is_valid"] is False
    assert res["null_key_count"] > 0
    assert any("contains 1 null values" in v for v in res["violations"])


def test_relationship_valid(sample_courses_df, sample_student_info_df):
    res = validate_relationship(
        parent_name="courses",
        parent_df=sample_courses_df,
        child_name="student_info",
        child_df=sample_student_info_df,
        join_keys=["code_module", "code_presentation"],
        allow_orphans=False,
    )
    assert res["is_valid"] is True
    assert res["orphan_child_rows"] == 0


def test_relationship_orphans_detected(sample_courses_df, sample_student_info_df):
    orphan_df = sample_student_info_df.copy()
    orphan_row = orphan_df.iloc[[0]].copy()
    orphan_row["code_module"] = "GGG"
    orphan_df = pd.concat([orphan_df, orphan_row], ignore_index=True)

    res = validate_relationship(
        parent_name="courses",
        parent_df=sample_courses_df,
        child_name="student_info",
        child_df=orphan_df,
        join_keys=["code_module", "code_presentation"],
        allow_orphans=False,
    )
    assert res["is_valid"] is False
    assert res["orphan_child_rows"] == 1
    assert any("not found in parent" in v for v in res["violations"])


def test_relationship_vle_to_student_vle_composite(sample_vle_df, sample_student_vle_df):
    # Valid composite relationship
    res = validate_relationship(
        parent_name="vle",
        parent_df=sample_vle_df,
        child_name="student_vle",
        child_df=sample_student_vle_df,
        join_keys=["id_site", "code_module", "code_presentation"],
        allow_orphans=False,
    )
    assert res["is_valid"] is True

    # Deliberately mismatched presentation in student_vle
    mismatched_sv = sample_student_vle_df.copy()
    mismatched_sv.loc[0, "code_presentation"] = "2014J"  # site 546943 is 2013J in vle
    res_bad = validate_relationship(
        parent_name="vle",
        parent_df=sample_vle_df,
        child_name="student_vle",
        child_df=mismatched_sv,
        join_keys=["id_site", "code_module", "code_presentation"],
        allow_orphans=False,
    )
    assert res_bad["is_valid"] is False
    assert res_bad["orphan_child_rows"] == 1
    assert any("not found in parent" in v for v in res_bad["violations"])


def test_relationship_student_info_to_student_vle(sample_student_info_df, sample_student_vle_df):
    res = validate_relationship(
        parent_name="student_info",
        parent_df=sample_student_info_df,
        child_name="student_vle",
        child_df=sample_student_vle_df,
        join_keys=["code_module", "code_presentation", "id_student"],
        allow_orphans=False,
    )
    assert res["is_valid"] is True

    # Mismatched student enrollment
    mismatched_sv = sample_student_vle_df.copy()
    mismatched_sv.loc[0, "id_student"] = 999999
    res_bad = validate_relationship(
        parent_name="student_info",
        parent_df=sample_student_info_df,
        child_name="student_vle",
        child_df=mismatched_sv,
        join_keys=["code_module", "code_presentation", "id_student"],
        allow_orphans=False,
    )
    assert res_bad["is_valid"] is False
    assert res_bad["orphan_child_rows"] == 1


def test_student_assessment_enrollment_compatibility(
    sample_student_assessment_df,
    sample_assessments_df,
    sample_student_info_df,
):
    # Valid setup: all assessments match student_info enrollment
    res = validate_student_assessment_enrollment(
        student_assessment_df=sample_student_assessment_df,
        assessments_df=sample_assessments_df,
        student_info_df=sample_student_info_df,
        allow_orphans=False,
    )
    assert res["is_valid"] is True
    assert res["orphan_child_rows"] == 0

    # Deliberately mismatched assessment-student combination
    # Assessment 1759 is BBB / 2013J. Student 11391 is enrolled in AAA / 2013J, NOT BBB / 2013J.
    bad_sa = sample_student_assessment_df.copy()
    bad_sa.loc[0, "id_assessment"] = 1759  # student 11391 taking BBB assessment
    res_bad = validate_student_assessment_enrollment(
        student_assessment_df=bad_sa,
        assessments_df=sample_assessments_df,
        student_info_df=sample_student_info_df,
        allow_orphans=False,
    )
    assert res_bad["is_valid"] is False
    assert res_bad["orphan_child_rows"] == 1
    assert any("does not match an active student_info enrollment" in v for v in res_bad["violations"])


def test_relationship_cardinality_1_to_1(sample_student_info_df, sample_student_registration_df):
    res = validate_relationship(
        parent_name="student_info",
        parent_df=sample_student_info_df,
        child_name="student_registration",
        child_df=sample_student_registration_df,
        join_keys=["code_module", "code_presentation", "id_student"],
        cardinality="1:1",
    )
    assert res["is_valid"] is True


def test_validate_all_keys_and_relationships(sample_tables):
    key_res = validate_all_primary_keys(sample_tables)
    assert key_res["is_valid"] is True

    rel_res = validate_all_relationships(sample_tables)
    assert rel_res["is_valid"] is True
    assert rel_res["configured_rules_count"] == 9
    assert rel_res["executed_rules_count"] == 9
    assert len(rel_res["relationships"]) == 9
    assert rel_res["total_violations"] == 0


def test_validate_all_relationships_missing_bridged_table(sample_tables):
    # If a table needed for bridged check (e.g. assessments) is missing, the rule must be reported as failed
    incomplete_tables = {k: v for k, v in sample_tables.items() if k != "assessments"}
    rel_res = validate_all_relationships(incomplete_tables)
    assert rel_res["is_valid"] is False
    # Check that bridged relationship rule is present in results and marked is_valid: False
    bridged_results = [r for r in rel_res["relationships"] if r.get("via") == "assessments"]
    assert len(bridged_results) == 1
    assert bridged_results[0]["is_valid"] is False
    assert any("Missing required table" in v for v in bridged_results[0]["violations"])


def test_validate_all_relationships_unsupported_rule(sample_tables):
    custom_cfg = {
        "source_quality_rules": {
            "referential_integrity": [
                {"parent": "courses", "child": "assessments"},  # missing join_keys
            ]
        }
    }
    rel_res = validate_all_relationships(sample_tables, rules_cfg=custom_cfg)
    assert rel_res["is_valid"] is False
    assert rel_res["configured_rules_count"] == 1
    assert rel_res["executed_rules_count"] == 1
    assert rel_res["relationships"][0]["is_valid"] is False
    assert any("unsupported" in v for v in rel_res["relationships"][0]["violations"])
