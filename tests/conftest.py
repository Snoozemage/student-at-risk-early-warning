"""Pytest fixtures for student_retention test suite."""
import pandas as pd
import pytest


@pytest.fixture
def sample_courses_df():
    return pd.DataFrame({
        "code_module": ["AAA", "BBB"],
        "code_presentation": ["2013J", "2013J"],
        "module_presentation_length": [268, 268],
    })


@pytest.fixture
def sample_assessments_df():
    return pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_assessment": [1757, 1758, 1759],
        "assessment_type": ["TMA", "TMA", "CMA"],
        "date": [19, 54, 54],
        "weight": [10.0, 20.0, 25.0],
    })


@pytest.fixture
def sample_vle_df():
    return pd.DataFrame({
        "id_site": [546943, 546714, 546715],
        "code_module": ["AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "activity_type": ["resource", "oucontent", "forumng"],
        "week_from": pd.Series([pd.NA, pd.NA, 1], dtype="Int64"),
        "week_to": pd.Series([pd.NA, pd.NA, 5], dtype="Int64"),
    })


@pytest.fixture
def sample_student_info_df():
    return pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [11391, 28400, 31604],
        "gender": ["M", "F", "F"],
        "region": ["East Anglian Region", "Scotland", "South East Region"],
        "highest_education": ["HE Qualification", "HE Qualification", "A Level or Equivalent"],
        "imd_band": ["90-100%", "20-30%", "50-60%"],
        "age_band": ["55<=", "35-55", "0-35"],
        "num_of_prev_attempts": [0, 0, 0],
        "studied_credits": [240, 60, 60],
        "disability": ["N", "N", "N"],
        "final_result": ["Pass", "Pass", "Pass"],
    })


@pytest.fixture
def sample_student_registration_df():
    return pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [11391, 28400, 31604],
        "date_registration": [-159, -53, -52],
        "date_unregistration": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Int64"),
    })


@pytest.fixture
def sample_student_assessment_df():
    return pd.DataFrame({
        "id_assessment": [1757, 1758, 1759],
        "id_student": [11391, 28400, 31604],
        "date_submitted": [18, 52, 53],
        "is_banked": [0, 0, 0],
        "score": [78.0, 70.0, 85.0],
    })


@pytest.fixture
def sample_student_vle_df():
    return pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [11391, 28400, 31604],
        "id_site": [546943, 546714, 546715],
        "date": [-10, 0, 15],
        "sum_click": [4, 1, 10],
    })


@pytest.fixture
def sample_tables(
    sample_courses_df,
    sample_assessments_df,
    sample_vle_df,
    sample_student_info_df,
    sample_student_registration_df,
    sample_student_assessment_df,
    sample_student_vle_df,
):
    return {
        "courses": sample_courses_df,
        "assessments": sample_assessments_df,
        "vle": sample_vle_df,
        "student_info": sample_student_info_df,
        "student_registration": sample_student_registration_df,
        "student_assessment": sample_student_assessment_df,
        "student_vle": sample_student_vle_df,
    }


@pytest.fixture
def mock_raw_data_dir(tmp_path, sample_tables):
    """Write sample tables as CSV files with OULAD naming into a temp directory."""
    name_map = {
        "courses": "courses.csv",
        "assessments": "assessments.csv",
        "vle": "vle.csv",
        "student_info": "studentInfo.csv",
        "student_registration": "studentRegistration.csv",
        "student_assessment": "studentAssessment.csv",
        "student_vle": "studentVle.csv",
    }
    for tbl_name, filename in name_map.items():
        csv_path = tmp_path / filename
        sample_tables[tbl_name].to_csv(csv_path, index=False)
    return tmp_path
