"""Unit tests for Phase 5 temporal feature engineering pipeline."""
import numpy as np
import pandas as pd
import pytest

from student_retention.features.feature_definitions import (
    CANONICAL_GRAIN,
    PRIMARY_FEATURE_ALLOWLIST,
    PRIMARY_FEATURE_DEFINITIONS,
    FeatureCategory,
)
from student_retention.features.feature_selection import (
    select_primary_features,
    verify_no_leakage_columns,
)
from student_retention.features.feature_validation import validate_feature_table
from student_retention.paths import QUALITY_REPORTS_DIR
from student_retention.transformation.assessment_features import (
    extract_assessment_features,
)
from student_retention.transformation.build_feature_table import FeatureTableBuilder
from student_retention.transformation.registration_features import (
    extract_registration_features,
)
from student_retention.transformation.vle_features import extract_vle_features

# =====================================================================
# 1. Tests for Feature Definitions & Allowlist
# =====================================================================

def test_feature_allowlist_count_and_locking():
    """Verify that primary feature allowlist is strictly locked at 22 features."""
    assert len(PRIMARY_FEATURE_ALLOWLIST) == 22
    assert len(PRIMARY_FEATURE_DEFINITIONS) == 22

    # Verify counts per functional category
    categories = [defn.category for defn in PRIMARY_FEATURE_DEFINITIONS.values()]
    assert categories.count(FeatureCategory.COURSE_CONTEXT) == 4
    assert categories.count(FeatureCategory.REGISTRATION) == 2
    assert categories.count(FeatureCategory.VLE_ENGAGEMENT) == 11
    assert categories.count(FeatureCategory.ASSESSMENT_PROGRESS) == 5


def test_feature_definitions_metadata_completeness():
    """Verify all 22 feature definitions have non-empty metadata fields."""
    for name, defn in PRIMARY_FEATURE_DEFINITIONS.items():
        assert defn.name == name
        assert defn.description
        assert defn.source_table
        assert defn.source_field
        assert defn.data_type


# =====================================================================
# 2. Tests for Registration Features
# =====================================================================

def test_registration_features_lead_time_and_late_flags():
    """Verify lead time sign conventions and late registration indicator logic."""
    cohort_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3, 4],
        "date_registration": [-30.0, 0.0, 15.0, np.nan],
    })

    reg_feats = extract_registration_features(cohort_df=cohort_df, cutoff_day=35)

    assert len(reg_feats) == 4
    # Student 1: registered 30 days in advance (-(-30) = +30)
    assert reg_feats.loc[reg_feats["id_student"] == 1, "registration_lead_time_days"].iloc[0] == 30.0
    assert reg_feats.loc[reg_feats["id_student"] == 1, "is_late_registration"].iloc[0] == 0.0

    # Student 2: registered on Day 0
    assert reg_feats.loc[reg_feats["id_student"] == 2, "registration_lead_time_days"].iloc[0] == 0.0
    assert reg_feats.loc[reg_feats["id_student"] == 2, "is_late_registration"].iloc[0] == 0.0

    # Student 3: late registration on Day 15 (lead time -15, is_late 1.0)
    assert reg_feats.loc[reg_feats["id_student"] == 3, "registration_lead_time_days"].iloc[0] == -15.0
    assert reg_feats.loc[reg_feats["id_student"] == 3, "is_late_registration"].iloc[0] == 1.0

    # Student 4: missing registration date (Policy B sensitivity)
    assert np.isnan(reg_feats.loc[reg_feats["id_student"] == 4, "registration_lead_time_days"].iloc[0])
    assert np.isnan(reg_feats.loc[reg_feats["id_student"] == 4, "is_late_registration"].iloc[0])


# =====================================================================
# 3. Tests for VLE Engagement Features
# =====================================================================

def test_vle_features_window_filtering_and_aggregations():
    """Verify VLE event filtering by window, pre-course vs recent vs total clicks."""
    cohort_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [101, 102],
    })

    # Student 101: clicks on Day -20, Day -5, Day 10, Day 25, Day 30, and future Day 40 (must be filtered out)
    # Student 102: zero activity
    student_vle_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA", "AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2013J", "2013J"],
        "id_student": [101, 101, 101, 101, 101, 101],
        "id_site": [1, 2, 1, 3, 4, 1],
        "date": [-20, -5, 10, 25, 30, 40],  # Day 40 must be ignored
        "sum_click": [5, 10, 20, 15, 25, 100],
    })

    vle_df = pd.DataFrame({
        "id_site": [1, 2, 3, 4],
        "activity_type": ["oucontent", "forumng", "quiz", "oucontent"],
    })

    vle_feats = extract_vle_features(
        cohort_df=cohort_df,
        student_vle_df=student_vle_df,
        vle_df=vle_df,
        cutoff_day=35,
    )

    s101 = vle_feats[vle_feats["id_student"] == 101].iloc[0]
    # Total clicks in [-25, 35]: 5 + 10 + 20 + 15 + 25 = 75 (Day 40 ignored)
    assert s101["vle_total_clicks_window"] == 75
    # Distinct dates in window: -20, -5, 10, 25, 30 -> 5 active days
    assert s101["vle_active_days_window"] == 5
    # Unique sites in window: 1, 2, 3, 4 -> 4 unique sites
    assert s101["vle_unique_sites_accessed"] == 4
    # Pre-course clicks [-25, -1]: 5 + 10 = 15
    assert s101["vle_pre_course_clicks"] == 15
    # Recent clicks [22, 35]: Day 25 (15) + Day 30 (25) = 40
    assert s101["vle_recent_clicks_14d"] == 40
    # Inactivity recency: max date <= 35 is 30 -> 35 - 30 = 5 days
    assert s101["days_since_last_vle_interaction"] == 5
    assert s101["has_no_vle_activity_by_cutoff"] == 0

    # Activity types for Student 101:
    # forumng: Day -5 (10 clicks)
    assert s101["vle_clicks_forumng"] == 10
    # oucontent: Day -20 (5) + Day 10 (20) + Day 30 (25) = 50
    assert s101["vle_clicks_oucontent"] == 50
    # quiz: Day 25 (15 clicks)
    assert s101["vle_clicks_quiz"] == 15

    # Inactive Student 102:
    s102 = vle_feats[vle_feats["id_student"] == 102].iloc[0]
    assert s102["vle_total_clicks_window"] == 0
    assert s102["vle_active_days_window"] == 0
    assert s102["vle_unique_sites_accessed"] == 0
    assert s102["days_since_last_vle_interaction"] == 65  # Sentinel 65
    assert s102["has_no_vle_activity_by_cutoff"] == 1
    assert s102["vle_velocity_ratio"] == 0.0


def test_vle_velocity_ratio_formula():
    """Verify exact formula for vle_velocity_ratio."""
    cohort_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [201],
    })

    # Student 201:
    # 29 days baseline [0, 28]: 290 clicks -> average daily baseline = 10.0
    # 7 days window [29, 35]: 140 clicks -> average daily velocity = 20.0
    # Velocity ratio = 20.0 / (10.0 + 1.0) = 20.0 / 11.0 = 1.8182
    student_vle_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [201, 201],
        "id_site": [1, 1],
        "date": [10, 32],
        "sum_click": [290, 140],
    })
    vle_df = pd.DataFrame({"id_site": [1], "activity_type": ["oucontent"]})

    vle_feats = extract_vle_features(
        cohort_df=cohort_df,
        student_vle_df=student_vle_df,
        vle_df=vle_df,
        cutoff_day=35,
    )
    ratio = vle_feats.iloc[0]["vle_velocity_ratio"]
    assert np.isclose(ratio, 1.8182, atol=1e-4)


# =====================================================================
# 4. Tests for Assessment Progress Features
# =====================================================================

def test_assessment_features_due_universe_and_timeliness():
    """Verify assessment progress features, due universe filtering, and timeliness."""
    cohort_df = pd.DataFrame({
        "code_module": ["BBB", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [301, 302, 303],
        "date_registration": [-20.0, -20.0, -20.0],
    })

    # Assessment schedule:
    # A1: TMA due Day 20 (in due universe)
    # A2: TMA due Day 30 (in due universe)
    # A3: TMA due Day 50 (FUTURE assessment, deadline > 35, must NOT count towards due)
    # A4: Exam due Day 30 (Exam, must NOT count towards continuous due universe)
    assessments_df = pd.DataFrame({
        "id_assessment": [1001, 1002, 1003, 1004],
        "code_module": ["BBB", "BBB", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J"],
        "assessment_type": ["TMA", "TMA", "TMA", "Exam"],
        "date": [20.0, 30.0, 50.0, 30.0],
    })

    # Submissions:
    # Student 301:
    # A1 submitted Day 18 (on time)
    # A2 submitted Day 33 (late but <= 35)
    # A3 submitted early on Day 25 (future assignment; must NOT inflate due progress counts)
    # Student 302:
    # A1 banked credit (is_banked = 1)
    # A2 not submitted
    # Student 303:
    # A1 not submitted, A2 not submitted (0 submissions)
    student_assessment_df = pd.DataFrame({
        "id_assessment": [1001, 1002, 1003, 1001],
        "id_student": [301, 301, 301, 302],
        "date_submitted": [18.0, 33.0, 25.0, -1.0],
        "is_banked": [0, 0, 0, 1],
    })

    assess_feats = extract_assessment_features(
        cohort_df=cohort_df,
        student_assessment_df=student_assessment_df,
        assessments_df=assessments_df,
        cutoff_day=35,
    )

    # Student 301:
    s301 = assess_feats[assess_feats["id_student"] == 301].iloc[0]
    assert s301["assessments_due_count"] == 2  # A1 and A2
    assert s301["submissions_count_lte_cutoff"] == 2  # A1 and A2
    assert s301["submitted_on_time_count"] == 1  # A1
    assert s301["late_submissions_count_lte_cutoff"] == 1  # A2
    assert s301["missing_due_submissions_count"] == 0

    # Invariant: on_time + late == submissions
    assert s301["submitted_on_time_count"] + s301["late_submissions_count_lte_cutoff"] == s301["submissions_count_lte_cutoff"]

    # Student 302 (banked A1):
    s302 = assess_feats[assess_feats["id_student"] == 302].iloc[0]
    assert s302["assessments_due_count"] == 2
    assert s302["submissions_count_lte_cutoff"] == 0  # Banked records do not count as current study volume
    assert s302["missing_due_submissions_count"] == 1  # Only A2 is missing; A1 satisfied via banked credit

    # Student 303 (missing all):
    s303 = assess_feats[assess_feats["id_student"] == 303].iloc[0]
    assert s303["assessments_due_count"] == 2
    assert s303["submissions_count_lte_cutoff"] == 0
    assert s303["missing_due_submissions_count"] == 2


def test_assessment_features_reasonable_opportunity_policy():
    """Verify late registrants are not penalized for assessments due before registration."""
    # Student 401 registered Day 25, while assessment was due Day 15
    cohort_df = pd.DataFrame({
        "code_module": ["CCC"],
        "code_presentation": ["2014J"],
        "id_student": [401],
        "date_registration": [25.0],
    })

    assessments_df = pd.DataFrame({
        "id_assessment": [2001],
        "code_module": ["CCC"],
        "code_presentation": ["2014J"],
        "assessment_type": ["TMA"],
        "date": [15.0],  # Due BEFORE registration
    })

    student_assessment_df = pd.DataFrame(columns=["id_assessment", "id_student", "date_submitted", "is_banked"])

    assess_feats = extract_assessment_features(
        cohort_df=cohort_df,
        student_assessment_df=student_assessment_df,
        assessments_df=assessments_df,
        cutoff_day=35,
        enforce_registration_opportunity=True,
    )

    s401 = assess_feats.iloc[0]
    # Under reasonable opportunity, unsubmitted pre-registration assessment is excluded from due count
    assert s401["assessments_due_count"] == 0
    assert s401["missing_due_submissions_count"] == 0


# =====================================================================
# 5. Tests for Feature Validation Engine
# =====================================================================

def test_feature_validation_detects_violations():
    """Verify that validate_feature_table detects schema and invariant errors."""
    valid_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "presentation_season": ["J"],
        "studied_credits": [60],
        "num_of_prev_attempts": [0],
        "registration_lead_time_days": [30.0],
        "is_late_registration": [0.0],
        "vle_total_clicks_window": [100],
        "vle_active_days_window": [10],
        "vle_unique_sites_accessed": [5],
        "vle_pre_course_clicks": [20],
        "vle_recent_clicks_14d": [30],
        "vle_velocity_ratio": [1.5],
        "days_since_last_vle_interaction": [2],
        "has_no_vle_activity_by_cutoff": [0],
        "vle_clicks_forumng": [10],
        "vle_clicks_oucontent": [70],
        "vle_clicks_quiz": [20],
        "assessments_due_count": [1],
        "submissions_count_lte_cutoff": [1],
        "submitted_on_time_count": [1],
        "late_submissions_count_lte_cutoff": [0],
        "missing_due_submissions_count": [0],
        "withdrawn_after_cutoff": [0],
    })

    # Valid table passes cleanly
    report = validate_feature_table(valid_df, expected_row_count=1)
    assert report["is_valid"] is True

    # 1. Missing allowlist feature
    invalid_df1 = valid_df.drop(columns=["vle_velocity_ratio"])
    with pytest.raises(ValueError, match="Missing required primary allowlist features"):
        validate_feature_table(invalid_df1)

    # 2. Invariant violation: on_time + late != submissions
    invalid_df2 = valid_df.copy()
    invalid_df2["late_submissions_count_lte_cutoff"] = 2  # 1 + 2 != 1
    with pytest.raises(ValueError, match="Assessment partition invariant violated"):
        validate_feature_table(invalid_df2)

    # 3. Inactive student with wrong sentinel value
    invalid_df3 = valid_df.copy()
    invalid_df3["has_no_vle_activity_by_cutoff"] = 1
    invalid_df3["vle_total_clicks_window"] = 0
    invalid_df3["days_since_last_vle_interaction"] = 10  # Must be 65
    with pytest.raises(ValueError, match="expected days_since_last_vle_interaction == 65"):
        validate_feature_table(invalid_df3)


# =====================================================================
# 6. Tests for Feature Selection
# =====================================================================

def test_feature_selection_and_anti_leakage():
    """Verify feature selection and anti-leakage column checks."""
    df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        **{f: [0] for f in PRIMARY_FEATURE_ALLOWLIST if f not in CANONICAL_GRAIN},
        "withdrawn_after_cutoff": [0],
        "final_result": ["Pass"],  # Leakage column
    })

    # Selecting primary features excludes 'final_result'
    selected = select_primary_features(df, include_target=True)
    assert "final_result" not in selected.columns
    # Canonical grain (3) + 22 allowlist (code_module shared) + target (1) = 25 unique columns
    assert len(selected.columns) == 25

    # verify_no_leakage_columns raises on forbidden columns
    with pytest.raises(ValueError, match="Forbidden leakage columns detected"):
        verify_no_leakage_columns(["studied_credits", "final_result"])


# =====================================================================
# 7. Tests for Configurable Cutoff Enforcement & Path Isolation
# =====================================================================

def test_unsupported_cutoff_day_rejected():
    """Verify that unsupported cutoff_day != 35 is rejected with a clear error across all modules."""
    cohort_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "date_registration": [-20.0],
    })
    vle_df = pd.DataFrame({"id_site": [1], "activity_type": ["quiz"]})
    student_vle = pd.DataFrame(
        columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    )
    assess_df = pd.DataFrame({
        "id_assessment": [1],
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "assessment_type": ["TMA"],
        "date": [20.0],
    })
    student_assess = pd.DataFrame(
        columns=["id_assessment", "id_student", "date_submitted", "is_banked"]
    )

    # FeatureTableBuilder constructor
    with pytest.raises(ValueError, match="Unsupported cutoff_day=40"):
        FeatureTableBuilder(cutoff_day=40)

    # VLE features extractor
    with pytest.raises(ValueError, match="Unsupported cutoff_day=40"):
        extract_vle_features(cohort_df, student_vle, vle_df, cutoff_day=40)

    # Registration features extractor
    with pytest.raises(ValueError, match="Unsupported cutoff_day=40"):
        extract_registration_features(cohort_df, cutoff_day=40)

    # Assessment features extractor
    with pytest.raises(ValueError, match="Unsupported cutoff_day=40"):
        extract_assessment_features(cohort_df, student_assess, assess_df, cutoff_day=40)


def test_quality_report_custom_path_isolation_does_not_overwrite_production(tmp_path):
    """Regression test proving custom/test run writes report to custom dir and never overwrites production."""
    prod_report_path = QUALITY_REPORTS_DIR / "feature_quality_report.json"
    prod_mtime_before = prod_report_path.stat().st_mtime if prod_report_path.exists() else None
    prod_content_before = prod_report_path.read_text(encoding="utf-8") if prod_report_path.exists() else None

    # Setup isolated test inputs and outputs
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    custom_output_dir = tmp_path / "custom_features"
    custom_output_dir.mkdir()

    # Create minimal synthetic datasets
    cohort = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1],
        "gender": ["M"],
        "region": ["East"],
        "highest_education": ["HE"],
        "imd_band": ["50%"],
        "age_band": ["0-35"],
        "num_of_prev_attempts": [0],
        "studied_credits": [60],
        "disability": ["N"],
        "date_registration": [-30.0],
        "withdrawn_after_cutoff": [0],
        "policy_version": ["Policy C"],
    })
    cohort.to_parquet(input_dir / "cohort.parquet")

    pd.DataFrame(
        columns=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    ).to_parquet(input_dir / "student_vle.parquet")
    pd.DataFrame({"id_site": [1], "activity_type": ["oucontent"]}).to_parquet(
        input_dir / "vle.parquet"
    )
    pd.DataFrame(
        columns=["id_assessment", "id_student", "date_submitted", "is_banked"]
    ).to_parquet(input_dir / "student_assessment.parquet")
    pd.DataFrame({
        "id_assessment": [1],
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "assessment_type": ["TMA"],
        "date": [20.0],
    }).to_parquet(input_dir / "assessments.parquet")

    builder = FeatureTableBuilder(cutoff_day=35)
    _feat_df, _report = builder.run_pipeline(
        cohort_path=input_dir / "cohort.parquet",
        interim_dir=input_dir,
        output_dir=custom_output_dir,
        policy_version="Policy C",
    )

    # 1. Custom output directory must contain the generated report
    custom_report_path = custom_output_dir / "feature_quality_report.json"
    assert custom_report_path.exists(), "Custom quality report was not saved in custom output_dir"
    assert (custom_output_dir / "feature_matrix_policy_c.parquet").exists()

    # 2. Production quality report must NOT have been overwritten or modified
    if prod_report_path.exists():
        prod_mtime_after = prod_report_path.stat().st_mtime
        prod_content_after = prod_report_path.read_text(encoding="utf-8")
        assert prod_mtime_before == prod_mtime_after, "Production report mtime was modified by custom run!"
        assert prod_content_before == prod_content_after, "Production report content was altered by custom run!"
