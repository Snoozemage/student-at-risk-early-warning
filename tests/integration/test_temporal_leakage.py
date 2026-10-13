"""Integration tests for temporal anti-leakage invariants and real-data pipeline integrity."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_retention.features.feature_definitions import (
    CANONICAL_GRAIN,
    CANONICAL_TARGET,
    PRIMARY_FEATURE_ALLOWLIST,
)
from student_retention.features.feature_selection import (
    verify_no_leakage_columns,
)
from student_retention.features.feature_validation import validate_feature_table
from student_retention.paths import (
    INTERIM_NORMALIZED_DIR,
    PROCESSED_COHORT_DIR,
    PROCESSED_FEATURES_DIR,
)
from student_retention.transformation.assessment_features import (
    extract_assessment_features,
)
from student_retention.transformation.build_feature_table import FeatureTableBuilder
from student_retention.transformation.vle_features import extract_vle_features

# =====================================================================
# 1. Temporal Anti-Leakage Invariant Tests
# =====================================================================

def test_vle_future_events_cannot_alter_cutoff_features():
    """Prove that adding VLE interactions strictly after Day 35 does not change Day 35 features."""
    cohort_df = pd.DataFrame({
        "code_module": ["AAA"],
        "code_presentation": ["2013J"],
        "id_student": [1001],
    })

    # Baseline VLE events on Day 10, Day 20, Day 30
    baseline_vle = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [1001, 1001, 1001],
        "id_site": [10, 20, 30],
        "date": [10, 20, 30],
        "sum_click": [5, 15, 25],
    })
    vle_metadata = pd.DataFrame({
        "id_site": [10, 20, 30, 99],
        "activity_type": ["oucontent", "forumng", "quiz", "oucontent"],
    })

    baseline_feats = extract_vle_features(
        cohort_df=cohort_df,
        student_vle_df=baseline_vle,
        vle_df=vle_metadata,
        cutoff_day=35,
    )

    # Injected future VLE event: Day 36 with 500 clicks
    future_vle = pd.concat([
        baseline_vle,
        pd.DataFrame({
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "id_student": [1001],
            "id_site": [99],
            "date": [36],  # Future event strictly > Day 35
            "sum_click": [500],
        }),
    ], ignore_index=True)

    future_feats = extract_vle_features(
        cohort_df=cohort_df,
        student_vle_df=future_vle,
        vle_df=vle_metadata,
        cutoff_day=35,
    )

    # All Day 35 features must be 100% identical
    vle_feature_cols = [
        col for col in baseline_feats.columns if col.startswith(("vle_", "days_", "has_"))
    ]
    pd.testing.assert_frame_equal(
        baseline_feats[vle_feature_cols],
        future_feats[vle_feature_cols],
        check_dtype=True,
    )


def test_assessment_future_submissions_cannot_alter_cutoff_features():
    """Prove that adding assessment submissions strictly after Day 35 does not change Day 35 features."""
    cohort_df = pd.DataFrame({
        "code_module": ["BBB"],
        "code_presentation": ["2013J"],
        "id_student": [2001],
        "date_registration": [-30.0],
    })

    assessments_df = pd.DataFrame({
        "id_assessment": [501],
        "code_module": ["BBB"],
        "code_presentation": ["2013J"],
        "assessment_type": ["TMA"],
        "date": [25.0],  # Due Day 25
    })

    # Case A: Student has NOT submitted by Day 35
    empty_subs = pd.DataFrame(columns=["id_assessment", "id_student", "date_submitted", "is_banked"])
    feats_before = extract_assessment_features(
        cohort_df=cohort_df,
        student_assessment_df=empty_subs,
        assessments_df=assessments_df,
        cutoff_day=35,
    )

    # Case B: Student submits late on Day 45 (after cutoff)
    future_subs = pd.DataFrame({
        "id_assessment": [501],
        "id_student": [2001],
        "date_submitted": [45.0],  # Strictly > Day 35
        "is_banked": [0],
    })
    feats_after = extract_assessment_features(
        cohort_df=cohort_df,
        student_assessment_df=future_subs,
        assessments_df=assessments_df,
        cutoff_day=35,
    )

    # Features at Day 35 must not change: student still has 0 submissions and 1 missing submission
    assess_cols = [
        "assessments_due_count",
        "submissions_count_lte_cutoff",
        "submitted_on_time_count",
        "late_submissions_count_lte_cutoff",
        "missing_due_submissions_count",
    ]
    pd.testing.assert_frame_equal(
        feats_before[assess_cols],
        feats_after[assess_cols],
        check_dtype=True,
    )


def test_future_outcome_modification_does_not_alter_features():
    """Prove that modifying final_result or date_unregistration does not alter predictive features."""
    builder = FeatureTableBuilder(cutoff_day=35)

    # Cohort with Retained / Pass
    cohort_pass = pd.DataFrame({
        "code_module": ["DDD"],
        "code_presentation": ["2013B"],
        "id_student": [3001],
        "studied_credits": [60],
        "num_of_prev_attempts": [0],
        "date_registration": [-25.0],
        "date_unregistration": [np.nan],
        "final_result": ["Pass"],
        "withdrawn_after_cutoff": [0],
    })

    # Exactly same student but with future unregistration on Day 120 (Withdrawn)
    cohort_withdrawn = pd.DataFrame({
        "code_module": ["DDD"],
        "code_presentation": ["2013B"],
        "id_student": [3001],
        "studied_credits": [60],
        "num_of_prev_attempts": [0],
        "date_registration": [-25.0],
        "date_unregistration": [120.0],
        "final_result": ["Withdrawn"],
        "withdrawn_after_cutoff": [1],
    })

    vle_data = pd.DataFrame({
        "code_module": ["DDD"],
        "code_presentation": ["2013B"],
        "id_student": [3001],
        "id_site": [1],
        "date": [15],
        "sum_click": [10],
    })
    vle_meta = pd.DataFrame({"id_site": [1], "activity_type": ["oucontent"]})
    assess_data = pd.DataFrame(columns=["id_assessment", "id_student", "date_submitted", "is_banked"])
    assess_meta = pd.DataFrame({
        "id_assessment": [999],
        "code_module": ["DDD"],
        "code_presentation": ["2013B"],
        "assessment_type": ["TMA"],
        "date": [20.0],
    })

    feats_pass = builder.build_feature_table(
        cohort_df=cohort_pass,
        student_vle_df=vle_data,
        vle_df=vle_meta,
        student_assessment_df=assess_data,
        assessments_df=assess_meta,
        include_demographics=False,
    )

    feats_withdrawn = builder.build_feature_table(
        cohort_df=cohort_withdrawn,
        student_vle_df=vle_data,
        vle_df=vle_meta,
        student_assessment_df=assess_data,
        assessments_df=assess_meta,
        include_demographics=False,
    )

    # All 22 predictive features must be strictly identical
    pd.testing.assert_frame_equal(
        feats_pass[PRIMARY_FEATURE_ALLOWLIST],
        feats_withdrawn[PRIMARY_FEATURE_ALLOWLIST],
        check_dtype=True,
    )

    # Only target label changes
    assert feats_pass[CANONICAL_TARGET].iloc[0] == 0
    assert feats_withdrawn[CANONICAL_TARGET].iloc[0] == 1


# =====================================================================
# 2. End-to-End Real-Data Pipeline Integration Tests
# =====================================================================

REQUIRED_INTERIM_TABLES = (
    "student_vle.parquet",
    "vle.parquet",
    "student_assessment.parquet",
    "assessments.parquet",
)


def has_preprocessed_pipeline_inputs(
    cohort_dir: Path = PROCESSED_COHORT_DIR,
    interim_dir: Path = INTERIM_NORMALIZED_DIR,
) -> bool:
    """Check if all preprocessed inputs required to build the feature matrix are available."""
    cohort_path = cohort_dir / "cohort_policy_c.parquet"
    if not cohort_path.exists():
        return False
    return all((interim_dir / table).exists() for table in REQUIRED_INTERIM_TABLES)


def can_run_real_data_audit(
    features_dir: Path = PROCESSED_FEATURES_DIR,
    cohort_dir: Path = PROCESSED_COHORT_DIR,
    interim_dir: Path = INTERIM_NORMALIZED_DIR,
) -> bool:
    """Check if pre-generated feature matrix exists or all preprocessed inputs are available."""
    feat_path = features_dir / "feature_matrix_policy_c.parquet"
    if feat_path.exists():
        return True
    return has_preprocessed_pipeline_inputs(cohort_dir=cohort_dir, interim_dir=interim_dir)


def test_real_data_audit_prerequisite_conditions(tmp_path: Path):
    """Regression test proving prerequisite check handles missing inputs, raw-only presence, and complete setups."""
    empty_features_dir = tmp_path / "features"
    empty_cohort_dir = tmp_path / "cohort"
    empty_interim_dir = tmp_path / "interim"
    for d in (empty_features_dir, empty_cohort_dir, empty_interim_dir):
        d.mkdir(parents=True)

    # 1. Clean checkout state: all directories empty -> cannot run audit
    assert not can_run_real_data_audit(
        features_dir=empty_features_dir,
        cohort_dir=empty_cohort_dir,
        interim_dir=empty_interim_dir,
    )

    # 2. Presence of a single raw CSV (e.g. studentInfo.csv) must NOT satisfy prerequisites
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "studentInfo.csv").write_text("code_module,id_student\nAAA,1\n", encoding="utf-8")
    assert not has_preprocessed_pipeline_inputs(
        cohort_dir=empty_cohort_dir,
        interim_dir=empty_interim_dir,
    )
    assert not can_run_real_data_audit(
        features_dir=empty_features_dir,
        cohort_dir=empty_cohort_dir,
        interim_dir=empty_interim_dir,
    )

    # 3. Partial inputs: cohort exists, but interim normalized tables missing -> cannot run
    (empty_cohort_dir / "cohort_policy_c.parquet").write_text("dummy", encoding="utf-8")
    assert not has_preprocessed_pipeline_inputs(
        cohort_dir=empty_cohort_dir,
        interim_dir=empty_interim_dir,
    )
    assert not can_run_real_data_audit(
        features_dir=empty_features_dir,
        cohort_dir=empty_cohort_dir,
        interim_dir=empty_interim_dir,
    )

    # 4. Partial interim inputs: 3 of 4 interim tables exist -> cannot run
    for table in ["student_vle.parquet", "vle.parquet", "student_assessment.parquet"]:
        (empty_interim_dir / table).write_text("dummy", encoding="utf-8")
    assert not has_preprocessed_pipeline_inputs(
        cohort_dir=empty_cohort_dir,
        interim_dir=empty_interim_dir,
    )
    assert not can_run_real_data_audit(
        features_dir=empty_features_dir,
        cohort_dir=empty_cohort_dir,
        interim_dir=empty_interim_dir,
    )

    # 5. Complete preprocessed inputs: all 4 interim tables + cohort exist -> CAN run
    (empty_interim_dir / "assessments.parquet").write_text("dummy", encoding="utf-8")
    assert has_preprocessed_pipeline_inputs(
        cohort_dir=empty_cohort_dir,
        interim_dir=empty_interim_dir,
    )
    assert can_run_real_data_audit(
        features_dir=empty_features_dir,
        cohort_dir=empty_cohort_dir,
        interim_dir=empty_interim_dir,
    )

    # 6. Pre-computed feature matrix exists -> CAN run directly even without raw/interim
    (empty_features_dir / "feature_matrix_policy_c.parquet").write_text("dummy", encoding="utf-8")
    assert can_run_real_data_audit(
        features_dir=empty_features_dir,
        cohort_dir=tmp_path / "nonexistent_cohort",
        interim_dir=tmp_path / "nonexistent_interim",
    )


@pytest.mark.skipif(
    not can_run_real_data_audit(),
    reason=(
        "Neither pre-computed feature_matrix_policy_c.parquet nor required preprocessed inputs "
        "(cohort_policy_c.parquet and normalized interim parquet tables) are available"
    ),
)
def test_real_data_policy_c_feature_matrix_integrity(tmp_path: Path):
    """Verify that generated feature_matrix_policy_c satisfies all acceptance criteria."""
    feat_path = PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet"
    if feat_path.exists():
        # Pre-computed production matrix is available: audit it directly (read-only, no modifications)
        df = pd.read_parquet(feat_path)
    elif has_preprocessed_pipeline_inputs():
        # Feature matrix not pre-computed, but required preprocessed inputs exist:
        # Generate on the fly into isolated tmp_path so production artifacts are not modified
        builder = FeatureTableBuilder(cutoff_day=35)
        df, _report = builder.run_pipeline(
            cohort_path=PROCESSED_COHORT_DIR / "cohort_policy_c.parquet",
            interim_dir=INTERIM_NORMALIZED_DIR,
            output_dir=tmp_path,
            policy_version="Policy C",
        )
    else:
        pytest.skip(
            "Required inputs unavailable: neither feature_matrix_policy_c.parquet nor "
            "cohort_policy_c.parquet with normalized interim parquet tables exist"
        )

    # 1. Exactly 27,128 rows for Primary Policy C cohort
    assert len(df) == 27128, f"Expected 27,128 rows, got {len(df)}"

    # 2. Canonical grain uniqueness
    assert df.duplicated(subset=CANONICAL_GRAIN).sum() == 0, "Duplicate grain keys found"

    # 3. All 22 primary allowlist features present
    for feature in PRIMARY_FEATURE_ALLOWLIST:
        assert feature in df.columns, f"Allowlist feature missing: {feature}"

    # 4. Zero nulls across all 22 primary features under Policy C
    nulls = df[PRIMARY_FEATURE_ALLOWLIST].isna().sum()
    assert nulls.sum() == 0, f"Unexpected nulls found in Policy C features:\n{nulls[nulls > 0]}"

    # 5. Canonical target variable present and verified
    assert CANONICAL_TARGET in df.columns
    base_rate = df[CANONICAL_TARGET].mean()
    assert np.isclose(base_rate, 4718 / 27128, atol=1e-5), f"Base rate mismatch: {base_rate}"

    # 6. Anti-leakage: forbidden columns absent from predictive feature set
    verify_no_leakage_columns(PRIMARY_FEATURE_ALLOWLIST)

    # 7. Assessment counter invariants hold across all 27,128 records
    assert (
        df["submissions_count_lte_cutoff"]
        == df["submitted_on_time_count"] + df["late_submissions_count_lte_cutoff"]
    ).all(), "Assessment partition invariant violated"

    assert (
        df["submissions_count_lte_cutoff"] <= df["assessments_due_count"]
    ).all(), "Submissions exceed due assessments"

    assert (
        df["missing_due_submissions_count"] <= df["assessments_due_count"]
    ).all(), "Missing submissions exceed due assessments"

    # 8. VLE inactivity invariants hold across all records
    zero_clicks = df["vle_total_clicks_window"] == 0
    assert (
        df["has_no_vle_activity_by_cutoff"] == zero_clicks.astype(int)
    ).all(), "has_no_vle_activity_by_cutoff mismatch"

    assert (
        df.loc[zero_clicks, "days_since_last_vle_interaction"] == 65
    ).all(), "Inactive students lacking sentinel 65"

    # 9. Full table validation report passes
    val_report = validate_feature_table(df, expected_row_count=27128, policy_version="Policy C")
    assert val_report["is_valid"] is True
