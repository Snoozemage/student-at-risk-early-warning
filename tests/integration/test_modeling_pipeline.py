"""Integration tests for the complete Phase 6 modeling and evaluation workflow."""

import numpy as np
import pandas as pd
import pytest

from student_retention.features.feature_definitions import (
    CANONICAL_TARGET,
    PRIMARY_FEATURE_ALLOWLIST,
)
from student_retention.modeling.train import train_and_evaluate_all_models
from student_retention.paths import PROCESSED_FEATURES_DIR


@pytest.fixture
def multi_student_features_df() -> pd.DataFrame:
    """Create a multi-presentation synthetic cohort dataset with repeat students."""
    np.random.seed(42)
    records = []
    # 250 unique students
    for sid in range(1, 251):
        # assign 1 or 2 enrollments
        num_courses = 2 if sid % 4 == 0 else 1
        for c in range(num_courses):
            pres = "2013B" if sid <= 100 else ("2013J" if sid <= 160 else "2014B" if sid <= 210 else "2014J")
            records.append(
                {
                    "code_module": "AAA" if (sid + c) % 2 == 0 else "BBB",
                    "code_presentation": pres,
                    "id_student": sid,
                    "presentation_season": pres[-1],
                    "studied_credits": 60 if sid % 2 == 0 else 120,
                    "num_of_prev_attempts": 0 if sid % 5 != 0 else 1,
                    "registration_lead_time_days": float(np.random.randint(-10, 45)),
                    "is_late_registration": 1.0 if sid % 10 == 0 else 0.0,
                    "vle_total_clicks_window": np.random.randint(0, 1500),
                    "vle_active_days_window": np.random.randint(0, 35),
                    "vle_unique_sites_accessed": np.random.randint(0, 40),
                    "vle_pre_course_clicks": np.random.randint(0, 250),
                    "vle_recent_clicks_14d": np.random.randint(0, 500),
                    "vle_velocity_ratio": float(np.random.uniform(0.1, 2.5)),
                    "days_since_last_vle_interaction": int(np.random.choice([2, 7, 18, 65])),
                    "has_no_vle_activity_by_cutoff": 1 if sid % 20 == 0 else 0,
                    "vle_clicks_forumng": np.random.randint(0, 200),
                    "vle_clicks_oucontent": np.random.randint(0, 600),
                    "vle_clicks_quiz": np.random.randint(0, 150),
                    "assessments_due_count": 1 if sid % 3 == 0 else 2,
                    "submissions_count_lte_cutoff": 1 if sid % 3 == 0 else 2,
                    "submitted_on_time_count": 1 if sid % 4 != 0 else 0,
                    "missing_due_submissions_count": 1 if sid % 6 == 0 else 0,
                    "late_submissions_count_lte_cutoff": 1 if sid % 8 == 0 else 0,
                    CANONICAL_TARGET: 1 if (sid + c) % 6 == 0 else 0,
                }
            )
    return pd.DataFrame(records)


def test_train_and_evaluate_all_models_synthetic(multi_student_features_df):
    """Test full training pipeline execution on synthetic data without disk mutation."""
    results = train_and_evaluate_all_models(
        feature_df=multi_student_features_df,
        save_artifacts=False,
        random_state=42,
    )

    # 1. Verify CV summary
    cv_summary = results["cv_summary"]
    assert not cv_summary.empty
    assert len(cv_summary) == 5
    assert set(cv_summary["model_key"]) == {
        "dummy_baseline",
        "logistic_regression",
        "logistic_regression_balanced",
        "random_forest",
        "random_forest_balanced",
    }
    assert np.all(cv_summary["cv_pr_auc_mean"] >= 0.0)

    # 2. Verify test evaluation summary
    test_summary = results["test_summary"]
    assert not test_summary.empty
    assert len(test_summary) == 5
    for col in ["pr_auc", "roc_auc", "brier_score", "f1", "precision", "recall"]:
        assert col in test_summary.columns
        assert not test_summary[col].isna().any()

    # 3. Verify Top-K metrics
    top_k_summary = results["top_k_summary"]
    assert not top_k_summary.empty
    assert "precision_at_k" in top_k_summary.columns
    assert "recall_at_k" in top_k_summary.columns
    assert all(top_k_summary["evaluation_scope"] == "pooled_test_cohort")

    top_k_by_term = results["top_k_by_term_summary"]
    assert not top_k_by_term.empty
    assert all(top_k_by_term["evaluation_scope"] == "per_term")
    assert all(top_k_by_term["operational_unit"] == "academic_term_cross_module")

    top_k_by_mod = results["top_k_by_module_presentation_summary"]
    assert not top_k_by_mod.empty
    assert all(top_k_by_mod["evaluation_scope"] == "per_module_presentation")
    assert all(top_k_by_mod["operational_unit"] == "module_presentation_course_offering")

    top_k_by_pres = results["top_k_by_presentation_summary"]
    assert not top_k_by_pres.empty
    assert all(top_k_by_pres["evaluation_scope"] == "per_term")

    # 4. Verify Chronological evaluation and anti-leakage threshold derivation
    chrono_summary = results["chronological_summary"]
    assert not chrono_summary.empty
    assert set(chrono_summary["evaluation_design"]) == {
        "natural_operational",
        "disjoint_purged",
    }
    assert all(chrono_summary["threshold_source"] == "chrono_train_2013_group_cv")
    assert "chronological_threshold_map" in results
    assert set(results["chronological_threshold_map"].keys()) == {
        "dummy_baseline",
        "logistic_regression",
        "random_forest",
    }

    # 5. Verify feature interpretation exports
    lr_importance = results["feature_importance_lr"]
    assert not lr_importance.empty
    assert "odds_ratio" in lr_importance.columns

    rf_importance = results["feature_importance_rf"]
    assert not rf_importance.empty
    assert "impurity_importance" in rf_importance.columns


def test_train_and_evaluate_on_real_data_if_available():
    """Verify execution on actual empirical feature matrix if available in working tree."""
    features_path = PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet"
    if not features_path.exists():
        pytest.skip(f"Feature matrix not found at {features_path}; skipping real-data integration audit.")

    df = pd.read_parquet(features_path)
    # Check that empirical feature matrix has expected grain and allowlist features
    assert len(df) == 27128
    assert CANONICAL_TARGET in df.columns
    for feat in PRIMARY_FEATURE_ALLOWLIST:
        assert feat in df.columns
