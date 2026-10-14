"""Integration tests for Phase 7 fairness audit and decision support pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_retention.decision_support.prioritize_students import (
    QUEUE_CANONICAL_COLUMNS,
    audit_support_queue_capacity,
    build_student_support_queue,
)
from student_retention.fairness.subgroup_audit import (
    audit_all_protected_attributes,
    audit_operational_support_queue,
)
from student_retention.features.feature_selection import select_primary_features
from student_retention.modeling.calibration import load_calibrated_model
from student_retention.modeling.predict import load_model, load_operational_threshold
from student_retention.modeling.splits import split_by_student_group
from student_retention.paths import MODELS_BASELINE_DIR, PROCESSED_FEATURES_DIR


@pytest.fixture(scope="module")
def empirical_feature_sample() -> pd.DataFrame:
    """Load empirical feature matrix if available, else skip."""
    features_path = PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet"
    if not features_path.exists():
        pytest.skip(f"Empirical feature matrix not found at {features_path}")
    df = pd.read_parquet(features_path)
    return df


@pytest.fixture(scope="module")
def empirical_rf_model():
    """Load trained Random Forest baseline model if available, else skip."""
    rf_path = MODELS_BASELINE_DIR / "random_forest.joblib"
    if not rf_path.exists():
        pytest.skip(f"Trained Random Forest model not found at {rf_path}")
    return load_model(rf_path)


@pytest.fixture(scope="module")
def empirical_calibrated_rf_model():
    """Load calibrated Random Forest model if available, else skip."""
    cal_path = MODELS_BASELINE_DIR / "random_forest_calibrated.joblib"
    if not cal_path.exists():
        pytest.skip(f"Calibrated Random Forest model not found at {cal_path}")
    return load_calibrated_model(cal_path)


def test_empirical_fairness_audit_pipeline(empirical_feature_sample: pd.DataFrame, empirical_rf_model):
    """Verify that full subgroup audit runs on empirical held-out test data across all 6 demographic attributes."""
    df = empirical_feature_sample
    rf_model = empirical_rf_model

    # Group split on id_student
    _, test_df = split_by_student_group(df, test_size=0.20, random_state=42)
    clean_test_x = select_primary_features(test_df, include_target=False, include_identifiers=False)

    test_probs = rf_model.predict_proba(clean_test_x)[:, 1]
    operational_threshold = load_operational_threshold("random_forest")
    test_preds = (test_probs >= operational_threshold).astype(int)

    test_eval_df = test_df.copy()
    test_eval_df["predicted_label"] = test_preds
    test_eval_df["risk_probability"] = test_probs

    audit_df = audit_all_protected_attributes(
        df=test_eval_df,
        target_col="withdrawn_after_cutoff",
        pred_col="predicted_label",
        prob_col="risk_probability",
    )

    # Invariants:
    # 1. Audited all 6 protected attributes
    expected_attrs = {"gender", "age_band", "disability", "imd_band", "highest_education", "region"}
    assert expected_attrs.issubset(set(audit_df["protected_attribute"]))

    # 2. Total subgroups >= 30
    assert len(audit_df) >= 30

    # 3. Sum of sample sizes for each attribute equals test cohort size (5,414)
    for attr in expected_attrs:
        attr_total = audit_df[audit_df["protected_attribute"] == attr]["sample_size"].sum()
        assert attr_total == len(test_df)

    # 4. Reference group flags exist exactly once per attribute
    ref_counts = audit_df.groupby("protected_attribute")["is_reference_group"].sum()
    assert (ref_counts == 1).all()


def test_empirical_support_queue_pipeline(empirical_feature_sample: pd.DataFrame, empirical_rf_model, tmp_path: Path):
    """Verify that student support queue is generated with correct schema, ordering, and capacity limits."""
    df = empirical_feature_sample
    rf_model = empirical_rf_model

    # Generate strictly on held-out test cohort (anti-leakage design)
    _, test_df = split_by_student_group(df, test_size=0.20, random_state=42)

    dest_csv = tmp_path / "student_support_queue.csv"

    queue_df = build_student_support_queue(
        feature_df=test_df,
        model=rf_model,
        urgent_threshold=0.70,
        watch_threshold=0.40,
        capacity_k=50,
        save_path=dest_csv,
    )

    cap_audit = audit_support_queue_capacity(queue_df)
    assert len(cap_audit) == 5  # 4 presentations + 1 ALL_TERMS_TOTAL

    # Invariants:
    # 1. CSV exists and is non-empty
    assert dest_csv.exists()
    assert dest_csv.stat().st_size > 0

    # 2. Exactly K=50 per presentation * 4 presentations = 200 records
    assert len(queue_df) == 200

    # 3. Canonical columns exact match
    assert list(queue_df.columns) == QUEUE_CANONICAL_COLUMNS

    # 4. Cutoff day is universally 35
    assert (queue_df["prediction_cutoff_day"] == 35).all()

    # 5. Probabilities are sorted descending within each presentation
    for pres in queue_df["code_presentation"].unique():
        pres_probs = queue_df[queue_df["code_presentation"] == pres]["risk_probability"].to_numpy()
        assert (np.diff(pres_probs) <= 0).all()

    # 6. Support reasons are non-empty strings
    assert (queue_df["support_reason_summary"].str.len() > 5).all()


def test_empirical_operational_queue_fairness_audit(
    empirical_feature_sample: pd.DataFrame,
    empirical_calibrated_rf_model,
    empirical_rf_model,
):
    """Verify that operational queue fairness audit evaluates queue inclusion with accurate decision policy metadata."""
    df = empirical_feature_sample
    cal_model = empirical_calibrated_rf_model

    _, test_df = split_by_student_group(df, test_size=0.20, random_state=42)

    # 1. Operational calibrated model with Top-500 capacity
    queue_df_500 = build_student_support_queue(
        feature_df=test_df,
        model=cal_model,
        capacity_k=500,
    )

    queue_audit_500 = audit_operational_support_queue(
        queue_df=queue_df_500,
        cohort_df=test_df,
        decision_policy="top_500_per_presentation_calibrated",
    )

    # Invariants for operational calibrated queue:
    # 1. 36 rows (6 protected attributes)
    assert len(queue_audit_500) == 36
    assert (queue_audit_500["audit_scope"] == "operational_support_queue").all()
    assert (queue_audit_500["decision_policy"] == "top_500_per_presentation_calibrated").all()

    # 2. Total queued records across test cohort matches sum of selection counts (2,000)
    expected_total_500 = len(queue_df_500)
    gender_queued_500 = queue_audit_500[queue_audit_500["protected_attribute"] == "gender"]["selected_count"].sum()
    assert gender_queued_500 == expected_total_500 == 2000

    # 3. Dynamic warnings are calculated without crashing
    assert "fairness_warning_flag" in queue_audit_500.columns
    assert "disparate_impact_ratio" in queue_audit_500.columns

    # 4. Raw model with custom capacity (K=100) must reflect its explicit policy provenance
    raw_rf_model = empirical_rf_model
    queue_df_raw_100 = build_student_support_queue(
        feature_df=test_df,
        model=raw_rf_model,
        capacity_k=100,
    )
    raw_policy_name = "top_100_per_presentation_raw"
    queue_audit_raw_100 = audit_operational_support_queue(
        queue_df=queue_df_raw_100,
        cohort_df=test_df,
        decision_policy=raw_policy_name,
    )
    assert (queue_audit_raw_100["decision_policy"] == raw_policy_name).all()
    assert (
        queue_audit_raw_100[queue_audit_raw_100["protected_attribute"] == "gender"]["selected_count"].sum()
        == len(queue_df_raw_100)
    )
