"""Unit tests for ML pipelines, threshold optimization, and model interpretation."""
import numpy as np
import pandas as pd
import pytest

from student_retention.features.feature_definitions import (
    CANONICAL_TARGET,
    PRIMARY_FEATURE_ALLOWLIST,
)
from student_retention.modeling.interpretation import (
    extract_logistic_regression_coefficients,
    extract_random_forest_importance,
)
from student_retention.modeling.pipelines import (
    build_model_pipeline,
)
from student_retention.modeling.predict import (
    load_operational_threshold,
    predict_withdrawal_risk,
)
from student_retention.modeling.threshold import find_optimal_threshold


@pytest.fixture
def synthetic_features_df() -> pd.DataFrame:
    """Create a synthetic dataset strictly conforming to the 22-feature allowlist."""
    np.random.seed(42)
    n = 200
    data = {
        "code_module": np.random.choice(["AAA", "BBB", "CCC"], size=n),
        "code_presentation": np.random.choice(["2013B", "2013J"], size=n),
        "id_student": np.arange(1001, 1001 + n),
        "presentation_season": np.random.choice(["B", "J"], size=n),
        "studied_credits": np.random.choice([60, 120, 180], size=n),
        "num_of_prev_attempts": np.random.choice([0, 1, 2], size=n),
        # Include realistic missingness in registration lead time
        "registration_lead_time_days": [
            np.nan if i % 10 == 0 else float(np.random.randint(-5, 60)) for i in range(n)
        ],
        "is_late_registration": np.random.choice([0.0, 1.0], size=n),
        "vle_total_clicks_window": np.random.randint(0, 1000, size=n),
        "vle_active_days_window": np.random.randint(0, 35, size=n),
        "vle_unique_sites_accessed": np.random.randint(0, 50, size=n),
        "vle_pre_course_clicks": np.random.randint(0, 200, size=n),
        "vle_recent_clicks_14d": np.random.randint(0, 400, size=n),
        "vle_velocity_ratio": np.random.uniform(0.0, 3.0, size=n),
        "days_since_last_vle_interaction": np.random.choice([1, 5, 12, 65], size=n),
        "has_no_vle_activity_by_cutoff": np.random.choice([0, 1], size=n),
        "vle_clicks_forumng": np.random.randint(0, 200, size=n),
        "vle_clicks_oucontent": np.random.randint(0, 500, size=n),
        "vle_clicks_quiz": np.random.randint(0, 100, size=n),
        "assessments_due_count": np.random.choice([0, 1, 2], size=n),
        "submissions_count_lte_cutoff": np.random.choice([0, 1, 2], size=n),
        "submitted_on_time_count": np.random.choice([0, 1, 2], size=n),
        "missing_due_submissions_count": np.random.choice([0, 1], size=n),
        "late_submissions_count_lte_cutoff": np.random.choice([0, 1], size=n),
        CANONICAL_TARGET: np.random.binomial(1, 0.20, size=n),
    }
    return pd.DataFrame(data)


def test_build_model_pipeline_and_fit(synthetic_features_df):
    """Test building, fitting, and predicting with scikit-learn pipelines."""
    X = synthetic_features_df[PRIMARY_FEATURE_ALLOWLIST]
    y = synthetic_features_df[CANONICAL_TARGET]

    for model_name in ["dummy", "logistic_regression", "random_forest"]:
        pipe = build_model_pipeline(model_name=model_name, random_state=42)
        pipe.fit(X, y)
        probs = pipe.predict_proba(X)

        assert probs.shape == (len(X), 2)
        # Probabilities must sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
        # Range must be [0, 1]
        assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_find_optimal_threshold_meets_constraint():
    """Test that optimal threshold selection satisfies precision >= 0.35 constraint."""
    np.random.seed(42)
    # 500 samples, 100 positives
    y_true = np.zeros(500, dtype=int)
    y_true[:100] = 1
    # Well-calibrated probabilities
    y_prob = np.concatenate([
        np.random.beta(5, 2, size=100),   # True positives get higher scores
        np.random.beta(1, 5, size=400),   # Negatives get lower scores
    ])

    result = find_optimal_threshold(y_true, y_prob, min_precision=0.35)

    assert result["constraint_met"] is True
    assert result["precision"] >= 0.35
    assert 0.01 <= result["threshold"] <= 0.99
    assert result["f1"] > 0.0


def test_find_optimal_threshold_fallback_on_uninformative():
    """Test graceful fallback when precision constraint cannot be met."""
    y_true = np.zeros(100, dtype=int)
    y_true[:5] = 1  # 5% base rate
    y_prob = np.random.uniform(0.01, 0.02, size=100)  # Uninformative scores

    # Constraint requires precision >= 0.90 which cannot be met
    result = find_optimal_threshold(y_true, y_prob, min_precision=0.90)

    assert result["constraint_met"] is False
    assert 0.01 <= result["threshold"] <= 0.99


def test_feature_interpretation_extraction(synthetic_features_df):
    """Test extraction of odds ratios and Random Forest importances."""
    X = synthetic_features_df[PRIMARY_FEATURE_ALLOWLIST]
    y = synthetic_features_df[CANONICAL_TARGET]

    # Logistic Regression
    lr_pipe = build_model_pipeline("logistic_regression", random_state=42)
    lr_pipe.fit(X, y)
    lr_coefs = extract_logistic_regression_coefficients(lr_pipe)
    assert not lr_coefs.empty
    assert "odds_ratio" in lr_coefs.columns
    assert "coefficient_log_odds" in lr_coefs.columns
    assert np.all(lr_coefs["odds_ratio"] > 0)

    # Random Forest
    rf_pipe = build_model_pipeline("random_forest", random_state=42)
    rf_pipe.fit(X, y)
    rf_imp = extract_random_forest_importance(rf_pipe)
    assert not rf_imp.empty
    assert "impurity_importance" in rf_imp.columns
    assert np.isclose(rf_imp["impurity_importance"].sum(), 1.0, rtol=1e-3)


def test_predict_withdrawal_risk_with_explicit_threshold(synthetic_features_df):
    """Test batch prediction scoring helper with explicit operational threshold."""
    X = synthetic_features_df[PRIMARY_FEATURE_ALLOWLIST]
    y = synthetic_features_df[CANONICAL_TARGET]

    pipe = build_model_pipeline("random_forest", random_state=42)
    pipe.fit(X, y)

    scored_df = predict_withdrawal_risk(pipe, synthetic_features_df, threshold=0.25)

    assert "risk_probability" in scored_df.columns
    assert "predicted_risk_flag" in scored_df.columns
    assert "operational_threshold" in scored_df.columns
    assert "risk_percentile" in scored_df.columns
    assert len(scored_df) == len(synthetic_features_df)
    assert set(scored_df["predicted_risk_flag"].unique()).issubset({0, 1})
    assert np.all(scored_df["operational_threshold"] == 0.25)


def test_predict_withdrawal_risk_requires_threshold():
    """Test that calling predict_withdrawal_risk without threshold or model_key raises ValueError.

    Downstream callers must not silently default to an arbitrary 0.50 threshold.
    """
    pipe = build_model_pipeline("random_forest", random_state=42)
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Operational decision threshold must be explicitly provided"):
        predict_withdrawal_risk(pipe, df)


def test_predict_withdrawal_risk_with_metadata(tmp_path, synthetic_features_df):
    """Test resolving operational threshold from saved training summary metadata."""
    import json

    metadata_path = tmp_path / "model_training_summary.json"
    metadata_path.write_text(
        json.dumps({"threshold_map": {"random_forest": 0.28, "logistic_regression": 0.32}}),
        encoding="utf-8",
    )

    X = synthetic_features_df[PRIMARY_FEATURE_ALLOWLIST]
    y = synthetic_features_df[CANONICAL_TARGET]
    pipe = build_model_pipeline("random_forest", random_state=42)
    pipe.fit(X, y)

    scored_df = predict_withdrawal_risk(
        pipe,
        synthetic_features_df,
        model_key="random_forest",
        threshold_metadata_path=metadata_path,
    )

    assert "operational_threshold" in scored_df.columns
    assert np.all(scored_df["operational_threshold"] == 0.28)


def test_load_operational_threshold_errors(tmp_path):
    """Test load_operational_threshold error cases for missing file and missing key."""
    import json

    # Missing file
    missing_path = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError, match="Model training summary not found"):
        load_operational_threshold("random_forest", summary_path=missing_path)

    # Missing key in file
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"threshold_map": {"dummy": 0.50}}), encoding="utf-8")
    with pytest.raises(KeyError, match="Model key 'random_forest' not found"):
        load_operational_threshold("random_forest", summary_path=summary_path)
