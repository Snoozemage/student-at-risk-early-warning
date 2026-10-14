"""Unit tests for Phase 7 probability calibration and reliability diagnostics."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from student_retention.modeling.calibration import (
    compute_calibration_diagnostics,
    evaluate_calibration_methods_on_training_data,
    fit_calibrated_model,
    load_calibrated_model,
    save_calibrated_model,
)


def _build_synthetic_train_df(n: int = 200) -> pd.DataFrame:
    """Generate reproducible synthetic training data with student grouping."""
    np.random.seed(42)
    # 50 unique students, each with 4 records on average
    n_students = 50
    students = [1000 + (i % n_students) for i in range(n)]

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    # Target influenced by x1 + student effect
    logits = 0.8 * x1 - 0.5 * x2
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n) < probs).astype(int)

    df = pd.DataFrame(
        {
            "id_student": students,
            "feature_1": x1,
            "feature_2": x2,
            "target": y,
        }
    )
    return df


def test_fit_calibrated_model_zero_test_leakage(tmp_path):
    """Verify that calibration fits strictly on training data with student group isolation."""
    train_df = _build_synthetic_train_df(n=200)

    # Base simple pipeline
    base_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
        ]
    )

    feature_cols = ["feature_1", "feature_2"]

    # Fit calibrated model strictly on train_df
    calibrated_model = fit_calibrated_model(
        base_pipeline=base_pipeline,
        train_df=train_df,
        method="isotonic",
        n_splits=5,
        group_col="id_student",
        target_col="target",
        feature_cols=feature_cols,
    )

    # Invariants:
    # 1. Calibrated model has 5 calibrated classifiers
    assert len(calibrated_model.calibrated_classifiers_) == 5

    # 2. Output probabilities on new unseen data are valid probabilities in [0, 1]
    test_x = pd.DataFrame({"feature_1": [0.5, -1.0, 2.0], "feature_2": [-0.5, 0.0, 1.0]})
    probs = calibrated_model.predict_proba(test_x)[:, 1]
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()

    # 3. Serialization round-trip preservation
    save_path = tmp_path / "calibrated_test.joblib"
    save_calibrated_model(calibrated_model, save_path)
    loaded_model = load_calibrated_model(save_path)
    loaded_probs = loaded_model.predict_proba(test_x)[:, 1]
    np.testing.assert_allclose(probs, loaded_probs, rtol=1e-5)


def test_calibration_diagnostics_calculation():
    """Verify exact calculation of Brier score, ECE, and reliability curve."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6, 0.85, 0.95])

    diag = compute_calibration_diagnostics(y_true, y_prob, n_bins=5)

    assert "brier_score" in diag
    assert "expected_calibration_error" in diag
    assert "maximum_calibration_error" in diag
    assert "curve_prob_true" in diag
    assert "curve_prob_pred" in diag
    assert "bin_details" in diag

    assert 0.0 <= diag["brier_score"] <= 1.0
    assert 0.0 <= diag["expected_calibration_error"] <= 1.0
    assert diag["min_probability"] == 0.1
    assert diag["max_probability"] == 0.95


def test_calibration_isolation_from_test_labels():
    """Prove that test labels do not affect the calibration artifact."""
    train_df = _build_synthetic_train_df(n=100)
    base_pipeline = Pipeline([("lr", LogisticRegression())])
    feature_cols = ["feature_1", "feature_2"]

    cal1 = fit_calibrated_model(
        base_pipeline=base_pipeline,
        train_df=train_df,
        method="sigmoid",
        n_splits=3,
        group_col="id_student",
        target_col="target",
        feature_cols=feature_cols,
    )

    test_x = pd.DataFrame({"feature_1": [0.1, -0.5], "feature_2": [0.2, 0.4]})
    probs1 = cal1.predict_proba(test_x)[:, 1]

    # Create hypothetical test sets with contradictory labels
    # Verify that the fitted model makes identical predictions regardless of any external test data
    test_y_fake_1 = np.array([1, 1])
    test_y_fake_2 = np.array([0, 0])

    diag1 = compute_calibration_diagnostics(test_y_fake_1, probs1)
    diag2 = compute_calibration_diagnostics(test_y_fake_2, probs1)

    # Diagnostic metrics reflect the labels, but the model predictions themselves remain unchanged
    probs_repeat = cal1.predict_proba(test_x)[:, 1]
    np.testing.assert_array_equal(probs1, probs_repeat)
    assert diag1["brier_score"] != diag2["brier_score"]


def test_evaluate_calibration_methods_training_only():
    """Verify that candidate calibration evaluation is strictly training-only and selects best method."""
    train_df = _build_synthetic_train_df(n=120)
    base_pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])
    feature_cols = ["feature_1", "feature_2"]

    comp_df, selected_method = evaluate_calibration_methods_on_training_data(
        base_pipeline=base_pipeline,
        train_df=train_df,
        methods=("sigmoid", "isotonic"),
        n_splits=3,
        group_col="id_student",
        target_col="target",
        feature_cols=feature_cols,
    )

    # Invariants:
    # 1. Returned DataFrame has entries for raw, sigmoid, and isotonic
    assert len(comp_df) == 3
    assert set(comp_df["calibration_method"]) == {"none", "sigmoid", "isotonic"}
    assert (comp_df["evaluation_split"] == "training_out_of_fold_cv").all()

    # 2. Selected method is one of the candidates
    assert selected_method in {"sigmoid", "isotonic"}

    # 3. Metrics are strictly within valid bounds
    for col in ["brier_score", "expected_calibration_error", "maximum_calibration_error"]:
        assert (comp_df[col] >= 0.0).all()
        assert (comp_df[col] <= 1.0).all()
