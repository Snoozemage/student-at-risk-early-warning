"""Probability calibration utilities for Day-35 student withdrawal risk models.

Implements leakage-free probability calibration fitted strictly on training data
using GroupKFold cross-validation to respect student group isolation (id_student).
Never consumes held-out test labels.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from student_retention.features.feature_definitions import (
    CANONICAL_TARGET,
    PRIMARY_FEATURE_ALLOWLIST,
)
from student_retention.logging_config import get_logger
from student_retention.paths import MODELS_BASELINE_DIR

logger = get_logger(__name__)


def evaluate_calibration_methods_on_training_data(
    base_pipeline: Pipeline,
    train_df: pd.DataFrame,
    methods: tuple[str, ...] = ("sigmoid", "isotonic"),
    n_splits: int = 5,
    group_col: str = "id_student",
    target_col: str = CANONICAL_TARGET,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Evaluate candidate calibration methods strictly on training data using out-of-fold GroupKFold CV.

    Guarantees zero leakage:
    1. Only records from `train_df` are evaluated. Held-out test labels are never accessed or consumed.
    2. Candidate methods ('sigmoid', 'isotonic') are scored using out-of-fold validation on internal folds.
    3. The calibration method is selected based entirely on training-only empirical evidence prior to
       evaluating the held-out test cohort.

    Args:
        base_pipeline: Template scikit-learn Pipeline exposing predict_proba.
        train_df: Training partition DataFrame.
        methods: Candidate calibration methods to compare.
        n_splits: Number of cross-validation folds. Defaults to 5.
        group_col: Student identifier column for group isolation. Defaults to 'id_student'.
        target_col: Binary target column name.
        feature_cols: Feature column names. Defaults to PRIMARY_FEATURE_ALLOWLIST.

    Returns:
        tuple[pd.DataFrame, str]: Tabular out-of-fold calibration diagnostics and the selected method name.
    """
    if feature_cols is None:
        feature_cols = PRIMARY_FEATURE_ALLOWLIST

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy().astype(int)
    groups = train_df[group_col].to_numpy()

    logger.info(
        "Evaluating candidate calibration methods strictly on training data (N=%d, %d folds)...",
        len(train_df),
        n_splits,
    )

    gkf = GroupKFold(n_splits=n_splits)
    custom_splits = list(gkf.split(X_train, y_train, groups=groups))

    oof_predictions: dict[str, np.ndarray] = {
        "raw": np.zeros(len(train_df)),
    }
    for m in methods:
        oof_predictions[m] = np.zeros(len(train_df))

    for fold_idx, (trn_idx, val_idx) in enumerate(custom_splits):
        logger.debug("Running calibration training evaluation fold %d/%d...", fold_idx + 1, n_splits)
        # 1. Fit base pipeline on fold training data
        fold_base = clone(base_pipeline)
        fold_base.fit(X_train.iloc[trn_idx], y_train[trn_idx])
        oof_predictions["raw"][val_idx] = fold_base.predict_proba(X_train.iloc[val_idx])[:, 1]

        # 2. Nested sub-splits for fitting candidate calibrators on trn_idx
        sub_gkf = GroupKFold(n_splits=3)
        sub_groups = groups[trn_idx]
        sub_splits = list(sub_gkf.split(X_train.iloc[trn_idx], y_train[trn_idx], groups=sub_groups))

        for m in methods:
            cal_clf = CalibratedClassifierCV(
                estimator=clone(base_pipeline),
                method=m,
                cv=sub_splits,
            )
            cal_clf.fit(X_train.iloc[trn_idx], y_train[trn_idx])
            oof_predictions[m][val_idx] = cal_clf.predict_proba(X_train.iloc[val_idx])[:, 1]

    # Compute diagnostics for each candidate on training out-of-fold predictions
    records = []
    candidates = [("raw", "Random Forest (Raw Probabilities)", "none")] + [
        (m, f"Random Forest ({m.capitalize()} Calibrated)", m) for m in methods
    ]

    for key, label, method_tag in candidates:
        probs = oof_predictions[key]
        diag = compute_calibration_diagnostics(y_train, probs)
        records.append(
            {
                "model_configuration": label,
                "calibration_method": method_tag,
                "evaluation_split": "training_out_of_fold_cv",
                "brier_score": diag["brier_score"],
                "expected_calibration_error": diag["expected_calibration_error"],
                "maximum_calibration_error": diag["maximum_calibration_error"],
                "min_probability": diag["min_probability"],
                "max_probability": diag["max_probability"],
                "mean_probability": diag["mean_probability"],
            }
        )

    results_df = pd.DataFrame(records)

    # Method selection rule: lowest ECE among calibrated candidates that uncompress probabilities
    cal_candidates = results_df[results_df["calibration_method"].isin(methods)]
    best_row = cal_candidates.sort_values(by=["expected_calibration_error", "brier_score"]).iloc[0]
    selected_method = str(best_row["calibration_method"])

    logger.info(
        "Training calibration method selection complete: selected '%s' (Training OOF ECE=%.5f, Brier=%.5f, Max P=%.4f).",
        selected_method,
        best_row["expected_calibration_error"],
        best_row["brier_score"],
        best_row["max_probability"],
    )
    return results_df, selected_method


def fit_calibrated_model(
    base_pipeline: Pipeline,
    train_df: pd.DataFrame,
    method: str = "isotonic",
    n_splits: int = 5,
    group_col: str = "id_student",
    target_col: str = CANONICAL_TARGET,
    feature_cols: list[str] | None = None,
) -> CalibratedClassifierCV:
    """Fit a calibrated classifier strictly on training data using group-aware cross-validation.

    Anti-Leakage Safeguards:
    1. Only records from `train_df` are used. Held-out test labels are never accessed.
    2. Folds are constructed using `GroupKFold` on `group_col` (`id_student`), ensuring
       that records for the same student never cross validation folds during calibration.
    3. Preprocessing inside `base_pipeline` is fit per-fold by `CalibratedClassifierCV`.

    Args:
        base_pipeline: Unfitted or template scikit-learn Pipeline exposing predict_proba.
        train_df: Training partition DataFrame.
        method: Calibration method: 'isotonic' (non-parametric) or 'sigmoid' (Platt scaling).
        n_splits: Number of cross-validation folds. Defaults to 5.
        group_col: Student identifier column for group isolation. Defaults to 'id_student'.
        target_col: Binary target column name.
        feature_cols: List of feature names to use. Defaults to PRIMARY_FEATURE_ALLOWLIST.

    Returns:
        CalibratedClassifierCV: Fitted calibrated model ensemble.
    """
    if feature_cols is None:
        feature_cols = PRIMARY_FEATURE_ALLOWLIST

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy().astype(int)
    groups = train_df[group_col].to_numpy()

    logger.info(
        "Fitting %s probability calibration using %d-fold GroupKFold on training data (N=%d, %d unique students)...",
        method,
        n_splits,
        len(train_df),
        len(np.unique(groups)),
    )

    gkf = GroupKFold(n_splits=n_splits)
    custom_splits = list(gkf.split(X_train, y_train, groups=groups))

    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method=method,
        cv=custom_splits,
    )
    calibrated_model.fit(X_train, y_train)

    logger.info(
        "Successfully fit %s calibrated model with %d calibrated base estimators.",
        method,
        len(calibrated_model.calibrated_classifiers_),
    )
    return calibrated_model


def compute_calibration_diagnostics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Compute calibration curve, Brier score, and expected calibration error (ECE).

    Args:
        y_true: Ground truth binary targets (0/1).
        y_prob: Predicted risk probabilities in [0.0, 1.0].
        n_bins: Number of discretization bins for reliability curve. Defaults to 10.
        strategy: Discretization strategy ('uniform' or 'quantile'). Defaults to 'uniform'.

    Returns:
        dict[str, Any]: Calibration diagnostic metrics and curve points.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    prob_true, prob_pred = calibration_curve(
        y_true_arr,
        y_prob_arr,
        n_bins=n_bins,
        strategy=strategy,
    )

    # Compute Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob_arr, bin_edges[1:-1])

    ece = 0.0
    max_ce = 0.0
    bin_details = []

    for b in range(n_bins):
        mask = bin_assignments == b
        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true_arr[mask]))
            bin_conf = float(np.mean(y_prob_arr[mask]))
            diff = abs(bin_acc - bin_conf)
            ece += (bin_size / len(y_prob_arr)) * diff
            max_ce = max(max_ce, diff)
            bin_details.append(
                {
                    "bin_index": b,
                    "bin_size": bin_size,
                    "mean_predicted_prob": round(bin_conf, 4),
                    "observed_positive_rate": round(bin_acc, 4),
                    "calibration_gap": round(diff, 4),
                }
            )

    return {
        "brier_score": round(brier, 5),
        "expected_calibration_error": round(float(ece), 5),
        "maximum_calibration_error": round(float(max_ce), 5),
        "min_probability": round(float(np.min(y_prob_arr)), 4),
        "max_probability": round(float(np.max(y_prob_arr)), 4),
        "mean_probability": round(float(np.mean(y_prob_arr)), 4),
        "curve_prob_true": [round(float(v), 4) for v in prob_true],
        "curve_prob_pred": [round(float(v), 4) for v in prob_pred],
        "bin_details": bin_details,
    }


def save_calibrated_model(
    model: CalibratedClassifierCV,
    output_path: Path | str | None = None,
) -> Path:
    """Serialize fitted calibrated model artifact to disk.

    Args:
        model: Fitted CalibratedClassifierCV model.
        output_path: Destination path. Defaults to models/baseline/random_forest_calibrated.joblib.

    Returns:
        Path: Path to serialized artifact.
    """
    path = Path(output_path) if output_path else (MODELS_BASELINE_DIR / "random_forest_calibrated.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved calibrated model artifact to %s", path)
    return path


def load_calibrated_model(
    model_path: Path | str | None = None,
) -> CalibratedClassifierCV:
    """Load serialized calibrated model artifact from disk.

    Args:
        model_path: Path to serialized artifact. Defaults to models/baseline/random_forest_calibrated.joblib.

    Returns:
        CalibratedClassifierCV: Loaded model artifact.

    Raises:
        FileNotFoundError: If artifact does not exist.
    """
    path = Path(model_path) if model_path else (MODELS_BASELINE_DIR / "random_forest_calibrated.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Calibrated model artifact not found at {path}")
    model = joblib.load(path)
    logger.info("Loaded calibrated model artifact from %s", path)
    return model
