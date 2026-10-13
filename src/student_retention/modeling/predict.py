"""Inference and scoring utilities for Day-35 student withdrawal risk."""
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from student_retention.features.feature_definitions import (
    CANONICAL_GRAIN,
    PRIMARY_FEATURE_ALLOWLIST,
)
from student_retention.logging_config import get_logger
from student_retention.paths import TABLES_DIR

logger = get_logger(__name__)


def load_model(model_path: str | Path) -> Pipeline:
    """Load a serialized scikit-learn model pipeline from disk.

    Args:
        model_path: Path to the .joblib artifact.

    Returns:
        Pipeline: Loaded pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    model = joblib.load(path)
    logger.info("Loaded model artifact from %s", path)
    return model


def load_operational_threshold(
    model_key: str = "random_forest",
    summary_path: str | Path | None = None,
) -> float:
    """Load the validation-optimized operational decision threshold from saved summary metadata.

    Args:
        model_key: Key in threshold_map (e.g. 'random_forest', 'logistic_regression').
        summary_path: Path to model_training_summary.json (default: outputs/tables/model_training_summary.json).

    Returns:
        float: Operational threshold.

    Raises:
        FileNotFoundError: If the summary file does not exist.
        KeyError: If model_key is not in threshold_map.
    """
    path = Path(summary_path) if summary_path else (TABLES_DIR / "model_training_summary.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Model training summary not found at {path}. "
            "Please provide an explicit threshold or run model training first."
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    threshold_map = data.get("threshold_map", {})
    if model_key not in threshold_map:
        raise KeyError(
            f"Model key '{model_key}' not found in saved threshold_map: {list(threshold_map.keys())}"
        )

    thresh = float(threshold_map[model_key])
    logger.info("Loaded operational threshold %.4f for '%s' from %s", thresh, model_key, path)
    return thresh


def predict_withdrawal_risk(
    model: Pipeline,
    df: pd.DataFrame,
    threshold: float | None = None,
    model_key: str | None = None,
    threshold_metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate risk probabilities and binary flags for incoming learner observations.

    Notice: An explicit operational decision threshold (e.g. threshold=0.25) or a resolvable
    model_key from training metadata must be provided. Generic static thresholds such as 0.50
    are non-operational for imbalanced student withdrawal risk (~17.4% base rate) and will
    cause high false-negative omission rates.

    Args:
        model: Fitted model pipeline.
        df: Input DataFrame containing primary features (and optionally grain identifiers).
        threshold: Explicit decision threshold for discrete alert assignment.
            If None, attempts to resolve from saved metadata via `model_key`.
        model_key: Model key to look up in threshold metadata if threshold is None.
        threshold_metadata_path: Path to model_training_summary.json.

    Returns:
        pd.DataFrame: DataFrame containing grain identifiers (if present),
            'risk_probability', 'predicted_risk_flag', 'operational_threshold', and 'risk_percentile'.

    Raises:
        ValueError: If neither an explicit threshold nor a resolvable model_key metadata is provided.
    """
    if threshold is None:
        if model_key is not None or (threshold_metadata_path is not None):
            threshold = load_operational_threshold(
                model_key=model_key or "random_forest",
                summary_path=threshold_metadata_path,
            )
        else:
            raise ValueError(
                "Operational decision threshold must be explicitly provided (e.g. threshold=0.25) "
                "or resolved via model_key (e.g. model_key='random_forest'). A generic 0.50 default "
                "is non-operational and causes severe false-negative triage omission."
            )

    # Ensure all primary features are present
    missing = [f for f in PRIMARY_FEATURE_ALLOWLIST if f not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required features: {missing}")

    X = df[PRIMARY_FEATURE_ALLOWLIST].copy()

    # Probability estimation
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        # Fallback if model only supports discrete decision
        preds = model.predict(X)
        probs = preds.astype(float)

    preds = (probs >= threshold).astype(int)

    # Calculate empirical risk percentiles
    percentiles = pd.Series(probs).rank(pct=True).values

    out_cols: dict[str, Any] = {}
    for col in CANONICAL_GRAIN:
        if col in df.columns:
            out_cols[col] = df[col].values

    out_cols["risk_probability"] = probs
    out_cols["predicted_risk_flag"] = preds
    out_cols["operational_threshold"] = float(threshold)
    out_cols["risk_percentile"] = percentiles

    result_df = pd.DataFrame(out_cols)
    return result_df
