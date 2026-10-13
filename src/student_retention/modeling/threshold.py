"""Operational decision threshold optimization and scan utilities."""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from student_retention.logging_config import get_logger

logger = get_logger(__name__)


def scan_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate classification metrics across a range of decision thresholds.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted positive-class probabilities.
        thresholds: Array of candidate thresholds. Defaults to 99 points from 0.01 to 0.99.

    Returns:
        pd.DataFrame: Metric curves indexed by candidate threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    records = []
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        flagged = tp + fp

        records.append(
            {
                "threshold": float(thresh),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "flagged_count": flagged,
                "flagged_rate": float(flagged / len(y_true)) if len(y_true) > 0 else 0.0,
            }
        )

    return pd.DataFrame(records)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.35,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Find the operational classification threshold maximizing F1 subject to precision constraint.

    Per config/evaluation.yaml:
        policy: optimize_f1_on_validation
        constraint: precision >= 0.35

    Must strictly be executed on cross-validation or out-of-fold validation sets,
    never on the test partition.

    Args:
        y_true: Validation ground truth binary labels.
        y_prob: Predicted probabilities on validation data.
        min_precision: Minimum acceptable precision constraint (default: 0.35).
        thresholds: Candidate decision thresholds.

    Returns:
        dict[str, Any]: Dictionary detailing selected threshold and validation metrics.
    """
    scan_df = scan_thresholds(y_true, y_prob, thresholds=thresholds)

    # Filter to candidates meeting the operational precision constraint
    valid_candidates = scan_df[scan_df["precision"] >= min_precision]

    if not valid_candidates.empty:
        # Choose candidate maximizing F1-score; if tie, choose higher precision
        best_row = valid_candidates.sort_values(
            by=["f1", "precision", "recall"], ascending=[False, False, False]
        ).iloc[0]
        constraint_met = True
    else:
        # Graceful fallback: unconstrained F1 maximization if precision >= 0.35 cannot be attained
        best_row = scan_df.sort_values(
            by=["f1", "precision"], ascending=[False, False]
        ).iloc[0]
        constraint_met = False
        logger.warning(
            "Precision constraint (>= %.2f) could not be met across candidate thresholds. "
            "Falling back to unconstrained max-F1 threshold: %.4f (precision=%.4f, f1=%.4f).",
            min_precision,
            best_row["threshold"],
            best_row["precision"],
            best_row["f1"],
        )

    return {
        "threshold": float(best_row["threshold"]),
        "f1": float(best_row["f1"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "constraint_met": constraint_met,
        "min_precision_target": float(min_precision),
        "policy": "optimize_f1_on_validation",
        "flagged_count": int(best_row["flagged_count"]),
        "flagged_rate": float(best_row["flagged_rate"]),
    }
