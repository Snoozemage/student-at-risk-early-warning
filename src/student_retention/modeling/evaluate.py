"""Comprehensive model evaluation metrics, top-K analysis, and error auditing."""
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from student_retention.logging_config import get_logger

logger = get_logger(__name__)


def _calculate_top_k_slice(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_k: int,
    total_positives: int,
    n_samples: int,
) -> tuple[float, float, float, float]:
    """Calculate expected positives, precision, recall, and cumulative gain at rank k with tie handling.

    When predicted probabilities contain ties (including constant-score baselines),
    the expectation is computed analytically under uniform tie-breaking to avoid arbitrary
    ordering artifacts from numerical argsort.

    Returns:
        tuple[float, float, float, float]: (expected_positives, precision_at_k, recall_at_k, cumulative_gain)
    """
    if target_k <= 0 or n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Case 1: Pure constant score predictor (e.g. DummyClassifier predicting prior probability)
    if np.min(y_prob) == np.max(y_prob):
        base_rate = float(total_positives / n_samples)
        expected_pos = float(target_k * base_rate)
        p_at_k = base_rate
        r_at_k = float(target_k / n_samples) if total_positives > 0 else 0.0
        return expected_pos, p_at_k, r_at_k, r_at_k

    # Case 2: Predictor with distinct or partially tied probabilities
    sorted_order = np.argsort(-y_prob)
    threshold_k = y_prob[sorted_order[target_k - 1]]

    mask_above = y_prob > threshold_k
    n_above = int(np.sum(mask_above))
    pos_above = float(np.sum(y_true[mask_above]))

    mask_tied = y_prob == threshold_k
    n_tied = int(np.sum(mask_tied))
    pos_tied = float(np.sum(y_true[mask_tied]))

    if n_tied > 0 and n_above < target_k:
        k_needed = target_k - n_above
        fraction_needed = min(1.0, k_needed / n_tied)
        expected_pos = pos_above + fraction_needed * pos_tied
    else:
        expected_pos = pos_above

    p_at_k = float(expected_pos / target_k)
    r_at_k = float(expected_pos / total_positives) if total_positives > 0 else 0.0
    return expected_pos, p_at_k, r_at_k, r_at_k


def _calculate_top_k_unique_students(
    students: np.ndarray,
    y_prob: np.ndarray,
    target_k: int,
) -> tuple[float, float, bool]:
    """Calculate expected unique students and duplicate slots at rank K with uniform tie-breaking.

    When predicted probabilities contain ties crossing the K-th rank boundary,
    the expectation of distinct students is computed analytically under uniform
    random tie-breaking to avoid arbitrary ordering artifacts from input row permutations.

    Args:
        students: Array of student identifiers aligned with rows.
        y_prob: Predicted probability array.
        target_k: Operational capacity cutoff.

    Returns:
        tuple[float, float, bool]: (expected_unique_students, expected_duplicate_slots, is_tied_boundary)
    """
    n_samples = len(students)
    if target_k <= 0 or n_samples == 0:
        return 0.0, 0.0, False

    sorted_order = np.argsort(-y_prob)
    threshold_k = y_prob[sorted_order[target_k - 1]]

    mask_above = y_prob > threshold_k
    n_above = int(np.sum(mask_above))

    mask_tied = y_prob == threshold_k
    n_tied = int(np.sum(mask_tied))

    # Case 1: No ties crossing the boundary (deterministic partition)
    if n_above + n_tied == target_k:
        selected_mask = mask_above | mask_tied
        selected_students = students[selected_mask]
        n_uniq = len(set(selected_students))
        n_dups = target_k - n_uniq
        return float(n_uniq), float(n_dups), False

    # Case 2: Ties cross the boundary (uniform random tie-breaking expectation)
    m = target_k - n_above
    students_above = set(students[mask_above])
    n_uniq_above = len(students_above)

    tied_students = students[mask_tied]
    tied_counts = Counter(tied_students)

    exp_additional = 0.0
    for s, c_s in tied_counts.items():
        if s in students_above:
            continue
        # Probability that none of s's c_s records are chosen in m draws out of n_tied without replacement:
        # p_none = prod_{j=0}^{c_s - 1} (n_tied - m - j) / (n_tied - j)
        p_none = 1.0
        for j in range(c_s):
            if n_tied - j > 0:
                p_none *= max(0.0, (n_tied - m - j) / (n_tied - j))
            else:
                p_none = 0.0
        exp_additional += (1.0 - p_none)

    exp_unique = n_uniq_above + exp_additional
    exp_dups = target_k - exp_unique
    return float(exp_unique), float(exp_dups), True


def compute_top_k_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_values: Sequence[int] = (100, 250, 500, 1000),
    pct_values: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    evaluation_scope: str = "pooled_test_cohort",
    presentation_code: str = "ALL_POOLED",
) -> pd.DataFrame:
    """Compute decision support ranking metrics for top-K intervention capacities.

    Handles tied scores and constant predictions analytically to provide unbiased
    expected precision and recall values rather than arbitrary sorting artifacts.

    Args:
        y_true: Binary ground truth outcomes (0 or 1).
        y_prob: Predicted withdrawal risk probabilities.
        k_values: Absolute capacity cutoffs (default: 100, 250, 500, 1000).
        pct_values: Relative capacity percentiles (default: 5%, 10%, 15%, 20%).
        evaluation_scope: Operational context label (e.g. 'pooled_test_cohort' or 'per_presentation').
        presentation_code: Presentation identifier (e.g. 'ALL_POOLED', '2014B').

    Returns:
        pd.DataFrame: Table with columns [evaluation_scope, presentation_code, capacity_type,
            k, capacity_label, is_valid_capacity, precision_at_k, recall_at_k, cumulative_gain, ...].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_samples = len(y_true)
    total_positives = int(np.sum(y_true))

    records = []

    # 1. Absolute K capacities
    for k in k_values:
        int_k = int(k)
        if int_k <= 0:
            continue

        if int_k > n_samples:
            # Insufficient observations in partition: do NOT cap K and fabricate an estimate
            records.append(
                {
                    "evaluation_scope": evaluation_scope,
                    "presentation_code": presentation_code,
                    "capacity_type": "absolute_k",
                    "k": int_k,
                    "capacity_label": f"Top {int_k}",
                    "is_valid_capacity": False,
                    "precision_at_k": float("nan"),
                    "recall_at_k": float("nan"),
                    "cumulative_gain": float("nan"),
                    "positives_captured": float("nan"),
                    "total_positives": total_positives,
                    "n_samples": n_samples,
                    "notes": f"Partition size N={n_samples} is smaller than requested capacity K={int_k}; omitted.",
                }
            )
            continue

        exp_pos, p_at_k, r_at_k, cg = _calculate_top_k_slice(
            y_true=y_true,
            y_prob=y_prob,
            target_k=int_k,
            total_positives=total_positives,
            n_samples=n_samples,
        )

        records.append(
            {
                "evaluation_scope": evaluation_scope,
                "presentation_code": presentation_code,
                "capacity_type": "absolute_k",
                "k": int_k,
                "capacity_label": f"Top {int_k}",
                "is_valid_capacity": True,
                "precision_at_k": p_at_k,
                "recall_at_k": r_at_k,
                "cumulative_gain": cg,
                "positives_captured": exp_pos,
                "total_positives": total_positives,
                "n_samples": n_samples,
                "notes": "Valid capacity",
            }
        )

    # 2. Relative percentile capacities (scale with cohort volume)
    for pct in pct_values:
        actual_k = max(1, min(int(np.round(n_samples * pct)), n_samples))
        exp_pos, p_at_k, r_at_k, cg = _calculate_top_k_slice(
            y_true=y_true,
            y_prob=y_prob,
            target_k=actual_k,
            total_positives=total_positives,
            n_samples=n_samples,
        )

        records.append(
            {
                "evaluation_scope": evaluation_scope,
                "presentation_code": presentation_code,
                "capacity_type": "percentile",
                "k": actual_k,
                "capacity_label": f"Top {pct * 100:.1f}% (N={actual_k})",
                "is_valid_capacity": True,
                "precision_at_k": p_at_k,
                "recall_at_k": r_at_k,
                "cumulative_gain": cg,
                "positives_captured": exp_pos,
                "total_positives": total_positives,
                "n_samples": n_samples,
                "notes": "Valid percentile capacity",
            }
        )

    return pd.DataFrame(records)


def compute_top_k_metrics_by_term(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    term_col: str = "code_presentation",
    target_col: str = "withdrawn_after_cutoff",
    student_col: str | None = "id_student",
    k_values: Sequence[int] = (100, 250, 500, 1000),
    pct_values: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
) -> pd.DataFrame:
    """Compute per-term operational Top-K triage metrics across pooled modules.

    Evaluates ranking within each academic calendar term (code_presentation, e.g. 2014B, 2014J)
    independently. This reflects an operational model where a frontline student support advisor
    (Persona A) manages a caseload of up to K cases across multiple course modules within that term.

    When student_col is provided and present in df, this function audits duplicate student instances
    in the triage queue caused by concurrent multi-module enrollments, reporting unique_students_in_top_k
    and duplicate_student_slots alongside the record-level triage metrics.

    Args:
        df: Input DataFrame containing presentation column, target, and optional student identifier.
        y_prob: Array of predicted probabilities aligned with df rows.
        term_col: Column identifying academic term (default: 'code_presentation').
        target_col: Binary target column name (default: 'withdrawn_after_cutoff').
        student_col: Optional student identifier column (default: 'id_student').
        k_values: Absolute capacity cutoffs.
        pct_values: Relative capacity percentiles.

    Returns:
        pd.DataFrame: Table of Top-K metrics disaggregated by academic term (labeled 'per_term').
    """
    if term_col not in df.columns:
        raise ValueError(f"Term column '{term_col}' missing from DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from DataFrame.")

    y_prob = np.asarray(y_prob, dtype=float)
    if len(df) != len(y_prob):
        raise ValueError(f"Row count mismatch between DataFrame ({len(df)}) and y_prob ({len(y_prob)}).")

    has_student = student_col is not None and student_col in df.columns
    all_dfs = []

    for term in sorted(df[term_col].unique()):
        sub_mask = (df[term_col] == term).values
        sub_df = df.loc[sub_mask]
        sub_y_true = sub_df[target_col].values
        sub_y_prob = y_prob[sub_mask]

        term_df = compute_top_k_metrics(
            y_true=sub_y_true,
            y_prob=sub_y_prob,
            k_values=k_values,
            pct_values=pct_values,
            evaluation_scope="per_term",
            presentation_code=str(term),
        )
        term_df["operational_unit"] = "academic_term_cross_module"
        term_df["queue_unit"] = "module_presentation_enrollment_record"

        if has_student:
            sub_students = sub_df[student_col].values
            term_unique_students = int(sub_df[student_col].nunique())
            term_df["term_total_unique_students"] = term_unique_students

            unique_k_list = []
            dup_slots_list = []
            updated_notes = []

            for _, row in term_df.iterrows():
                is_valid = bool(row["is_valid_capacity"])
                k_val = int(row["k"])
                orig_note = str(row["notes"])

                if not is_valid:
                    unique_k_list.append(np.nan)
                    dup_slots_list.append(np.nan)
                    updated_notes.append(f"{orig_note} (Pools all modules within academic term)")
                else:
                    exp_uniq, exp_dups, is_tied = _calculate_top_k_unique_students(
                        students=sub_students,
                        y_prob=sub_y_prob,
                        target_k=k_val,
                    )
                    unique_k_list.append(exp_uniq)
                    dup_slots_list.append(exp_dups)

                    if not is_tied:
                        n_uniq_int = round(exp_uniq)
                        n_dups_int = round(exp_dups)
                        dup_clause = (
                            f"{n_uniq_int} unique students, {n_dups_int} duplicate slots"
                            if n_dups_int > 0
                            else f"{n_uniq_int} unique students, 0 duplicate slots"
                        )
                        updated_notes.append(
                            f"{orig_note} [{dup_clause}; enrollment-level triage queue pooling all modules within academic term]"
                        )
                    else:
                        updated_notes.append(
                            f"{orig_note} [{exp_uniq:.2f} expected unique students, {exp_dups:.2f} expected duplicate slots under uniform tie-breaking; "
                            "enrollment-level triage queue pooling all modules within academic term]"
                        )

            term_df["unique_students_in_top_k"] = unique_k_list
            term_df["duplicate_student_slots"] = dup_slots_list
            term_df["notes"] = updated_notes
        else:
            term_df["term_total_unique_students"] = np.nan
            term_df["unique_students_in_top_k"] = np.nan
            term_df["duplicate_student_slots"] = np.nan
            term_df["notes"] = term_df["notes"] + " (Pools all modules within academic term)"

        all_dfs.append(term_df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def compute_top_k_metrics_by_presentation(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    presentation_col: str = "code_presentation",
    target_col: str = "withdrawn_after_cutoff",
    student_col: str | None = "id_student",
    k_values: Sequence[int] = (100, 250, 500, 1000),
    pct_values: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    evaluation_scope: str = "per_term",
) -> pd.DataFrame:
    """Compute per-presentation / per-term operational Top-K triage metrics.

    Maintained for backwards-compatibility; delegates to compute_top_k_metrics_by_term.
    """
    res_df = compute_top_k_metrics_by_term(
        df=df,
        y_prob=y_prob,
        term_col=presentation_col,
        target_col=target_col,
        student_col=student_col,
        k_values=k_values,
        pct_values=pct_values,
    )
    if evaluation_scope != "per_term":
        res_df["evaluation_scope"] = evaluation_scope
    return res_df


def compute_top_k_metrics_by_module_presentation(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    module_col: str = "code_module",
    presentation_col: str = "code_presentation",
    target_col: str = "withdrawn_after_cutoff",
    k_values: Sequence[int] = (100, 250, 500, 1000),
    pct_values: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
) -> pd.DataFrame:
    """Compute per-module-presentation operational Top-K triage metrics.

    Evaluates ranking within each course module offering (code_module, code_presentation)
    independently, modeling a Module Presentation Director (Persona B) monitoring attrition
    within their specific course offering.

    Note on operational capacity: In empirical cohorts and test partitions, individual
    module-presentations often have enrollment sizes N smaller than institutional advisor
    capacities (e.g. N < 500). In such cases, is_valid_capacity is set to False and metrics
    are recorded as NaN with an explanatory diagnostic note.

    Args:
        df: Input DataFrame containing module and presentation columns and ground truth target.
        y_prob: Array of predicted probabilities aligned with df rows.
        module_col: Column identifying course module (default: 'code_module').
        presentation_col: Column identifying presentation (default: 'code_presentation').
        target_col: Binary target column name (default: 'withdrawn_after_cutoff').
        k_values: Absolute capacity cutoffs.
        pct_values: Relative capacity percentiles.

    Returns:
        pd.DataFrame: Table of Top-K metrics disaggregated by (code_module, code_presentation)
            labeled 'per_module_presentation'.
    """
    if module_col not in df.columns:
        raise ValueError(f"Module column '{module_col}' missing from DataFrame.")
    if presentation_col not in df.columns:
        raise ValueError(f"Presentation column '{presentation_col}' missing from DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from DataFrame.")

    y_prob = np.asarray(y_prob, dtype=float)
    if len(df) != len(y_prob):
        raise ValueError(f"Row count mismatch between DataFrame ({len(df)}) and y_prob ({len(y_prob)}).")

    all_dfs = []
    # Sort distinct module-presentation pairs deterministically
    unique_pairs = (
        df[[module_col, presentation_col]]
        .drop_duplicates()
        .sort_values(by=[module_col, presentation_col])
        .values
    )
    for mod, pres in unique_pairs:
        sub_mask = ((df[module_col] == mod) & (df[presentation_col] == pres)).values
        sub_y_true = df.loc[sub_mask, target_col].values
        sub_y_prob = y_prob[sub_mask]

        offering_code = f"{mod}_{pres}"
        mod_pres_df = compute_top_k_metrics(
            y_true=sub_y_true,
            y_prob=sub_y_prob,
            k_values=k_values,
            pct_values=pct_values,
            evaluation_scope="per_module_presentation",
            presentation_code=offering_code,
        )
        mod_pres_df["code_module"] = mod
        mod_pres_df["code_presentation"] = pres
        mod_pres_df["operational_unit"] = "module_presentation_course_offering"
        all_dfs.append(mod_pres_df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Calculate scalar classification, ranking, and calibration metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probability for positive class.
        threshold: Decision threshold for discrete classification.

    Returns:
        dict[str, Any]: Dictionary containing all scalar metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Invariant check
    n_samples = len(y_true)
    n_positives = int(np.sum(y_true))
    base_rate = float(n_positives / n_samples) if n_samples > 0 else 0.0

    # Probability ranking metrics (threshold-independent)
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except (ValueError, TypeError) as e:
        logger.warning("Failed to compute PR-AUC: %s", e)
        pr_auc = float("nan")

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except (ValueError, TypeError) as e:
        logger.warning("Failed to compute ROC-AUC: %s", e)
        roc_auc = float("nan")

    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except (ValueError, TypeError) as e:
        logger.warning("Failed to compute Brier score: %s", e)
        brier = float("nan")

    # Threshold-dependent binary metrics
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n_samples": n_samples,
        "n_positives": n_positives,
        "base_rate": base_rate,
        "decision_threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "flagged_count": int(tp + fp),
        "flagged_rate": float((tp + fp) / n_samples) if n_samples > 0 else 0.0,
    }


def find_error_cases(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    threshold: float = 0.50,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify severe false negative and false positive error cases for auditing.

    False Negatives (high-risk omissions):
        Learners who withdrew (y_true == 1) but were assigned the lowest predicted risk.
    False Positives (spurious alarms):
        Learners who completed/persisted (y_true == 0) but were assigned the highest predicted risk.

    Args:
        df: Input DataFrame containing identifiers, predictors, and predicted probabilities.
        y_true_col: Ground truth binary target column.
        y_prob_col: Predicted probability column.
        threshold: Decision threshold.
        top_n: Number of extreme cases to extract.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (severe_false_negatives, severe_false_positives)
    """
    fn_mask = (df[y_true_col] == 1) & (df[y_prob_col] < threshold)
    fp_mask = (df[y_true_col] == 0) & (df[y_prob_col] >= threshold)

    severe_fn = (
        df[fn_mask]
        .sort_values(by=y_prob_col, ascending=True)
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )

    severe_fp = (
        df[fp_mask]
        .sort_values(by=y_prob_col, ascending=False)
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )

    return severe_fn, severe_fp


def generate_evaluation_figures(
    test_df: pd.DataFrame,
    models: dict[str, Any],
    figures_dir: Any,
) -> None:
    """Generate high-resolution publication-quality evaluation figures.

    Figures generated:
        - pr_curves.png: Precision-Recall curves across baseline models.
        - roc_curves.png: ROC curves with random guessing reference.
        - calibration_curves.png: Reliability calibration curves.
        - top_k_gains.png: Cumulative positive capture rate across capacities.
        - feature_importance_lr.png: Odds Ratios for Logistic Regression.
        - feature_importance_rf.png: Impurity importances for Random Forest.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    from student_retention.features.feature_definitions import (
        CANONICAL_TARGET,
        PRIMARY_FEATURE_ALLOWLIST,
    )
    from student_retention.modeling.interpretation import (
        extract_logistic_regression_coefficients,
        extract_random_forest_importance,
    )

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    X_test = test_df[PRIMARY_FEATURE_ALLOWLIST]
    y_test = test_df[CANONICAL_TARGET].values.astype(int)

    probs = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs[name] = model.predict_proba(X_test)[:, 1]
        else:
            probs[name] = model.predict(X_test).astype(float)

    # 1. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, lw=2, label=f"{name}")
    base_rate = np.mean(y_test)
    ax.axhline(base_rate, color="gray", linestyle="--", label=f"Base Rate ({base_rate:.2%})")
    ax.set_title("Precision-Recall Curves (Held-Out Test Partition)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "pr_curves.png")
    plt.close(fig)

    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier (AUC = 0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curves.png")
    plt.close(fig)

    # 3. Reliability / Calibration Curves
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for name, p in probs.items():
        if name == "dummy_baseline":
            continue
        prob_true, prob_pred = calibration_curve(y_test, p, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", lw=2, label=f"{name}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect Calibration")
    ax.set_title("Reliability Diagrams / Calibration Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of True Withdrawals", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "calibration_curves.png")
    plt.close(fig)

    # 4. Top-K Cumulative Gain Curves
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    k_range = np.linspace(50, len(y_test), 50, dtype=int)
    total_pos = np.sum(y_test)
    for name, p in probs.items():
        gains = [
            _calculate_top_k_slice(y_test, p, int(k), total_pos, len(y_test))[2]
            for k in k_range
        ]
        ax.plot(k_range, gains, lw=2, label=f"{name}")
    ax.plot(
        [0, len(y_test)],
        [0, 1.0],
        color="gray",
        linestyle="--",
        label="Random Selection Baseline",
    )
    ax.set_title("Cumulative Withdrawal Capture Rate vs Review Capacity", fontsize=13, fontweight="bold")
    ax.set_xlabel("Top K Learners Flagged for Advisor Review", fontsize=11)
    ax.set_ylabel("Cumulative Recall (Captured Withdrawals)", fontsize=11)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "top_k_gains.png")
    plt.close(fig)

    # 5. Logistic Regression Odds Ratios Bar Chart
    if "logistic_regression" in models:
        lr_coef_df = extract_logistic_regression_coefficients(models["logistic_regression"])
        top_coefs = lr_coef_df.head(15).iloc[::-1]

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        colors = ["#d9534f" if c > 0 else "#5cb85c" for c in top_coefs["coefficient_log_odds"]]
        ax.barh(top_coefs["feature"], top_coefs["coefficient_log_odds"], color=colors)
        ax.axvline(0, color="black", linestyle="--", lw=1)
        ax.set_title("Logistic Regression Log-Odds Coefficients (Top 15 Features)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Log-Odds Coefficient (Positive = Higher Risk, Negative = Protective)", fontsize=11)
        fig.tight_layout()
        fig.savefig(figures_dir / "feature_importance_lr.png")
        plt.close(fig)

    # 6. Random Forest Impurity Importance Bar Chart
    if "random_forest" in models:
        rf_imp_df = extract_random_forest_importance(models["random_forest"])
        top_rf = rf_imp_df.head(15).iloc[::-1]

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        ax.barh(top_rf["feature"], top_rf["impurity_importance"], color="#337ab7")
        ax.set_title("Random Forest Gini Impurity Importance (Top 15 Features)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Relative Impurity Importance", fontsize=11)
        fig.tight_layout()
        fig.savefig(figures_dir / "feature_importance_rf.png")
        plt.close(fig)

    logger.info("Successfully exported 6 evaluation figures to %s", figures_dir)
