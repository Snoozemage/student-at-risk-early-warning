"""Subgroup performance audit and demographic sensitivity analysis for ethical decision support."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from student_retention.config import get_fairness_config
from student_retention.features.feature_definitions import (
    CANONICAL_TARGET,
    DEMOGRAPHIC_AUDIT_FEATURES,
)
from student_retention.logging_config import get_logger
from student_retention.modeling.pipelines import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_model_pipeline,
)
from student_retention.modeling.splits import split_by_student_group

logger = get_logger(__name__)


def audit_subgroup_metrics(
    df: pd.DataFrame,
    attribute: str,
    reference_group: str | None = None,
    target_col: str = CANONICAL_TARGET,
    pred_col: str = "predicted_label",
    prob_col: str = "risk_probability",
    min_sample_size: int = 50,
    disparate_impact_bounds: tuple[float, float] = (0.80, 1.25),
    fpr_ratio_threshold: float = 1.50,
) -> pd.DataFrame:
    """Compute comprehensive fairness, accuracy, and error-rate parity metrics across subgroups of a protected attribute.

    Evaluates selection rate, base rate, TPR/recall, TNR/specificity, FPR, FNR, precision,
    disparate impact ratio, and FPR ratio relative to the specified reference group.
    Applies small sample warnings when N < min_sample_size.

    Args:
        df: DataFrame containing ground truth target, binary predictions, and the protected attribute column.
        attribute: Name of the protected attribute column (e.g. 'gender', 'age_band', 'disability').
        reference_group: Subgroup value serving as benchmark for ratio calculations. If None, the largest group is used.
        target_col: Column name of actual binary labels (0/1).
        pred_col: Column name of predicted binary labels (0/1).
        prob_col: Column name of predicted risk probabilities.
        min_sample_size: Minimum sample count threshold for statistical stability.
        disparate_impact_bounds: Acceptable band for selection rate parity ratio [lower, upper].
        fpr_ratio_threshold: Disparity threshold above which FPR ratio triggers an elevated disparity warning.

    Returns:
        pd.DataFrame: Tabular audit containing one row per subgroup with full metrics and warning indicators.
    """
    if attribute not in df.columns:
        raise ValueError(f"Protected attribute column '{attribute}' not found in DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if pred_col not in df.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found in DataFrame.")

    # Clean and standardize subgroup representations (fill NA with 'Missing')
    subgroup_series = df[attribute].fillna("Missing").astype(str)
    unique_groups = sorted(subgroup_series.unique().tolist())

    if not unique_groups:
        raise ValueError(f"No valid subgroups found in attribute '{attribute}'.")

    # If reference group is unspecified or not present, select the largest subgroup
    group_counts = subgroup_series.value_counts()
    if reference_group is None or reference_group not in group_counts:
        reference_group = group_counts.index[0]
        logger.info(
            "Reference group for '%s' defaulted to largest category: '%s' (N=%d)",
            attribute,
            reference_group,
            group_counts[reference_group],
        )

    total_cohort_n = len(df)
    records = []

    for grp in unique_groups:
        mask = subgroup_series == grp
        n_samples = int(mask.sum())
        sample_share = float(n_samples / total_cohort_n) if total_cohort_n > 0 else 0.0

        y_true = df.loc[mask, target_col].to_numpy()
        y_pred = df.loc[mask, pred_col].to_numpy()

        n_positives = int(np.sum(y_true == 1))
        n_selected = int(np.sum(y_pred == 1))

        base_rate = float(n_positives / n_samples) if n_samples > 0 else 0.0
        selection_rate = float(n_selected / n_samples) if n_samples > 0 else 0.0

        # Confusion matrix counts
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        # Rates
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        # Mean predicted risk
        mean_prob = (
            float(df.loc[mask, prob_col].mean())
            if prob_col in df.columns and n_samples > 0
            else np.nan
        )

        is_stable = bool(n_samples >= min_sample_size)

        records.append(
            {
                "protected_attribute": attribute,
                "subgroup": grp,
                "is_reference_group": bool(grp == reference_group),
                "sample_size": n_samples,
                "sample_share": round(sample_share, 4),
                "positives_count": n_positives,
                "base_rate": round(base_rate, 4),
                "selected_count": n_selected,
                "selection_rate": round(selection_rate, 4),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
                "mean_predicted_risk": round(mean_prob, 4) if not np.isnan(mean_prob) else np.nan,
                "is_statistically_stable": is_stable,
            }
        )

    audit_df = pd.DataFrame(records)

    # Calculate ratios relative to reference group
    ref_row = audit_df[audit_df["is_reference_group"]]
    if not ref_row.empty:
        ref_selection_rate = ref_row.iloc[0]["selection_rate"]
        ref_fpr = ref_row.iloc[0]["false_positive_rate"]
    else:
        ref_selection_rate = 0.0
        ref_fpr = 0.0

    dir_ratios = []
    fpr_ratios = []
    warnings = []

    for _, row in audit_df.iterrows():
        # Disparate impact ratio (selection rate ratio)
        if ref_selection_rate > 0:
            di_ratio = round(row["selection_rate"] / ref_selection_rate, 4)
        else:
            di_ratio = np.nan
        dir_ratios.append(di_ratio)

        # FPR ratio
        if ref_fpr > 0:
            fpr_ratio = round(row["false_positive_rate"] / ref_fpr, 4)
        else:
            fpr_ratio = np.nan
        fpr_ratios.append(fpr_ratio)

        # Audit warning flags
        flags = []
        if not row["is_statistically_stable"]:
            flags.append("small_sample_warning_N_lt_50")
        else:
            if not np.isnan(di_ratio) and (
                di_ratio < disparate_impact_bounds[0] or di_ratio > disparate_impact_bounds[1]
            ):
                flags.append("disparate_impact_warning")
            if not np.isnan(fpr_ratio) and fpr_ratio > fpr_ratio_threshold:
                flags.append("elevated_fpr_ratio_warning")

        warnings.append("; ".join(flags) if flags else "within_tolerance")

    audit_df["disparate_impact_ratio"] = dir_ratios
    audit_df["fpr_ratio"] = fpr_ratios
    audit_df["fairness_warning_flag"] = warnings

    return audit_df


def audit_all_protected_attributes(
    df: pd.DataFrame,
    fairness_config: dict[str, Any] | None = None,
    target_col: str = CANONICAL_TARGET,
    pred_col: str = "predicted_label",
    prob_col: str = "risk_probability",
    min_sample_size: int = 50,
    audit_scope: str | None = None,
    decision_policy: str | None = None,
) -> pd.DataFrame:
    """Execute complete subgroup disparity audit across all configured protected attributes.

    Reads reference group definitions and tolerance thresholds from configuration, iterates
    over each attribute, and returns a single concatenated audit report.

    Args:
        df: DataFrame containing predictions, labels, and demographic attributes.
        fairness_config: Fairness configuration dict. If None, loaded from config/fairness.yaml.
        target_col: Column name of actual binary labels.
        pred_col: Column name of predicted binary labels.
        prob_col: Column name of predicted risk probabilities.
        min_sample_size: Minimum sample count threshold for statistical stability.
        audit_scope: Optional tag describing the evaluation scope (e.g. 'classifier_level', 'operational_queue').
        decision_policy: Optional tag describing the decision threshold or ranking rule.

    Returns:
        pd.DataFrame: Comprehensive audit table with all protected attributes and subgroups.
    """
    if fairness_config is None:
        fairness_config = get_fairness_config()

    protected_cfg = fairness_config.get("protected_attributes", {})
    rules_cfg = fairness_config.get("subgroup_reporting_rules", {})
    tolerance_cfg = rules_cfg.get("tolerance_thresholds", {})

    di_bounds = tuple(tolerance_cfg.get("disparate_impact_warning_band", [0.80, 1.25]))
    fpr_thresh = float(tolerance_cfg.get("fpr_ratio_warning_threshold", 1.50))
    cfg_min_n = int(rules_cfg.get("min_subgroup_sample_size", min_sample_size))

    all_audits = []

    for attr_key, attr_info in protected_cfg.items():
        field_name = attr_info.get("field", attr_key)
        ref_group = attr_info.get("reference_group")

        if field_name not in df.columns:
            logger.warning(
                "Configured protected attribute '%s' (field: '%s') not present in DataFrame. Skipping.",
                attr_key,
                field_name,
            )
            continue

        attr_audit = audit_subgroup_metrics(
            df=df,
            attribute=field_name,
            reference_group=ref_group,
            target_col=target_col,
            pred_col=pred_col,
            prob_col=prob_col,
            min_sample_size=cfg_min_n,
            disparate_impact_bounds=di_bounds,
            fpr_ratio_threshold=fpr_thresh,
        )
        all_audits.append(attr_audit)

    if not all_audits:
        raise ValueError("Zero configured protected attributes were found in the provided DataFrame.")

    combined_df = pd.concat(all_audits, ignore_index=True)

    if decision_policy is not None:
        combined_df.insert(0, "decision_policy", decision_policy)
    if audit_scope is not None:
        combined_df.insert(0, "audit_scope", audit_scope)

    logger.info(
        "Completed subgroup audit across %d protected attributes (%d total subgroups).",
        len(all_audits),
        len(combined_df),
    )
    return combined_df


def audit_operational_support_queue(
    queue_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    fairness_config: dict[str, Any] | None = None,
    target_col: str = CANONICAL_TARGET,
    prob_col: str = "calibrated_risk_probability",
    min_sample_size: int = 50,
    decision_policy: str = "top_500_per_presentation_calibrated",
    audit_scope: str = "operational_support_queue",
) -> pd.DataFrame:
    """Audit subgroup fairness of the operational decision-support triage queue.

    Evaluates the actual binary operational decision: queue inclusion (e.g. Top-500 capacity allocation
    per presentation, scored with the calibrated model) versus exclusion across demographic subgroups.

    Anti-Leakage & Provenance Guarantees:
    - Scored strictly on the held-out group-aware test cohort (or prospective operational cohort).
    - Preserves course-enrollment prediction grain (code_module, code_presentation, id_student).
    - Reports selection/inclusion rates, disparate impact ratios, precision, recall, FPR, FPR ratios,
      and dynamically computed warning flags.

    Args:
        queue_df: Generated support queue DataFrame containing student_key, code_module, code_presentation.
        cohort_df: Full evaluation cohort DataFrame containing demographic features, identifiers, and ground truth target.
        fairness_config: Fairness configuration dict. If None, loaded from config/fairness.yaml.
        target_col: Binary target column name. Defaults to CANONICAL_TARGET.
        prob_col: Column containing predicted risk probabilities. Defaults to 'calibrated_risk_probability'.
        min_sample_size: Minimum sample count for statistical stability. Defaults to 50.
        decision_policy: Descriptor of operational triage policy. Defaults to 'top_500_per_presentation_calibrated'.
        audit_scope: Evaluation scope tag. Defaults to 'operational_support_queue'.

    Returns:
        pd.DataFrame: Operational queue subgroup audit table with full disparity metrics and warning flags.
    """
    for req in ["code_module", "code_presentation", "student_key"]:
        if req not in queue_df.columns:
            raise ValueError(f"Required column '{req}' missing from queue DataFrame.")
    for req in ["code_module", "code_presentation", "id_student", target_col]:
        if req not in cohort_df.columns:
            raise ValueError(f"Required column '{req}' missing from cohort DataFrame.")

    cohort_keys = (
        cohort_df["code_module"].astype(str)
        + "_"
        + cohort_df["code_presentation"].astype(str)
        + "_"
        + cohort_df["id_student"].astype(str)
    )
    queue_keys = set(
        queue_df["code_module"].astype(str)
        + "_"
        + queue_df["code_presentation"].astype(str)
        + "_"
        + queue_df["student_key"].astype(str)
    )

    eval_df = cohort_df.copy()
    eval_df["in_support_queue"] = cohort_keys.isin(queue_keys).astype(int)

    if prob_col not in eval_df.columns:
        if "risk_probability" in eval_df.columns:
            eval_df[prob_col] = eval_df["risk_probability"]
        else:
            eval_df[prob_col] = np.nan

    logger.info(
        "Auditing operational support queue fairness (Cohort N=%d, Queue Selected N=%d, Policy=%s)...",
        len(eval_df),
        eval_df["in_support_queue"].sum(),
        decision_policy,
    )

    return audit_all_protected_attributes(
        df=eval_df,
        fairness_config=fairness_config,
        target_col=target_col,
        pred_col="in_support_queue",
        prob_col=prob_col,
        min_sample_size=min_sample_size,
        audit_scope=audit_scope,
        decision_policy=decision_policy,
    )


def run_demographic_sensitivity_experiment(
    feature_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    operational_threshold: float = 0.28,
) -> dict[str, Any]:
    """Execute empirical sensitivity experiment comparing baseline models vs models with demographic predictors.

    Evaluates:
      1. Primary baseline (22 primary features, demographics excluded as per policy).
      2. Demographic-aware model (22 primary features + 6 demographic audit features).

    Measures incremental performance gain (PR-AUC, ROC-AUC, Brier score) and changes
    in subgroup parity metrics (disparate impact and FPR ratios).

    Args:
        feature_df: Complete feature matrix containing primary features, demographics, target, and id_student.
        test_size: Held-out group split test size.
        random_state: Reproducible seed.
        operational_threshold: Decision threshold for computing binary metrics.

    Returns:
        dict[str, Any]: Dictionary containing:
            - 'model_comparison': DataFrame comparing overall test performance.
            - 'baseline_subgroup_audit': Subgroup audit on baseline model.
            - 'demographic_subgroup_audit': Subgroup audit on demographic model.
            - 'parity_delta_table': Subgroup-level delta in disparate impact and FPR ratios.
            - 'summary_findings': Structured qualitative and quantitative takeaways.
    """
    logger.info("Executing Phase 7 Demographic Sensitivity Experiment...")

    # Ensure clean student group split
    train_df, test_df = split_by_student_group(
        feature_df,
        test_size=test_size,
        random_state=random_state,
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Standardize demographic categorical columns (fill NA with 'Missing' and cast to string)
    demo_available = [col for col in DEMOGRAPHIC_AUDIT_FEATURES if col in train_df.columns]
    for col in demo_available:
        train_df[col] = train_df[col].fillna("Missing").astype(str)
        test_df[col] = test_df[col].fillna("Missing").astype(str)

    y_train = train_df[CANONICAL_TARGET].to_numpy()
    y_test = test_df[CANONICAL_TARGET].to_numpy()

    # 1. Baseline Model (Primary Features Only)
    baseline_cat = list(CATEGORICAL_FEATURES)
    baseline_num = list(NUMERIC_FEATURES)
    baseline_features = baseline_cat + baseline_num

    baseline_pipeline = build_model_pipeline(
        model_name="random_forest",
        numeric_features=baseline_num,
        categorical_features=baseline_cat,
        random_state=random_state,
    )

    logger.info("Fitting primary baseline Random Forest (22 features, demographics excluded)...")
    baseline_pipeline.fit(train_df[baseline_features], y_train)

    prob_baseline = baseline_pipeline.predict_proba(test_df[baseline_features])[:, 1]
    pred_baseline = (prob_baseline >= operational_threshold).astype(int)

    # 2. Demographic-Aware Model (Primary Features + 6 Demographic Audit Features)
    demo_cat = baseline_cat + demo_available
    demo_features = demo_cat + baseline_num

    demo_pipeline = build_model_pipeline(
        model_name="random_forest",
        numeric_features=baseline_num,
        categorical_features=demo_cat,
        random_state=random_state,
    )

    logger.info(
        "Fitting demographic-aware Random Forest (28 features, including %d demographic features)...",
        len(demo_available),
    )
    demo_pipeline.fit(train_df[demo_features], y_train)

    prob_demo = demo_pipeline.predict_proba(test_df[demo_features])[:, 1]
    pred_demo = (prob_demo >= operational_threshold).astype(int)

    # Compute overall performance metrics
    def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "brier_score": round(brier, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    m_baseline = _compute_metrics(y_test, prob_baseline, pred_baseline)
    m_demo = _compute_metrics(y_test, prob_demo, pred_demo)

    comparison_df = pd.DataFrame(
        [
            {
                "model_configuration": "Primary Baseline (Demographics Excluded)",
                "feature_count": len(baseline_features),
                "operational_threshold": operational_threshold,
                **m_baseline,
            },
            {
                "model_configuration": "Demographic Sensitivity (Demographics Included)",
                "feature_count": len(demo_features),
                "operational_threshold": operational_threshold,
                **m_demo,
            },
        ]
    )

    # Evaluate subgroup disparity for both models on test set
    test_eval_df = test_df.copy()

    test_eval_df["predicted_label_baseline"] = pred_baseline
    test_eval_df["risk_prob_baseline"] = prob_baseline
    audit_baseline = audit_all_protected_attributes(
        df=test_eval_df,
        target_col=CANONICAL_TARGET,
        pred_col="predicted_label_baseline",
        prob_col="risk_prob_baseline",
    )

    test_eval_df["predicted_label_demo"] = pred_demo
    test_eval_df["risk_prob_demo"] = prob_demo
    audit_demo = audit_all_protected_attributes(
        df=test_eval_df,
        target_col=CANONICAL_TARGET,
        pred_col="predicted_label_demo",
        prob_col="risk_prob_demo",
    )

    # Compute parity delta table (Demographic - Baseline)
    delta_cols = [
        "protected_attribute",
        "subgroup",
        "sample_size",
        "is_statistically_stable",
        "disparate_impact_ratio",
        "fpr_ratio",
        "selection_rate",
        "recall",
        "fairness_warning_flag",
    ]
    merged_audit = audit_baseline[delta_cols].merge(
        audit_demo[delta_cols],
        on=["protected_attribute", "subgroup", "sample_size", "is_statistically_stable"],
        suffixes=("_baseline", "_demo"),
    )
    merged_audit["delta_disparate_impact"] = (
        merged_audit["disparate_impact_ratio_demo"] - merged_audit["disparate_impact_ratio_baseline"]
    ).round(4)
    merged_audit["delta_fpr_ratio"] = (
        merged_audit["fpr_ratio_demo"] - merged_audit["fpr_ratio_baseline"]
    ).round(4)
    merged_audit["delta_selection_rate"] = (
        merged_audit["selection_rate_demo"] - merged_audit["selection_rate_baseline"]
    ).round(4)

    pr_auc_diff = round(m_demo["pr_auc"] - m_baseline["pr_auc"], 4)
    summary_findings = {
        "pr_auc_delta": pr_auc_diff,
        "roc_auc_delta": round(m_demo["roc_auc"] - m_baseline["roc_auc"], 4),
        "brier_delta": round(m_demo["brier_score"] - m_baseline["brier_score"], 4),
        "policy_recommendation": (
            "Excluding demographic features from predictors is strongly upheld: adding demographic "
            f"features yielded an incremental PR-AUC change of {pr_auc_diff:+.4f}, but introduces the "
            "severe risk of direct disparate treatment and institutional bias reinforcement. Continuous "
            "subgroup auditing without direct demographic conditioning provides the ethically soundest posture."
        ),
    }

    return {
        "model_comparison": comparison_df,
        "baseline_subgroup_audit": audit_baseline,
        "demographic_subgroup_audit": audit_demo,
        "parity_delta_table": merged_audit,
        "summary_findings": summary_findings,
    }
