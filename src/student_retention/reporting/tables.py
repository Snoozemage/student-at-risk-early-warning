"""Reporting table formatting and metric extraction utilities for reports and dashboard."""

import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from student_retention.paths import (
    get_predictions_dir,
    get_quality_reports_dir,
    get_tables_dir,
)

logger = logging.getLogger(__name__)


def _format_finite(value: Any, fmt: str) -> str:
    """Format a finite numeric value, keeping NaN/infinity visibly unavailable."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return format(numeric, fmt) if math.isfinite(numeric) else "N/A"


def _finite_float(value: Any) -> float | None:
    """Return a finite numeric value or None for missing, malformed, or infinite input."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _format_finite_percent(value: Any) -> str:
    numeric = _finite_float(value)
    formatted = _format_finite(numeric * 100, ".2f") if numeric is not None else "N/A"
    return f"{formatted}%" if formatted != "N/A" else formatted


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean GitHub-flavored Markdown table without external dependencies."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join([":---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join([header_line, separator_line] + rows)


def load_model_training_summary(summary_path: Path | None = None) -> dict[str, Any]:
    """Load model training summary JSON metadata."""
    path = summary_path or (get_tables_dir() / "model_training_summary.json")
    if not path.exists():
        logger.warning("Model training summary not found at %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Error reading model training summary at %s: %s", path, e)
        return {}


def load_calibration_diagnostics(diag_path: Path | None = None) -> dict[str, Any]:
    """Load calibration diagnostics JSON metadata."""
    path = diag_path or (get_tables_dir() / "calibration_diagnostics.json")
    if not path.exists():
        logger.warning("Calibration diagnostics not found at %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Error reading calibration diagnostics at %s: %s", path, e)
        return {}


def format_model_comparison_table(test_metrics_path: Path | None = None) -> pd.DataFrame:
    """Load and format model comparison test metrics for clean display.

    Rounds metrics, formats percentages, and renames columns for readability.
    """
    path = test_metrics_path or (get_tables_dir() / "model_comparison_test.csv")
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    display_df = df.copy()

    # Column renames for display
    col_mapping = {
        "model_key": "Model",
        "model": "Model",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "brier_score": "Brier Score",
        "ece": "ECE",
        "threshold": "Decision Threshold",
        "decision_threshold": "Decision Threshold",
    }
    present_cols = [c for c in col_mapping if c in display_df.columns]
    display_df = display_df[present_cols].rename(columns=col_mapping)

    # Clean model names
    name_map = {
        "dummy_baseline": "Dummy (Class Prior)",
        "dummy_stratified": "Dummy (Stratified)",
        "dummy_most_frequent": "Dummy (Most Frequent)",
        "dummy_prior": "Dummy (Class Prior)",
        "logistic_regression": "Logistic Regression (Unweighted)",
        "logistic_regression_balanced": "Logistic Regression (Balanced)",
        "random_forest": "Random Forest (Unweighted)",
        "random_forest_balanced": "Random Forest (Balanced)",
        "random_forest_calibrated": "Random Forest (Calibrated)",
    }
    if "Model" in display_df.columns:
        display_df["Model"] = display_df["Model"].map(lambda x: name_map.get(x, str(x).replace("_", " ").title()))

    # Format numeric columns
    for col in ["Precision", "Recall", "F1 Score", "PR-AUC", "ROC-AUC"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "N/A")

    for col in ["Brier Score", "ECE"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda v: f"{v:.5f}" if pd.notna(v) else "N/A")

    if "Decision Threshold" in display_df.columns:
        display_df["Decision Threshold"] = display_df["Decision Threshold"].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
        )

    return display_df


def format_calibration_table(calibration_path: Path | None = None) -> pd.DataFrame:
    """Load and format calibration comparison metrics."""
    path = calibration_path or (get_tables_dir() / "calibration_comparison.csv")
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    display_df = df.copy()

    col_map = {
        "model_configuration": "Model Configuration",
        "calibration_method": "Calibration Method",
        "brier_score": "Brier Score",
        "expected_calibration_error": "ECE",
        "maximum_calibration_error": "MCE",
        "min_probability": "Min Prob",
        "max_probability": "Max Prob",
        "mean_probability": "Mean Prob",
    }
    present = [c for c in col_map if c in display_df.columns]
    display_df = display_df[present].rename(columns=col_map)

    for col in ["Brier Score", "ECE", "MCE"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda v: f"{v:.5f}" if pd.notna(v) else "N/A")

    for col in ["Min Prob", "Max Prob", "Mean Prob"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "N/A")

    return display_df


def format_subgroup_fairness_table(
    audit_path: Path | None = None,
    filter_stable_only: bool = True,
) -> pd.DataFrame:
    """Load and format demographic subgroup fairness audit table.

    Args:
        audit_path: Path to subgroup fairness CSV. Defaults to support_queue_fairness_audit.csv
                    if available, else subgroup_fairness_audit.csv.
        filter_stable_only: If True, filters to subgroups meeting sample threshold (N >= 50).
    """
    if audit_path is None:
        queue_path = get_tables_dir() / "support_queue_fairness_audit.csv"
        cls_path = get_tables_dir() / "subgroup_fairness_audit.csv"
        path = queue_path if queue_path.exists() else cls_path
    else:
        path = audit_path

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    stability_col = "is_statistically_stable" if "is_statistically_stable" in df.columns else ("is_sample_size_sufficient" if "is_sample_size_sufficient" in df.columns else None)
    if filter_stable_only:
        if stability_col:
            df = df[df[stability_col].astype(str).str.lower().isin(["true", "1"])].copy()
        else:
            # Subgroup statistical stability cannot be verified; do not assume every subgroup is stable
            return pd.DataFrame()

    display_df = df.copy()

    # Friendly formatting
    if "selection_rate" in display_df.columns:
        display_df["selection_rate_pct"] = display_df["selection_rate"].apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "N/A"
        )
    if "base_rate" in display_df.columns:
        display_df["base_rate_pct"] = display_df["base_rate"].apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "N/A"
        )
    if "recall" in display_df.columns:
        display_df["recall_pct"] = display_df["recall"].apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "N/A"
        )
    if "precision" in display_df.columns:
        display_df["precision_pct"] = display_df["precision"].apply(
            lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "N/A"
        )
    if "disparate_impact_ratio" in display_df.columns:
        display_df["di_ratio_fmt"] = display_df["disparate_impact_ratio"].apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"
        )
        if "is_reference_group" in display_df.columns:
            display_df.loc[display_df["is_reference_group"].astype(str).str.lower().isin(["true", "1"]), "di_ratio_fmt"] = "Ref"

    if "disparate_impact_ratio" in display_df.columns:
        di_values = pd.to_numeric(display_df["disparate_impact_ratio"], errors="coerce")
        di_warning = ((di_values < 0.80) | (di_values > 1.25)).astype("boolean")
        di_warning[di_values.isna()] = pd.NA
        display_df["disparate_impact_warning"] = di_warning
    else:
        display_df["disparate_impact_warning"] = pd.NA

    if "fpr_ratio" in display_df.columns:
        fpr_values = pd.to_numeric(display_df["fpr_ratio"], errors="coerce")
        fpr_warning = (fpr_values > 1.50).astype("boolean")
        fpr_warning[fpr_values.isna()] = pd.NA
        display_df["elevated_fpr_warning"] = fpr_warning
    else:
        display_df["elevated_fpr_warning"] = pd.NA

    return display_df


def count_stable_metric_warnings(
    audit_df: pd.DataFrame,
    metric_col: str,
    *,
    stability_col: str = "is_statistically_stable",
    lower: float | None = None,
    upper: float | None = None,
) -> int | None:
    """Count warnings only when all stable subgroup metric values are defined.

    ``None`` means the audit cannot support a complete count (no stable rows,
    missing metric column, or undefined values among stable rows).
    """
    if stability_col not in audit_df.columns or metric_col not in audit_df.columns:
        return None
    stable = audit_df[stability_col].astype(str).str.lower().isin(["true", "1"])
    stable_df = audit_df.loc[stable]
    if stable_df.empty:
        return None
    values = pd.to_numeric(stable_df[metric_col], errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all():
        return None
    warned = pd.Series(False, index=values.index)
    if lower is not None:
        warned |= values < lower
    if upper is not None:
        warned |= values > upper
    return int(warned.sum())


def format_top_k_table(top_k_path: Path | None = None) -> pd.DataFrame:
    """Load pooled test-cohort Top-K diagnostics, excluding operational term rows."""
    path = top_k_path or (get_tables_dir() / "top_k_evaluation.csv")
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # This file is a pooled random-test diagnostic. Keep its scope explicit and
    # avoid mixing it with per-term operational evaluations if schemas evolve.
    if "evaluation_scope" in df.columns:
        df = df[df["evaluation_scope"] == "pooled_test_cohort"].copy()
    if "presentation_code" in df.columns:
        df = df[df["presentation_code"] == "ALL_POOLED"].copy()
    if "model_key" in df.columns:
        df = df[df["model_key"] == "random_forest"].copy()
    if df.empty:
        return pd.DataFrame()

    display_df = df.copy()

    # Format numbers
    if "precision_at_k" in display_df.columns:
        precision = pd.to_numeric(display_df["precision_at_k"], errors="coerce")
        display_df["precision_at_k_pct"] = precision.apply(_format_finite_percent)
    if "recall_at_k" in display_df.columns:
        recall = pd.to_numeric(display_df["recall_at_k"], errors="coerce")
        display_df["recall_at_k_pct"] = recall.apply(_format_finite_percent)

    # Derive random precision baseline if missing
    if "random_precision_baseline" in display_df.columns:
        baseline = pd.to_numeric(display_df["random_precision_baseline"], errors="coerce")
        display_df["baseline_pct"] = baseline.apply(_format_finite_percent)
    elif "total_positives" in display_df.columns and "n_samples" in display_df.columns:
        sample_n = pd.to_numeric(display_df["n_samples"], errors="coerce")
        positives = pd.to_numeric(display_df["total_positives"], errors="coerce")
        baseline = positives.div(sample_n.where(sample_n > 0))
        display_df["baseline_pct"] = baseline.apply(_format_finite_percent)
    else:
        display_df["baseline_pct"] = "N/A"

    # Derive enrichment factor strictly from authoritative baseline
    if "enrichment_factor" in display_df.columns:
        enrichment = pd.to_numeric(display_df["enrichment_factor"], errors="coerce")
        display_df["enrichment_fmt"] = enrichment.apply(
            lambda v: f"{_format_finite(v, '.2f')}x" if pd.notna(v) else "N/A"
        )
    elif "precision_at_k" in display_df.columns and "random_precision_baseline" in display_df.columns:
        base = pd.to_numeric(display_df["random_precision_baseline"], errors="coerce")
        precision = pd.to_numeric(display_df["precision_at_k"], errors="coerce")
        valid_base = base.where(base.map(math.isfinite) & (base > 0))
        enrichment = precision.div(valid_base)
        display_df["enrichment_fmt"] = enrichment.apply(lambda v: f"{_format_finite(v, '.2f')}x" if pd.notna(v) else "N/A")
    elif "precision_at_k" in display_df.columns and "total_positives" in display_df.columns and "n_samples" in display_df.columns:
        n = pd.to_numeric(display_df["n_samples"], errors="coerce")
        positives = pd.to_numeric(display_df["total_positives"], errors="coerce")
        baseline = positives.div(n.where(n > 0))
        precision = pd.to_numeric(display_df["precision_at_k"], errors="coerce")
        enrichment = precision.div(baseline.where(baseline > 0))
        display_df["enrichment_fmt"] = enrichment.apply(lambda v: f"{_format_finite(v, '.2f')}x" if pd.notna(v) else "N/A")
    else:
        display_df["enrichment_fmt"] = "N/A"

    return display_df


def load_chronological_term_top_k(
    top_k_path: Path | None = None,
    k: int = 500,
) -> pd.DataFrame:
    """Return the Random Forest's full chronological 2014 per-term capacity rows.

    These rows represent the operational time split, unlike pooled random-test
    diagnostics. The evaluation remains enrollment-record level; multi-course
    student aggregation is not defined by the project.
    """
    path = top_k_path or (get_tables_dir() / "top_k_evaluation_by_term.csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    required = {"evaluation_scope", "cohort_partition", "model_key", "k", "presentation_code"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    out = df[
        (df["evaluation_scope"] == "per_term")
        & (df["cohort_partition"] == "chronological_full_presentation_2014")
        & (df["model_key"] == "random_forest")
        & (pd.to_numeric(df["k"], errors="coerce") == k)
        & (df["capacity_type"] == "absolute_k" if "capacity_type" in df.columns else True)
    ].copy()
    if out.empty:
        return out
    for col in ("precision_at_k", "unique_students_in_top_k", "duplicate_student_slots", "n_samples"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("presentation_code").reset_index(drop=True)


def load_executive_kpis(
    summary_path: Path | None = None,
    test_metrics_path: Path | None = None,
    top_k_path: Path | None = None,
    queue_path: Path | None = None,
    cohort_report_path: Path | None = None,
    calibration_path: Path | None = None,
    calibration_diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    """Compile high-level executive KPIs strictly from validated source outputs.

    Never falls back to hardcoded empirical values. If a source file is missing
    or invalid, empirical fields are set to None.
    """
    # Policy / configuration constants (these are not empirically estimated)
    cutoff_day: int = 35
    advisor_capacity_k: int = 500

    cohort_size: int | None = None
    base_rate_pct: float | None = None
    test_cohort_size: int | None = None
    test_unique_students: int | None = None
    model_pr_auc: float | None = None
    dummy_pr_auc: float | None = None
    pr_auc_lift_pct: float | None = None
    model_roc_auc: float | None = None
    precision_at_500: float | None = None
    precision_at_500_baseline: float | None = None
    enrichment_at_500: float | None = None
    queue_total_records: int | None = None
    queue_unique_students: int | None = None
    queue_duplicate_slots: int | None = None
    tier_1_urgent_count: int | None = None
    tier_2_watch_count: int | None = None
    tier_3_standard_count: int | None = None
    ece_raw: float | None = None
    ece_calibrated: float | None = None
    ece_reduction_pct: float | None = None
    selected_calibration_method: str | None = None

    # 1. Cohort and Base Rate from cohort audit report
    c_path = cohort_report_path or (get_quality_reports_dir() / "cohort_policy_c_audit_report.json")
    if c_path.exists():
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            rec = c_data.get("reconciliation", {})
            target_dist = c_data.get("target_distribution", {})
            if "eligible_cohort_records" in rec and rec["eligible_cohort_records"] is not None:
                cohort_size = int(rec["eligible_cohort_records"])
            if "positive_base_rate_pct" in target_dist and target_dist["positive_base_rate_pct"] is not None:
                base_rate_pct = float(target_dist["positive_base_rate_pct"])
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Error reading cohort audit report at %s: %s", c_path, e)

    # 2. Model Training Summary
    summary = load_model_training_summary(summary_path)
    if summary:
        if cohort_size is None and "n_total" in summary and summary["n_total"] is not None:
            cohort_size = int(summary["n_total"])
        if "n_test" in summary and summary["n_test"] is not None:
            test_cohort_size = int(summary["n_test"])

        prov = summary.get("provisional_target_audit", {})
        if isinstance(prov, dict):
            if "test_cohort_size" in prov and prov["test_cohort_size"] is not None:
                test_cohort_size = int(prov["test_cohort_size"])
            if "test_unique_students" in prov and prov["test_unique_students"] is not None:
                test_unique_students = int(prov["test_unique_students"])

            pr_info = prov.get("pr_auc", {})
            if isinstance(pr_info, dict):
                if model_pr_auc is None and "achieved_test_metric" in pr_info and pr_info["achieved_test_metric"] is not None:
                    model_pr_auc = float(pr_info["achieved_test_metric"])
                if dummy_pr_auc is None and "baseline_dummy_pr_auc" in pr_info and pr_info["baseline_dummy_pr_auc"] is not None:
                    dummy_pr_auc = float(pr_info["baseline_dummy_pr_auc"])

            p500_info = prov.get("precision_at_500_pooled_diagnostic", {}) or prov.get("precision_at_500_pooled", {})
            if isinstance(p500_info, dict) and precision_at_500 is None and "achieved_test_metric" in p500_info and p500_info["achieved_test_metric"] is not None:
                precision_at_500 = float(p500_info["achieved_test_metric"])

        test_pr = summary.get("test_pr_auc", {})
        if isinstance(test_pr, dict):
            if model_pr_auc is None and "random_forest" in test_pr and test_pr["random_forest"] is not None:
                model_pr_auc = float(test_pr["random_forest"])
            if dummy_pr_auc is None and "dummy_baseline" in test_pr and test_pr["dummy_baseline"] is not None:
                dummy_pr_auc = float(test_pr["dummy_baseline"])

    # 3. Model Comparison Test CSV
    m_path = test_metrics_path or (get_tables_dir() / "model_comparison_test.csv")
    if m_path.exists():
        try:
            m_df = pd.read_csv(m_path)
            if "model_key" in m_df.columns:
                # Explicitly match primary candidate model by exact model_key identity (never order-dependent)
                rf_rows = m_df[m_df["model_key"] == "random_forest"]
                if rf_rows.empty:
                    # Select calibrated model only if explicit raw model row is absent
                    rf_rows = m_df[m_df["model_key"] == "random_forest_calibrated"]
                if not rf_rows.empty:
                    if "pr_auc" in m_df.columns and pd.notna(rf_rows["pr_auc"].iloc[0]):
                        model_pr_auc = float(rf_rows["pr_auc"].iloc[0])
                    if "roc_auc" in m_df.columns and pd.notna(rf_rows["roc_auc"].iloc[0]):
                        model_roc_auc = float(rf_rows["roc_auc"].iloc[0])

                dummy_rows = m_df[m_df["model_key"] == "dummy_baseline"]
                if dummy_rows.empty:
                    dummy_rows = m_df[m_df["model_key"] == "dummy_prior"]
                if not dummy_rows.empty and "pr_auc" in m_df.columns and pd.notna(dummy_rows["pr_auc"].iloc[0]):
                    dummy_pr_auc = float(dummy_rows["pr_auc"].iloc[0])

            if test_cohort_size is None and "n_samples" in m_df.columns and not m_df.empty and pd.notna(m_df["n_samples"].iloc[0]):
                test_cohort_size = int(m_df["n_samples"].iloc[0])
            if base_rate_pct is None and "base_rate" in m_df.columns and not m_df.empty and pd.notna(m_df["base_rate"].iloc[0]):
                base_rate_pct = float(m_df["base_rate"].iloc[0]) * 100.0
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
            logger.warning("Error reading model comparison test CSV at %s: %s", m_path, e)

    # 4. Top-K Evaluation CSV
    tk_path = top_k_path or (get_tables_dir() / "top_k_evaluation.csv")
    if tk_path.exists():
        try:
            tk_df = pd.read_csv(tk_path)
            if "k" in tk_df.columns and "precision_at_k" in tk_df.columns:
                # Explicit model key matching (never order-dependent)
                if "model_key" in tk_df.columns:
                    subset = tk_df[(tk_df["k"] == 500) & (tk_df["model_key"] == "random_forest")]
                else:
                    subset = tk_df[tk_df["k"] == 500]
                if "evaluation_scope" in subset.columns:
                    subset = subset[subset["evaluation_scope"] == "pooled_test_cohort"]
                if "presentation_code" in subset.columns:
                    subset = subset[subset["presentation_code"] == "ALL_POOLED"]

                if not subset.empty:
                    precision_at_500 = _finite_float(subset["precision_at_k"].iloc[0])

                    # Authoritative random precision baseline
                    base_prec: float | None = None
                    if "random_precision_baseline" in subset.columns:
                        base_prec = _finite_float(subset["random_precision_baseline"].iloc[0])
                    if base_prec is None and "total_positives" in subset.columns and "n_samples" in subset.columns:
                        tot_pos = subset["total_positives"].iloc[0]
                        tot_n = subset["n_samples"].iloc[0]
                        if pd.notna(tot_pos) and pd.notna(tot_n) and float(tot_n) > 0:
                            base_prec = _finite_float(float(tot_pos) / float(tot_n))
                    precision_at_500_baseline = base_prec

                    if "enrichment_factor" in subset.columns:
                        enrichment_at_500 = _finite_float(subset["enrichment_factor"].iloc[0])
                    if enrichment_at_500 is None and precision_at_500 is not None and base_prec is not None and base_prec > 0:
                        enrichment_at_500 = precision_at_500 / base_prec
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
            logger.warning("Error reading top_k evaluation CSV at %s: %s", tk_path, e)

    # 5. Calibration Comparison CSV
    cal_path = calibration_path or (get_tables_dir() / "calibration_comparison.csv")
    diag_path = calibration_diagnostics_path or (cal_path.parent / "calibration_diagnostics.json")
    cal_diagnostics = load_calibration_diagnostics(diag_path)
    selection = cal_diagnostics.get("training_calibration_selection", {})
    if isinstance(selection, dict) and selection.get("selected_method"):
        selected_calibration_method = str(selection["selected_method"]).strip().lower()
    if cal_path.exists():
        try:
            cal_df = pd.read_csv(cal_path)
            if "calibration_method" in cal_df.columns and "expected_calibration_error" in cal_df.columns:
                raw_rows = cal_df[cal_df["calibration_method"].astype(str).str.lower().isin(["none", "uncalibrated", "raw"])]
                method_values = cal_df["calibration_method"].astype(str).str.lower()
                if selected_calibration_method:
                    cal_rows = cal_df[method_values.str.startswith(selected_calibration_method)]
                else:
                    candidate_mask = method_values.str.contains("isotonic|sigmoid|platt")
                    candidate_rows = cal_df[candidate_mask]
                    cal_rows = candidate_rows if len(candidate_rows) == 1 else cal_df.iloc[0:0]
                if not raw_rows.empty and pd.notna(raw_rows["expected_calibration_error"].iloc[0]):
                    ece_raw = float(raw_rows["expected_calibration_error"].iloc[0])
                if not cal_rows.empty and pd.notna(cal_rows["expected_calibration_error"].iloc[0]):
                    if selected_calibration_method is None:
                        selected_calibration_method = method_values.loc[cal_rows.index[0]].split("_")[0]
                    ece_calibrated = float(cal_rows["expected_calibration_error"].iloc[0])
                if ece_raw is not None and ece_calibrated is not None:
                    if ece_raw > 0:
                        ece_reduction_pct = ((ece_raw - ece_calibrated) / ece_raw) * 100.0
                    else:
                        # Undefined relative reduction when raw ECE is zero
                        ece_reduction_pct = None
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
            logger.warning("Error reading calibration comparison CSV at %s: %s", cal_path, e)

    # 6. Actual queue counts from queue CSV if available
    q_path = queue_path or (get_predictions_dir() / "student_support_queue.csv")
    if q_path.exists():
        try:
            q_df = pd.read_csv(q_path)
            if not q_df.empty:
                queue_total_records = len(q_df)
                if "student_key" in q_df.columns and q_df["student_key"].notna().all():
                    queue_unique_students = int(q_df["student_key"].nunique())
                    queue_duplicate_slots = queue_total_records - queue_unique_students
                known_tiers = {"Tier 1 (Urgent)", "Tier 2 (Watch)", "Tier 3 (Standard)"}
                if "priority_tier" in q_df.columns and q_df["priority_tier"].notna().all() and q_df["priority_tier"].isin(known_tiers).all():
                    counts = q_df["priority_tier"].value_counts().to_dict()
                    tier_1_urgent_count = int(counts.get("Tier 1 (Urgent)", 0))
                    tier_2_watch_count = int(counts.get("Tier 2 (Watch)", 0))
                    tier_3_standard_count = int(counts.get("Tier 3 (Standard)", 0))
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
            logger.warning("Error reading queue for KPIs at %s: %s", q_path, e)

    # 7. Dynamically calculate derived lift if not yet derived
    if model_pr_auc is not None and dummy_pr_auc is not None and dummy_pr_auc > 0:
        pr_auc_lift_pct = ((model_pr_auc - dummy_pr_auc) / dummy_pr_auc) * 100.0

    return {
        "cutoff_day": cutoff_day,
        "advisor_capacity_k": advisor_capacity_k,
        "cohort_size": cohort_size,
        "base_rate_pct": base_rate_pct,
        "test_cohort_size": test_cohort_size,
        "test_unique_students": test_unique_students,
        "model_pr_auc": model_pr_auc,
        "dummy_pr_auc": dummy_pr_auc,
        "pr_auc_lift_pct": pr_auc_lift_pct,
        "model_roc_auc": model_roc_auc,
        "precision_at_500": precision_at_500,
        "precision_at_500_baseline": precision_at_500_baseline,
        "enrichment_at_500": enrichment_at_500,
        "queue_total_records": queue_total_records,
        "queue_unique_students": queue_unique_students,
        "queue_duplicate_slots": queue_duplicate_slots,
        "tier_1_urgent_count": tier_1_urgent_count,
        "tier_2_watch_count": tier_2_watch_count,
        "tier_3_standard_count": tier_3_standard_count,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_calibrated,
        "ece_reduction_pct": ece_reduction_pct,
        "selected_calibration_method": selected_calibration_method,
        "precision_at_500_scope": "pooled_test_cohort_diagnostic",
    }
