"""Generate high-level markdown deliverables from real empirical outputs."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from student_retention.paths import (
    get_deliverables_dir,
    get_predictions_dir,
    get_quality_reports_dir,
    get_tables_dir,
)
from student_retention.reporting.tables import (
    count_stable_metric_warnings,
    df_to_markdown,
    format_calibration_table,
    format_model_comparison_table,
    load_chronological_term_top_k,
    load_executive_kpis,
)

logger = logging.getLogger(__name__)


class ReportingDataError(Exception):
    """Raised when required empirical inputs for reporting deliverables are missing or invalid."""


def _calibration_method_label(kpis: dict[str, Any]) -> str:
    """Return a user-facing name from the persisted training-selection metadata."""
    method = str(kpis.get("selected_calibration_method") or "").strip().lower()
    labels = {"isotonic": "Isotonic", "sigmoid": "Sigmoid (Platt scaling)", "platt": "Sigmoid (Platt scaling)"}
    return labels.get(method, method.title() if method else "selected")


def _chronological_term_capacity_text(top_k_path: Path | None = None) -> str:
    """Summarize chronological per-term K=500 results without implying unique-student capacity."""
    df = load_chronological_term_top_k(top_k_path)
    if df.empty:
        return "Chronological per-term K=500 evaluation results are unavailable."
    parts = []
    for _, row in df.iterrows():
        parts.append(
            f"{row['presentation_code']}: Precision@500={row['precision_at_k']:.3f} "
            f"({row['positives_captured']:.0f} positive enrollment records captured; "
            f"{row['unique_students_in_top_k']:.0f} expected unique students selected; "
            f"{row['duplicate_student_slots']:.0f} expected duplicate enrollment slots)"
        )
    return "; ".join(parts) + ". Student-level capacity remains unresolved pending an approved multi-course aggregation policy."


def validate_reporting_inputs(
    tables_dir: Path | None = None,
    predictions_dir: Path | None = None,
    quality_reports_dir: Path | None = None,
) -> None:
    """Validate presence and schema of empirical inputs required for reporting deliverables.

    Raises:
        FileNotFoundError: If a required empirical source file does not exist.
        ReportingDataError: If a required source file is empty or missing necessary columns.
    """
    t_dir = tables_dir or get_tables_dir()
    p_dir = predictions_dir or get_predictions_dir()
    q_dir = quality_reports_dir or get_quality_reports_dir()

    required_files = [
        (t_dir / "model_training_summary.json", "scripts/05_train_models.py (train_models stage)"),
        (t_dir / "model_comparison_test.csv", "scripts/05_train_models.py (evaluate_models stage)"),
        (t_dir / "top_k_evaluation.csv", "scripts/07_run_fairness_and_decision_support.py (fairness_and_decisions stage)"),
        (t_dir / "calibration_comparison.csv", "scripts/07_run_fairness_and_decision_support.py (fairness_and_decisions stage)"),
        (t_dir / "support_queue_fairness_audit.csv", "scripts/07_run_fairness_and_decision_support.py (fairness_and_decisions stage)"),
        (p_dir / "student_support_queue.csv", "scripts/07_run_fairness_and_decision_support.py (fairness_and_decisions stage)"),
        (q_dir / "cohort_policy_c_audit_report.json", "scripts/03_build_cohort.py (build_cohort stage)"),
    ]

    for file_path, stage_hint in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing required empirical reporting input: '{file_path}'. "
                f"Deliverables cannot be generated from missing data. "
                f"Please run pipeline stage: {stage_hint}."
            )
        if file_path.stat().st_size == 0:
            raise ReportingDataError(
                f"Required reporting input is empty: '{file_path}'. "
                f"Please regenerate via: {stage_hint}."
            )

    # 1. Check required columns in model comparison CSV
    try:
        df_comp = pd.read_csv(t_dir / "model_comparison_test.csv")
        if df_comp.empty:
            raise ReportingDataError("model_comparison_test.csv is empty.")
        for col in ["model_key", "pr_auc"]:
            if col not in df_comp.columns:
                raise ReportingDataError(f"model_comparison_test.csv missing required column: '{col}'")
        comp_numeric = pd.to_numeric(df_comp["pr_auc"], errors="coerce")
        if comp_numeric.isna().all():
            raise ReportingDataError("model_comparison_test.csv 'pr_auc' column contains no valid numeric values.")
    except Exception as e:
        if isinstance(e, ReportingDataError):
            raise
        raise ReportingDataError(f"Failed to read model_comparison_test.csv: {e}") from e

    # 2. Check required columns in top_k evaluation CSV
    try:
        df_topk = pd.read_csv(t_dir / "top_k_evaluation.csv")
        if df_topk.empty:
            raise ReportingDataError("top_k_evaluation.csv is empty.")
        for col in ["k", "precision_at_k"]:
            if col not in df_topk.columns:
                raise ReportingDataError(f"top_k_evaluation.csv missing required column: '{col}'")
        prec_numeric = pd.to_numeric(df_topk["precision_at_k"], errors="coerce")
        if prec_numeric.isna().all():
            raise ReportingDataError("top_k_evaluation.csv 'precision_at_k' column contains no valid numeric values.")

        has_baseline = False
        if "random_precision_baseline" in df_topk.columns:
            base_s = pd.to_numeric(df_topk["random_precision_baseline"], errors="coerce")
            if (base_s > 0).any():
                has_baseline = True
        if not has_baseline and "total_positives" in df_topk.columns and "n_samples" in df_topk.columns:
            n_s = pd.to_numeric(df_topk["n_samples"], errors="coerce")
            tp_s = pd.to_numeric(df_topk["total_positives"], errors="coerce")
            if (n_s > 0).any() and not tp_s.isna().all():
                has_baseline = True
        if not has_baseline and "enrichment_factor" in df_topk.columns:
            enr_s = pd.to_numeric(df_topk["enrichment_factor"], errors="coerce")
            if not enr_s.isna().all():
                has_baseline = True

        if not has_baseline:
            raise ReportingDataError(
                "top_k_evaluation.csv lacks required baseline columns ('random_precision_baseline', "
                "'total_positives'/'n_samples', or 'enrichment_factor') with valid numeric values."
            )
        if not (df_topk["k"] == 500).any():
            raise ReportingDataError("top_k_evaluation.csv lacks required evaluation row for k=500.")
    except Exception as e:
        if isinstance(e, ReportingDataError):
            raise
        raise ReportingDataError(f"Failed to read top_k_evaluation.csv: {e}") from e

    # 3. Check calibration comparison CSV
    try:
        df_cal = pd.read_csv(t_dir / "calibration_comparison.csv")
        if df_cal.empty:
            raise ReportingDataError("calibration_comparison.csv is empty.")
        for col in ["calibration_method", "expected_calibration_error"]:
            if col not in df_cal.columns:
                raise ReportingDataError(f"calibration_comparison.csv missing required column: '{col}'")
        ece_s = pd.to_numeric(df_cal["expected_calibration_error"], errors="coerce")
        if ece_s.isna().all():
            raise ReportingDataError("calibration_comparison.csv contains no valid numeric 'expected_calibration_error' values.")
    except Exception as e:
        if isinstance(e, ReportingDataError):
            raise
        raise ReportingDataError(f"Failed to read calibration_comparison.csv: {e}") from e

    # 4. Check support queue fairness audit CSV
    try:
        df_fa = pd.read_csv(t_dir / "support_queue_fairness_audit.csv")
        if df_fa.empty:
            raise ReportingDataError("support_queue_fairness_audit.csv is empty.")
        for col in ["protected_attribute", "subgroup", "selection_rate", "disparate_impact_ratio", "fpr_ratio", "is_statistically_stable"]:
            if col not in df_fa.columns:
                raise ReportingDataError(f"support_queue_fairness_audit.csv missing required column: '{col}'")
        for num_col in ["selection_rate", "disparate_impact_ratio", "fpr_ratio"]:
            s = pd.to_numeric(df_fa[num_col], errors="coerce")
            if s.isna().all():
                raise ReportingDataError(f"support_queue_fairness_audit.csv '{num_col}' column contains no valid numeric values.")
    except Exception as e:
        if isinstance(e, ReportingDataError):
            raise
        raise ReportingDataError(f"Failed to read support_queue_fairness_audit.csv: {e}") from e

    # 5. Check student support queue CSV
    try:
        df_queue = pd.read_csv(p_dir / "student_support_queue.csv")
        if df_queue.empty:
            raise ReportingDataError("student_support_queue.csv is empty.")
        for col in ["student_key", "priority_tier", "risk_probability"]:
            if col not in df_queue.columns:
                raise ReportingDataError(f"student_support_queue.csv missing required column: '{col}'")
        prob_s = pd.to_numeric(df_queue["risk_probability"], errors="coerce")
        if prob_s.isna().all():
            raise ReportingDataError("student_support_queue.csv 'risk_probability' contains no valid numeric values.")
        if ((prob_s < 0.0) | (prob_s > 1.0)).any():
            raise ReportingDataError("student_support_queue.csv 'risk_probability' values out of valid probability range [0, 1].")
    except Exception as e:
        if isinstance(e, ReportingDataError):
            raise
        raise ReportingDataError(f"Failed to read student_support_queue.csv: {e}") from e

    # 6. Check cohort report JSON
    try:
        with open(q_dir / "cohort_policy_c_audit_report.json", "r", encoding="utf-8") as f:
            c_json = json.load(f)
        if not isinstance(c_json, dict) or "reconciliation" not in c_json:
            raise ReportingDataError("cohort_policy_c_audit_report.json missing required 'reconciliation' section.")
        rec_sec = c_json["reconciliation"]
        if "eligible_cohort_records" not in rec_sec:
            raise ReportingDataError("cohort_policy_c_audit_report.json missing 'eligible_cohort_records' in reconciliation.")
        el_recs = rec_sec["eligible_cohort_records"]
        if not isinstance(el_recs, (int, float)) or el_recs <= 0:
            raise ReportingDataError("cohort_policy_c_audit_report.json 'eligible_cohort_records' must be a positive number.")
        if "total_source_records" in rec_sec:
            tot_src = rec_sec["total_source_records"]
            if isinstance(tot_src, (int, float)) and tot_src < el_recs:
                raise ReportingDataError(
                    f"Incompatible cohort reconciliation: total_source_records ({tot_src}) < eligible_cohort_records ({el_recs})."
                )
    except Exception as e:
        if isinstance(e, ReportingDataError):
            raise
        raise ReportingDataError(f"Failed to parse cohort_policy_c_audit_report.json: {e}") from e


def _format_subgroup_parity_findings(
    queue_audit_path: Path,
    cls_audit_path: Path | None = None,
) -> str:
    """Dynamically format subgroup parity audit findings box for reports."""
    lines = [
        "+-----------------------------------------------------------------------------------------+",
        "|                                SUBGROUP PARITY FINDINGS                                 |",
        "+-----------------------------------------------------------------------------------------+",
    ]

    # 1. Classifier Diagnostic
    cls_p = cls_audit_path or (queue_audit_path.parent / "subgroup_fairness_audit.csv")
    if cls_p.exists():
        try:
            cls_df = pd.read_csv(cls_p)
            lines.append("|  1. Classifier Diagnostic (Fixed Threshold theta = 0.25):                               |")
            if "is_statistically_stable" not in cls_df.columns:
                lines.append("|     - Subgroup Stability: Unavailable ('is_statistically_stable' missing)".ljust(90) + "|")
                lines.append("|     - Parity Warnings: Incomplete audit (subgroup stability unverified)".ljust(90) + "|")
            else:
                if "disparate_impact_ratio" in cls_df.columns:
                    di_cnt = count_stable_metric_warnings(cls_df, "disparate_impact_ratio", lower=0.80, upper=1.25)
                    if di_cnt is None:
                        lines.append("|     - Disparate Impact Warnings: Incomplete (no stable or defined metrics)".ljust(90) + "|")
                    else:
                        lines.append(f"|     - Disparate Impact Warnings: {di_cnt} stable subgroups outside [0.80, 1.25]".ljust(90) + "|")
                else:
                    lines.append("|     - Disparate Impact Warnings: Unavailable (missing 'disparate_impact_ratio')".ljust(90) + "|")
                if "fpr_ratio" in cls_df.columns:
                    fpr_cnt = count_stable_metric_warnings(cls_df, "fpr_ratio", upper=1.50)
                    if fpr_cnt is None:
                        lines.append("|     - Elevated FPR Warnings: Incomplete (no stable or defined metrics)".ljust(90) + "|")
                    else:
                        lines.append(f"|     - Elevated FPR Warnings: {fpr_cnt} stable subgroups exceeding FPR ratio 1.50".ljust(90) + "|")
                else:
                    lines.append("|     - Elevated FPR Warnings: Unavailable (missing 'fpr_ratio')".ljust(90) + "|")
            lines.append("|".ljust(90) + "|")
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
            logger.warning("Error parsing classifier fairness audit for summary: %s", e)
            lines.append("|  1. Classifier Diagnostic: Error parsing audit data".ljust(90) + "|")
            lines.append("|".ljust(90) + "|")
    else:
        lines.extend([
            "|  1. Classifier Diagnostic: Audit data not available".ljust(90) + "|",
            "|".ljust(90) + "|",
        ])

    # 2. Operational Support Queue Policy
    if queue_audit_path.exists():
        try:
            q_df = pd.read_csv(queue_audit_path)
            lines.append("|  2. Operational Support Queue Policy (Top-500 Calibrated Allocation):".ljust(90) + "|")
            if "is_statistically_stable" not in q_df.columns:
                lines.append("|     - Subgroup Stability: Unavailable ('is_statistically_stable' missing)".ljust(90) + "|")
                lines.append("|     - Parity Warnings: Incomplete audit (subgroup stability unverified)".ljust(90) + "|")
            else:
                stable_q = q_df[q_df["is_statistically_stable"].astype(str).str.lower().isin(["true", "1"])]

                if "disparate_impact_ratio" in q_df.columns:
                    di_cnt = count_stable_metric_warnings(q_df, "disparate_impact_ratio", lower=0.80, upper=1.25)
                    if di_cnt is None:
                        lines.append("|     - Disparate Impact Warnings: Incomplete (no stable or defined metrics)".ljust(90) + "|")
                    else:
                        di_values = pd.to_numeric(stable_q["disparate_impact_ratio"], errors="coerce")
                        di_rows = stable_q.loc[(di_values < 0.80) | (di_values > 1.25)]
                        lines.append(f"|     - Disparate Impact Warnings: {di_cnt} stable subgroups".ljust(90) + "|")
                        for _, r in di_rows.iterrows():
                            sub = f"{r.get('protected_attribute', '')} {r.get('subgroup', '')}".strip()
                            dir_str = f"DIR = {r['disparate_impact_ratio']:.4f}"
                            sel_val = r.get("selection_rate")
                            sel_str = f"Selection Rate {sel_val * 100:.2f}%" if (sel_val is not None and pd.notna(sel_val)) else "Selection Rate N/A"
                            lines.append(f"|       * {sub}: {dir_str} ({sel_str})".ljust(90) + "|")
                else:
                    lines.append("|     - Disparate Impact Warnings: Unavailable (missing 'disparate_impact_ratio')".ljust(90) + "|")

                if "fpr_ratio" in q_df.columns:
                    fpr_cnt = count_stable_metric_warnings(q_df, "fpr_ratio", upper=1.50)
                    if fpr_cnt is None:
                        lines.append("|     - Elevated FPR Warnings: Incomplete (no stable or defined metrics)".ljust(90) + "|")
                    else:
                        lines.append(f"|     - Elevated FPR Warnings: {fpr_cnt} stable subgroups exceeding FPR ratio 1.50".ljust(90) + "|")
                else:
                    lines.append("|     - Elevated FPR Warnings: Unavailable (missing 'fpr_ratio')".ljust(90) + "|")
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
            logger.warning("Error parsing support queue fairness audit for summary: %s", e)
            lines.append("|  2. Operational Support Queue Policy: Error parsing audit data                          |")
    else:
        lines.append("|  2. Operational Support Queue Policy: Audit data not available                          |")

    lines.append("+-----------------------------------------------------------------------------------------+")
    return "\n".join(lines)


def _format_caseload_breakdown(queue_csv: Path) -> str:
    """Format per-presentation student support queue counts dynamically."""
    if not queue_csv.exists():
        return "Support queue dataset not available."
    try:
        q_df = pd.read_csv(queue_csv)
        if "code_presentation" not in q_df.columns:
            return f"Total demonstration queue records: {len(q_df):,}."
        lines = []
        for pres, grp in q_df.groupby("code_presentation", sort=True):
            n_slots = len(grp)
            n_uniq = grp["student_key"].nunique() if "student_key" in grp.columns else n_slots
            n_dup = n_slots - n_uniq
            lines.append(f"- `{pres}`: {n_slots:,} slots, {n_uniq:,} unique students, {n_dup:,} duplicate slots.")
        tot_slots = len(q_df)
        tot_uniq = q_df["student_key"].nunique() if "student_key" in q_df.columns else tot_slots
        tot_dup = tot_slots - tot_uniq
        lines.append(f"- **Total:** {tot_slots:,} slots, {tot_uniq:,} unique students, {tot_dup:,} duplicate slots across terms.")
        return "\n".join(lines)
    except (OSError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
        return f"Error loading caseload: {e}"


def _format_exclusion_hierarchy(cohort_report_path: Path) -> str:
    """Format exclusion precedence hierarchy dynamically from cohort audit report.

    Strictly derives all statistics from authoritative inputs. Never falls back to
    hardcoded empirical counts or percentages.
    """
    if not cohort_report_path.exists():
        return "Exclusion precedence audit data not available."
    try:
        with open(cohort_report_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        rec = c_data.get("reconciliation", {})
        steps = c_data.get("exclusion_precedence_hierarchy", {})

        raw_count = rec.get("total_source_records")
        raw_count_str = f"{raw_count:,}" if (raw_count is not None and isinstance(raw_count, (int, float))) else "N/A"

        lines = [
            f"[Raw Source Records: {raw_count_str}]",
            "         |",
        ]

        if isinstance(steps, dict) and steps:
            for key in sorted(steps.keys()):
                step_info = steps[key]
                st_num = step_info.get("step", key)
                st_reason = step_info.get("reason", "Excluded")
                st_cnt = step_info.get("count")
                st_cnt_str = f"{st_cnt:,}" if (st_cnt is not None and isinstance(st_cnt, (int, float))) else "N/A"
                lines.append(f"         +--> Step {st_num}: {st_reason} ------> Excluded: {st_cnt_str} records")
                lines.append("         |")
        else:
            lines.append("         +--> Exclusion Precedence Hierarchy: (Not available in report)")
            lines.append("         |")

        el_count = rec.get("eligible_cohort_records")
        el_count_str = f"{el_count:,}" if (el_count is not None and isinstance(el_count, (int, float))) else "N/A"

        # Derive percentage strictly if both numerator and denominator exist and are valid
        if "pct_eligible" in rec and rec["pct_eligible"] is not None:
            pct_str = f" ({float(rec['pct_eligible']):.2f}%)"
        elif (
            el_count is not None
            and raw_count is not None
            and isinstance(el_count, (int, float))
            and isinstance(raw_count, (int, float))
            and raw_count > 0
        ):
            pct_str = f" ({(el_count / raw_count) * 100:.2f}%)"
        else:
            pct_str = ""

        lines.extend([
            "         v",
            f"[Eligible Cohort Policy C: {el_count_str} records{pct_str}]",
        ])

        t_dist = c_data.get("target_distribution")
        if isinstance(t_dist, dict) and t_dist:
            pos_cnt = t_dist.get("positive_cases")
            pos_pct = t_dist.get("positive_base_rate_pct")
            if pos_pct is None and pos_cnt is not None and el_count and el_count > 0:
                pos_pct = (pos_cnt / el_count) * 100.0

            neg_cnt = t_dist.get("negative_cases")
            neg_pct = t_dist.get("negative_class_pct")
            if neg_pct is None and neg_cnt is not None and el_count and el_count > 0:
                neg_pct = (neg_cnt / el_count) * 100.0

            if pos_cnt is not None:
                pos_pct_fmt = f" ({float(pos_pct):.4f}%)" if pos_pct is not None else ""
                lines.append(f"  - Positive Cases (Withdrawn > Day 35): {pos_cnt:,} records{pos_pct_fmt}")
            if neg_cnt is not None:
                neg_pct_fmt = f" ({float(neg_pct):.4f}%)" if neg_pct is not None else ""
                lines.append(f"  - Negative Cases (Pass / Distinction / Fail): {neg_cnt:,} records{neg_pct_fmt}")

        return "\n".join(lines)
    except (OSError, pd.errors.EmptyDataError, ValueError, KeyError, json.JSONDecodeError) as e:
        return f"Error loading exclusion hierarchy: {e}"


def generate_executive_summary(
    kpis: dict[str, Any],
    output_path: Path | None = None,
    queue_audit_path: Path | None = None,
    top_k_by_term_path: Path | None = None,
) -> Path:
    """Generate executive_summary.md deliverable with verified empirical figures."""
    dest = output_path or (get_deliverables_dir() / "executive_summary.md")
    dest.parent.mkdir(parents=True, exist_ok=True)

    cohort_str = f"{kpis['cohort_size']:,}" if kpis.get("cohort_size") is not None else "N/A"
    base_rate_str = f"{kpis['base_rate_pct']:.2f}%" if kpis.get("base_rate_pct") is not None else "N/A"
    pos_count_str = f"{round(kpis['cohort_size'] * (kpis['base_rate_pct'] / 100)):,}" if (kpis.get("cohort_size") is not None and kpis.get("base_rate_pct") is not None) else "N/A"
    test_cohort_str = f"{kpis['test_cohort_size']:,}" if kpis.get("test_cohort_size") is not None else "N/A"
    model_pr_auc_str = f"{kpis['model_pr_auc']:.4f}" if kpis.get("model_pr_auc") is not None else "N/A"
    dummy_pr_auc_str = f"{kpis['dummy_pr_auc']:.4f}" if kpis.get("dummy_pr_auc") is not None else "N/A"
    prec_500_str = f"{kpis['precision_at_500'] * 100:.1f}%" if kpis.get("precision_at_500") is not None else "N/A"
    pos_at_500_str = f"{round(kpis['precision_at_500'] * 500)}" if kpis.get("precision_at_500") is not None else "N/A"
    topk_baseline_str = f"{kpis['precision_at_500_baseline'] * 100:.2f}%" if kpis.get("precision_at_500_baseline") is not None else "N/A"
    q_tot_str = f"{kpis['queue_total_records']:,}" if kpis.get("queue_total_records") is not None else "N/A"
    q_uniq_str = f"{kpis['queue_unique_students']:,}" if kpis.get("queue_unique_students") is not None else "N/A"
    q_dup_str = f"{kpis['queue_duplicate_slots']:,}" if kpis.get("queue_duplicate_slots") is not None else "N/A"
    term_capacity_text = _chronological_term_capacity_text(top_k_by_term_path)
    calibration_label = _calibration_method_label(kpis)

    pr_lift = kpis.get("pr_auc_lift_pct")
    if pr_lift is not None and dummy_pr_auc_str != "N/A":
        if pr_lift > 0:
            pr_auc_lift_str = f"+{pr_lift:.1f}%"
            pr_auc_lift_context = f"Measured relative lift over class-prior dummy baseline ({dummy_pr_auc_str})"
        elif pr_lift == 0:
            pr_auc_lift_str = "0.0%"
            pr_auc_lift_context = f"No observed lift over class-prior dummy baseline ({dummy_pr_auc_str})"
        else:
            pr_auc_lift_str = f"{pr_lift:.1f}%"
            pr_auc_lift_context = f"Measured deficit relative to class-prior dummy baseline ({dummy_pr_auc_str})"
    else:
        pr_auc_lift_str = "N/A"
        pr_auc_lift_context = "Relative lift over dummy baseline could not be evaluated"

    enrich = kpis.get("enrichment_at_500")
    if enrich is not None and topk_baseline_str != "N/A":
        enrich_str = f"{enrich:.2f}x"
        if topk_baseline_str == "N/A":
            enrich_context = "Enrichment factor was present in the source data, but the corresponding baseline value was unavailable."
        elif enrich > 2.0:
            enrich_context = "More than double the test-split random-ranking baseline in this pooled diagnostic"
        elif enrich > 1.0:
            enrich_context = f"Measured {enrich:.2f}x relative to the test-split random-ranking baseline"
        elif enrich == 1.0:
            enrich_context = "Equivalent to the test-split random-ranking baseline (1.00x)"
        else:
            enrich_context = f"Below the test-split random-ranking baseline ({enrich:.2f}x)"
    else:
        enrich_str = "N/A"
        enrich_context = "Triage enrichment factor was not evaluated (insufficient baseline data)"

    q_tot = kpis.get("queue_total_records")
    t1_cnt = kpis.get("tier_1_urgent_count")
    t2_cnt = kpis.get("tier_2_watch_count")
    t3_cnt = kpis.get("tier_3_standard_count")

    if (
        q_tot is not None
        and isinstance(q_tot, (int, float))
        and q_tot > 0
        and t1_cnt is not None
        and t2_cnt is not None
        and t3_cnt is not None
    ):
        t1_pct = (t1_cnt / q_tot) * 100.0
        t2_pct = (t2_cnt / q_tot) * 100.0
        t3_pct = (t3_cnt / q_tot) * 100.0
        tier_section = (
            f"- **Urgent Cases (Tier 1):** {t1_cnt:,} records ({t1_pct:.1f}% of demonstration queue) fall in the proposed highest-score review band.\n"
            f"- **Watch Cases (Tier 2):** {t2_cnt:,} records ({t2_pct:.1f}% of demonstration queue) fall in the proposed intermediate-score review band.\n"
            f"- **Standard Cases (Tier 3):** {t3_cnt:,} records ({t3_pct:.1f}% of demonstration queue) fall below the proposed review bands."
        )
    elif t1_cnt is not None or t2_cnt is not None or t3_cnt is not None:
        t1_disp = f"{t1_cnt:,} records" if t1_cnt is not None else "Unavailable"
        t2_disp = f"{t2_cnt:,} records" if t2_cnt is not None else "Unavailable"
        t3_disp = f"{t3_cnt:,} records" if t3_cnt is not None else "Unavailable"
        tier_section = (
            f"- **Urgent Cases (Tier 1):** {t1_disp} (queue percentage unavailable) fall in the proposed highest-score review band.\n"
            f"- **Watch Cases (Tier 2):** {t2_disp} (queue percentage unavailable) fall in the proposed intermediate-score review band.\n"
            f"- **Standard Cases (Tier 3):** {t3_disp} (queue percentage unavailable) fall below the proposed review bands."
        )
    else:
        tier_section = (
            "- **Urgent Cases (Tier 1):** Unavailable (queue data not loaded or evaluated).\n"
            "- **Watch Cases (Tier 2):** Unavailable (queue data not loaded or evaluated).\n"
            "- **Standard Cases (Tier 3):** Unavailable (queue data not loaded or evaluated)."
        )

    ece_raw = kpis.get("ece_raw")
    ece_cal = kpis.get("ece_calibrated")
    ece_red = kpis.get("ece_reduction_pct")
    calibration_label = _calibration_method_label(kpis)

    if ece_raw is not None and ece_cal is not None:
        ece_cal_fmt = f"{ece_cal:.4f}"
        if ece_cal < ece_raw:
            red_str = f"{ece_red:.1f}%" if ece_red is not None else "measured"
            ece_row_val = f"**{ece_cal_fmt} ({red_str} error reduction)**"
            ece_row_context = f"{calibration_label} calibration improved empirical probability alignment on this test split"
        elif ece_cal == ece_raw:
            ece_row_val = f"**{ece_cal_fmt} (no change)**"
            ece_row_context = "Calibration yielded no change in Expected Calibration Error"
        else:
            inc_str = f"+{abs(ece_red):.1f}%" if ece_red is not None else "measured"
            ece_row_val = f"**{ece_cal_fmt} ({inc_str} error increase)**"
            ece_row_context = "Calibration deteriorated empirical probability alignment on this split"
    elif ece_cal is not None:
        ece_row_val = f"**{ece_cal:.4f}**"
        ece_row_context = "Probability calibration diagnostics available without baseline comparison"
    else:
        ece_row_val = "**N/A**"
        ece_row_context = "Probability calibration diagnostics unavailable (comparison unavailable)"

    # Make Section 2 lead-in strictly conditional on metric availability and direction
    if pr_lift is not None and pr_lift > 0 and enrich is not None and enrich > 1.0:
        lead_in = f"Operating under primary **Policy C (Strict Clean-Data)** on {cohort_str} eligible student enrollment records, retrospective evaluation showed ranking lift and pooled-test top-K enrichment:"
    elif pr_lift is not None and enrich is not None:
        lead_in = f"Operating under primary **Policy C (Strict Clean-Data)** on {cohort_str} eligible student enrollment records, retrospective evaluation yielded the following predictive ranking and pooled-test top-K diagnostic metrics:"
    else:
        lead_in = f"Operating under primary **Policy C (Strict Clean-Data)** on {cohort_str} eligible student enrollment records, retrospective evaluation was conducted under the following parameters (some comparative metrics unavailable):"

    q_audit_p = queue_audit_path or (get_tables_dir() / "support_queue_fairness_audit.csv")
    fairness_audit_claim = "Fairness audit data is unavailable; operational demographic parity could not be evaluated."
    if q_audit_p.exists():
        try:
            q_df = pd.read_csv(q_audit_p)
            if not q_df.empty and "protected_attribute" in q_df.columns:
                n_attrs = int(q_df["protected_attribute"].nunique())
                n_subs = len(q_df)
                attrs_list = ", ".join(sorted(q_df["protected_attribute"].unique()))

                if "is_statistically_stable" not in q_df.columns:
                    fairness_audit_claim = (
                        f"Operational support queue audit evaluated across {n_attrs} demographic attributes "
                        f"({attrs_list}; {n_subs} subgroups total), but subgroup statistical stability information "
                        "('is_statistically_stable') is absent; stable-subgroup parity conclusions cannot be established."
                    )
                elif "fpr_ratio" not in q_df.columns:
                    fairness_audit_claim = (
                        f"Operational support queue audit evaluated across {n_attrs} demographic attributes "
                        f"({attrs_list}; {n_subs} subgroups total), but false positive ratio metrics ('fpr_ratio') "
                        "are unavailable; parity conclusions cannot be established."
                    )
                else:
                    stable_mask = q_df["is_statistically_stable"].astype(str).str.lower().isin(["true", "1"])
                    stable_q = q_df[stable_mask]
                    n_small = len(q_df) - len(stable_q)
                    fpr_warns = count_stable_metric_warnings(q_df, "fpr_ratio", upper=1.50)
                    if fpr_warns is None:
                        fairness_audit_claim = (
                            f"Operational support queue audit covered {n_attrs} demographic attributes "
                            f"({attrs_list}; {n_subs} subgroups), but a complete stable-subgroup FPR warning count "
                            "is unavailable because no stable subgroups or defined FPR ratios were present."
                        )
                    else:
                        small_sample_note = f" (with {n_small} subgroups flagged as statistically unstable, N < 50)" if n_small > 0 else ""
                        warn_str = f"{fpr_warns} demographic subgroup{'s' if fpr_warns != 1 else ''} exhibit elevated False Positive Ratios ($> 1.50$)"
                        if fpr_warns == 0:
                            warn_str = "zero demographic subgroups exhibit elevated False Positive Ratios ($> 1.50$)"
                        fairness_audit_claim = (
                            f"Operational support queue fairness was evaluated across {n_attrs} protected demographic attributes "
                            f"({attrs_list}; {n_subs} subgroups total). Under the calibrated support queue, {warn_str} "
                            f"among statistically stable subgroups{small_sample_note}. "
                            "Ongoing monitoring is required as zero detected warnings does not guarantee the complete absence of disparate impact."
                        )
            else:
                fairness_audit_claim = "Operational support queue fairness audit artifact is empty or missing required schema."
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError):
            fairness_audit_claim = "Error parsing operational support queue fairness audit data."

    content = rf"""# Executive Summary: Smart Student Retention & Early Warning System

**Project:** Smart Student Retention & Early Warning System<br>
**Evaluation Scope:** Open University Learning Analytics Dataset (OULAD)<br>
**Prediction Horizon:** Early-warning cutoff at relative Day 35 (first 5 weeks of study)<br>
**Target Variable:** Post-cutoff course withdrawal (`withdrawn_after_cutoff`)<br>

---

## 1. The Strategic Retention Challenge

In distance higher education, disengagement may be less visible in routine classroom interactions. Retrospective withdrawal records describe course outcomes; they do not establish when an advisor could have noticed a student's difficulties or whether those difficulties were recoverable.

The **Smart Student Retention System** was evaluated at **relative Day 35**—the end of the fifth study week—to estimate and rank post-cutoff withdrawal risk. This retrospective evaluation does not show that outreach at Day 35 changes outcomes, that this is the most effective intervention time, or that departure is inevitable.

---

## 2. Key Empirical Findings

{lead_in}

| Evaluation Metric | Empirical Value | Context & Significance |
| :--- | :--- | :--- |
| **Eligible Student Cohort** | **{cohort_str} records** | Complete courses across 22 module-presentations |
| **Withdrawal Base Rate** | **{base_rate_str}** | {pos_count_str} post-cutoff withdrawals (class imbalance ~1:5) |
| **Evaluation Design** | **Zero-Leakage Group Split** | 80% train / 20% test ($N={test_cohort_str}$) with 0 student overlap |
| **Model PR-AUC (Random Forest)** | **{model_pr_auc_str}** | Primary ranking metric on held-out test cohort |
| **PR-AUC Relative Lift** | **{pr_auc_lift_str}** | {pr_auc_lift_context} |
| **Pooled-Test Precision@500 Diagnostic (Raw Random Forest Ranking)** | **{prec_500_str}** | {pos_at_500_str} positive enrollment records among the top 500 in the pooled random test split; not a per-term capacity estimate |
| **Pooled-Test Enrichment Diagnostic** | **{enrich_str}** | {enrich_context} |
| **Probability Calibration (ECE)** | {ece_row_val} | {ece_row_context} |

### Critical Early Warning Indicators

1. **Coursework Pacing:** Missed due assessments and submission progress were examined as retrospective associations with course withdrawal.
2. **VLE Engagement:** Activity level, recency and resource diversity were examined as candidate early signals.
3. **Registration Timing:** The continuous registration-date comparison does not establish a late-registration cutoff; no binary threshold is validated.

*Note: These analyses are observational and retrospective. They do not establish causation or show that an intervention based on a signal improves retention.*

---

## 3. Proposed Human-Reviewed Triage Workflow (Demonstration)

The demonstration queue contains held-out test enrollment records, capped at up to $K=500$ enrollment slots per presentation ({q_tot_str} slots across the represented terms). It is not an estimate of a future per-term unique-student caseload. The 0.70/0.40 tier cut points and 3/7-day response targets below are proposed workflow settings, not empirically validated decision thresholds or service levels.

Chronological out-of-time per-term K=500 evaluation using uncalibrated Random Forest ranking scores: {term_capacity_text}

```
+-----------------------------------------------------------------------------------------+
|                                ADVISOR TRIAGE PIPELINE                                  |
+-----------------------------------------------------------------------------------------+
|  Tier 1: Urgent (Risk >= 70%)     --> Proposed 3-day human-review response target       |
|  Tier 2: Watch (40% <= Risk < 70%) --> Proposed 7-day supportive check-in target         |
|  Tier 3: Standard (proposed Risk < 40%) --> Routine course communications & regular pacing |
+-----------------------------------------------------------------------------------------+
```

{tier_section}

---

**Workflow status:** Human review and supportive contact are proposed uses of this retrospective ranking. Intervention effectiveness has not been evaluated; any thresholds and response targets require local governance and prospective evaluation before operational adoption.

## 4. Responsible AI & Ethical Boundaries

To prevent algorithmic harm and protect student agency, the system enforces four foundational ethical boundaries:

1. **Exclusively Supportive Interventions:** Algorithmic risk predictions must never trigger punitive actions, academic probation, or financial aid cancellation.
2. **Advisor Discretion:** Early-warning scores represent observational triage signals, not definitive academic destinies. Human advisors maintain absolute discretion to override or dismiss alerts.
3. **Non-Stigmatizing Communication:** Communications with students focus strictly on available resources and learning support; students are never told they were flagged by a "predictive algorithm."
4. **Transparent Fairness Audits:** {fairness_audit_claim}

---

## 5. Next Steps for Production Implementation

1. **Institutional Aggregation Policy:** Formalize university governance on how multi-course students ({q_dup_str} duplicate slots across {q_uniq_str} unique students) are prioritized across faculties.
2. **Prospective Randomized Pilot:** Deploy in a single academic department to test whether proactive outreach measurably lifts retention over control cohorts.
3. **Advisor Feedback Integration:** Incorporate qualitative advisor outcomes back into the training feedback loop.
"""

    dest.write_text(content, encoding="utf-8")
    logger.info("Generated Executive Summary at %s", dest)
    return dest


def generate_technical_report(
    kpis: dict[str, Any],
    output_path: Path | None = None,
    test_metrics_path: Path | None = None,
    calibration_path: Path | None = None,
    queue_audit_path: Path | None = None,
    cls_audit_path: Path | None = None,
    queue_csv: Path | None = None,
    cohort_report_path: Path | None = None,
) -> Path:
    """Generate technical_report.md deliverable documenting end-to-end data science methodology."""
    dest = output_path or (get_deliverables_dir() / "technical_report.md")
    dest.parent.mkdir(parents=True, exist_ok=True)

    t_metrics_p = test_metrics_path or (get_tables_dir() / "model_comparison_test.csv")
    cal_p = calibration_path or (get_tables_dir() / "calibration_comparison.csv")
    q_audit_p = queue_audit_path or (get_tables_dir() / "support_queue_fairness_audit.csv")
    cls_audit_p = cls_audit_path or (get_tables_dir() / "subgroup_fairness_audit.csv")
    q_csv_p = queue_csv or (get_predictions_dir() / "student_support_queue.csv")
    c_rep_p = cohort_report_path or (get_quality_reports_dir() / "cohort_policy_c_audit_report.json")

    model_comp_table_md = df_to_markdown(format_model_comparison_table(t_metrics_p))
    calibration_table_md = df_to_markdown(format_calibration_table(cal_p))
    subgroup_parity_md = _format_subgroup_parity_findings(q_audit_p, cls_audit_p)
    caseload_breakdown_md = _format_caseload_breakdown(q_csv_p)
    exclusion_hierarchy_md = _format_exclusion_hierarchy(c_rep_p)

    cohort_str = f"{kpis['cohort_size']:,}" if kpis.get("cohort_size") is not None else "N/A"
    test_cohort_str = f"{kpis['test_cohort_size']:,}" if kpis.get("test_cohort_size") is not None else "N/A"
    train_size = (kpis["cohort_size"] - kpis["test_cohort_size"]) if (kpis.get("cohort_size") is not None and kpis.get("test_cohort_size") is not None) else None
    train_cohort_str = f"{train_size:,}" if train_size is not None else "N/A"
    uniq_students_str = f"{kpis['test_unique_students']:,}" if kpis.get("test_unique_students") is not None else "N/A"

    if uniq_students_str != "N/A":
        split_eval_desc = f"Train $N={train_cohort_str}$ ({test_cohort_str} test records across {uniq_students_str} unique students)."
    else:
        split_eval_desc = f"Train $N={train_cohort_str}$ ({test_cohort_str} test records)."

    ece_raw = kpis.get("ece_raw")
    ece_cal = kpis.get("ece_calibrated")
    ece_red = kpis.get("ece_reduction_pct")
    calibration_label = _calibration_method_label(kpis)

    if ece_raw is not None and ece_cal is not None:
        ece_raw_fmt = f"{ece_raw:.5f}"
        ece_cal_fmt = f"{ece_cal:.5f}"
        if ece_cal < ece_raw:
            red_str = f"{ece_red:.1f}%" if ece_red is not None else "measured"
            cal_eval_desc = f"- On untouched test data, {calibration_label} calibration improved empirical probability alignment, achieving a **{red_str} error reduction** in Expected Calibration Error (${ece_raw_fmt} \\to {ece_cal_fmt}$)."
        elif ece_cal == ece_raw:
            cal_eval_desc = f"- On untouched test data, {calibration_label} calibration yielded no change in Expected Calibration Error (${ece_cal_fmt}$)."
        else:
            inc_str = f"+{abs(ece_red):.1f}%" if ece_red is not None else "measured"
            cal_eval_desc = f"- On untouched test data, {calibration_label} calibration deteriorated empirical probability alignment, exhibiting a **{inc_str} error increase** in Expected Calibration Error (${ece_raw_fmt} \\to {ece_cal_fmt}$)."
    else:
        cal_eval_desc = "- Test-set calibration diagnostics are unavailable; empirical probability alignment could not be verified."

    content = rf"""# Technical Report: End-to-End Retention Modeling & Decision Support Architecture

**Version:** 1.0 (Phase 8 Production Review)<br>
**Dataset:** Open University Learning Analytics Dataset (OULAD)<br>
**Author:** AI Engineering & Learning Analytics Team<br>

---

## 1. System Overview & Problem Formulation

The early-warning retention system predicts the probability that an enrolled student will withdraw from a course module strictly after relative **Day 35** of the academic presentation:

$$\hat{{y}} = P(\text{{withdrawn\_after\_cutoff}} = 1 \mid \mathbf{{x}}_{{\le 35}})$$

- **Observation Unit & Grain:** `(code_module, code_presentation, id_student)` across {cohort_str} eligible student registrations.
- **Temporal Cutoff:** Day 35 (end of week 5, inclusive).
- **Prohibited Leakage:** Zero post-cutoff VLE interactions, zero post-cutoff coursework submissions, zero assessment grades, and zero demographic features in the primary model.

---

## 2. Ingestion, Cleaning & Exclusion Precedence

Starting from raw enrollment records, the pipeline enforces a strictly deterministic 5-step exclusion precedence hierarchy:

```
{exclusion_hierarchy_md}
```

---

## 3. Feature Engineering & Strict Temporal Safeguards

The primary feature store comprises **22 baseline features** across four functional categories:

1. **Course Context (4 features):** `code_module` (7 categories), `presentation_season` (B vs J), `studied_credits`, `num_of_prev_attempts`.
2. **Registration Dynamics (2 features):** `registration_lead_time_days` (clamped to $[-365, 35]$), `is_late_registration` (binary flag for registration after Day 0).
3. **Digital VLE Engagement (11 features):**
   - Cumulative volumes: `vle_total_clicks_window`, `vle_active_days_window`, `vle_unique_sites_accessed`.
   - Temporal windows: `vle_pre_course_clicks` ($[-25, -1]$), `vle_recent_clicks_14d` ($[22, 35]$).
   - Resource diversity: `vle_clicks_forumng`, `vle_clicks_oucontent`, `vle_clicks_quiz`.
   - Behavioral disengagement: `has_no_vle_activity_by_cutoff`, `days_since_last_vle_interaction` (imputed with deterministic sentinel value **65.0** for inactive students).
   - Pacing momentum: `vle_velocity_ratio` = $\frac{{\text{{recent clicks}} / 14}}{{\text{{total clicks}} / 61}}$.
4. **Assessment Progress Milestones (5 features):**
   - Restricted strictly to scheduled coursework due $\le$ Day 35 ($A_{{\text{{due}}}}$): `assessments_due_count`, `submissions_count_lte_cutoff`, `submitted_on_time_count`, `late_submissions_count_lte_cutoff`, `missing_due_submissions_count`.
   - Explicit exclusion of assignment scores and feedback content to prevent target leakage.

---

## 4. Modeling, Evaluation & Probability Calibration

### Evaluation Splitting Protocol
- Group-aware 80/20 train/test split strictly on `id_student` via `GroupShuffleSplit` (random seed 42).
- Zero student overlap between partitions: {split_eval_desc}

### Benchmark Comparison (Held-Out Test Cohort)

{model_comp_table_md}

### Probability Calibration Diagnostics

{calibration_table_md}

- Evaluated candidate calibration procedures strictly on the training partition via 5-fold `GroupKFold` out-of-fold cross-validation.
- {_calibration_method_label(kpis)} calibration was selected based on training-only selection metadata.
{cal_eval_desc}

---

## 5. Subgroup Fairness & Disparity Analysis

Conducted dual subgroup audits across protected demographic characteristics (evaluating classifier sensitivity at a fixed diagnostic threshold and held-out demonstration queue inclusion):

```
{subgroup_parity_md}
```

---

## 6. Decision Support Caseload & Operational Grain Audit

The demonstration support queue deliverable (`outputs/predictions/student_support_queue.csv`) was generated strictly on the held-out test cohort ($N={test_cohort_str}$) with capacity capped at $K=500$ enrollment slots per presentation:

{caseload_breakdown_md}

**Technical vs Governance Boundary:** The modeling grain is the course enrollment. Transforming enrollment slots into an unduplicated 500-unique-student operational caseload requires institutional policy approval regarding multi-course risk aggregation.
"""

    dest.write_text(content, encoding="utf-8")
    logger.info("Generated Technical Report at %s", dest)
    return dest


def generate_recommendations(
    kpis: dict[str, Any] | None = None,
    output_path: Path | None = None,
    statistical_comparisons_path: Path | None = None,
    presentation_withdrawal_rates_path: Path | None = None,
) -> Path:
    """Generate recommendations.md deliverable with evidence-based institutional guidelines."""
    dest = output_path or (get_deliverables_dir() / "recommendations.md")
    dest.parent.mkdir(parents=True, exist_ok=True)

    stat_p = statistical_comparisons_path or (get_tables_dir() / "statistical_comparisons.csv")
    rec1_evidence = (
        "In module-presentations with coursework due on or before Day 35, the predictive relationship "
        "between missing scheduled continuous assessments and eventual course departure could not be "
        "verified (statistical comparison metrics unavailable; prospective evaluation required)."
    )
    rec3_evidence = (
        "Registration timing has not been established as a basis for risk-directed support; group-level "
        "statistics are unavailable, so no enrollment-timing cohort is justified."
    )
    rec2_evidence = (
        "The available analysis does not validate a 14-day VLE inactivity threshold; "
        "threshold-specific evidence is unavailable, so prospective evaluation is required."
    )
    if stat_p.exists():
        try:
            stat_df = pd.read_csv(stat_p)
            if not stat_df.empty and "feature" in stat_df.columns and "p_value" in stat_df.columns:
                ass_row = stat_df[stat_df["feature"] == "has_missed_due_assessment"]
                if not ass_row.empty and pd.notna(ass_row["p_value"].iloc[0]):
                    p1 = float(ass_row["p_value"].iloc[0])
                    p1_str = f"p = {p1:.2e}" if p1 < 0.001 else f"p = {p1:.4f}"
                    rec1_evidence = (
                        f"In module-presentations with coursework due on or before Day 35, missing scheduled "
                        f"continuous assessments is strongly associated with eventual course departure "
                        f"(${p1_str}$, Mann-Whitney $U$ test)."
                    )
                reg_row = stat_df[stat_df["feature"] == "date_registration"]
                if not reg_row.empty:
                    reg = reg_row.iloc[0]
                    required = {"p_value", "n_retained", "n_withdrawn", "retained_median", "withdrawn_median"}
                    if required.issubset(reg.index) and all(pd.notna(reg[col]) for col in required):
                        p3 = float(reg["p_value"])
                        p3_str = f"p = {p3:.2e}" if p3 < 0.001 else f"p = {p3:.4f}"
                        retained_median = float(reg["retained_median"])
                        withdrawn_median = float(reg["withdrawn_median"])
                        if withdrawn_median < retained_median:
                            direction = (
                                f"The withdrawn group registered earlier at the median "
                                f"({withdrawn_median:g} vs {retained_median:g} days relative to Day 0; "
                                f"equivalent lead times {-withdrawn_median:g} vs {-retained_median:g} days)."
                            )
                        elif withdrawn_median > retained_median:
                            direction = (
                                f"The withdrawn group registered later at the median "
                                f"({withdrawn_median:g} vs {retained_median:g} days relative to Day 0; "
                                f"equivalent lead times {-withdrawn_median:g} vs {-retained_median:g} days)."
                            )
                        else:
                            direction = "The groups have equal median registration dates."
                        rec3_evidence = (
                            f"An exploratory Mann-Whitney U comparison of relative `date_registration` "
                            f"(negative values are days before course start, Day 0; N={int(reg['n_retained']):,} "
                            f"retained and N={int(reg['n_withdrawn']):,} withdrawn) found a distributional "
                            f"difference (${p3_str}$). {direction} `registration_lead_time_days` is "
                            f"`-date_registration`, so larger positive values mean earlier registration. "
                            f"The p-value does not determine direction; this "
                            "small observational difference does not justify targeting either group or "
                            "establish a causal effect or registration cutoff."
                        )
                vle_row = stat_df[stat_df["feature"] == "days_since_last_vle"]
                if not vle_row.empty and pd.notna(vle_row["p_value"].iloc[0]):
                    p2 = float(vle_row["p_value"].iloc[0])
                    p2_str = f"p = {p2:.2e}" if p2 < 0.001 else f"p = {p2:.4f}"
                    rec2_evidence = (
                        "In this retrospective cohort, days since last VLE interaction differed between "
                        f"withdrawn and retained learners (${p2_str}$, Mann-Whitney $U$ test). "
                        "This continuous comparison does not validate a 14-day threshold."
                    )
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError):
            pass

    wdr_p = presentation_withdrawal_rates_path or (get_tables_dir() / "presentation_withdrawal_rates.csv")
    rec4_evidence = (
        "Discipline-specific variation in post-cutoff attrition rates could not be verified "
        "(presentation withdrawal rate metrics unavailable; cross-discipline comparison requires validation)."
    )
    if wdr_p.exists():
        try:
            wdr_df = pd.read_csv(wdr_p)
            if not wdr_df.empty and "withdrawal_rate_pct" in wdr_df.columns:
                stem_mask = (wdr_df["faculty"] == "STEM") if "faculty" in wdr_df.columns else wdr_df["code_module"].isin(["CCC", "DDD", "EEE", "FFF"])
                soc_mask = (wdr_df["faculty"] == "Social Sciences") if "faculty" in wdr_df.columns else wdr_df["code_module"].isin(["AAA", "BBB", "GGG"])
                stem_avg = wdr_df[stem_mask]["withdrawal_rate_pct"].mean()
                soc_avg = wdr_df[soc_mask]["withdrawal_rate_pct"].mean()
                if pd.notna(stem_avg) and pd.notna(soc_avg):
                    rec4_evidence = (
                        f"Post-cutoff attrition rates vary substantially across academic disciplines: evaluated "
                        f"STEM modules average ~{stem_avg:.1f}% withdrawal across presentations, whereas "
                        f"Social Sciences and Arts modules average ~{soc_avg:.1f}%."
                    )
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError):
            pass

    content = rf"""# Evidence-Based Recommendations & Decision Support Guidelines

**Project:** Smart Student Retention & Early Warning System<br>
**Framework:** Decision Support & Human-in-the-Loop Advising<br>

---

## 1. Structured Institutional Recommendations

Each recommendation connects observed empirical patterns to concrete institutional actions, detailing supporting evidence, target cohorts, uncertainties, and necessary verification.

### Recommendation 1: Pilot Early Milestone Review for Missed Coursework

- **Empirical Evidence:** {rec1_evidence}
- **Proposed Review Cohort:** Records in the proposed highest-score demonstration band with `missing_due_submissions_count >= 1`; the score band is a pilot configuration, not a validated risk threshold.
- **Proposed Pilot Action:** Test a rapid-response coursework extension protocol: when an assessment deadline passes without submission, offer a supportive 7-day penalty-free extension and academic tutoring. A 3-business-day advisor-review target may be evaluated as a pilot setting; it is not a validated service level, and the effectiveness of outreach or extensions remains unknown pending prospective evaluation.
- **Uncertainties & Alternative Explanations:** Missing an early assessment may reflect external personal crises (illness, bereavement, work pressures) rather than academic disinterest. An extension alone may not address underlying life disruptions.
- **Additional Data Needed:** Post-extension submission completion rates and advisor contact logs to determine if coursework flexibility translates into long-term course completion.

---

### Recommendation 2: Review of Extended VLE Inactivity

- **Empirical Evidence:** {rec2_evidence}
- **Proposed Review Cohort:** Students in Tier 2 (Watch) or Tier 1 (Urgent) with longer observed inactivity, using the measure as a continuous review signal. No inactivity threshold has been validated; evaluate any cutoff before operational use.
- **Proposed Pilot Action:** Evaluate supportive email or SMS check-ins about platform access barriers (e.g. login difficulties, textbook availability), with an optional link to book an advisor consultation. Their effect on retention is unknown until prospectively evaluated.
- **Uncertainties & Alternative Explanations:** Some students download offline learning materials, PDF study guides, or read physical course textbooks, appearing inactive on the web platform while remaining actively engaged offline.
- **Additional Data Needed:** Offline resource usage surveys and advisor check-in feedback to distinguish true disengagement from intentional offline study habits.

---

### Recommendation 3: Universal Access to Academic Onboarding Support

- **Empirical Evidence:** {rec3_evidence}
- **Affected Target Cohort:** All students may access onboarding and academic scaffolding; offer additional help through student request or advisor review, not registration timing.
- **Recommended Action:** Provide an optional 2-week onboarding pathway with LMS orientation, study schedule templates, and peer-mentor pairing during Weeks 1-3.
- **Uncertainties & Alternative Explanations:** The retrospective registration comparison is small and observational. Its group medians do not establish that registration timing causes withdrawal, and they do not support prioritizing late registrants.
- **Additional Data Needed:** Prospectively evaluate onboarding uptake and outcomes before considering any risk-directed targeting.

---

### Recommendation 4: Faculty-Specific Caseload Allocation & Risk Calibration

- **Empirical Evidence:** {rec4_evidence}
- **Affected Target Cohort:** Central Student Support Services and Faculty Advising Leads.
- **Recommended Action:** Disaggregate advisor caseload quotas. Instead of a flat cross-faculty allocation, assign advising FTE capacity proportionally based on course-level attrition history and calibrated risk volume.
- **Uncertainties & Alternative Explanations:** High attrition in STEM courses may be driven by steep early mathematical prerequisites rather than advising shortages.
- **Additional Data Needed:** Assessment-level prerequisite diagnostics and course-specific curriculum feedback.

---

## 2. Decision Support Workflow & Human Advisor Runbook

```
                         DAY-35 MODEL SCORING
                                   |
                                   v
             [ Calibrated Score & Proposed Priority Band ]
                                   |
          +------------------------+------------------------+
          |                                                 |
          v                                                 v
  Tier 1: Urgent (proposed >= 70%)                  Tier 2: Watch (proposed 40-70%)
          |                                                 |
          v                                                 v
  [ Proposed 3-Day Review Target ]                   [ Proposed 7-Day Check-In Target ]
          |                                                 |
          v                                                 v
  Example: Personal Advisor Review                 Example: Supportive Check-In Review
  - Phone / video call                              - Learning resource links
  - Coursework deadline flexibility                 - Tutor forum alert
  - Pastoral support referral                       - Advisor office hours invite
          |                                                 |
          +------------------------+------------------------+
                                   |
                                   v
                   ADVISOR QUALITATIVE DISCRETION
                     - Override / dismiss alert
                     - Record qualitative context
                     - Log student outcome
```

---

## 3. Strict Prohibitions & Harm Mitigation

1. **No Automated Sanctions:** Risk scores must NEVER be used to automate course deregistration, financial aid denial, or grade capping.
2. **No Algorithmic Labeling:** Advisors and automated templates must NEVER tell a student: *"You were flagged by an algorithm as high risk."*
3. **Mandatory Human Discretion:** Human judgment supersedes model recommendations in all circumstances.
4. **Observational Bounds:** Documented behavioral associations represent predictive triage cues rather than established causal drivers.
"""

    dest.write_text(content, encoding="utf-8")
    logger.info("Generated Recommendations at %s", dest)
    return dest


def generate_presentation_outline(
    kpis: dict[str, Any],
    output_path: Path | None = None,
    queue_audit_path: Path | None = None,
    top_k_by_term_path: Path | None = None,
) -> Path:
    """Generate presentation_outline.md deliverable formatted for stakeholder or academic review."""
    dest = output_path or (get_deliverables_dir() / "presentation_outline.md")
    dest.parent.mkdir(parents=True, exist_ok=True)

    cohort_str = f"{kpis['cohort_size']:,}" if kpis.get("cohort_size") is not None else "N/A"
    base_rate_str = f"{kpis['base_rate_pct']:.2f}%" if kpis.get("base_rate_pct") is not None else "N/A"
    model_pr_auc_str = f"{kpis['model_pr_auc']:.4f}" if kpis.get("model_pr_auc") is not None else "N/A"
    model_roc_auc_str = f"{kpis['model_roc_auc']:.4f}" if kpis.get("model_roc_auc") is not None else "N/A"
    pr_auc_lift_str = f"+{kpis['pr_auc_lift_pct']:.1f}%" if kpis.get("pr_auc_lift_pct") is not None else "N/A"
    dummy_pr_auc_str = f"{kpis['dummy_pr_auc']:.4f}" if kpis.get("dummy_pr_auc") is not None else "N/A"
    prec_500_str = f"{kpis['precision_at_500'] * 100:.1f}%" if kpis.get("precision_at_500") is not None else "N/A"
    pos_at_500_str = f"{round(kpis['precision_at_500'] * 500)}" if kpis.get("precision_at_500") is not None else "N/A"
    topk_baseline_str = f"{kpis['precision_at_500_baseline'] * 100:.2f}%" if kpis.get("precision_at_500_baseline") is not None else "N/A"
    q_tot_str = f"{kpis['queue_total_records']:,}" if kpis.get("queue_total_records") is not None else "N/A"
    term_capacity_text = _chronological_term_capacity_text(top_k_by_term_path)
    calibration_label = _calibration_method_label(kpis)

    pr_lift = kpis.get("pr_auc_lift_pct")
    if pr_lift is not None and dummy_pr_auc_str != "N/A":
        if pr_lift > 0:
            pr_auc_lift_str = f"+{pr_lift:.1f}% relative lift over dummy baseline {dummy_pr_auc_str}"
        elif pr_lift == 0:
            pr_auc_lift_str = f"no observed lift over dummy baseline {dummy_pr_auc_str}"
        else:
            pr_auc_lift_str = f"{pr_lift:.1f}% relative deficit vs dummy baseline {dummy_pr_auc_str}"
    else:
        pr_auc_lift_str = "relative lift over dummy baseline unavailable"

    enrich = kpis.get("enrichment_at_500")
    if enrich is not None and topk_baseline_str != "N/A":
        enrich_str = f"{enrich:.2f}x"
        if enrich > 1.0:
            slide7_enrich_point = f"- **Pooled-test enrichment:** **{enrich_str}** relative to random ranking ({pos_at_500_str} positive enrollment records among the pooled top 500)."
            slide12_takeaway_1 = f"Pooled-test Top-500 diagnostic showed {enrich_str} enrichment over random ranking."
        elif enrich == 1.0:
            slide7_enrich_point = f"- **Enrichment Factor:** **{enrich_str} efficiency factor** (equivalent to random outreach baseline)."
            slide12_takeaway_1 = "Pooled-test Top-500 diagnostic matched the random-ranking baseline (1.00x)."
        else:
            slide7_enrich_point = f"- **Enrichment Factor:** **{enrich_str} efficiency factor** (below random outreach baseline)."
            slide12_takeaway_1 = f"Pooled-test Top-500 diagnostic yielded {enrich_str} enrichment relative to random ranking."
    else:
        enrich_str = "N/A"
        slide7_enrich_point = "- **Enrichment Factor:** Pooled-test enrichment was unavailable because the matching baseline value was missing."
        slide12_takeaway_1 = "Pooled-test Top-K enrichment was unavailable because its comparison baseline was missing."

    q_tot = kpis.get("queue_total_records")
    t1_cnt = kpis.get("tier_1_urgent_count")
    t2_cnt = kpis.get("tier_2_watch_count")
    t3_cnt = kpis.get("tier_3_standard_count")

    if (
        q_tot is not None
        and isinstance(q_tot, (int, float))
        and q_tot > 0
        and t1_cnt is not None
        and t2_cnt is not None
        and t3_cnt is not None
    ):
        slide8_body = (
            f"  - Demonstration queue deliverable: {q_tot:,} cases across academic terms.\n"
            f"  - **Tier 1 (proposed cutoff $\\ge 70\\%$):** {t1_cnt:,} cases; proposed 3-day human-review response target.\n"
            f"  - **Tier 2 (proposed 40-70%):** {t2_cnt:,} cases; proposed 7-day supportive check-in target.\n"
            f"  - **Tier 3 (Standard, $< 40\\%$):** {t3_cnt:,} cases $\\to$ Routine monitoring."
        )
    elif t1_cnt is not None or t2_cnt is not None or t3_cnt is not None:
        t1_str = f"{t1_cnt:,} cases" if t1_cnt is not None else "Unavailable"
        t2_str = f"{t2_cnt:,} cases" if t2_cnt is not None else "Unavailable"
        t3_str = f"{t3_cnt:,} cases" if t3_cnt is not None else "Unavailable"
        slide8_body = (
            f"  - Demonstration queue deliverable: {q_tot_str} cases.\n"
            f"  - **Tier 1 (proposed cutoff $\\ge 70\\%$):** {t1_str}; proposed 3-day human-review response target.\n"
            f"  - **Tier 2 (proposed 40-70%):** {t2_str}; proposed 7-day supportive check-in target.\n"
            f"  - **Tier 3 (Standard, $< 40\\%$):** {t3_str} $\\to$ Routine monitoring."
        )
    else:
        slide8_body = (
            "  - Demonstration queue deliverable: Queue data unavailable; operational triage distribution not evaluated.\n"
            "  - Proposed demonstration settings: Tier 1 (Risk $\\ge 70\\%$, 3-day response target), Tier 2 (40-70%, 7-day response target), Tier 3 ($< 40\\%$, routine); none are validated intervention thresholds or SLAs."
        )

    ece_raw = kpis.get("ece_raw")
    ece_cal = kpis.get("ece_calibrated")
    ece_red = kpis.get("ece_reduction_pct")

    if ece_raw is not None and ece_cal is not None:
        ece_raw_fmt = f"{ece_raw:.4f}"
        ece_cal_fmt = f"{ece_cal:.4f}"
        if ece_cal < ece_raw:
            red_str = f"{ece_red:.1f}%" if ece_red is not None else "measured"
            slide6_ece_point = f"- **Out-of-Time Test Result:** Expected Calibration Error (ECE) dropped from {ece_raw_fmt} to **{ece_cal_fmt} ({red_str} error reduction)**."
        elif ece_cal == ece_raw:
            slide6_ece_point = f"- **Out-of-Time Test Result:** Expected Calibration Error (ECE) showed no change ({ece_cal_fmt})."
        else:
            inc_str = f"+{abs(ece_red):.1f}%" if ece_red is not None else "measured"
            slide6_ece_point = f"- **Out-of-Time Test Result:** Expected Calibration Error (ECE) increased from {ece_raw_fmt} to **{ece_cal_fmt} ({inc_str} error increase)**."
    else:
        slide6_ece_point = "- **Out-of-Time Test Result:** Calibration diagnostics unavailable; empirical probability alignment not evaluated."

    calibration_label = _calibration_method_label(kpis)
    if ece_cal is not None and ece_raw is not None and ece_cal < ece_raw:
        slide12_cal_point = f"{calibration_label} calibration improved empirical probability alignment ({ece_cal:.4f} ECE) on the held-out evaluation split."
    elif ece_cal is not None:
        slide12_cal_point = f"Empirical probability calibration evaluated ({ece_cal:.4f} ECE) for operational priority tiers."
    else:
        slide12_cal_point = "Probability calibration diagnostics unavailable."

    q_audit_p = queue_audit_path or (get_tables_dir() / "support_queue_fairness_audit.csv")
    slide9_scope = "Demographic subgroup audit data is unavailable; fairness could not be evaluated."
    slide9_fpr_point = "Operational FPR parity metrics unavailable."
    if q_audit_p.exists():
        try:
            q_df = pd.read_csv(q_audit_p)
            if not q_df.empty and "protected_attribute" in q_df.columns:
                n_attrs = int(q_df["protected_attribute"].nunique())
                n_subs = len(q_df)
                attrs_list = ", ".join(sorted(q_df["protected_attribute"].unique()))
                slide9_scope = f"Audited {n_subs} demographic subgroups across {n_attrs} protected characteristics ({attrs_list})."

                if "is_statistically_stable" not in q_df.columns:
                    slide9_fpr_point = "Operational FPR parity metrics incomplete: subgroup statistical stability ('is_statistically_stable') is unverified."
                elif "fpr_ratio" not in q_df.columns:
                    slide9_fpr_point = "Operational FPR parity metrics unavailable: 'fpr_ratio' is missing."
                else:
                    fpr_cnt = count_stable_metric_warnings(q_df, "fpr_ratio", upper=1.50)
                    if fpr_cnt is None:
                        slide9_fpr_point = "Operational FPR parity metrics incomplete: no stable subgroups or defined FPR ratios were available."
                    elif fpr_cnt == 0:
                        slide9_fpr_point = "Under the calibrated support queue, **zero stable subgroups exhibit elevated False Positive Ratios ($> 1.50$)**."
                    else:
                        slide9_fpr_point = f"Under the calibrated support queue, **{fpr_cnt} stable subgroups exhibit elevated False Positive Ratios ($> 1.50$)**."
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError):
            pass

    content = rf"""# Presentation Outline: Smart Student Retention & Early Warning System

**Format:** 12-Slide Executive & Academic Stakeholder Presentation<br>
**Target Audience:** University Leadership, Faculty Deans, Advising Directors, Academic Evaluators<br>
**Duration:** 20 minutes presentation + 10 minutes Q&A<br>

---

## Slide 1: Title & Executive Vision
- **Slide Title:** Smart Student Retention: Operational Early Warning & Advising Triage
- **Subtitle:** Machine Learning Decision Support for Timely Student Success at Relative Day 35
- **Presenter:** AI Engineering & Learning Analytics Team
- **Key Message:** Early behavioral signals at Day 35 can identify students at elevated risk of course departure; intervention effectiveness in preventing attrition requires prospective, randomized evaluation.

---

## Slide 2: The Core Educational Problem
- **Talking Points:**
  - Distance learning suffers from high, silent attrition ({base_rate_str} withdraw after Week 5).
  - Traditional advising discovers disengagement too late (after missed exams or formal dropout).
  - The model predicts post-cutoff risk from information available by Day 35; this study did not compare intervention timing or establish whether later outreach is unrecoverable.
- **Visual:** Flowchart comparing reactive advising vs proactive early warning.

---

## Slide 3: Analytical Formulation & Anti-Leakage Safeguards
- **Talking Points:**
  - **Cutoff Horizon:** Strictly bounded at relative Day 35 (first 5 weeks of study).
  - **Dataset:** Open University Learning Analytics Dataset ({cohort_str} eligible enrollment records).
  - **Anti-Leakage Safeguards:** Zero post-cutoff clicks, zero post-cutoff submissions, zero assessment grades, and zero demographic attributes in the primary predictive feature store.
- **Visual:** Timeline diagram illustrating the closed Day 35 observation window and forward target horizon.

---

## Slide 4: Early Behavioral Signals of Student Attrition
- **Talking Points:**
  - **Coursework Pacing:** Missed due assessments and submission progress were examined as retrospective associations.
  - **VLE Engagement:** Activity and resource-diversity measures are candidate signals for further validation.
  - **Registration Horizon:** The continuous comparison does not validate a binary late-registration cutoff.
  - These observations are not causal and do not establish intervention effectiveness.
- **Visual:** Pair of charts: Weekly VLE activity curves + Assessment milestone progress.

---

## Slide 5: Rigorous Machine Learning Evaluation
- **Talking Points:**
  - Strict group-aware splitting on student ID (zero student overlap between train and test).
  - Heavy class imbalance ({base_rate_str} positive base rate) makes accuracy misleading; PR-AUC is primary.
  - **Random Forest Performance:** PR-AUC **{model_pr_auc_str}** ({pr_auc_lift_str}).
  - ROC-AUC: {model_roc_auc_str}.
- **Visual:** Precision-Recall Curves comparing Dummy, Logistic Regression, and Random Forest.

---

## Slide 6: Probability Calibration & Decision Quality
- **Talking Points:**
  - Raw tree ensemble probabilities are compressed and unsuited for direct risk thresholds.
  - **Methodology:** Training-only cross-validation selected {calibration_label} calibration based on persisted selection metadata.
  {slide6_ece_point}
  - The 0.70 Urgent and 0.40 Watch cut points are proposed demonstration settings; probability calibration does not validate decision thresholds or intervention benefit.
- **Visual:** Reliability diagram (Calibration curve) showing 10-bin calibration before and after.

---

## Slide 7: Raw Random Forest Top-K Diagnostic & Chronological Capacity Evaluation
- **Talking Points:**
  - Pooled random-test diagnostic using uncalibrated Random Forest scores: top-500 precision is **{prec_500_str}** versus {topk_baseline_str} test-split baseline; this is not a per-term capacity estimate.
  {slide7_enrich_point}
- **Chronological per-term K=500 results:** {term_capacity_text}
- **Visual:** Pooled-test ranking diagnostic and separate chronological per-term results.

---

## Slide 8: Student Support Queue & Triage Workflow
- **Talking Points:**
{slide8_body}
- **Visual:** Interactive Streamlit dashboard screenshot showing the Student Support Queue.

---

## Slide 9: Subgroup Fairness & Algorithmic Accountability
- **Talking Points:**
  - {slide9_scope}
  - {slide9_fpr_point}
  - Observational disparities reflect historical platform usage, not causal deficiencies.
  - Regular fairness re-auditing monitors for emerging disparities across learner subgroups.
- **Visual:** Horizontal bar chart of Disparate Impact Ratios with 4/5ths reference bands.

---

## Slide 10: Ethical Boundaries & Harm Prevention
- **Talking Points:**
  - Exclusively supportive: algorithm CANNOT trigger disciplinary or financial sanctions.
  - Human-in-the-loop: academic advisors hold 100% discretion to override model flags.
  - Stigma-free messaging: students are contacted regarding support resources, never told they are "flagged."
- **Visual:** 3 Core Pillars of Ethical Early Warning diagram.

---

## Slide 11: Implementation Roadmap & Governance
- **Talking Points:**
  - **Phase A (Immediate):** University governance approves multi-course student caseload aggregation policy.
  - **Phase B (Pilot):** 1-semester randomized controlled trial across two STEM departments.
  - **Phase C (Consider broader deployment):** Proceed only if prospective evaluation supports benefit and institutional governance approves the policy and integration.
- **Visual:** 3-phase deployment timeline with clear milestones.

---

## Slide 12: Summary & Discussion
- **Summary Takeaways:**
  1. {slide12_takeaway_1}
  2. {slide12_cal_point}
  3. Strict human-in-the-loop guardrails uphold supportive, stigma-free advising protocols.
- **Open for Questions:** Discussion on advisor capacity, multi-course aggregation, and pilot deployment.
"""

    dest.write_text(content, encoding="utf-8")
    logger.info("Generated Presentation Outline at %s", dest)
    return dest


def generate_all_deliverables(
    deliverables_dir: Path | None = None,
    tables_dir: Path | None = None,
    predictions_dir: Path | None = None,
    quality_reports_dir: Path | None = None,
) -> dict[str, Path]:
    """Compile high-level KPIs and generate all four project deliverables strictly from validated outputs."""
    t_dir = tables_dir or get_tables_dir()
    p_dir = predictions_dir or get_predictions_dir()
    q_dir = quality_reports_dir or get_quality_reports_dir()

    # Explicit input validation
    validate_reporting_inputs(
        tables_dir=t_dir,
        predictions_dir=p_dir,
        quality_reports_dir=q_dir,
    )

    kpis = load_executive_kpis(
        summary_path=t_dir / "model_training_summary.json",
        test_metrics_path=t_dir / "model_comparison_test.csv",
        top_k_path=t_dir / "top_k_evaluation.csv",
        queue_path=p_dir / "student_support_queue.csv",
        cohort_report_path=q_dir / "cohort_policy_c_audit_report.json",
        calibration_path=t_dir / "calibration_comparison.csv",
    )

    dest_dir = deliverables_dir or get_deliverables_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    summary_path = generate_executive_summary(
        kpis,
        dest_dir / "executive_summary.md",
        queue_audit_path=t_dir / "support_queue_fairness_audit.csv",
        top_k_by_term_path=t_dir / "top_k_evaluation_by_term.csv",
    )
    tech_path = generate_technical_report(
        kpis,
        dest_dir / "technical_report.md",
        test_metrics_path=t_dir / "model_comparison_test.csv",
        calibration_path=t_dir / "calibration_comparison.csv",
        queue_audit_path=t_dir / "support_queue_fairness_audit.csv",
        cls_audit_path=t_dir / "subgroup_fairness_audit.csv",
        queue_csv=p_dir / "student_support_queue.csv",
        cohort_report_path=q_dir / "cohort_policy_c_audit_report.json",
    )
    rec_path = generate_recommendations(
        kpis,
        dest_dir / "recommendations.md",
        statistical_comparisons_path=t_dir / "statistical_comparisons.csv",
        presentation_withdrawal_rates_path=t_dir / "presentation_withdrawal_rates.csv",
    )
    pres_path = generate_presentation_outline(
        kpis,
        dest_dir / "presentation_outline.md",
        queue_audit_path=t_dir / "support_queue_fairness_audit.csv",
        top_k_by_term_path=t_dir / "top_k_evaluation_by_term.csv",
    )

    return {
        "executive_summary": summary_path,
        "technical_report": tech_path,
        "recommendations": rec_path,
        "presentation_outline": pres_path,
    }
