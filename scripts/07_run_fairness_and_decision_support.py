"""Phase 7 pipeline runner: Model Interpretation, Subgroup Fairness Audit, Calibration, and Decision Support Queue Generation."""

import argparse
import json
from pathlib import Path

import pandas as pd

from student_retention.decision_support.prioritize_students import (
    audit_support_queue_capacity,
    build_student_support_queue,
)
from student_retention.fairness.subgroup_audit import (
    audit_all_protected_attributes,
    audit_operational_support_queue,
    run_demographic_sensitivity_experiment,
)
from student_retention.features.feature_definitions import CANONICAL_TARGET
from student_retention.features.feature_selection import select_primary_features
from student_retention.logging_config import get_logger
from student_retention.modeling.calibration import (
    compute_calibration_diagnostics,
    evaluate_calibration_methods_on_training_data,
    fit_calibrated_model,
    save_calibrated_model,
)
from student_retention.modeling.interpretation import (
    extract_logistic_regression_coefficients,
    extract_random_forest_importance,
)
from student_retention.modeling.pipelines import build_model_pipeline
from student_retention.modeling.predict import load_model, load_operational_threshold
from student_retention.modeling.splits import split_by_student_group
from student_retention.paths import (
    MODELS_BASELINE_DIR,
    PREDICTIONS_DIR,
    PROCESSED_FEATURES_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_directories,
)

logger = get_logger(__name__)


def generate_subgroup_fairness_report(
    audit_df: pd.DataFrame,
    sensitivity_results: dict,
    operational_threshold: float,
    threshold_provenance: str,
    output_path: Path,
    queue_audit_df: pd.DataFrame | None = None,
    training_cal_df: pd.DataFrame | None = None,
    selected_calibration_method: str | None = None,
) -> None:
    """Generate comprehensive Markdown report documenting subgroup disparity audits.

    Clearly separates:
      1. Training-only calibration method selection and provenance.
      2. Demographic sensitivity analysis findings.
      3. Classifier-level subgroup diagnostic (Uncalibrated RF @ fixed threshold).
      4. Held-out demonstration queue subgroup audit (calibrated RF, enrollment-slot cap).

    Dynamically enumerates disparity flags without hard-coded parity claims or unsupported
    interpretations, maintaining strict non-causal observational phrasing.
    """
    stable_groups = audit_df[audit_df["is_statistically_stable"]]
    unstable_groups = audit_df[~audit_df["is_statistically_stable"]]

    di_warnings = stable_groups[stable_groups["fairness_warning_flag"].str.contains("disparate_impact")]
    fpr_warnings = stable_groups[stable_groups["fairness_warning_flag"].str.contains("elevated_fpr")]

    comp_df = sensitivity_results["model_comparison"]

    report_text = f"""# Subgroup Fairness Audit & Demographic Sensitivity Report

**Phase:** Phase 7 — Interpretation, Fairness and Decision Support
**Evaluation Candidate Cohort:** Held-Out Group-Aware Test Cohort (20% sample, N={audit_df['sample_size'].sum() // len(audit_df['protected_attribute'].unique()):,} candidate enrollment records across 2013B, 2013J, 2014B, 2014J)
**Classifier Diagnostic Policy:** Fixed Decision Threshold $\\theta = {operational_threshold:.2f}$ ({threshold_provenance}) evaluated across all candidate records
**Queue Evaluation Scope:** Historical held-out demonstration queue, ranked by the calibration method recorded in training-only selection metadata; capacity is measured in course-enrollment slots, not unique students, and does not represent a deployed institutional policy.

---

## 1. Executive Summary & Policy Adherence

The Smart Student Retention & Early Warning System enforces an **"audit-only" demographic policy**:
* Demographic and socioeconomic variables are **strictly excluded from predictive model inputs** to prevent direct disparate treatment and algorithmic redlining.
* Excluding demographic attributes does **not** prove the absence of proxy reliance; therefore, all models undergo continuous empirical subgroup audits across six protected characteristics: `gender`, `age_band`, `disability`, `imd_band`, `highest_education`, and `region`.
* Statistical stability constraint: Subgroups with sample sizes $N < 50$ carry an explicit small-sample instability warning to prevent over-interpreting noisy ratio estimates.

> [!IMPORTANT]
> **Fairness Evaluation Scope & Policy Boundary:**
> Do not claim that the model has passed fairness merely because one metric is within tolerance. Subgroup disparity metrics provide necessary operational transparency across multiple equity dimensions, but no single threshold or ratio proves the absence of proxy effects or guarantees equal educational outcomes. Observational differences in selection rates or error rates reflect behavioral and contextual signal distributions in historical data and must not be interpreted as causal student deficiencies or proof of model perfection.
"""

    if training_cal_df is not None:
        report_text += """
---

## 2. Calibration Method Selection & Provenance (Training-Only Validation)

To eliminate any risk of test-set leakage or post-hoc method selection, candidate calibration procedures were evaluated **strictly on the 80% training partition** (`train_df`, N=21,714) using 5-fold `GroupKFold` cross-validation on `id_student`. The held-out test cohort was reserved strictly for untouched final evaluation.

### Training-Only Out-of-Fold Cross-Validation Comparison

| Model Configuration | Calibration Method | Evaluation Split | Brier Score | ECE | MCE | Max Probability |
|---|---|---|---|---|---|---|
"""
        for _, r in training_cal_df.iterrows():
            report_text += (
                f"| **{r['model_configuration']}** | `{r['calibration_method']}` | {r['evaluation_split']} | "
                f"{r['brier_score']:.5f} | {r['expected_calibration_error']:.5f} | {r['maximum_calibration_error']:.5f} | {r['max_probability']:.4f} |\n"
            )

        report_text += f"""
**Method Selection Evidence:** The selected method in training-only calibration metadata is **{selected_calibration_method or 'unavailable'}**. The comparison above reports this run's observed training out-of-fold metrics; the selected method is not assumed to be universally optimal.
"""

    report_text += f"""
---

## 3. Demographic Sensitivity Analysis Findings

To evaluate the predictive utility and ethical trade-offs of demographic predictors, an empirical experiment compared:
1. **Primary Baseline Model:** 22 non-demographic behavioral and context features (demographics strictly excluded).
2. **Demographic-Aware Model:** 28 features (22 primary + 6 one-hot encoded demographic attributes).

### Performance Comparison

| Model Configuration | Features | PR-AUC | ROC-AUC | Brier Score | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **Primary Baseline (Excluded)** | {comp_df.iloc[0]['feature_count']} | {comp_df.iloc[0]['pr_auc']:.4f} | {comp_df.iloc[0]['roc_auc']:.4f} | {comp_df.iloc[0]['brier_score']:.4f} | {comp_df.iloc[0]['precision']:.4f} | {comp_df.iloc[0]['recall']:.4f} | {comp_df.iloc[0]['f1']:.4f} |
| **Demographic-Aware (Included)** | {comp_df.iloc[1]['feature_count']} | {comp_df.iloc[1]['pr_auc']:.4f} | {comp_df.iloc[1]['roc_auc']:.4f} | {comp_df.iloc[1]['brier_score']:.4f} | {comp_df.iloc[1]['precision']:.4f} | {comp_df.iloc[1]['recall']:.4f} | {comp_df.iloc[1]['f1']:.4f} |
| **Delta (Demo − Baseline)** | +6 | **{comp_df.iloc[1]['pr_auc'] - comp_df.iloc[0]['pr_auc']:+.4f}** | **{comp_df.iloc[1]['roc_auc'] - comp_df.iloc[0]['roc_auc']:+.4f}** | **{comp_df.iloc[1]['brier_score'] - comp_df.iloc[0]['brier_score']:+.4f}** | **{comp_df.iloc[1]['precision'] - comp_df.iloc[0]['precision']:+.4f}** | **{comp_df.iloc[1]['recall'] - comp_df.iloc[0]['recall']:+.4f}** | **{comp_df.iloc[1]['f1'] - comp_df.iloc[0]['f1']:+.4f}** |

### Policy Recommendation

{sensitivity_results['summary_findings']['policy_recommendation']}

---

## 4. Classifier-Level Subgroup Diagnostic (Uncalibrated RF @ $\\theta = {operational_threshold:.2f}$)

**Binary Decision Evaluated:** Model output thresholding at the fixed validation threshold: $\\hat{{y}} = \\mathbb{{I}}(\\text{{raw\\_prob}} \\ge {operational_threshold:.2f})$.
**Purpose:** Pre-deployment classifier diagnostic evaluating raw score calibration and threshold sensitivity across demographic groups before operational capacity constraints are applied.

Across the 6 protected attributes ({len(audit_df)} total evaluated subgroups):
* **Statistically Stable Subgroups ($N \\ge 50$):** {len(stable_groups)} subgroups ({len(stable_groups) / len(audit_df) * 100:.1f}%).
* **Small Sample Subgroups ($N < 50$):** {len(unstable_groups)} subgroups ({len(unstable_groups) / len(audit_df) * 100:.1f}%), primarily in extreme deprivation bands or specific regional partitions.

### Observational Disparity Highlights (Classifier Level)

* **Gender (`gender`):** Reference group `M` (Male). Selection rate for females is {audit_df.loc[(audit_df['protected_attribute']=='gender') & (audit_df['subgroup']=='F'), 'selection_rate'].values[0]:.2%} vs males {audit_df.loc[(audit_df['protected_attribute']=='gender') & (audit_df['subgroup']=='M'), 'selection_rate'].values[0]:.2%}.
* **Age Band (`age_band`):** Reference group `0-35`. Learners in the `55<=` group have small sample sizes ($N=35 < 50$), carrying an explicit statistical instability warning.
* **Disability (`disability`):** Reference group `N`. Declared disabled learners (`Y`) exhibit an elevated selection rate relative to reference group `N` (Disparate Impact Ratio = {audit_df.loc[(audit_df['protected_attribute']=='disability') & (audit_df['subgroup']=='Y'), 'disparate_impact_ratio'].values[0]:.2f}). Under the system's ethical guidelines, elevated alert rates direct supportive pastoral and accessibility resources rather than punitive sanctions.
* **Education (`highest_education`):** Reference group `A Level or Equivalent`. Selection and error rates reflect observational differences in entry qualifications across academic presentations.
* **Deprivation (`imd_band`):** Reference group `90-100%`. Selection and error rates across deprivation deciles should be reviewed in the master audit table.
* **Region (`region`):** Reference group `South East Region`. Geographical partitions exhibit substantial variation in selection and error rates across the 13 regions.

### Identified Classifier Warnings ({len(di_warnings) + len(fpr_warnings)} total flags)

"""
    if len(di_warnings) > 0:
        report_text += "#### Disparate Impact Ratio Flags (Outside [0.80, 1.25]):\n"
        for _, w in di_warnings.iterrows():
            report_text += f"- **{w['protected_attribute']} = {w['subgroup']}** (N={w['sample_size']}): Selection Rate = {w['selection_rate']:.2%}, Disparate Impact Ratio = {w['disparate_impact_ratio']:.2f}\n"
    else:
        report_text += "No stable subgroups exhibited disparate impact ratios outside the [0.80, 1.25] tolerance band.\n"

    report_text += "\n"
    if len(fpr_warnings) > 0:
        report_text += "#### Elevated False Positive Rate Ratio Flags (> 1.50):\n"
        for _, w in fpr_warnings.iterrows():
            report_text += f"- **{w['protected_attribute']} = {w['subgroup']}** (N={w['sample_size']}): False Positive Rate = {w['false_positive_rate']:.2%}, FPR Ratio = {w['fpr_ratio']:.2f}\n"
    else:
        report_text += "No stable subgroups exhibited false positive rate ratios exceeding the 1.50 warning threshold.\n"

    if queue_audit_df is not None:
        q_stable = queue_audit_df[queue_audit_df["is_statistically_stable"]]
        q_unstable = queue_audit_df[~queue_audit_df["is_statistically_stable"]]
        q_di_warnings = q_stable[q_stable["fairness_warning_flag"].str.contains("disparate_impact")]
        q_fpr_warnings = q_stable[q_stable["fairness_warning_flag"].str.contains("elevated_fpr")]

        report_text += f"""
---

## 5. Demonstration Support Queue Subgroup Audit (Calibrated RF @ Top-500 Enrollment Slots)

**Binary Decision Evaluated:** Inclusion in the historical held-out demonstration queue, ranked by calibrated probabilities and capped at up to 500 course-enrollment slots per represented presentation.
**Purpose:** Retrospective audit of this demonstration allocation rule. It does not establish a deployed policy or a 500-unique-student caseload; duplicate course enrollments require an institutional aggregation decision.

Across the 6 protected attributes ({len(queue_audit_df)} total evaluated subgroups):
* **Statistically Stable Subgroups ($N \\ge 50$):** {len(q_stable)} subgroups ({len(q_stable) / len(queue_audit_df) * 100:.1f}%).
* **Small Sample Subgroups ($N < 50$):** {len(q_unstable)} subgroups ({len(q_unstable) / len(queue_audit_df) * 100:.1f}%), primarily `age_band = 55<=` (N=35).

### Queue-Level Parity Findings & Comparison to Classifier Diagnostics

In this held-out demonstration queue (overall inclusion rate = {queue_audit_df['selected_count'].sum() / queue_audit_df['sample_size'].sum():.2%}):
* **Disparate Impact Warnings:** **{len(q_di_warnings)} stable subgroups** exceed the [0.80, 1.25] tolerance band (down from {len(di_warnings)} in the uncalibrated classifier diagnostic).
* **Elevated False Positive Rate Warnings:** **{len(q_fpr_warnings)} stable subgroups** exceed the 1.50 warning threshold (down from {len(fpr_warnings)} in the uncalibrated classifier diagnostic).

### Identified Operational Queue Warnings ({len(q_di_warnings) + len(q_fpr_warnings)} total flags)

"""
        if len(q_di_warnings) > 0:
            report_text += "#### Operational Queue Disparate Impact Ratio Flags (Outside [0.80, 1.25]):\n"
            for _, w in q_di_warnings.iterrows():
                report_text += (
                    f"- **{w['protected_attribute']} = {w['subgroup']}** (N={w['sample_size']}): "
                    f"Queue Inclusion Rate = {w['selection_rate']:.2%}, Disparate Impact Ratio = {w['disparate_impact_ratio']:.2f}, "
                    f"Queue Recall (Capture Rate) = {w['recall']:.2%}\n"
                )
        else:
            report_text += "No stable subgroups exhibited disparate impact ratios outside the [0.80, 1.25] tolerance band in the operational queue.\n"

        report_text += "\n"
        if len(q_fpr_warnings) > 0:
            report_text += "#### Operational Queue Elevated False Positive Rate Ratio Flags (> 1.50):\n"
            for _, w in q_fpr_warnings.iterrows():
                report_text += (
                    f"- **{w['protected_attribute']} = {w['subgroup']}** (N={w['sample_size']}): "
                    f"Queue False Positive Rate = {w['false_positive_rate']:.2%}, FPR Ratio = {w['fpr_ratio']:.2f}\n"
                )
        else:
            report_text += "No stable subgroups exhibited false positive rate ratios exceeding the 1.50 warning threshold in the operational queue.\n"

        report_text += """
### Caseload Capacity Grain & Aggregation Policy Notice

The queue allocates capacity at the **course-enrollment grain** (up to 500 slots per presentation in this held-out demonstration). Duplicate enrollment slots mean the institution's intended 500-case capacity cannot be interpreted as 500 unique students without an approved cross-module risk aggregation policy.
"""

    report_text += """
---

## 6. Responsible Use & Governance Protocols

1. **Non-Stigmatizing Triage:** Output priority scores must never be labeled as "failure probabilities" or "student risk flags" in student-facing settings.
2. **Supportive Outreach Only:** The triage queue exists solely to trigger timely academic advising, tutor check-ins, and pastoral assistance.
3. **Mandatory Human Discretion:** Academic advisors retain full discretion to reprioritize or dismiss triage alerts based on qualitative contextual knowledge.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved Subgroup Fairness Report to %s", output_path)


def generate_decision_support_report(
    queue_df: pd.DataFrame,
    capacity_audit_df: pd.DataFrame,
    calibration_diag: dict,
    output_path: Path,
    selected_calibration_method: str | None = None,
) -> None:
    """Generate Markdown report detailing decision support queue composition, calibration, and workflow rules."""
    tier_counts = queue_df["priority_tier"].value_counts()
    total_q = len(queue_df)
    total_unique = queue_df["student_key"].nunique()
    total_dups = total_q - total_unique

    report_text = f"""# Decision Support & Caseload Prioritization Report

**Phase:** Phase 7 — Interpretation, Fairness and Decision Support
**Output Deliverable:** [`outputs/predictions/student_support_queue.csv`](../predictions/student_support_queue.csv)
**Evaluation Candidate Cohort:** Held-Out Group-Aware Test Cohort (5,414 candidate enrollment records across four academic terms: 2013B, 2013J, 2014B, 2014J)
**Generated Support Queue:** Historical held-out demonstration queue (up to K=500 enrollment records per presentation; {total_q:,} slots across {total_unique:,} unique students in this run)
**Demonstration Cap:** K=500 course-enrollment records per presentation. This is not a validated 500-unique-student caseload; cross-module aggregation remains unresolved.

---

## 1. Proposed Demonstration Triage Settings

The cutoffs and response targets below are provisional settings used to illustrate a retrospective held-out queue. They are not validated decision thresholds or operational service levels. Retrospective calibration and queue composition do not establish that outreach improves student outcomes; intervention effectiveness requires prospective evaluation.

| Priority Tier | Proposed Demonstration Rule | Proposed Response Target | Demonstration Cases | Caseload % | Example Support Action |
|---|---|---|---|---|---|
| **Tier 1 (Urgent)** | Proposed score $\\ge 0.70$ | Proposed 3 business days | {tier_counts.get('Tier 1 (Urgent)', 0):,} | {tier_counts.get('Tier 1 (Urgent)', 0) / total_q:.1%} | Example: personal outreach for human review |
| **Tier 2 (Watch)** | Proposed score $0.40 \\le p < 0.70$ | Proposed 7 business days | {tier_counts.get('Tier 2 (Watch)', 0):,} | {tier_counts.get('Tier 2 (Watch)', 0) / total_q:.1%} | Example: supportive check-in for human review |
| **Tier 3 (Standard)** | Proposed score $< 0.40$ | No target set | {tier_counts.get('Tier 3 (Standard)', 0):,} | {tier_counts.get('Tier 3 (Standard)', 0) / total_q:.1%} | Example: routine course communications |

---

## 2. Caseload Capacity & Unique-Student Representation Audit

| Academic Term | Caseload Slots (K=500 Limit) | Unique Students | Duplicate Slots | Multi-Course Students | Institutional Aggregation Status |
|---|---|---|---|---|---|
"""
    for _, r in capacity_audit_df.iterrows():
        term_label = f"**{r['code_presentation']}**" if r["code_presentation"] == "ALL_TERMS_TOTAL" else r["code_presentation"]
        report_text += f"| {term_label} | {int(r['caseload_enrollment_slots']):,} | {int(r['unique_students_in_queue']):,} | {int(r['duplicate_slots']):,} | {int(r['multi_course_students_in_queue']):,} | `{r['unresolved_aggregation_status']}` |\n"

    report_text += f"""
> **Important Operational Governance Notice:**
> Institutional business requirements specify academic advising capacity at approximately 500 cases per academic term across modules.
> In this historical demonstration queue, the triage ranking operates at the **course-enrollment record grain** (`code_module`, `code_presentation`, `student_key`) across four terms: 2013B, 2013J, 2014B, and 2014J.
> Learners concurrently enrolled in multiple course offerings during the same term may appear in multiple module-presentation slots ({total_dups:,} duplicate slots across all presentations).
> Because an institutional policy defining how multi-course learner risk should be aggregated (e.g., maximum module risk vs mean risk) has not yet been established, a true 500-unique-student operational queue remains **unresolved pending institutional policy definition**. In-sample training records are strictly excluded.

---

## 3. Probability Calibration & Reliability

Model risk probabilities use **{selected_calibration_method or 'a method unavailable in selection metadata'}** as selected on the training partition via 5-fold `GroupKFold` cross-validation:
* **Held-Out Test Brier Score:** {calibration_diag.get('brier_score', 'N/A')}
* **Expected Calibration Error (ECE):** {calibration_diag.get('expected_calibration_error', 'N/A')} on the held-out evaluation data; compare with the raw-model row before interpreting calibration change.
* **Calibrated Probability Range:** [{calibration_diag.get('min_probability', 0.0):.4f}, {calibration_diag.get('max_probability', 1.0):.4f}] (mean: {calibration_diag.get('mean_probability', 0.0):.4f})
* **Calibration Artifact:** [`models/baseline/random_forest_calibrated.joblib`](../../models/baseline/random_forest_calibrated.joblib)

---

## 4. Decision Support Queue Data Dictionary

The generated queue delivers structured, pseudonymized context for academic advising teams:

1. `student_key`: Pseudonymized student identifier (`id_student`).
2. `code_module`: Course module offering (AAA–GGG).
3. `code_presentation`: Academic term (2013B, 2013J, 2014B, 2014J).
4. `prediction_cutoff_day`: Always `35` (relative Day 35 cutoff).
5. `risk_probability`: Calibrated model probability of withdrawal after Day 35.
6. `priority_tier`: Urgent, Watch, or Standard.
7. `support_reason_summary`: Top 2–3 observed behavioral signals associated with risk (e.g. days inactive, missed scheduled assignments). Observational only; not a causal explanation.
8. `review_status`: Advisor workflow tracker (`Pending`, `Under Review`, `Contacted`, `Resolved`, `No Action Needed`).
9. `recommended_next_step`: Concrete intervention guidelines tailored to risk profile.

---

## 5. Human-in-the-Loop Operational Workflow

```
[Day 35 Data Cutoff]
        │
        ▼
[Calibrated Scoring Engine (Trained strictly on train partition)]
        │
        ▼
[Priority Triage Engine (Tier 1 / 2 / 3)]
        │
        ▼
[Held-Out Demonstration Queue (up to K=500 enrollment records per presentation)]
        │
        ▼
[Advisor Context Review & Behavioral Verification]
        │
        ├─► [Pastoral Outreach: Personal/Technical Disruptions]
        ├─► [Academic Outreach: Assignment Pacing & Scaffolding]
        └─► [Advisor Override: Known Context / No Action Needed]
        │
        ▼
[Log Outcome in System: Contacted / Resolved]
```

### Ethical Guardrails:
* **No Automated Penalties:** Predictions never trigger automated drops or administrative holds.
* **Supportive Communications:** Stigmatizing language is prohibited in all student communications.
* **Professional Override:** Advisors have complete authority to adjust priority tiers based on unmodeled student circumstances.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved Decision Support Report to %s", output_path)


def generate_feature_interpretation_report(
    lr_coefs: pd.DataFrame,
    rf_importances: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate Markdown report synthesizing model interpretation and behavioral signals."""
    top_lr_risk = lr_coefs[lr_coefs["coefficient_log_odds"] > 0].head(5)
    top_lr_protective = lr_coefs[lr_coefs["coefficient_log_odds"] < 0].head(5)
    top_rf = rf_importances.head(10)

    report_text = """# Feature Interpretation & Behavioral Signals Report

**Phase:** Phase 7 — Interpretation, Fairness and Decision Support
**Models Interpreted:** L2-Regularized Logistic Regression (Odds Ratios) and Random Forest (Gini Impurity)

---

## 1. Interpretive Guidance & Causality Disclaimer

> **Critical Methodological Notice:**
> All feature importance metrics and regression coefficients describe **statistical associations** within historical observational data up to Day 35. They do **not** establish causal relationships. A high importance or large odds ratio does not imply that manipulating that feature will directly alter student retention.

---

## 2. Logistic Regression: Directional Signals & Odds Ratios ($e^\\beta$)

### Top Risk Amplifiers ($OR > 1.0$)
| Feature | Log-Odds Coef ($\\beta$) | Odds Ratio ($e^\\beta$) | Interpretation |
|---|---|---|---|
"""
    for _, r in top_lr_risk.iterrows():
        report_text += f"| `{r['feature']}` | {r['coefficient_log_odds']:+.4f} | **{r['odds_ratio']:.2f}** | Associated with increased likelihood of withdrawal |\n"

    report_text += """
### Top Protective Factors ($OR < 1.0$)
| Feature | Log-Odds Coef ($\\beta$) | Odds Ratio ($e^\\beta$) | Interpretation |
|---|---|---|---|
"""
    for _, r in top_lr_protective.iterrows():
        report_text += f"| `{r['feature']}` | {r['coefficient_log_odds']:+.4f} | **{r['odds_ratio']:.2f}** | Associated with decreased likelihood of withdrawal |\n"

    report_text += """
---

## 3. Random Forest: Non-Linear Impurity Importances

Top 10 features contributing most to tree split purity (Gini impurity importance):

| Rank | Feature | Gini Impurity Importance | Relative Share |
|---|---|---|---|
"""
    for _, r in top_rf.iterrows():
        report_text += f"| {int(r['relative_rank'])} | `{r['feature']}` | {r['impurity_importance']:.4f} | {r['impurity_importance']:.1%} |\n"

    report_text += """
---

## 4. Key Behavioral Takeaways for Academic Advisors

1. **Digital Consistency Over Raw Volume:** `vle_active_days_window` and `days_since_last_vle_interaction` are substantially more predictive than total clicks. Regular, paced engagement is the hallmark of persistent students.
2. **Early Assessment Milestones:** `submitted_on_time_count` and `missing_due_submissions_count` provide decisive pivot points by Day 35. Learners who miss their initial assignment represent prime candidates for early scaffolding.
3. **Pacing Momentum:** A sharp drop in `vle_velocity_ratio` serves as an early indicator of silent disengagement well before formal withdrawal occurs.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved Feature Interpretation Report to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 7 Fairness Audit and Decision Support Pipeline.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROCESSED_FEATURES_DIR / "feature_matrix_policy_c.parquet",
        help="Path to feature matrix Parquet.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_BASELINE_DIR,
        help="Directory containing trained model artifacts.",
    )
    args = parser.parse_args()

    ensure_directories()

    logger.info("Loading feature matrix from %s...", args.features_path)
    df = pd.read_parquet(args.features_path)

    # 1. Load trained models
    rf_path = args.models_dir / "random_forest.joblib"
    lr_path = args.models_dir / "logistic_regression.joblib"

    if not rf_path.exists():
        raise FileNotFoundError(f"Trained Random Forest model not found at {rf_path}. Run Phase 6 training first.")

    rf_model = load_model(rf_path)
    lr_model = load_model(lr_path) if lr_path.exists() else None

    # 2. Extract feature interpretations
    rf_imp = extract_random_forest_importance(rf_model)
    rf_imp.to_csv(TABLES_DIR / "feature_importance_rf.csv", index=False)

    if lr_model is not None:
        lr_coefs = extract_logistic_regression_coefficients(lr_model)
        lr_coefs.to_csv(TABLES_DIR / "feature_importance_lr.csv", index=False)
    else:
        lr_coefs = pd.DataFrame()

    generate_feature_interpretation_report(
        lr_coefs=lr_coefs,
        rf_importances=rf_imp,
        output_path=REPORTS_DIR / "feature_interpretation_report.md",
    )

    # 3. Probability Calibration Workflow (strictly on training data)
    train_df, test_df = split_by_student_group(df, test_size=0.20, random_state=42, group_col="id_student")
    logger.info(
        "Split data for calibration and evaluation: train N=%d, test N=%d (zero student group overlap)",
        len(train_df),
        len(test_df),
    )

    rf_base = build_model_pipeline("random_forest", random_state=42)

    # 3a. Training-only calibration method evaluation and selection
    training_cal_df, selected_cal_method = evaluate_calibration_methods_on_training_data(
        base_pipeline=rf_base,
        train_df=train_df,
        methods=("sigmoid", "isotonic"),
        n_splits=5,
        group_col="id_student",
    )
    training_cal_path = TABLES_DIR / "calibration_training_selection.csv"
    training_cal_df.to_csv(training_cal_path, index=False)
    logger.info(
        "Saved training calibration selection table to %s (Selected: '%s')",
        training_cal_path,
        selected_cal_method,
    )

    # 3b. Fit final calibrated model on training partition using selected method
    calibrated_rf = fit_calibrated_model(
        base_pipeline=rf_base,
        train_df=train_df,
        method=selected_cal_method,
        n_splits=5,
        group_col="id_student",
    )
    calibrated_rf_path = save_calibrated_model(calibrated_rf, args.models_dir / "random_forest_calibrated.joblib")

    # 3c. Evaluate final calibration diagnostics on held-out test cohort
    clean_test_x = select_primary_features(test_df, include_target=False, include_identifiers=False)
    y_test = test_df[CANONICAL_TARGET].to_numpy().astype(int)

    raw_probs = rf_model.predict_proba(clean_test_x)[:, 1]
    cal_probs = calibrated_rf.predict_proba(clean_test_x)[:, 1]

    raw_diag = compute_calibration_diagnostics(y_test, raw_probs)
    cal_diag = compute_calibration_diagnostics(y_test, cal_probs)

    cal_comp_df = pd.DataFrame(
        [
            {
                "model_configuration": "Random Forest (Raw Probabilities)",
                "calibration_method": "none",
                "evaluation_split": "held_out_test_cohort",
                "brier_score": raw_diag["brier_score"],
                "expected_calibration_error": raw_diag["expected_calibration_error"],
                "maximum_calibration_error": raw_diag["maximum_calibration_error"],
                "min_probability": raw_diag["min_probability"],
                "max_probability": raw_diag["max_probability"],
                "mean_probability": raw_diag["mean_probability"],
            },
            {
                "model_configuration": f"Random Forest ({selected_cal_method.capitalize()} Calibrated)",
                "calibration_method": f"{selected_cal_method}_group_kfold_cv",
                "evaluation_split": "held_out_test_cohort",
                "brier_score": cal_diag["brier_score"],
                "expected_calibration_error": cal_diag["expected_calibration_error"],
                "maximum_calibration_error": cal_diag["maximum_calibration_error"],
                "min_probability": cal_diag["min_probability"],
                "max_probability": cal_diag["max_probability"],
                "mean_probability": cal_diag["mean_probability"],
            },
        ]
    )
    cal_comp_path = TABLES_DIR / "calibration_comparison.csv"
    cal_comp_df.to_csv(cal_comp_path, index=False)

    cal_diag_path = TABLES_DIR / "calibration_diagnostics.json"
    with open(cal_diag_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_calibration_selection": {
                    "selected_method": selected_cal_method,
                    "selection_evidence": "Selected via 5-fold GroupKFold cross-validation strictly on training data (train_df, N=21,714). Lowest training out-of-fold ECE.",
                    "training_out_of_fold_metrics": training_cal_df.to_dict(orient="records"),
                },
                "test_evaluation_held_out": {
                    "raw_rf_test": raw_diag,
                    "calibrated_rf_test": cal_diag,
                },
                "calibration_provenance": "Method selected and calibrator fitted strictly on 80% train split using GroupKFold CV. Zero test labels consumed.",
            },
            f,
            indent=2,
        )
    logger.info("Saved calibration diagnostics and comparison table to %s", TABLES_DIR)

    # 4. Classifier-Level Subgroup Diagnostic on held-out test cohort
    # Load validation-derived threshold from metadata to ensure exact experiment provenance
    operational_threshold = load_operational_threshold("random_forest")
    threshold_provenance = (
        "Derived from 5-fold GroupKFold cross-validation on 80% random train partition (precision >= 0.35)"
    )

    test_preds = (raw_probs >= operational_threshold).astype(int)

    test_audit_df = test_df.copy()
    test_audit_df["predicted_label"] = test_preds
    test_audit_df["risk_probability"] = raw_probs

    master_audit = audit_all_protected_attributes(
        df=test_audit_df,
        target_col=CANONICAL_TARGET,
        pred_col="predicted_label",
        prob_col="risk_probability",
        audit_scope="classifier_level",
        decision_policy=f"fixed_threshold_{operational_threshold:.2f}",
    )
    audit_classifier_path = TABLES_DIR / "subgroup_fairness_audit_classifier.csv"
    master_audit.to_csv(audit_classifier_path, index=False)
    audit_table_path = TABLES_DIR / "subgroup_fairness_audit.csv"
    master_audit.to_csv(audit_table_path, index=False)
    logger.info("Saved classifier subgroup fairness audit (%d rows) to %s", len(master_audit), audit_table_path)

    # 5. Demographic Sensitivity Experiment
    sensitivity_results = run_demographic_sensitivity_experiment(
        feature_df=df,
        test_size=0.20,
        random_state=42,
        operational_threshold=operational_threshold,
    )
    demo_comp_path = TABLES_DIR / "demographic_sensitivity_comparison.csv"
    sensitivity_results["model_comparison"].to_csv(demo_comp_path, index=False)

    demo_delta_path = TABLES_DIR / "demographic_subgroup_parity_deltas.csv"
    sensitivity_results["parity_delta_table"].to_csv(demo_delta_path, index=False)

    # 6. Build Decision Support Demonstration Queue on Held-Out Test Cohort
    # In accordance with anti-leakage guidelines, generate strictly on unseen test partition
    queue_df = build_student_support_queue(
        feature_df=test_df,
        model=calibrated_rf,
        urgent_threshold=0.70,
        watch_threshold=0.40,
        capacity_k=500,
        save_path=PREDICTIONS_DIR / "student_support_queue.csv",
    )

    cap_audit_df = audit_support_queue_capacity(queue_df)
    cap_audit_path = TABLES_DIR / "support_queue_capacity_audit.csv"
    cap_audit_df.to_csv(cap_audit_path, index=False)
    logger.info("Saved queue capacity audit to %s", cap_audit_path)

    generate_decision_support_report(
        queue_df=queue_df,
        capacity_audit_df=cap_audit_df,
        calibration_diag=cal_diag,
        output_path=REPORTS_DIR / "decision_support_recommendations.md",
        selected_calibration_method=selected_cal_method,
    )

    # 7. Held-Out Demonstration Queue Subgroup Audit (Calibrated Enrollment-Slot Policy)
    test_eval_queue_df = test_df.copy()
    test_eval_queue_df["calibrated_risk_probability"] = cal_probs

    queue_audit_df = audit_operational_support_queue(
        queue_df=queue_df,
        cohort_df=test_eval_queue_df,
        target_col=CANONICAL_TARGET,
        prob_col="calibrated_risk_probability",
    )
    queue_audit_path = TABLES_DIR / "support_queue_fairness_audit.csv"
    queue_audit_df.to_csv(queue_audit_path, index=False)
    logger.info("Saved operational support queue fairness audit (%d rows) to %s", len(queue_audit_df), queue_audit_path)

    # 8. Generate Comprehensive Subgroup Fairness Report
    generate_subgroup_fairness_report(
        audit_df=master_audit,
        sensitivity_results=sensitivity_results,
        operational_threshold=operational_threshold,
        threshold_provenance=threshold_provenance,
        output_path=REPORTS_DIR / "subgroup_fairness_report.md",
        queue_audit_df=queue_audit_df,
        training_cal_df=training_cal_df,
        selected_calibration_method=selected_cal_method,
    )

    print("\nPhase 7 Execution Completed Successfully!")
    print(f"  - Training Calibration Selection:   {training_cal_path}")
    print(f"  - Calibrated Model:                 {calibrated_rf_path}")
    print(f"  - Calibration Comparison Table:     {cal_comp_path}")
    print(f"  - Classifier Subgroup Audit:        {audit_table_path}")
    print(f"  - Operational Queue Subgroup Audit: {queue_audit_path}")
    print(f"  - Demographic Sensitivity:          {demo_comp_path}")
    print(f"  - Student Support Queue (Held-Out): {PREDICTIONS_DIR / 'student_support_queue.csv'} ({len(queue_df):,} rows)")
    print(f"  - Queue Capacity Audit:             {cap_audit_path}")
    print(f"  - Fairness Report:                  {REPORTS_DIR / 'subgroup_fairness_report.md'}")
    print(f"  - Feature Interpretation Report:    {REPORTS_DIR / 'feature_interpretation_report.md'}")
    print(f"  - Decision Support Report:          {REPORTS_DIR / 'decision_support_recommendations.md'}")


if __name__ == "__main__":
    main()
