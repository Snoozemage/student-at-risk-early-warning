"""Model performance, engagement patterns, and fairness audit components for Streamlit."""

import pandas as pd
import streamlit as st

from student_retention.paths import (
    get_figures_dir,
    get_tables_dir,
)
from student_retention.reporting.charts import (
    create_subgroup_disparity_chart,
    create_top_k_enrichment_chart,
    create_weekly_activity_chart,
)
from student_retention.reporting.tables import (
    count_stable_metric_warnings,
    format_model_comparison_table,
    format_subgroup_fairness_table,
    format_top_k_table,
    load_chronological_term_top_k,
    load_executive_kpis,
)


def render_engagement_patterns() -> None:
    """Render Page 2: Engagement Patterns."""
    st.title("📈 Student Learning & Digital Engagement Patterns")
    st.markdown(
        "Empirical investigation of student interaction behaviors within the strictly bounded observation window "
        "up to **relative Day 35**. All metrics compare confirmed completers (Pass/Distinction/Fail) against "
        "students who withdrew strictly after the Day 35 cutoff."
    )

    st.markdown(
        "> 💡 **Observational Association Notice:**  \n"
        "> All relationships documented below represent observational statistical correlations in distance learning. "
        "> Mandating clicks without instructional scaffolding will not improve retention."
    )

    tab1, tab2, tab3 = st.tabs(["📊 Longitudinal Weekly Curves", "🎯 Coursework Milestone Pacing", "⏱️ Registration Horizon"])

    with tab1:
        st.markdown("### Longitudinal Digital Activity Trajectories (Weeks -3 to 5)")
        col_fig, col_text = st.columns([1.2, 1.0])

        with col_fig:
            fig3_path = get_figures_dir() / "fig3_weekly_vle_activity_curves.png"
            if fig3_path.exists():
                st.image(str(fig3_path), caption="Figure 3: Weekly mean VLE clicks across outcome groups up to Day 35.", width="stretch")
            else:
                fig = create_weekly_activity_chart()
                st.pyplot(fig)

        with col_text:
            st.markdown("#### Key Longitudinal Insights:")
            st.markdown(
                r"- **Orientation Indistinguishability:** During pre-course weeks (Weeks -3 to -1), interaction volumes between eventual completers and departing students are virtually identical." "\n"
                r"- **Early Divergence in Week 0:** Activity diverges immediately upon course start and the gap widens through Weeks 2, 3, 4, and 5." "\n"
                r"- **Platform Silence:** Departing students exhibit longer periods of platform inactivity elapsed since last click, indicating early disengagement."
            )

    with tab2:
        st.markdown(r"### Coursework Milestone Pacing (Due $\le$ Day 35)")
        col_fig5, col_text5 = st.columns([1.2, 1.0])

        with col_fig5:
            fig5_path = get_figures_dir() / "fig5_assessment_submission_progress.png"
            if fig5_path.exists():
                st.image(str(fig5_path), caption="Figure 5: Assessment submission rate and missed due milestone proportions.", width="stretch")
            else:
                st.info("Assessment submission progress figure not available.")

        with col_text5:
            st.markdown("#### Decisive Pacing Indicators:")
            ass_path = get_tables_dir() / "assessment_summary_by_outcome.csv"
            if ass_path.exists():
                try:
                    df_ass = pd.read_csv(ass_path)
                    st.dataframe(df_ass, width="stretch")
                    ret_rows = df_ass[df_ass["withdrawn_after_cutoff"] == 0] if "withdrawn_after_cutoff" in df_ass.columns else pd.DataFrame()
                    wdr_rows = df_ass[df_ass["withdrawn_after_cutoff"] == 1] if "withdrawn_after_cutoff" in df_ass.columns else pd.DataFrame()

                    pacing_points = [
                        "- **Candidate Pacing Signal:** Missed due coursework is compared retrospectively with outcome groups; this does not establish intervention effectiveness."
                    ]
                    if "students_with_due" in df_ass.columns:
                        tot_due = df_ass["students_with_due"].sum()
                        pacing_points.insert(0, f"- **Continuous Coursework Due:** Evaluated across {tot_due:,} student records with due assessments by Day 35.")
                    if (
                        not ret_rows.empty
                        and not wdr_rows.empty
                        and "mean_submission_rate" in df_ass.columns
                    ):
                        ret_sub = ret_rows["mean_submission_rate"].iloc[0] * 100
                        wdr_sub = wdr_rows["mean_submission_rate"].iloc[0] * 100
                        pacing_points.append(f"- **Pacing Adherence:** Retained students submit **{ret_sub:.1f}%** of due assessments, compared to **{wdr_sub:.1f}%** for students who eventually withdraw.")
                    st.markdown("\n".join(pacing_points))
                except (OSError, pd.errors.EmptyDataError, ValueError, KeyError):
                    st.info("Unable to parse assessment summary table.")
            else:
                st.info("Assessment milestone summary table not available.")

    with tab3:
        st.markdown("### Registration Timing & Commitment Horizon")
        col_fig4, col_text4 = st.columns([1.2, 1.0])

        with col_fig4:
            fig4_path = get_figures_dir() / "fig4_registration_lead_time_distribution.png"
            if fig4_path.exists():
                st.image(str(fig4_path), caption="Figure 4: Registration lead time distribution by outcome group.", width="stretch")
            else:
                st.info("Registration lead time distribution figure not available.")

        with col_text4:
            st.markdown("#### Commitment Lead Time:")
            stat_path = get_tables_dir() / "statistical_comparisons.csv"
            if stat_path.exists():
                try:
                    df_stat = pd.read_csv(stat_path)
                    reg_row = df_stat[df_stat["feature"] == "date_registration"]
                    if not reg_row.empty and all(
                        pd.notna(reg_row[col].iloc[0]) for col in ["retained_median", "withdrawn_median", "p_value"]
                    ):
                        ret_day = float(reg_row["retained_median"].iloc[0])
                        wdr_day = float(reg_row["withdrawn_median"].iloc[0])
                        p_value = float(reg_row["p_value"].iloc[0])
                        p_text = f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}"
                        st.markdown(
                            f"- Median relative registration date: retained **{ret_day:.1f}** days vs withdrawn **{wdr_day:.1f}** days (Mann-Whitney $p={p_text}$; negative values are before course start).\n"
                            "- This is a continuous retrospective comparison; it does not validate a binary late-registration cutoff or establish that registration timing causes withdrawal."
                        )
                    else:
                        st.info("Registration lead time data not found in statistical comparisons.")
                except (OSError, pd.errors.EmptyDataError, ValueError, KeyError):
                    st.info("Unable to parse statistical comparisons table.")
            else:
                st.info("Statistical comparisons table not available.")


def render_model_performance() -> None:
    """Render Page 4: Model Performance."""
    st.title("🤖 Model Evaluation, Probability Calibration & Error Diagnostics")
    st.markdown(
        "Comprehensive validation results comparing baseline models, tree ensembles, and probability calibration "
        "on the untouched held-out test cohort."
    )

    tab_eval, tab_cal, tab_topk, tab_err = st.tabs([
        "📋 Test Set Benchmarks",
        "🎯 Probability Calibration",
        "📈 Advisor Capacity (Top-K)",
        "🔍 Error Case Diagnostics",
    ])

    with tab_eval:
        st.markdown("### Head-to-Head Model Performance on Held-Out Test Cohort")
        comp_df = format_model_comparison_table()
        if not comp_df.empty:
            st.dataframe(comp_df, width="stretch")
        else:
            st.info("Model comparison test table not available. Please run Stage 6 (scripts/05_train_models.py).")

        st.markdown("#### Diagnostic Evaluation Curves")
        c_pr, c_roc = st.columns(2)
        with c_pr:
            pr_path = get_figures_dir() / "pr_curves.png"
            if pr_path.exists():
                st.image(str(pr_path), caption="Precision-Recall Curves across candidate models on held-out test cohort.", width="stretch")
            else:
                st.info("PR curves figure not available.")
        with c_roc:
            roc_path = get_figures_dir() / "roc_curves.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC Curves across candidate models on held-out test cohort.", width="stretch")
            else:
                st.info("ROC curves figure not available.")

    with tab_cal:
        kpis = load_executive_kpis()
        calibration_method = kpis.get("selected_calibration_method")
        method_label = {
            "isotonic": "Isotonic Regression",
            "sigmoid": "Sigmoid (Platt Scaling)",
            "platt": "Sigmoid (Platt Scaling)",
        }.get(calibration_method, "Selected Calibration Method" if calibration_method else "Calibration Method")
        st.markdown(f"### Probability Calibration ({method_label})")
        st.markdown(
            "Raw tree ensembles compress predicted probabilities toward the training base rate. "
            f"The training-only selection metadata identifies **{method_label}** as the selected calibration procedure. "
            "The displayed test metrics are diagnostic and do not establish intervention effectiveness."
        )

        col_cal_fig, col_cal_stats = st.columns([1.2, 1.0])
        with col_cal_fig:
            cal_fig_path = get_figures_dir() / "calibration_curves.png"
            if cal_fig_path.exists():
                st.image(str(cal_fig_path), caption="Reliability Diagram (Calibration Curves)", width="stretch")
            else:
                st.info("Calibration reliability curves figure not available.")

        with col_cal_stats:
            st.markdown("#### Test-Set Calibration Metrics:")
            cal_comp_path = get_tables_dir() / "calibration_comparison.csv"
            if cal_comp_path.exists():
                try:
                    cal_df = pd.read_csv(cal_comp_path)
                    st.dataframe(cal_df, width="stretch")

                    raw_row = cal_df[cal_df["calibration_method"] == "none"]
                    if calibration_method:
                        cal_row = cal_df[
                            cal_df["calibration_method"].astype(str).str.lower().str.startswith(calibration_method)
                        ]
                    else:
                        cal_row = pd.DataFrame()
                    if not raw_row.empty and not cal_row.empty:
                        raw_ece = float(raw_row["expected_calibration_error"].iloc[0])
                        cal_ece = float(cal_row["expected_calibration_error"].iloc[0])
                        raw_mce = float(raw_row["maximum_calibration_error"].iloc[0])
                        cal_mce = float(cal_row["maximum_calibration_error"].iloc[0])

                        if raw_ece > 0:
                            if cal_ece < raw_ece:
                                ece_drop = ((raw_ece - cal_ece) / raw_ece) * 100.0
                                ece_text = f"Dropped from {raw_ece:.4f} to **{cal_ece:.4f} ({ece_drop:.1f}% error reduction)**."
                            elif cal_ece == raw_ece:
                                ece_text = f"Unchanged at **{cal_ece:.4f} (no change)**."
                            else:
                                ece_inc = ((cal_ece - raw_ece) / raw_ece) * 100.0
                                ece_text = f"Increased from {raw_ece:.4f} to **{cal_ece:.4f} (+{ece_inc:.1f}% error increase)**."
                        else:
                            ece_text = f"Raw ECE is 0.0000; calibrated ECE is **{cal_ece:.4f}** (relative percentage change is undefined)."

                        st.markdown(
                            f"- **Expected Calibration Error (ECE):** {ece_text}\n"
                            f"- **Maximum Calibration Error (MCE):** Changed from {raw_mce:.4f} to **{cal_mce:.4f}**."
                        )
                    elif not calibration_method:
                        st.info("Selected calibration method is unavailable in calibration diagnostics metadata.")
                    elif cal_row.empty:
                        st.info(f"No test calibration row matches the selected method: {calibration_method}.")
                except (OSError, pd.errors.EmptyDataError, ValueError, KeyError):
                    st.info("Unable to parse calibration comparison table.")
            else:
                st.info("Calibration comparison table not available. Please run Stage 7.")

    with tab_topk:
        st.markdown("### Top-K Ranking Diagnostics and Chronological Capacity Evaluation")
        st.markdown(
            "Pooled Top-K metrics below use the uncalibrated Random Forest ranking on the 20% random held-out test cohort. "
            "They are diagnostics, not estimates of performance under a 500-student per-term operating capacity. "
            "Chronological 2014 term results are shown separately using the same uncalibrated ranking model."
        )

        col_topk_fig, col_topk_tbl = st.columns([1.2, 1.0])
        with col_topk_fig:
            topk_fig_path = get_figures_dir() / "top_k_gains.png"
            if topk_fig_path.exists():
                st.image(str(topk_fig_path), caption="Cumulative Recall Gains vs Advisor Capacity (K)", width="stretch")
            else:
                topk_raw_path = get_tables_dir() / "top_k_evaluation.csv"
                topk_df = pd.read_csv(topk_raw_path) if topk_raw_path.exists() else pd.DataFrame()
                if not topk_df.empty:
                    st.pyplot(create_top_k_enrichment_chart(topk_df))
                else:
                    st.info("Top-K visualization data not available.")

        with col_topk_tbl:
            st.markdown("#### Pooled Random-Test Diagnostic (not per-term capacity):")
            topk_fmt = format_top_k_table()
            if not topk_fmt.empty:
                display_cols = [c for c in ["k", "precision_at_k_pct", "baseline_pct", "enrichment_fmt", "recall_at_k_pct"] if c in topk_fmt.columns]
                st.dataframe(topk_fmt[display_cols], width="stretch")

                row_500 = topk_fmt[topk_fmt["k"] == 500]
                if not row_500.empty:
                    p500_str = row_500["precision_at_k_pct"].iloc[0] if "precision_at_k_pct" in row_500.columns else "N/A"
                    base_str = row_500["baseline_pct"].iloc[0] if "baseline_pct" in row_500.columns else "baseline"
                    enr_str = row_500["enrichment_fmt"].iloc[0] if "enrichment_fmt" in row_500.columns else "N/A"
                    enrich_claim = (
                        f"- **{enr_str} Enrichment @ K=500:** Precision is **{p500_str}** compared to the {base_str} random baseline."
                        if enr_str != "N/A"
                        else f"- **Precision @ K=500:** Precision is **{p500_str}** compared to the {base_str} random baseline."
                    )
                    st.markdown(
                        f"{enrich_claim}\n"
                    )
            else:
                st.info("Top-K capacity evaluation table not available.")

        st.markdown("#### Chronological 2014 evaluation (per-term, enrollment-record level):")
        term_df = load_chronological_term_top_k()
        if term_df.empty:
            st.info("Chronological per-term K=500 results are unavailable.")
        else:
            columns = {
                "presentation_code": "Presentation",
                "precision_at_k": "Precision@500 (enrollment records)",
                "positives_captured": "Positive records captured",
                "n_samples": "Test enrollment records",
                "unique_students_in_top_k": "Expected unique students selected",
                "duplicate_student_slots": "Expected duplicate slots",
            }
            present = [col for col in columns if col in term_df.columns]
            st.dataframe(term_df[present].rename(columns=columns), width="stretch")
            st.caption(
                "These chronological out-of-time results use uncalibrated Random Forest scores and count enrollment records. "
                "The demonstration support queue uses the separately calibrated model. Student-level capacity remains unresolved because the "
                "project has no approved rule for aggregating a student's risks across concurrent courses."
            )

    with tab_err:
        st.markdown("### Error Analysis Diagnostics (Severe Outliers)")
        st.markdown(
            "Inspecting the most extreme prediction errors on the held-out test cohort provides qualitative insight "
            "into unmodeled factors affecting student retention."
        )

        err_type = st.radio("Select Error Category to Inspect:", ["Severe False Negatives (High Engagement, Withdrew)", "Severe False Positives (Low Engagement, Completed)"])

        if "False Negatives" in err_type:
            fn_path = get_tables_dir() / "error_audit_severe_false_negatives.csv"
            if fn_path.exists():
                fn_df = pd.read_csv(fn_path)
                st.dataframe(fn_df.head(20), width="stretch")
                st.caption("Top severe false negatives: Students with active platform engagement who nevertheless withdrew later in the term (likely personal/life reasons).")
            else:
                st.info("Severe false negatives table not available.")
        else:
            fp_path = get_tables_dir() / "error_audit_severe_false_positives.csv"
            if fp_path.exists():
                fp_df = pd.read_csv(fp_path)
                st.dataframe(fp_df.head(20), width="stretch")
                st.caption("Top severe false positives: Students with low early clicks who nevertheless passed the course (offline studying or catch-up pacing).")
            else:
                st.info("Severe false positives table not available.")


def render_fairness_and_limitations() -> None:
    """Render Page 5: Fairness and Limitations."""
    st.title("⚖️ Subgroup Fairness Auditing & Ethical Limitations")
    st.markdown(
        "Algorithmic fairness audits across protected demographic characteristics "
        "to evaluate decision-support equity and monitor for potential subgroup disparities."
    )

    st.markdown(
        "> 🔒 **Proxy Reliance Notice:**  \n"
        "> Excluding demographic variables from model features does **NOT** prove the absence of proxy reliance or guarantee disparate-impact parity. "
        "> Disparity audits provide operational transparency; observational differences reflect historical signal distributions and must not be interpreted as student deficiencies."
    )

    audit_view = st.radio(
        "Select Fairness Audit Perspective:",
        ["Held-Out Demonstration Queue Audit (Not Treatment Exposure)", "Classifier Diagnostic (Fixed Threshold theta = 0.25)"],
    )

    if "Demonstration Queue" in audit_view:
        audit_path = get_tables_dir() / "support_queue_fairness_audit.csv"
        st.markdown("### Held-Out Demonstration Queue Subgroup Audit (Top-500 Calibrated Allocation)")
        st.markdown(
            "Describes retrospective held-out queue inclusion under a proposed advisor-capacity demonstration ($K=500$ per presentation), "
            "scored with calibrated probabilities. Queue inclusion does not show that a student received support or that an intervention changed outcomes."
        )
    else:
        audit_path = get_tables_dir() / "subgroup_fairness_audit.csv"
        st.markdown("### Classifier-Level Subgroup Diagnostic (Fixed Threshold $\\theta = 0.25$)")
        st.markdown(
            "Evaluates raw classifier threshold sensitivity across held-out test data prior to operational capacity capping."
        )

    if audit_path.exists():
        try:
            raw_audit_df = pd.read_csv(audit_path)
            has_stability = "is_statistically_stable" in raw_audit_df.columns or "is_sample_size_sufficient" in raw_audit_df.columns
            if not has_stability:
                st.warning(
                    "⚠️ **Incomplete Fairness Audit**: Subgroup statistical stability indicators ('is_statistically_stable') "
                    "are missing from this artifact. Findings cannot be verified for statistically stable subgroups."
                )

            formatted_audit_df = format_subgroup_fairness_table(audit_path=audit_path, filter_stable_only=has_stability)

            if not raw_audit_df.empty and "protected_attribute" in raw_audit_df.columns:
                attrs = sorted(raw_audit_df["protected_attribute"].unique().tolist())
                sel_attr = st.selectbox("Select Demographic Characteristic to Visualize:", options=attrs)

                fig_disparity = create_subgroup_disparity_chart(raw_audit_df, attribute=sel_attr)
                st.pyplot(fig_disparity)

                st.markdown(f"#### Complete Audit Metrics for: `{sel_attr}`")
                display_source = formatted_audit_df if not formatted_audit_df.empty else raw_audit_df
                attr_table = display_source[display_source["protected_attribute"] == sel_attr]
                display_cols = [c for c in ["subgroup", "sample_size", "base_rate_pct", "selection_rate_pct", "recall_pct", "precision_pct", "di_ratio_fmt", "disparate_impact_warning", "elevated_fpr_warning"] if c in attr_table.columns]
                st.dataframe(attr_table[display_cols], width="stretch")

                st.markdown("---")
                st.markdown("### 🔍 Summary of Parity Warnings Across Subgroups")

                w_col1, w_col2 = st.columns(2)
                if not has_stability:
                    with w_col1:
                        st.metric("Disparate Impact Warnings (Stable Subgroups)", "Incomplete")
                        st.info("Subgroup stability unverified ('is_statistically_stable' missing).")
                    with w_col2:
                        st.metric("Elevated FPR Warnings (Stable Subgroups)", "Incomplete")
                        st.info("Subgroup stability unverified ('is_statistically_stable' missing).")
                else:
                    stability_col = "is_statistically_stable" if "is_statistically_stable" in raw_audit_df.columns else "is_sample_size_sufficient"
                    stable_df = raw_audit_df[raw_audit_df[stability_col].astype(str).str.lower().isin(["true", "1"])]

                    di_warns = None
                    di_count = count_stable_metric_warnings(
                        raw_audit_df, "disparate_impact_ratio", lower=0.80, upper=1.25, stability_col=stability_col
                    )
                    if di_count is not None:
                        di_values = pd.to_numeric(stable_df["disparate_impact_ratio"], errors="coerce")
                        di_warns = stable_df[(di_values < 0.80) | (di_values > 1.25)]

                    fpr_warns = None
                    fpr_count = count_stable_metric_warnings(
                        raw_audit_df, "fpr_ratio", upper=1.50, stability_col=stability_col
                    )
                    if fpr_count is not None:
                        fpr_values = pd.to_numeric(stable_df["fpr_ratio"], errors="coerce")
                        fpr_warns = stable_df[fpr_values > 1.50]

                    with w_col1:
                        if di_warns is not None:
                            st.metric("Disparate Impact Warnings (DIR < 0.80 or > 1.25)", len(di_warns))
                            if not di_warns.empty:
                                disp_cols = [c for c in ["protected_attribute", "subgroup", "selection_rate", "disparate_impact_ratio"] if c in di_warns.columns]
                                st.dataframe(di_warns[disp_cols], width="stretch")
                            else:
                                st.success("Zero disparate impact warnings detected among statistically stable subgroups.")
                        else:
                            st.metric("Disparate Impact Warnings", "Unavailable")
                            st.info("Disparate impact metrics are not available in audit data.")

                    with w_col2:
                        if fpr_warns is not None:
                            st.metric("Elevated False Positive Rate Warnings (FPR Ratio > 1.50)", len(fpr_warns))
                            if not fpr_warns.empty:
                                disp_cols = [c for c in ["protected_attribute", "subgroup", "false_positive_rate", "fpr_ratio"] if c in fpr_warns.columns]
                                st.dataframe(fpr_warns[disp_cols], width="stretch")
                            else:
                                st.success("Zero elevated False Positive Rate warnings detected among statistically stable subgroups.")
                        else:
                            st.metric("Elevated False Positive Rate Warnings", "Unavailable")
                            st.info("False positive rate ratio metrics ('fpr_ratio') are not available in audit data.")

                st.caption("Note: Zero detected empirical warnings does not constitute proof of absolute fairness or absence of harm.")
            else:
                st.info("Fairness audit data is empty.")
        except (OSError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
            st.info(f"Unable to parse subgroup fairness audit data: {e}")
    else:
        st.info(f"Subgroup fairness audit data is not available at {audit_path}. Please run Stage 7.")
