"""Operational Overview dashboard page component."""

import streamlit as st

from student_retention.reporting.charts import create_tier_distribution_chart
from student_retention.reporting.tables import load_executive_kpis


def render_operational_overview() -> None:
    """Render Page 1: Operational Overview."""
    st.title("📊 Operational Overview: Student Retention Early Warning")
    st.markdown(
        "Welcome to the **Smart Student Retention Early Warning System**. This operational decision-support "
        "dashboard provides proactive, transparent, and capacity-constrained triage for academic advisors at "
        "**relative Day 35** (the end of the fifth week of study)."
    )

    kpis = load_executive_kpis()

    # Executive KPI Metric Cards
    st.markdown("### 🎯 Strategic Performance & Cohort Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        cohort_val = f"{kpis['cohort_size']:,}" if kpis.get("cohort_size") is not None else "N/A"
        st.metric(
            label="Eligible Cohort Size",
            value=cohort_val,
            help="Total eligible student course-presentation enrollments under Policy C (Strict Clean-Data).",
        )
    with col2:
        base_val = f"{kpis['base_rate_pct']:.2f}%" if kpis.get("base_rate_pct") is not None else "N/A"
        st.metric(
            label="Withdrawal Base Rate",
            value=base_val,
            delta=None,
            delta_color="off",
            help="Historical post-Day-35 course withdrawal prevalence across the cohort.",
        )
    with col3:
        cutoff_val = f"Day {kpis['cutoff_day']}" if kpis.get("cutoff_day") is not None else "Day 35"
        st.metric(
            label="Prediction Cutoff",
            value=cutoff_val,
            help="Observation window closes strictly at end of relative study Day 35 (first 5 weeks).",
        )
    with col4:
        pr_val = f"{kpis['model_pr_auc']:.4f}" if kpis.get("model_pr_auc") is not None else "N/A"
        pr_lift = kpis.get("pr_auc_lift_pct")
        if pr_lift is not None:
            lift_delta = f"+{pr_lift:.1f}% vs Dummy" if pr_lift >= 0 else f"{pr_lift:.1f}% vs Dummy"
        else:
            lift_delta = None
        st.metric(
            label="Model PR-AUC (Random Forest)",
            value=pr_val,
            delta=lift_delta,
            help="Precision-Recall AUC on held-out unseen test cohort (Random Forest benchmark).",
        )
    with col5:
        enrich_val = f"{kpis['enrichment_at_500']:.2f}x" if kpis.get("enrichment_at_500") is not None else "N/A"
        prec_delta = f"{kpis['precision_at_500'] * 100:.1f}% Prec" if kpis.get("precision_at_500") is not None else None
        st.metric(
            label="Pooled-Test Enrichment @ 500",
            value=enrich_val,
            delta=prec_delta,
            help="Diagnostic ranking enrichment among the pooled random held-out test records; it is not a per-term capacity estimate.",
        )

    st.markdown("---")

    # Priority Tier Distribution and Response Protocol
    col_chart, col_proto = st.columns([1.1, 1.0])

    with col_chart:
        st.markdown("### 📋 Caseload Distribution by Priority Tier")
        st.markdown(
            "Priority tiers in the demonstration queue categorize calibrated scores. The queue is a held-out evaluation artifact, "
            "not a live per-term caseload estimate:"
        )
        if (
            kpis.get("queue_total_records") is not None
            and isinstance(kpis["queue_total_records"], (int, float))
            and kpis["queue_total_records"] > 0
            and kpis.get("tier_1_urgent_count") is not None
            and kpis.get("tier_2_watch_count") is not None
            and kpis.get("tier_3_standard_count") is not None
        ):
            tier_counts = {
                "Tier 1 (Urgent)": kpis["tier_1_urgent_count"],
                "Tier 2 (Watch)": kpis["tier_2_watch_count"],
                "Tier 3 (Standard)": kpis["tier_3_standard_count"],
            }
            fig_tiers = create_tier_distribution_chart(tier_counts)
            st.pyplot(fig_tiers)

            uniq_str = f"{kpis['queue_unique_students']:,}" if kpis.get("queue_unique_students") is not None else "N/A"
            dup_str = f"{kpis['queue_duplicate_slots']:,}" if kpis.get("queue_duplicate_slots") is not None else "N/A"
            st.caption(
                f"**Demonstration Cohort:** {kpis['queue_total_records']:,} slots across {uniq_str} "
                f"unique students ({dup_str} multi-course slots) evaluated on held-out test data."
            )
        else:
            st.info("ℹ️ Queue distribution data is not available. Please run pipeline Stage 7 (scripts/07_run_fairness_and_decision_support.py).")

    with col_proto:
        st.markdown("### ⏱️ Proposed Human-Reviewed Workflow (Demonstration)")
        st.markdown(
            r"""
            These score bands and response targets are proposed demonstration settings, not locally validated thresholds or service levels. The retrospective model evaluation does not establish intervention effectiveness; that requires prospective evaluation.

            | Priority Tier | Proposed Score Band | Example Support Action | Proposed Review Target |
            | :--- | :--- | :--- | :--- |
            | **Tier 1 (Urgent)** | **Proposed score $\ge 70\%$** | Example: personal outreach for advisor review | **Proposed: 3 days** |
            | **Tier 2 (Watch)** | **Proposed score $40\% \le p < 70\%$** | Example: supportive check-in for advisor review | **Proposed: 7 days** |
            | **Tier 3 (Standard)** | **Proposed score $< 40\%$** | Example: routine course updates | No target set |
            """
        )

        st.markdown(
            "> ⚠️ **Institutional Aggregation Notice:**  \n"
            "> The current model evaluates risk at the individual course-enrollment level `(code_module, code_presentation, id_student)`. "
            "> Transitioning to an unduplicated 500-unique-student quota requires university governance approval on multi-course risk aggregation."
        )

    st.markdown("---")

    # Ethical Pillars & Governance Guardrails
    st.markdown("### 🛡️ Ethical Principles & Responsible Use Guidelines")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 1. Supportive, Not Punitive")
        st.write(
            "Algorithmic early-warning scores must **NEVER** be used to automate course deregistration, financial aid denial, "
            "or academic probation. All system outputs exist exclusively to deliver academic scaffolding and pastoral support."
        )

    with c2:
        st.markdown("#### 2. Human-in-the-Loop Discretion")
        st.write(
            "Academic advisors maintain **100% discretion** to override, adjust, or dismiss automated risk flags based on "
            "qualitative context, personal correspondence, and unmodeled student circumstances."
        )

    with c3:
        st.markdown("#### 3. Stigma-Free Communication")
        st.write(
            "Outreach to students must focus on available learning resources, extension policies, and tutoring services. "
            "Advisors and templates must never inform students they were flagged as 'high risk' by an algorithm."
        )
