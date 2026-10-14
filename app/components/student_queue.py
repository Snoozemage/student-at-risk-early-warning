import math
from pathlib import Path

import pandas as pd
import streamlit as st

from student_retention.paths import get_predictions_dir


def render_student_support_queue(filters: dict, queue_path: Path | None = None) -> None:
    """Render Page 3: Student Support Queue."""
    st.title("📋 Student Support & Advising Triage Queue")
    st.markdown(
        "This interactive queue displays student cases prioritized for academic advising outreach at **relative Day 35**. "
        "Advisors can filter by course offering, review behavioral disengagement drivers, and track intervention status."
    )

    st.warning(
        "⚠️ **Retrospective Demonstration Disclaimer:**  \n"
        "This triage queue is a retrospective demonstration derived from public Open University research data scored "
        "on the held-out test cohort. Algorithmic scores reflect observational patterns and must support—never replace—advisor judgment."
    )

    queue_csv = queue_path or (get_predictions_dir() / "student_support_queue.csv")
    if not queue_csv.exists():
        st.info(
            "ℹ️ **Student Support Queue output is not available.**  \n"
            f"Expected file at `{queue_csv}`. "
            "Please run the decision support pipeline stage (`scripts/07_run_fairness_and_decision_support.py`) to generate the triage queue."
        )
        return

    try:
        df = pd.read_csv(queue_csv)
    except (OSError, pd.errors.EmptyDataError):
        st.info("ℹ️ Student Support Queue file is empty. Please run pipeline Stage 7.")
        return

    if df.empty:
        st.info("ℹ️ Student Support Queue is empty. No cases to display.")
        return

    # Apply filters
    filtered_df = df.copy()

    if filters.get("presentations"):
        filtered_df = filtered_df[filtered_df["code_presentation"].isin(filters["presentations"])]

    if filters.get("modules"):
        filtered_df = filtered_df[filtered_df["code_module"].isin(filters["modules"])]

    if filters.get("tiers"):
        filtered_df = filtered_df[filtered_df["priority_tier"].isin(filters["tiers"])]

    if filters.get("search_query"):
        q = filters["search_query"].lower()
        filtered_df = filtered_df[
            filtered_df["student_key"].astype(str).str.lower().str.contains(q)
            | filtered_df["code_module"].str.lower().str.contains(q)
            | filtered_df["support_reason_summary"].str.lower().str.contains(q)
        ]

    # Apply sorting
    sort_choice = filters.get("sort_order", "Risk Probability (Highest First)")
    if sort_choice == "Risk Probability (Highest First)":
        filtered_df = filtered_df.sort_values(by="risk_probability", ascending=False)
    elif sort_choice == "Student Key":
        filtered_df = filtered_df.sort_values(by="student_key", ascending=True)
    elif sort_choice == "Priority Tier":
        tier_order = {"Tier 1 (Urgent)": 0, "Tier 2 (Watch)": 1, "Tier 3 (Standard)": 2}
        filtered_df["_sort_rank"] = filtered_df["priority_tier"].map(lambda t: tier_order.get(t, 9))
        filtered_df = filtered_df.sort_values(by=["_sort_rank", "risk_probability"], ascending=[True, False]).drop(columns=["_sort_rank"])

    # Summary Metrics for current filter
    total_shown = len(filtered_df)
    urgent_count = (filtered_df["priority_tier"] == "Tier 1 (Urgent)").sum()
    watch_count = (filtered_df["priority_tier"] == "Tier 2 (Watch)").sum()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Filtered Cases", f"{total_shown:,}")
    with m2:
        st.metric("Tier 1 (Urgent) Cases", f"{urgent_count:,}", delta="Proposed 3-day target" if urgent_count > 0 else None)
    with m3:
        st.metric("Tier 2 (Watch) Cases", f"{watch_count:,}", delta="Proposed 7-day target" if watch_count > 0 else None)
    with m4:
        risk_values = pd.to_numeric(filtered_df["risk_probability"], errors="coerce")
        risk_is_valid = total_shown > 0 and risk_values.map(math.isfinite).all()
        avg_risk = risk_values.mean() if risk_is_valid else None
        st.metric("Average Risk Probability", f"{avg_risk:.4f}" if avg_risk is not None else "N/A")

    st.markdown("---")

    # Limit display rows
    max_rows = filters.get("max_rows", 50)
    display_subset = filtered_df.head(max_rows)

    st.markdown(f"### 📑 Caseload Table (Showing Top {len(display_subset)} of {total_shown:,} Cases)")

    # Color format priority tier in dataframe display
    def highlight_tier(val):
        if "Urgent" in str(val):
            return "background-color: #FEE2E2; color: #991B1B; font-weight: bold;"
        elif "Watch" in str(val):
            return "background-color: #FEF3C7; color: #92400E; font-weight: bold;"
        return "background-color: #EFF6FF; color: #1E40AF;"

    table_cols = [
        "student_key",
        "code_module",
        "code_presentation",
        "risk_probability",
        "priority_tier",
        "support_reason_summary",
        "review_status",
    ]
    st.dataframe(
        display_subset[table_cols].style.map(highlight_tier, subset=["priority_tier"]),
        width="stretch",
        height=380,
    )

    st.markdown("---")

    # Detailed Student Case Inspector Drawer
    st.markdown("### 🔍 Student Case Inspector & Intervention Drawer")
    if not display_subset.empty:
        student_keys = display_subset["student_key"].astype(str).tolist()
        selected_student = st.selectbox("Select a Student Case to Inspect Details:", options=student_keys)

        student_row = display_subset[display_subset["student_key"].astype(str) == selected_student].iloc[0]

        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.markdown(f"#### Student: `{student_row['student_key']}`")
            st.markdown(f"**Course Module:** `{student_row['code_module']}` | **Presentation:** `{student_row['code_presentation']}`")
            st.markdown(f"**Calibrated Risk Probability:** `{student_row['risk_probability']:.4f}`")
            st.markdown(f"**Assigned Urgency Tier:** **{student_row['priority_tier']}**")
            st.markdown(f"**Observed Engagement Drivers:**  \n*{student_row['support_reason_summary']}*")

        with c_right:
            st.markdown("#### Recommended Advising Action")
            st.info(f"👉 **Example Support Option (for advisor review):**\n\n{student_row['recommended_next_step']}")

            tier_target = "Proposed 3-day review target" if "Urgent" in student_row["priority_tier"] else ("Proposed 7-day review target" if "Watch" in student_row["priority_tier"] else "No response target set")
            st.markdown(f"**Demonstration response setting (not validated):** `{tier_target}`")

            # Mock status update simulation
            st.markdown("**Advisor Review Status Tracker:**")
            new_status = st.selectbox(
                "Update Case Status:",
                options=["Pending", "Under Review", "Outreach Initiated", "Consultation Completed", "Dismissed (Qualitative Override)"],
                index=0,
                key=f"status_{selected_student}",
            )
            if st.button("Log Status Update (Simulation)"):
                st.success(f"Status for student `{selected_student}` updated to: **{new_status}**.")
    else:
        st.info("No student cases match the selected filter criteria.")
