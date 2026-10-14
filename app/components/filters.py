"""Sidebar and header filter components for the Streamlit dashboard."""

from typing import Any

import streamlit as st


def render_scope_sidebar(cohort_size: int | None = None) -> None:
    """Render persistent data scope and governance information in the sidebar."""
    st.sidebar.markdown("### 📌 System Scope & Governance")
    cohort_str = f", $N={cohort_size:,}$" if cohort_size is not None else ""
    st.sidebar.info(
        "**Prediction Horizon:** Day 35 (Week 5 Cutoff)\n\n"
        f"**Cohort Policy:** Policy C (Strict Clean-Data{cohort_str})\n\n"
        "**Dataset:** Open University Learning Analytics Dataset (OULAD)\n\n"
        "**Evaluation Design:** Group-disjoint student split (Zero leakage)\n\n"
        "**Notice:** Retrospective research demonstration. Algorithmic scores must support human advising and never trigger punitive administrative actions."
    )


def render_queue_filters(
    available_modules: list[str],
    available_presentations: list[str],
    available_tiers: list[str],
) -> dict[str, Any]:
    """Render interactive filters for the Student Support Queue page."""
    st.sidebar.markdown("### 🔍 Support Queue Filters")

    sel_presentations = st.sidebar.multiselect(
        "Academic Presentation (Term)",
        options=available_presentations,
        default=available_presentations,
        help="Select one or more academic presentation terms (e.g., 2014B, 2014J).",
    )

    sel_modules = st.sidebar.multiselect(
        "Course Module",
        options=available_modules,
        default=available_modules,
        help="Select course modules (AAA - GGG).",
    )

    sel_tiers = st.sidebar.multiselect(
        "Priority Tier",
        options=available_tiers,
        default=available_tiers,
        help="Filter by triage urgency tier.",
    )

    search_query = st.sidebar.text_input(
        "Search Student Key / ID",
        value="",
        help="Filter table by pseudonymized student identifier.",
    ).strip()

    sort_order = st.sidebar.radio(
        "Sort Queue By",
        options=["Risk Probability (Highest First)", "Student Key", "Priority Tier"],
        index=0,
    )

    max_rows = st.sidebar.slider(
        "Display Max Rows",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        help="Number of triage cases to display on page.",
    )

    return {
        "presentations": sel_presentations,
        "modules": sel_modules,
        "tiers": sel_tiers,
        "search_query": search_query,
        "sort_order": sort_order,
        "max_rows": max_rows,
    }
