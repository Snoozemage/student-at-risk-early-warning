"""Main entry point for the Smart Student Retention Early Warning Streamlit dashboard."""

import sys
from pathlib import Path

# Ensure project root and src are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import streamlit as st

from app.components.filters import render_queue_filters, render_scope_sidebar
from app.components.model_metrics import (
    render_engagement_patterns,
    render_fairness_and_limitations,
    render_model_performance,
)
from app.components.risk_summary import render_operational_overview
from app.components.student_queue import render_student_support_queue
from student_retention.paths import (
    get_predictions_dir,
)
from student_retention.reporting.tables import load_executive_kpis


def apply_custom_styling() -> None:
    """Inject modern styling and CSS tweaks for a clean dashboard look."""
    st.markdown(
        """
        <style>
        /* General page layout improvements */
        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }
        /* Metric card styling */
        [data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            color: #0F172A;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.90rem !important;
            font-weight: 600 !important;
            color: #475569;
        }
        /* Alert box styling */
        .stAlert {
            border-radius: 6px;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Streamlit application main dispatcher."""
    st.set_page_config(
        page_title="Smart Student Retention Early Warning System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styling()

    # Sidebar Header & Navigation
    st.sidebar.title("🎓 Student Retention")
    st.sidebar.caption("Day-35 Early Warning & Advising Triage")

    pages = [
        "Operational Overview",
        "Engagement Patterns",
        "Student Support Queue",
        "Model Performance",
        "Fairness and Limitations",
    ]

    selected_page = st.sidebar.radio(
        "Navigate Dashboard Pages:",
        options=pages,
        index=0,
    )

    st.sidebar.markdown("---")

    # If on Support Queue page, load unique values and render interactive filters
    queue_filters = {}
    if selected_page == "Student Support Queue":
        queue_csv = get_predictions_dir() / "student_support_queue.csv"
        if queue_csv.exists():
            try:
                df_q = pd.read_csv(queue_csv)
                modules = sorted(df_q["code_module"].unique().tolist()) if "code_module" in df_q.columns else []
                presentations = sorted(df_q["code_presentation"].unique().tolist()) if "code_presentation" in df_q.columns else []
                tiers = sorted(df_q["priority_tier"].unique().tolist()) if "priority_tier" in df_q.columns else []
            except (OSError, pd.errors.EmptyDataError, ValueError, KeyError):
                modules, presentations, tiers = [], [], []
        else:
            modules = []
            presentations = []
            tiers = []

        queue_filters = render_queue_filters(
            available_modules=modules,
            available_presentations=presentations,
            available_tiers=tiers,
        )

    # Persistent governance scope in sidebar with dynamic cohort size
    kpis = load_executive_kpis()
    render_scope_sidebar(cohort_size=kpis.get("cohort_size"))

    # Dispatch to appropriate page component
    if selected_page == "Operational Overview":
        render_operational_overview()
    elif selected_page == "Engagement Patterns":
        render_engagement_patterns()
    elif selected_page == "Student Support Queue":
        render_student_support_queue(queue_filters)
    elif selected_page == "Model Performance":
        render_model_performance()
    elif selected_page == "Fairness and Limitations":
        render_fairness_and_limitations()


if __name__ == "__main__":
    main()
