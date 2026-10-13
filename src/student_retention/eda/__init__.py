"""Exploratory Data Analysis (EDA) module for student retention early warning system."""
from student_retention.eda.assessment_analysis import (
    extract_cutoff_assessment_features,
    summarize_assessment_patterns,
)
from student_retention.eda.cohort_analysis import (
    compare_policy_sensitivities,
    summarize_cohort_demographics,
    summarize_presentation_withdrawal_rates,
)
from student_retention.eda.eda_runner import EDARunner
from student_retention.eda.engagement_analysis import (
    compute_weekly_vle_trajectory,
    extract_cutoff_vle_features,
)
from student_retention.eda.statistical_tests import (
    compare_missingness_by_outcome,
    compare_numeric_distributions,
    compute_cliffs_delta,
    compute_cohens_d,
    compute_odds_ratio,
)
from student_retention.eda.visualizations import (
    plot_assessment_submission_progress,
    plot_correlation_matrix,
    plot_presentation_withdrawal_rates,
    plot_registration_lead_time,
    plot_vle_clicks_distribution,
    plot_weekly_activity_curves,
)

__all__ = [
    "EDARunner",
    "compare_missingness_by_outcome",
    "compare_numeric_distributions",
    "compare_policy_sensitivities",
    "compute_cliffs_delta",
    "compute_cohens_d",
    "compute_odds_ratio",
    "compute_weekly_vle_trajectory",
    "extract_cutoff_assessment_features",
    "extract_cutoff_vle_features",
    "plot_assessment_submission_progress",
    "plot_correlation_matrix",
    "plot_presentation_withdrawal_rates",
    "plot_registration_lead_time",
    "plot_vle_clicks_distribution",
    "plot_weekly_activity_curves",
    "summarize_assessment_patterns",
    "summarize_cohort_demographics",
    "summarize_presentation_withdrawal_rates",
]
