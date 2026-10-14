"""Reporting, visualization, and executive deliverables generation package."""

from student_retention.reporting.charts import (
    create_subgroup_disparity_chart,
    create_tier_distribution_chart,
    create_top_k_enrichment_chart,
    create_weekly_activity_chart,
)
from student_retention.reporting.generate_report import (
    generate_all_deliverables,
    generate_executive_summary,
    generate_presentation_outline,
    generate_recommendations,
    generate_technical_report,
)
from student_retention.reporting.tables import (
    format_model_comparison_table,
    format_subgroup_fairness_table,
    format_top_k_table,
    load_calibration_diagnostics,
    load_executive_kpis,
    load_model_training_summary,
)

__all__ = [
    "create_subgroup_disparity_chart",
    "create_tier_distribution_chart",
    "create_top_k_enrichment_chart",
    "create_weekly_activity_chart",
    "format_model_comparison_table",
    "format_subgroup_fairness_table",
    "format_top_k_table",
    "generate_all_deliverables",
    "generate_executive_summary",
    "generate_presentation_outline",
    "generate_recommendations",
    "generate_technical_report",
    "load_calibration_diagnostics",
    "load_executive_kpis",
    "load_model_training_summary",
]
