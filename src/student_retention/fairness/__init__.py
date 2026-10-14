"""Subgroup fairness audit and demographic sensitivity analysis package."""

from student_retention.fairness.subgroup_audit import (
    audit_all_protected_attributes,
    audit_operational_support_queue,
    audit_subgroup_metrics,
    run_demographic_sensitivity_experiment,
)

__all__ = [
    "audit_all_protected_attributes",
    "audit_operational_support_queue",
    "audit_subgroup_metrics",
    "run_demographic_sensitivity_experiment",
]
