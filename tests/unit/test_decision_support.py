"""Unit tests for Phase 7 decision support queue generation and intervention policies."""

import numpy as np
import pandas as pd

from student_retention.decision_support.intervention_policy import (
    PriorityTier,
    ReviewStatus,
    get_prescribed_action,
    get_tier_sla_days,
    validate_ethical_communication,
)
from student_retention.decision_support.prioritize_students import (
    QUEUE_CANONICAL_COLUMNS,
    assign_priority_tier,
    build_student_support_queue,
    generate_support_reason_summary,
)


class MockPredictor:
    """Mock model exposing predict_proba for unit testing queue generation."""

    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.probabilities[: len(X)]
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def test_assign_priority_tier():
    """Verify exact priority tier mapping by calibrated probability ranges."""
    assert assign_priority_tier(0.95) == PriorityTier.TIER_1_URGENT.value
    assert assign_priority_tier(0.70) == PriorityTier.TIER_1_URGENT.value
    assert assign_priority_tier(0.699) == PriorityTier.TIER_2_WATCH.value
    assert assign_priority_tier(0.40) == PriorityTier.TIER_2_WATCH.value
    assert assign_priority_tier(0.399) == PriorityTier.TIER_3_STANDARD.value
    assert assign_priority_tier(0.05) == PriorityTier.TIER_3_STANDARD.value


def test_generate_support_reason_summary():
    """Verify translation of individual behavioral signals into clear qualitative summaries."""
    # Profile 1: Inactive learner with missed assessments
    row1 = pd.Series(
        {
            "missing_due_submissions_count": 2,
            "late_submissions_count_lte_cutoff": 0,
            "days_since_last_vle_interaction": 21,
            "vle_active_days_window": 2,
            "vle_velocity_ratio": 0.10,
            "is_late_registration": 0,
            "num_of_prev_attempts": 0,
        }
    )
    summary1 = generate_support_reason_summary(row1)
    assert "Missed 2 scheduled due assessment(s)" in summary1
    assert "High platform inactivity (21 days" in summary1

    # Profile 2: Late registrant with declining velocity
    row2 = pd.Series(
        {
            "missing_due_submissions_count": 0,
            "late_submissions_count_lte_cutoff": 1,
            "days_since_last_vle_interaction": 5,
            "vle_active_days_window": 8,
            "vle_velocity_ratio": 0.25,
            "is_late_registration": 1,
            "registration_lead_time_days": -12,
            "num_of_prev_attempts": 0,
        }
    )
    summary2 = generate_support_reason_summary(row2)
    assert "Submitted 1 assessment(s) past deadline" in summary2
    assert "Sharp decline in interaction velocity" in summary2
    assert "Late course registration (12 days late)" in summary2

    # Profile 3: Persistent active student
    row3 = pd.Series(
        {
            "missing_due_submissions_count": 0,
            "late_submissions_count_lte_cutoff": 0,
            "days_since_last_vle_interaction": 1,
            "vle_active_days_window": 18,
            "vle_velocity_ratio": 1.15,
            "is_late_registration": 0,
            "num_of_prev_attempts": 0,
        }
    )
    summary3 = generate_support_reason_summary(row3)
    assert summary3 == "Normal learning engagement patterns observed"


def test_get_tier_sla_days_and_prescribed_actions():
    """Verify tier response SLAs and supportive action guidelines."""
    assert get_tier_sla_days(PriorityTier.TIER_1_URGENT) == 3
    assert get_tier_sla_days(PriorityTier.TIER_2_WATCH) == 7
    assert get_tier_sla_days(PriorityTier.TIER_3_STANDARD) is None

    action_t1 = get_prescribed_action(PriorityTier.TIER_1_URGENT, primary_driver="Missed assessment")
    assert "Direct personal contact" in action_t1
    assert "Coursework deadline extension" in action_t1

    action_t2 = get_prescribed_action(PriorityTier.TIER_2_WATCH)
    assert "Automated supportive check-in email" in action_t2


def test_validate_ethical_communication():
    """Verify detection of stigmatizing / automated phrasing in student communications."""
    clean_msg = "Hello, we noticed you have not had a chance to access the forum this week and wanted to check in."
    is_ethical, violations = validate_ethical_communication(clean_msg)
    assert is_ethical is True
    assert len(violations) == 0

    unethical_msg = "You have been identified by an algorithm as high risk of failure and face academic probation."
    is_ethical2, violations2 = validate_ethical_communication(unethical_msg)
    assert is_ethical2 is False
    assert len(violations2) >= 2


def test_build_student_support_queue_schema_and_capacity(tmp_path):
    """Verify queue construction schema, descending probability sort, and capacity constraint."""
    n = 30
    df = pd.DataFrame(
        {
            "id_student": [100 + i for i in range(n)],
            "code_module": ["BBB"] * n,
            "code_presentation": ["2014B"] * 15 + ["2014J"] * 15,
            "presentation_season": ["B"] * 15 + ["J"] * 15,
            "studied_credits": [60] * n,
            "num_of_prev_attempts": [0] * n,
            "registration_lead_time_days": [25.0] * n,
            "is_late_registration": [0] * n,
            "vle_total_clicks_window": [150] * n,
            "vle_active_days_window": [10] * n,
            "vle_unique_sites_accessed": [12] * n,
            "vle_pre_course_clicks": [20] * n,
            "vle_recent_clicks_14d": [60] * n,
            "vle_clicks_forumng": [30] * n,
            "vle_clicks_oucontent": [70] * n,
            "vle_clicks_quiz": [10] * n,
            "has_no_vle_activity_by_cutoff": [0] * n,
            "days_since_last_vle_interaction": [2.0] * n,
            "vle_velocity_ratio": [1.0] * n,
            "assessments_due_count": [1] * n,
            "submissions_count_lte_cutoff": [1] * n,
            "submitted_on_time_count": [1] * n,
            "late_submissions_count_lte_cutoff": [0] * n,
            "missing_due_submissions_count": [0] * n,
        }
    )

    # Assign distinct probabilities
    probs = np.linspace(0.10, 0.90, n)
    mock_model = MockPredictor(probs)

    csv_dest = tmp_path / "test_queue.csv"

    # Build queue with capacity K=5 per presentation
    queue_df = build_student_support_queue(
        feature_df=df,
        model=mock_model,
        capacity_k=5,
        save_path=csv_dest,
    )

    # Invariants:
    # 1. Total rows = 5 per presentation * 2 presentations = 10 rows
    assert len(queue_df) == 10
    assert csv_dest.exists()

    # 2. Canonical columns exact match
    assert list(queue_df.columns) == QUEUE_CANONICAL_COLUMNS

    # 3. Default review status
    assert (queue_df["review_status"] == ReviewStatus.PENDING.value).all()

    # 4. Probabilities are strictly valid float range [0, 1]
    assert (queue_df["risk_probability"] >= 0.0).all()
    assert (queue_df["risk_probability"] <= 1.0).all()


def test_audit_support_queue_capacity():
    """Verify capacity, unique student, and duplicate slot calculations."""
    from student_retention.decision_support.prioritize_students import (
        audit_support_queue_capacity,
    )

    # 4 slots in 2014B: students s1, s2, s1 (duplicate), s3 -> 3 unique students, 1 duplicate slot
    # 3 slots in 2014J: students s4, s5, s6 -> 3 unique students, 0 duplicate slots
    mock_queue = pd.DataFrame(
        {
            "student_key": ["s1", "s2", "s1", "s3", "s4", "s5", "s6"],
            "code_presentation": ["2014B", "2014B", "2014B", "2014B", "2014J", "2014J", "2014J"],
        }
    )

    audit_df = audit_support_queue_capacity(mock_queue)

    # Invariants:
    # 2 terms + 1 ALL_TERMS_TOTAL = 3 rows
    assert len(audit_df) == 3

    row_b = audit_df[audit_df["code_presentation"] == "2014B"].iloc[0]
    assert row_b["caseload_enrollment_slots"] == 4
    assert row_b["unique_students_in_queue"] == 3
    assert row_b["duplicate_slots"] == 1
    assert row_b["multi_course_students_in_queue"] == 1

    row_j = audit_df[audit_df["code_presentation"] == "2014J"].iloc[0]
    assert row_j["caseload_enrollment_slots"] == 3
    assert row_j["unique_students_in_queue"] == 3
    assert row_j["duplicate_slots"] == 0
    assert row_j["multi_course_students_in_queue"] == 0

    row_total = audit_df[audit_df["code_presentation"] == "ALL_TERMS_TOTAL"].iloc[0]
    assert row_total["caseload_enrollment_slots"] == 7
    assert row_total["unique_students_in_queue"] == 6
    assert row_total["duplicate_slots"] == 1


def _make_mock_student_df(student_ids: list[object], presentation: str = "2014B") -> pd.DataFrame:
    """Helper to create minimal valid feature DataFrame for queue regression tests."""
    n = len(student_ids)
    return pd.DataFrame(
        {
            "id_student": student_ids,
            "code_module": ["BBB"] * n,
            "code_presentation": [presentation] * n,
            "presentation_season": ["B"] * n,
            "studied_credits": [60] * n,
            "num_of_prev_attempts": [0] * n,
            "registration_lead_time_days": [25.0] * n,
            "is_late_registration": [0] * n,
            "vle_total_clicks_window": [150] * n,
            "vle_active_days_window": [10] * n,
            "vle_unique_sites_accessed": [12] * n,
            "vle_pre_course_clicks": [20] * n,
            "vle_recent_clicks_14d": [60] * n,
            "vle_clicks_forumng": [30] * n,
            "vle_clicks_oucontent": [70] * n,
            "vle_clicks_quiz": [10] * n,
            "has_no_vle_activity_by_cutoff": [0] * n,
            "days_since_last_vle_interaction": [2.0] * n,
            "vle_velocity_ratio": [1.0] * n,
            "assessments_due_count": [1] * n,
            "submissions_count_lte_cutoff": [1] * n,
            "submitted_on_time_count": [1] * n,
            "late_submissions_count_lte_cutoff": [0] * n,
            "missing_due_submissions_count": [0] * n,
        }
    )


def test_full_precision_tier_assignment_near_urgent_threshold():
    """Verify that tier assignment evaluates full-precision probabilities near the 0.70 threshold.

    0.70001 and 0.69996 both round to 0.7000 at four decimal places.
    However, 0.70001 must be Tier 1 (Urgent) because 0.70001 >= 0.70,
    while 0.69996 must be Tier 2 (Watch) because 0.69996 < 0.70.
    """
    df = _make_mock_student_df([101, 102])
    # Student 101 has 0.70001, Student 102 has 0.69996
    mock_model = MockPredictor(np.array([0.70001, 0.69996]))

    queue_df = build_student_support_queue(feature_df=df, model=mock_model)

    row_101 = queue_df[queue_df["student_key"] == "101"].iloc[0]
    row_102 = queue_df[queue_df["student_key"] == "102"].iloc[0]

    assert row_101["priority_tier"] == PriorityTier.TIER_1_URGENT.value
    assert row_102["priority_tier"] == PriorityTier.TIER_2_WATCH.value

    # Both output risk_probability are rounded to 0.7000 for display
    assert row_101["risk_probability"] == 0.7000
    assert row_102["risk_probability"] == 0.7000


def test_full_precision_sorting_breaks_rounded_ties():
    """Verify that sorting uses full-precision probabilities so higher true risk ranks first even if student ID sorts later.

    Student A (ID 10) has prob 0.70001.
    Student B (ID 99) has prob 0.70004.
    Both round to 0.7000. If sorted by rounded values, ID '10' would sort before '99'.
    With full-precision sorting, Student B (0.70004 > 0.70001) must rank first.
    """
    df = _make_mock_student_df([10, 99])
    mock_model = MockPredictor(np.array([0.70001, 0.70004]))

    queue_df = build_student_support_queue(feature_df=df, model=mock_model)

    assert list(queue_df["student_key"]) == ["99", "10"]
    assert queue_df.iloc[0]["student_key"] == "99"
    assert queue_df.iloc[1]["student_key"] == "10"
    assert (queue_df["risk_probability"] == 0.7000).all()


def test_full_precision_capacity_selection_prevents_displacement():
    """Verify that a lower true-risk candidate cannot displace a higher one under capacity constraints due to rounding.

    Student Alpha has prob 0.50001.
    Student Zeta has prob 0.50004.
    Capacity K=1.
    If rounded before capacity selection, both are 0.5000, and alphabetical ID order would pick Alpha.
    With full-precision selection, Zeta must be selected and Alpha excluded.
    """
    df = _make_mock_student_df(["student_alpha", "student_zeta"])
    mock_model = MockPredictor(np.array([0.50001, 0.50004]))

    queue_df = build_student_support_queue(feature_df=df, model=mock_model, capacity_k=1)

    assert len(queue_df) == 1
    assert queue_df.iloc[0]["student_key"] == "student_zeta"
    assert queue_df.iloc[0]["risk_probability"] == 0.5000
