"""Intervention policy, triage Service Level Agreements (SLAs), and ethical communication protocols."""

from enum import Enum

from student_retention.logging_config import get_logger

logger = get_logger(__name__)


class PriorityTier(str, Enum):
    """Provisional triage priority tiers for student support."""

    TIER_1_URGENT = "Tier 1 (Urgent)"
    TIER_2_WATCH = "Tier 2 (Watch)"
    TIER_3_STANDARD = "Tier 3 (Standard)"


class ReviewStatus(str, Enum):
    """Operational workflow statuses for academic advisor caseload tracking."""

    PENDING = "Pending"
    UNDER_REVIEW = "Under Review"
    CONTACTED = "Contacted"
    RESOLVED = "Resolved"
    NO_ACTION_NEEDED = "No Action Needed"


# Provisional Service Level Agreements (business days)
PROVISIONAL_TIER_SLAS: dict[str, int | None] = {
    PriorityTier.TIER_1_URGENT.value: 3,
    PriorityTier.TIER_2_WATCH.value: 7,
    PriorityTier.TIER_3_STANDARD.value: None,
}

# Standard prescribed actions by priority tier
PRESCRIBED_TIER_ACTIONS: dict[str, str] = {
    PriorityTier.TIER_1_URGENT.value: (
        "Direct personal contact by academic advisor (phone or 1-on-1 email) "
        "to explore personal, technical, or learning barriers."
    ),
    PriorityTier.TIER_2_WATCH.value: (
        "Automated supportive check-in email and module tutor alert "
        "to monitor course forum participation and assignment pacing."
    ),
    PriorityTier.TIER_3_STANDARD.value: (
        "Standard course communications, routine study progress summaries, and regular tutor monitoring."
    ),
}

# Prohibited punitive phrases in advisor communications
PROHIBITED_COMMUNICATION_PHRASES: list[str] = [
    "algorithm flagged you",
    "identified by an algorithm",
    "predicted to fail",
    "high risk of failure",
    "risk score",
    "academic probation",
    "cancellation warning",
    "automatic withdrawal",
]


def get_tier_sla_days(tier: PriorityTier | str) -> int | None:
    """Retrieve the provisional response SLA in business days for a priority tier.

    Args:
        tier: Priority tier enum or string.

    Returns:
        int | None: Business days allowed for initial contact, or None for routine tracking.
    """
    tier_str = tier.value if isinstance(tier, PriorityTier) else str(tier)
    return PROVISIONAL_TIER_SLAS.get(tier_str)


def get_prescribed_action(
    tier: PriorityTier | str,
    primary_driver: str | None = None,
) -> str:
    """Retrieve the prescribed supportive intervention recommendation for a student case.

    Args:
        tier: Priority tier enum or string.
        primary_driver: Optional primary risk driver description to contextualize the action.

    Returns:
        str: Prescribed supportive action guideline for the academic advisor.
    """
    tier_str = tier.value if isinstance(tier, PriorityTier) else str(tier)
    base_action = PRESCRIBED_TIER_ACTIONS.get(tier_str, PRESCRIBED_TIER_ACTIONS[PriorityTier.TIER_3_STANDARD.value])

    if primary_driver and tier_str == PriorityTier.TIER_1_URGENT.value:
        if "assessment" in primary_driver.lower():
            return f"{base_action} Focus: Coursework deadline extension and tutor assignment support."
        if "inactivity" in primary_driver.lower() or "vle" in primary_driver.lower():
            return f"{base_action} Focus: VLE portal access check, study schedule review, and pastoral support."

    return base_action


def validate_ethical_communication(message_text: str) -> tuple[bool, list[str]]:
    """Verify that proposed outreach messaging conforms to supportive, non-punitive ethical principles.

    Checks for prohibited stigmatizing or automated language (e.g., 'identified by algorithm', 'high risk of failure').

    Args:
        message_text: Draft outreach communication text.

    Returns:
        tuple[bool, list[str]]: (is_ethical, list_of_violations)
    """
    violations = []
    text_lower = message_text.lower()

    for phrase in PROHIBITED_COMMUNICATION_PHRASES:
        if phrase in text_lower:
            violations.append(f"Contains prohibited phrasing: '{phrase}'")

    is_ethical = len(violations) == 0
    return is_ethical, violations


def get_advising_workflow_principles() -> dict[str, str]:
    """Return core governance and ethical decision-support principles."""
    return {
        "supportive_framing_only": (
            "All student communications must use supportive, empowering language. Never disclose algorithmic "
            "scores, risk probability percentiles, or failure classifications to students."
        ),
        "human_discretion_prevails": (
            "The priority queue is a decision-support guide, not an administrative mandate. Academic advisors "
            "retain complete professional authority to override priority tiers, adjust response timing, or "
            "close cases based on contextual qualitative knowledge."
        ),
        "no_automated_penalties": (
            "Model predictions must never trigger automatic administrative sanctions, financial aid revocations, "
            "registration blocks, or mandatory withdrawals."
        ),
        "transparent_context": (
            "Advisors must be provided with specific behavioral signals (such as days since last portal login or "
            "missed assessment milestones) to enable meaningful, empathetic outreach conversations."
        ),
    }
