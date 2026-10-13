"""Feature definitions, metadata schemas, and feature allowlists for Day 35 prediction."""
from dataclasses import dataclass
from enum import Enum
from typing import Any


class FeatureCategory(str, Enum):
    """Functional categories for features."""
    COURSE_CONTEXT = "course_context"
    REGISTRATION = "registration"
    VLE_ENGAGEMENT = "vle_engagement"
    ASSESSMENT_PROGRESS = "assessment_progress"
    DEMOGRAPHIC_AUDIT = "demographic_audit"


@dataclass(frozen=True)
class FeatureDefinition:
    """Formal metadata definition for a single predictive or audit feature."""
    name: str
    category: FeatureCategory
    data_type: str
    window: str
    description: str
    source_table: str
    source_field: str
    nullable: bool = False
    sentinel_value: Any = None


# Canonical observation grain for all predictions and features
CANONICAL_GRAIN = ["code_module", "code_presentation", "id_student"]

# Canonical target variable
CANONICAL_TARGET = "withdrawn_after_cutoff"

# Primary 22-feature allowlist definitions
PRIMARY_FEATURE_DEFINITIONS: dict[str, FeatureDefinition] = {
    # 1. Course Context (4 features)
    "code_module": FeatureDefinition(
        name="code_module",
        category=FeatureCategory.COURSE_CONTEXT,
        data_type="nominal_string",
        window="Static entry context",
        description="Module code identifier (AAA-GGG).",
        source_table="studentInfo",
        source_field="code_module",
    ),
    "presentation_season": FeatureDefinition(
        name="presentation_season",
        category=FeatureCategory.COURSE_CONTEXT,
        data_type="binary_string",
        window="Static entry context",
        description="Recurring seasonal presentation start: 'B' (February/Spring) vs 'J' (October/Autumn).",
        source_table="studentInfo",
        source_field="code_presentation (suffix character)",
    ),
    "studied_credits": FeatureDefinition(
        name="studied_credits",
        category=FeatureCategory.COURSE_CONTEXT,
        data_type="integer",
        window="Static entry context",
        description="Total credit volume student is studying concurrently in this presentation.",
        source_table="studentInfo",
        source_field="studied_credits",
    ),
    "num_of_prev_attempts": FeatureDefinition(
        name="num_of_prev_attempts",
        category=FeatureCategory.COURSE_CONTEXT,
        data_type="integer",
        window="Static entry context",
        description="Number of times student previously attempted this specific module.",
        source_table="studentInfo",
        source_field="num_of_prev_attempts",
    ),

    # 2. Registration (2 features)
    "registration_lead_time_days": FeatureDefinition(
        name="registration_lead_time_days",
        category=FeatureCategory.REGISTRATION,
        data_type="float",
        window="Relative registration date",
        description="Days between registration and Day 0: -date_registration. Positive for advance registration, zero for Day 0, negative for late.",
        source_table="studentRegistration",
        source_field="date_registration",
        nullable=True,  # Nullable strictly under Policy B sensitivity; 0 nulls under Policy C
    ),
    "is_late_registration": FeatureDefinition(
        name="is_late_registration",
        category=FeatureCategory.REGISTRATION,
        data_type="float",
        window="Relative registration date",
        description="Binary indicator: 1 if 0 < date_registration <= 35, 0 if date_registration <= 0, null if date_registration is missing.",
        source_table="studentRegistration",
        source_field="date_registration",
        nullable=True,  # Nullable strictly under Policy B sensitivity; 0 nulls under Policy C
    ),

    # 3. VLE Digital Engagement (11 features; filtered to date <= 35)
    "vle_total_clicks_window": FeatureDefinition(
        name="vle_total_clicks_window",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="integer",
        window="[-25, 35]",
        description="Total VLE interaction clicks across entire Day -25 to Day 35 observation window.",
        source_table="studentVle",
        source_field="sum_click",
    ),
    "vle_active_days_window": FeatureDefinition(
        name="vle_active_days_window",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="integer",
        window="[-25, 35]",
        description="Count of distinct active calendar days with >= 1 VLE interaction.",
        source_table="studentVle",
        source_field="date",
    ),
    "vle_unique_sites_accessed": FeatureDefinition(
        name="vle_unique_sites_accessed",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="integer",
        window="[-25, 35]",
        description="Count of distinct VLE learning materials/pages visited.",
        source_table="studentVle",
        source_field="id_site",
    ),
    "vle_pre_course_clicks": FeatureDefinition(
        name="vle_pre_course_clicks",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="integer",
        window="[-25, -1]",
        description="Total clicks during pre-course orientation strictly prior to Day 0.",
        source_table="studentVle",
        source_field="sum_click",
    ),
    "vle_recent_clicks_14d": FeatureDefinition(
        name="vle_recent_clicks_14d",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="integer",
        window="[22, 35]",
        description="Total clicks during the 14 days immediately preceding Day 35 cutoff.",
        source_table="studentVle",
        source_field="sum_click",
    ),
    "vle_velocity_ratio": FeatureDefinition(
        name="vle_velocity_ratio",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="float",
        window="[0, 35]",
        description="Ratio of average daily clicks [29, 35] to baseline [0, 28] with Laplace smoothing denominator.",
        source_table="studentVle",
        source_field="sum_click, date",
    ),
    "days_since_last_vle_interaction": FeatureDefinition(
        name="days_since_last_vle_interaction",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="integer",
        window="Cutoff relative day",
        description="Days elapsed since most recent VLE interaction: 35 - max(date <= 35). Sentinel 65 for inactive students.",
        source_table="studentVle",
        source_field="date",
        sentinel_value=65,
    ),
    "has_no_vle_activity_by_cutoff": FeatureDefinition(
        name="has_no_vle_activity_by_cutoff",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="integer",
        window="[-25, 35]",
        description="Binary indicator: 1 if 0 clicks recorded by Day 35, else 0.",
        source_table="studentVle",
        source_field="sum_click",
    ),
    "vle_clicks_forumng": FeatureDefinition(
        name="vle_clicks_forumng",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="integer",
        window="[-25, 35]",
        description="Total clicks on collaborative discussion forums (activity_type == 'forumng').",
        source_table="studentVle, vle",
        source_field="sum_click, activity_type",
    ),
    "vle_clicks_oucontent": FeatureDefinition(
        name="vle_clicks_oucontent",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="integer",
        window="[-25, 35]",
        description="Total clicks on core interactive learning syllabus content (activity_type == 'oucontent').",
        source_table="studentVle, vle",
        source_field="sum_click, activity_type",
    ),
    "vle_clicks_quiz": FeatureDefinition(
        name="vle_clicks_quiz",
        category=FeatureCategory.VLE_ENGAGEMENT,
        data_type="integer",
        window="[-25, 35]",
        description="Total clicks strictly on formative quizzes (activity_type == 'quiz').",
        source_table="studentVle, vle",
        source_field="sum_click, activity_type",
    ),

    # 4. Assessment Progress (5 features; due universe date <= 35)
    "assessments_due_count": FeatureDefinition(
        name="assessments_due_count",
        category=FeatureCategory.ASSESSMENT_PROGRESS,
        data_type="integer",
        window="date <= 35",
        description="Count of scheduled non-exam coursework assessments with deadline <= Day 35 for which learner had opportunity.",
        source_table="assessments",
        source_field="id_assessment, date, assessment_type",
    ),
    "submissions_count_lte_cutoff": FeatureDefinition(
        name="submissions_count_lte_cutoff",
        category=FeatureCategory.ASSESSMENT_PROGRESS,
        data_type="integer",
        window="date_submitted <= 35",
        description="Count of genuine submissions for due-universe assessments (date <= 35) with date_submitted <= 35 and is_banked == 0.",
        source_table="studentAssessment, assessments",
        source_field="id_assessment, date_submitted, is_banked",
    ),
    "submitted_on_time_count": FeatureDefinition(
        name="submitted_on_time_count",
        category=FeatureCategory.ASSESSMENT_PROGRESS,
        data_type="integer",
        window="date_submitted <= date <= 35",
        description="Count of genuine submissions for due-universe assessments submitted on or before deadline.",
        source_table="studentAssessment, assessments",
        source_field="id_assessment, date_submitted, date, is_banked",
    ),
    "late_submissions_count_lte_cutoff": FeatureDefinition(
        name="late_submissions_count_lte_cutoff",
        category=FeatureCategory.ASSESSMENT_PROGRESS,
        data_type="integer",
        window="date < date_submitted <= 35",
        description="Count of genuine submissions for due-universe assessments submitted after deadline but <= Day 35.",
        source_table="studentAssessment, assessments",
        source_field="id_assessment, date_submitted, date, is_banked",
    ),
    "missing_due_submissions_count": FeatureDefinition(
        name="missing_due_submissions_count",
        category=FeatureCategory.ASSESSMENT_PROGRESS,
        data_type="integer",
        window="Cutoff milestone",
        description="Count of scheduled due-universe assessments where learner has neither a genuine submission nor banked credit.",
        source_table="studentAssessment, assessments",
        source_field="id_assessment, date_submitted, is_banked",
    ),
}

# Locked 22-feature allowlist in exact order
PRIMARY_FEATURE_ALLOWLIST: list[str] = list(PRIMARY_FEATURE_DEFINITIONS.keys())

# Demographic attributes retained solely for fairness auditing and sensitivity experiments
DEMOGRAPHIC_AUDIT_FEATURES: list[str] = [
    "gender",
    "age_band",
    "disability",
    "imd_band",
    "highest_education",
    "region",
]

# Prohibited / outcome-derived features that must NEVER appear as predictive features
FORBIDDEN_FEATURES: list[str] = [
    "final_result",
    "date_unregistration",
    "score",
    "id_student",
    "withdrawn_after_cutoff",
]
