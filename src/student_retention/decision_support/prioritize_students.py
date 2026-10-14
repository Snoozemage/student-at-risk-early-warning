"""Student caseload prioritization and triage queue generator for Day-35 decision support."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from student_retention.decision_support.intervention_policy import (
    PriorityTier,
    ReviewStatus,
    get_prescribed_action,
)
from student_retention.features.feature_selection import select_primary_features
from student_retention.logging_config import get_logger

logger = get_logger(__name__)

# Expected canonical columns for the exported decision support queue
QUEUE_CANONICAL_COLUMNS: list[str] = [
    "student_key",
    "code_module",
    "code_presentation",
    "prediction_cutoff_day",
    "risk_probability",
    "priority_tier",
    "support_reason_summary",
    "review_status",
    "recommended_next_step",
]


def generate_support_reason_summary(row: pd.Series) -> str:
    """Generate a human-readable summary of top observed engagement signals for an individual learner.

    Translates behavioral and pacing signals at Day 35 into contextual observational cues
    for academic advisors (e.g. days since last login, missed scheduled assignments, declining activity momentum).
    Notice: This summary describes observed historical signals associated with risk; it does not
    constitute a guaranteed causal explanation of why the model produced a specific score.

    Args:
        row: Series containing student feature values.

    Returns:
        str: Semicolon-delimited summary of observed engagement signals.
    """
    signals: list[str] = []

    # 1. Assessment submission progress
    missing_assess = row.get("missing_due_submissions_count")
    if pd.notna(missing_assess) and missing_assess > 0:
        signals.append(f"Missed {int(missing_assess)} scheduled due assessment(s)")

    late_assess = row.get("late_submissions_count_lte_cutoff")
    if pd.notna(late_assess) and late_assess > 0:
        signals.append(f"Submitted {int(late_assess)} assessment(s) past deadline")

    # 2. Digital platform engagement & inactivity
    has_no_vle = row.get("has_no_vle_activity_by_cutoff")
    if pd.notna(has_no_vle) and has_no_vle == 1:
        signals.append("Zero online VLE interactions recorded by cutoff")
    else:
        inactivity_days = row.get("days_since_last_vle_interaction")
        if pd.notna(inactivity_days):
            if inactivity_days >= 60:
                signals.append("Extended platform inactivity (>60 days)")
            elif inactivity_days >= 14:
                signals.append(f"High platform inactivity ({int(inactivity_days)} days since last interaction)")

    # 3. Frequency & momentum
    active_days = row.get("vle_active_days_window")
    if pd.notna(active_days) and active_days <= 3:
        signals.append(f"Low engagement frequency ({int(active_days)} active days in 35-day window)")

    velocity_ratio = row.get("vle_velocity_ratio")
    if pd.notna(velocity_ratio) and velocity_ratio <= 0.35:
        signals.append(f"Sharp decline in interaction velocity ({velocity_ratio:.2f}x recent momentum)")

    # 4. Context & registration
    is_late_reg = row.get("is_late_registration")
    if pd.notna(is_late_reg) and is_late_reg == 1:
        lead_days = row.get("registration_lead_time_days")
        lead_str = f" ({int(abs(lead_days))} days late)" if pd.notna(lead_days) else ""
        signals.append(f"Late course registration{lead_str}")

    prev_attempts = row.get("num_of_prev_attempts")
    if pd.notna(prev_attempts) and prev_attempts >= 2:
        signals.append(f"Repeat module attempt ({int(prev_attempts)} prior attempts)")

    if not signals:
        return "Normal learning engagement patterns observed"

    # Select up to top 3 signals for clarity
    return "; ".join(signals[:3])


def assign_priority_tier(
    probability: float,
    urgent_threshold: float = 0.70,
    watch_threshold: float = 0.40,
) -> str:
    """Assign a provisional triage priority tier based on calibrated withdrawal risk probability.

    Args:
        probability: Calibrated risk probability score [0.0, 1.0].
        urgent_threshold: Minimum probability for Tier 1 (Urgent). Defaults to 0.70.
        watch_threshold: Minimum probability for Tier 2 (Watch). Defaults to 0.40.

    Returns:
        str: Priority tier string (e.g. 'Tier 1 (Urgent)').
    """
    if probability >= urgent_threshold:
        return PriorityTier.TIER_1_URGENT.value
    elif probability >= watch_threshold:
        return PriorityTier.TIER_2_WATCH.value
    else:
        return PriorityTier.TIER_3_STANDARD.value


def audit_support_queue_capacity(queue_df: pd.DataFrame) -> pd.DataFrame:
    """Audit caseload capacity, unique students, and duplicate slots per term in the queue.

    Preserves the course-enrollment prediction grain without inventing an unapproved
    aggregation rule across concurrent modules.

    Args:
        queue_df: Support queue DataFrame containing 'code_presentation' and 'student_key'.

    Returns:
        pd.DataFrame: Tabular summary of enrollment slots, unique students, and duplicate slots per term.
    """
    records = []
    presentations = sorted(queue_df["code_presentation"].unique().tolist())

    for pres in presentations:
        sub = queue_df[queue_df["code_presentation"] == pres]
        n_slots = len(sub)
        n_unique = sub["student_key"].nunique()
        n_dups = n_slots - n_unique
        # Count students with >1 slot
        student_counts = sub["student_key"].value_counts()
        multi_course_students = int((student_counts > 1).sum())

        records.append(
            {
                "code_presentation": pres,
                "caseload_enrollment_slots": n_slots,
                "unique_students_in_queue": n_unique,
                "duplicate_slots": n_dups,
                "multi_course_students_in_queue": multi_course_students,
                "unresolved_aggregation_status": "unresolved_pending_institutional_policy",
            }
        )

    # Total row across all presentations in queue
    total_slots = len(queue_df)
    total_unique = queue_df["student_key"].nunique()
    records.append(
        {
            "code_presentation": "ALL_TERMS_TOTAL",
            "caseload_enrollment_slots": total_slots,
            "unique_students_in_queue": total_unique,
            "duplicate_slots": total_slots - total_unique,
            "multi_course_students_in_queue": int((queue_df["student_key"].value_counts() > 1).sum()),
            "unresolved_aggregation_status": "unresolved_pending_institutional_policy",
        }
    )

    return pd.DataFrame(records)


def build_student_support_queue(
    feature_df: pd.DataFrame,
    model: Pipeline | Any,
    urgent_threshold: float = 0.70,
    watch_threshold: float = 0.40,
    capacity_k: int | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """Build the demonstration student support queue deliverable for academic advising triage.

    Computes calibrated predicted withdrawal risk probabilities for each student enrollment record,
    maps probabilities to provisional priority tiers, compiles observed engagement signal
    summaries, assigns prescribed supportive intervention guidelines, and enforces the canonical schema.

    Demonstration Cohort Notice:
    In accordance with leakage-free evaluation protocols, this demonstration queue should be
    generated on held-out test records or out-of-fold retrospective partitions. In-sample
    training records must never be presented as prospective operational triage.

    Caseload Capacity & Grain Notice:
    Capacity is enforced at the course-enrollment record grain (e.g. 500 enrollment slots per
    presentation). Concurrent module enrollments cause some multi-course students to occupy
    multiple slots. Until an approved institutional policy defines how risk should be aggregated
    across courses (e.g. max risk vs mean risk), a true 500-unique-student operational queue
    remains unresolved.

    Args:
        feature_df: DataFrame containing student identifiers, course context, and primary features.
        model: Fitted scikit-learn Pipeline or CalibratedClassifierCV exposing predict_proba.
        urgent_threshold: Cutoff for Tier 1 (Urgent).
        watch_threshold: Cutoff for Tier 2 (Watch).
        capacity_k: Optional maximum capacity limit per presentation. If provided, selects top K
                    records with highest risk probabilities within each code_presentation.
        save_path: Optional destination path to export CSV. Defaults to outputs/predictions/student_support_queue.csv.

    Returns:
        pd.DataFrame: Structured caseload queue with canonical columns.
    """
    logger.info("Generating Day-35 student support triage queue (N=%d)...", len(feature_df))

    # Required identifier and feature check
    for req_id in ["id_student", "code_module", "code_presentation"]:
        if req_id not in feature_df.columns:
            raise ValueError(f"Required identifier column '{req_id}' missing from input DataFrame.")

    # Prepare features for model scoring
    clean_features = select_primary_features(
        feature_df,
        include_target=False,
        include_identifiers=False,
    )

    # Compute risk probabilities
    if not hasattr(model, "predict_proba"):
        raise ValueError("Provided model pipeline does not support 'predict_proba'.")

    probabilities = model.predict_proba(clean_features)[:, 1]

    # Assemble queue DataFrame preserving full-precision scores for internal decisions
    queue_df = pd.DataFrame(
        {
            "student_key": feature_df["id_student"].astype(str),
            "code_module": feature_df["code_module"].astype(str),
            "code_presentation": feature_df["code_presentation"].astype(str),
            "prediction_cutoff_day": 35,
            "_full_precision_risk": probabilities,
        }
    )

    # Generate priority tiers using full-precision probabilities
    queue_df["priority_tier"] = queue_df["_full_precision_risk"].apply(
        lambda p: assign_priority_tier(p, urgent_threshold=urgent_threshold, watch_threshold=watch_threshold)
    )

    # Generate personalized support reason summaries from input feature attributes
    reasons = [generate_support_reason_summary(feature_df.iloc[i]) for i in range(len(feature_df))]
    queue_df["support_reason_summary"] = reasons

    # Assign default operational review status
    queue_df["review_status"] = ReviewStatus.PENDING.value

    # Assign prescribed supportive intervention recommendations
    queue_df["recommended_next_step"] = [
        get_prescribed_action(
            tier=queue_df.iloc[i]["priority_tier"],
            primary_driver=queue_df.iloc[i]["support_reason_summary"],
        )
        for i in range(len(queue_df))
    ]

    # Sort queue by full-precision risk probability descending (highest risk first)
    queue_df = queue_df.sort_values(
        by=["_full_precision_risk", "code_presentation", "code_module", "student_key"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    # Apply capacity filtering if requested (e.g. top K per presentation)
    if capacity_k is not None and capacity_k > 0:
        logger.info("Applying advisor capacity constraint of K=%d per presentation...", capacity_k)
        queue_df = (
            queue_df.groupby("code_presentation", as_index=False)
            .head(capacity_k)
            .reset_index(drop=True)
        )

    # Apply four-decimal rounding strictly to presentation output after all decisions
    queue_df["risk_probability"] = np.round(queue_df["_full_precision_risk"], 4)
    queue_df = queue_df.drop(columns=["_full_precision_risk"])

    # Audit capacity and unique students
    if "code_presentation" in queue_df.columns and "student_key" in queue_df.columns:
        total_records = len(queue_df)
        total_unique = queue_df["student_key"].nunique()
        dup_slots = total_records - total_unique
        logger.info(
            "Queue composition: %d enrollment records across %d unique students (%d duplicate slots). "
            "True unique-student operational queue remains unresolved pending institutional multi-course aggregation policy.",
            total_records,
            total_unique,
            dup_slots,
        )

    # Validate schema conformity
    missing_cols = set(QUEUE_CANONICAL_COLUMNS) - set(queue_df.columns)
    if missing_cols:
        raise ValueError(f"Decision queue missing canonical columns: {missing_cols}")

    queue_df = queue_df[QUEUE_CANONICAL_COLUMNS]

    # Save to disk if requested
    if save_path is not None:
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        queue_df.to_csv(target_path, index=False)
        logger.info("Saved validated student support queue (%d rows) to %s", len(queue_df), target_path)

    return queue_df
