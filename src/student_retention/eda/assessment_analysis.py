"""Assessment submission progress, deadline adherence, and milestone pacing metrics."""
import numpy as np
import pandas as pd


def extract_cutoff_assessment_features(
    cohort_df: pd.DataFrame,
    student_assessment_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    cutoff_day: int = 35,
    enforce_registration_opportunity: bool = True,
) -> pd.DataFrame:
    """Extract assessment submission progress and deadline adherence up to the cutoff.

    Identifies scheduled continuous assessments (TMA, CMA) with deadline date <= cutoff_day (A_due).
    Restricts submission evaluation strictly to submissions recorded on or before cutoff_day.
    Maintains the full canonical grain (code_module, code_presentation, id_student) throughout all joins
    and aggregations to prevent cross-course submission attribution for multi-course learners.

    Assessment Counter Invariants:
        - All counters (submissions_count_lte_cutoff, submitted_on_time_count, late_submissions_count_lte_cutoff,
          banked_submissions_count) measure strictly in units of unique qualifying assessments.
        - Multiple submission attempts or duplicate records for the same (id_student, id_assessment) are
          aggregated at the assessment level first: an assessment is on-time if any attempt met the deadline,
          and late if all attempts occurred after the deadline.
        - Invariant: submitted_on_time_count + late_submissions_count_lte_cutoff == submissions_count_lte_cutoff.

    Reasonable Opportunity Policy (enforce_registration_opportunity):
        - A scheduled assessment qualifies as an expected coursework milestone for a learner if the scheduled
          deadline occurred on or after the student's registration date (date >= date_registration) OR if the
          student actually completed/banked the assessment.
        - Students who registered late after an assessment deadline passed and who did not submit it are not
          penalized as having missed an assignment scheduled prior to their enrollment.

    For module-presentations with no continuous assessments due by cutoff (assessments_due_count == 0):
        - has_assessments_due is 0
        - submission_rate is undefined (np.nan), avoiding false 100% assumptions
        - has_missed_due_assessment is 0 (learner did not miss an assignment that was never scheduled)
        - unsubmitted_due_count is 0

    Args:
        cohort_df: Eligible cohort DataFrame with canonical keys.
        student_assessment_df: student_assessment table.
        assessments_df: assessments reference table.
        cutoff_day: Prediction cutoff day (default: 35).
        enforce_registration_opportunity: If True, adjusts due requirements for late registrants (default: True).

    Returns:
        DataFrame: Cohort rows with assessment pacing and progress features attached.
    """
    canonical_grain = ["code_module", "code_presentation", "id_student"]
    course_grain = ["code_module", "code_presentation"]

    # 1. Identify continuous non-exam assessments due on or before cutoff (A_due)
    due_assessments = assessments_df[
        (assessments_df["assessment_type"] != "Exam")
        & (assessments_df["date"].notna())
        & (assessments_df["date"] <= cutoff_day)
    ].copy()

    # 2. Recover module and presentation metadata through id_assessment link
    due_meta = (
        due_assessments[["id_assessment", "code_module", "code_presentation", "date"]]
        .drop_duplicates(subset=["id_assessment"])
    )

    # Link student submissions to due assessment metadata (strictly offering-specific)
    sub_with_meta = student_assessment_df.merge(
        due_meta,
        on="id_assessment",
        how="inner",
    )

    # 3. Filter submissions strictly to those turned in on or before cutoff_day
    valid_subs = sub_with_meta[
        sub_with_meta["date_submitted"].notna()
        & (sub_with_meta["date_submitted"] <= cutoff_day)
    ].copy()

    # Separate current-presentation submissions from banked records
    if "is_banked" in valid_subs.columns:
        current_subs = valid_subs[valid_subs["is_banked"] == 0].copy()
        banked_subs = valid_subs[valid_subs["is_banked"] == 1].copy()
    else:
        current_subs = valid_subs.copy()
        banked_subs = pd.DataFrame(columns=valid_subs.columns)

    # 4. Assessment-level deduplication to ensure consistent unique-assessment units
    # Determine on-time status per attempt (date_submitted <= deadline date)
    current_subs["is_on_time"] = current_subs["date_submitted"] <= current_subs["date"]

    assess_grain = ["code_module", "code_presentation", "id_student", "id_assessment"]
    assess_level = (
        current_subs.groupby(assess_grain, observed=True)
        .agg(is_on_time=("is_on_time", "any"))
        .reset_index()
    )
    assess_level["is_late"] = ~assess_level["is_on_time"]

    # Aggregate current submissions to student-offering grain
    sub_agg = (
        assess_level.groupby(canonical_grain, observed=True)
        .agg(
            submissions_count_lte_cutoff=("id_assessment", "count"),
            submitted_on_time_count=("is_on_time", lambda s: int(s.sum())),
            late_submissions_count_lte_cutoff=("is_late", lambda s: int(s.sum())),
        )
        .reset_index()
    )

    if not banked_subs.empty:
        banked_agg = (
            banked_subs.groupby(canonical_grain, observed=True)
            .agg(banked_submissions_count=("id_assessment", "nunique"))
            .reset_index()
        )
        sub_agg = sub_agg.merge(banked_agg, on=canonical_grain, how="outer")
        sub_agg["banked_submissions_count"] = sub_agg["banked_submissions_count"].fillna(0).astype("Int64")
    else:
        sub_agg["banked_submissions_count"] = 0

    # Total unique satisfied assessments across current submissions and banked credits
    all_satisfied = pd.concat([
        current_subs[["code_module", "code_presentation", "id_student", "id_assessment"]],
        banked_subs[["code_module", "code_presentation", "id_student", "id_assessment"]],
    ], ignore_index=True).drop_duplicates()

    sat_agg = (
        all_satisfied.groupby(canonical_grain, observed=True)
        .agg(unique_satisfied_count=("id_assessment", "count"))
        .reset_index()
    )
    sub_agg = sub_agg.merge(sat_agg, on=canonical_grain, how="left")
    sub_agg["unique_satisfied_count"] = sub_agg["unique_satisfied_count"].fillna(0).astype("Int64")

    # 5. Compute due assessments count (respecting registration opportunity if applicable)
    initial_len = len(cohort_df)
    if enforce_registration_opportunity and "date_registration" in cohort_df.columns:
        satisfied_set = set(zip(
            all_satisfied["code_module"],
            all_satisfied["code_presentation"],
            all_satisfied["id_student"],
            all_satisfied["id_assessment"],
        ))
        cohort_due_pairs = cohort_df[["code_module", "code_presentation", "id_student", "date_registration"]].merge(
            due_meta,
            on=["code_module", "code_presentation"],
            how="inner",
        )
        cohort_due_pairs["is_sat"] = [
            (m, p, s, a) in satisfied_set
            for m, p, s, a in zip(
                cohort_due_pairs["code_module"],
                cohort_due_pairs["code_presentation"],
                cohort_due_pairs["id_student"],
                cohort_due_pairs["id_assessment"],
            )
        ]
        cohort_due_pairs["has_opportunity"] = (
            cohort_due_pairs["date_registration"].isna()
            | (cohort_due_pairs["date"] >= cohort_due_pairs["date_registration"])
            | cohort_due_pairs["is_sat"]
        )
        student_due = (
            cohort_due_pairs[cohort_due_pairs["has_opportunity"]]
            .groupby(canonical_grain, observed=True)["id_assessment"]
            .nunique()
            .reset_index(name="assessments_due_count")
        )
        merged = cohort_df.merge(student_due, on=canonical_grain, how="left")
    else:
        pres_due = (
            due_assessments.groupby(course_grain, observed=True)["id_assessment"]
            .nunique()
            .reset_index(name="assessments_due_count")
        )
        merged = cohort_df.merge(pres_due, on=course_grain, how="left")

    merged["assessments_due_count"] = merged["assessments_due_count"].fillna(0).astype("Int64")
    merged["has_assessments_due"] = (merged["assessments_due_count"] > 0).astype("Int64")

    # Merge student submissions using full canonical grain
    merged = merged.merge(sub_agg, on=canonical_grain, how="left")

    # Reconcile row count: verify zero row multiplication or silent record loss
    if len(merged) != initial_len:
        raise ValueError(
            f"Assessment feature merge resulted in row multiplication/loss: "
            f"expected {initial_len} rows, got {len(merged)}"
        )

    for col in [
        "submissions_count_lte_cutoff",
        "submitted_on_time_count",
        "late_submissions_count_lte_cutoff",
        "banked_submissions_count",
        "unique_satisfied_count",
    ]:
        merged[col] = merged[col].fillna(0).astype("Int64")

    # 6. Unsubmitted due count & missed due assessment flag
    # Total satisfied due coursework accounts for deduplicated current submissions and banked credits
    total_satisfied = np.minimum(merged["assessments_due_count"], merged["unique_satisfied_count"])
    unsubmitted = merged["assessments_due_count"] - total_satisfied
    merged["unsubmitted_due_count"] = pd.Series(unsubmitted, index=merged.index).astype("Int64")

    # has_missed_due_assessment:
    # Strictly 1 if the student had due assessments AND left at least one unsubmitted
    # Strictly 0 if all due assessments were submitted OR if zero assessments were due
    has_missed = np.where(
        merged["assessments_due_count"] > 0,
        (merged["unsubmitted_due_count"] > 0).astype(int),
        0,
    )
    merged["has_missed_due_assessment"] = pd.Series(has_missed, index=merged.index).astype("Int64")

    # 7. Submission rate calculation
    # Proportion of required due coursework satisfied (via current submission or banked credit)
    # When assessments_due_count > 0: total_satisfied / assessments_due_count
    # When assessments_due_count == 0: np.nan (undefined rate, NOT 1.0)
    has_due = merged["assessments_due_count"] > 0
    sub_rate = np.where(
        has_due,
        (total_satisfied.astype(float) / merged["assessments_due_count"].astype(float)).round(4),
        np.nan,
    )
    merged["submission_rate"] = pd.Series(sub_rate, index=merged.index, dtype="float64")

    return merged


def summarize_assessment_patterns(cohort_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize submission progress and milestone failure rates by outcome group.

    Restricts summary strictly to eligible students enrolled in offerings with >= 1 continuous assessment
    due on or before the cutoff (assessments_due_count > 0). For students in offerings with zero assessments due,
    submission_rate is undefined (NaN) and they are excluded from coursework denominator.
    """
    has_due_mask = cohort_metrics_df["assessments_due_count"] > 0
    subset = cohort_metrics_df[has_due_mask]

    summary = (
        subset.groupby("withdrawn_after_cutoff", observed=True)
        .agg(
            students_with_due=("id_student", "count"),
            mean_due=("assessments_due_count", "mean"),
            mean_submitted=("submissions_count_lte_cutoff", "mean"),
            mean_on_time=("submitted_on_time_count", "mean"),
            mean_late=("late_submissions_count_lte_cutoff", "mean"),
            mean_unsubmitted=("unsubmitted_due_count", "mean"),
            mean_submission_rate=("submission_rate", "mean"),
            pct_missed_due=("has_missed_due_assessment", lambda s: float((s == 1).mean() * 100)),
        )
        .reset_index()
    )

    summary["outcome_label"] = summary["withdrawn_after_cutoff"].map(
        {0: "Retained (Completers)", 1: "Withdrawn After Day 35"}
    )
    return summary.round(3)
