"""Assessment submission progress and milestone pacing feature extraction for Day 35 cutoff."""
import numpy as np
import pandas as pd


def extract_assessment_features(
    cohort_df: pd.DataFrame,
    student_assessment_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    cutoff_day: int = 35,
    enforce_registration_opportunity: bool = True,
) -> pd.DataFrame:
    """Extract the 5 locked assessment progress features bounded by Day 35 cutoff.

    Due Universe Specification (A_due):
        - Restricted strictly to continuous non-exam coursework (TMA, CMA) with scheduled deadline date <= cutoff_day.
        - Excludes final examinations and unscheduled components.
        - Early submissions for future assessments (deadline > cutoff_day) are excluded from due-progress
          counts to prevent on-time submissions exceeding due counts.

    Reasonable Opportunity Policy:
        - Late registrants (registered after Day 0) are only held accountable for assessments whose
          deadline occurred on/after their registration date, or which they actually satisfied.

    Banked Coursework Policy (is_banked == 1):
        - Transferred assessment credits from prior presentations satisfy assessment requirements
          (not penalized as missing), but do not count toward current-presentation submission volume.

    Mathematical Invariants Enforced:
        1. submitted_on_time_count + late_submissions_count_lte_cutoff == submissions_count_lte_cutoff
        2. submissions_count_lte_cutoff <= assessments_due_count
        3. missing_due_submissions_count <= assessments_due_count

    Args:
        cohort_df: Eligible cohort DataFrame with canonical keys.
        student_assessment_df: Raw or normalized student_assessment table.
        assessments_df: Raw or normalized assessments reference table.
        cutoff_day: Relative day cutoff (default: 35).
        enforce_registration_opportunity: Whether to adjust due counts for late registrants (default: True).

    Returns:
        DataFrame: Canonical keys + 5 assessment features.
    """
    if cutoff_day != 35:
        raise ValueError(
            f"Unsupported cutoff_day={cutoff_day}. The Day-35 early warning feature engineering pipeline "
            f"strictly specifies and enforces cutoff_day=35."
        )

    canonical_grain = ["code_module", "code_presentation", "id_student"]
    course_grain = ["code_module", "code_presentation"]

    # Base cohort scaffold
    base = cohort_df[canonical_grain].copy()
    initial_len = len(base)

    # 1. Identify due continuous non-exam assessments (A_due)
    due_assessments = assessments_df[
        (assessments_df["assessment_type"] != "Exam")
        & (assessments_df["date"].notna())
        & (assessments_df["date"] <= cutoff_day)
    ].copy()

    # Link due assessments to course-presentation metadata
    due_meta = (
        due_assessments[["id_assessment", "code_module", "code_presentation", "date"]]
        .drop_duplicates(subset=["id_assessment"])
    )

    # 2. Filter student submissions to due assessments turned in on or before cutoff_day
    sub_with_meta = student_assessment_df.merge(
        due_meta,
        on="id_assessment",
        how="inner",
    )

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

    # 3. Assessment-level deduplication to ensure unique assessment units
    current_subs["is_on_time"] = current_subs["date_submitted"] <= current_subs["date"]

    assess_grain = ["code_module", "code_presentation", "id_student", "id_assessment"]
    assess_level = (
        current_subs.groupby(assess_grain, observed=True)
        .agg(is_on_time=("is_on_time", "any"))
        .reset_index()
    )
    assess_level["is_late"] = ~assess_level["is_on_time"]

    # Student-offering aggregation of current submissions
    sub_agg = (
        assess_level.groupby(canonical_grain, observed=True)
        .agg(
            submissions_count_lte_cutoff=("id_assessment", "count"),
            submitted_on_time_count=("is_on_time", lambda s: int(s.sum())),
            late_submissions_count_lte_cutoff=("is_late", lambda s: int(s.sum())),
        )
        .reset_index()
    )

    # Deduplicated satisfied coursework (current + banked)
    all_satisfied = pd.concat([
        current_subs[["code_module", "code_presentation", "id_student", "id_assessment"]],
        banked_subs[["code_module", "code_presentation", "id_student", "id_assessment"]],
    ], ignore_index=True).drop_duplicates()

    sat_agg = (
        all_satisfied.groupby(canonical_grain, observed=True)
        .agg(unique_satisfied_count=("id_assessment", "count"))
        .reset_index()
    )
    sub_agg = sub_agg.merge(sat_agg, on=canonical_grain, how="outer")
    for col in [
        "submissions_count_lte_cutoff",
        "submitted_on_time_count",
        "late_submissions_count_lte_cutoff",
        "unique_satisfied_count",
    ]:
        sub_agg[col] = sub_agg[col].fillna(0).astype("Int64")

    # 4. Compute due assessments count per student (with reasonable opportunity policy)
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
        merged = base.merge(student_due, on=canonical_grain, how="left")
    else:
        pres_due = (
            due_assessments.groupby(course_grain, observed=True)["id_assessment"]
            .nunique()
            .reset_index(name="assessments_due_count")
        )
        merged = base.merge(pres_due, on=course_grain, how="left")

    merged["assessments_due_count"] = merged["assessments_due_count"].fillna(0).astype("Int64")

    # 5. Merge student submissions using full canonical grain
    merged = merged.merge(sub_agg, on=canonical_grain, how="left")

    if len(merged) != initial_len:
        raise ValueError(
            f"Assessment feature merge resulted in row count mismatch: "
            f"expected {initial_len}, got {len(merged)}"
        )

    for col in [
        "submissions_count_lte_cutoff",
        "submitted_on_time_count",
        "late_submissions_count_lte_cutoff",
        "unique_satisfied_count",
    ]:
        merged[col] = merged[col].fillna(0).astype("Int64")

    # 6. Calculate missing due submissions count
    # Due assessments lacking both genuine current submission and banked credit
    total_satisfied = np.minimum(merged["assessments_due_count"], merged["unique_satisfied_count"])
    missing_due = merged["assessments_due_count"] - total_satisfied
    merged["missing_due_submissions_count"] = pd.Series(missing_due, index=merged.index).astype("Int64")

    # Drop temporary calculation column
    merged.drop(columns=["unique_satisfied_count"], inplace=True)

    return merged
