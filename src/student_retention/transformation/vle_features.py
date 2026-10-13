"""VLE digital engagement feature extraction strictly bounded by Day 35 cutoff."""
import numpy as np
import pandas as pd


def extract_vle_features(
    cohort_df: pd.DataFrame,
    student_vle_df: pd.DataFrame,
    vle_df: pd.DataFrame,
    cutoff_day: int = 35,
) -> pd.DataFrame:
    """Extract the 11 locked VLE digital engagement features for the cohort.

    All features are derived strictly from interactions on or before cutoff_day.
    Zero activity learners are assigned sentinel recency 65 and indicator 1.

    Features extracted:
        1. vle_total_clicks_window: Clicks in [-25, cutoff_day]
        2. vle_active_days_window: Distinct interaction dates in [-25, cutoff_day]
        3. vle_unique_sites_accessed: Distinct id_site accessed in [-25, cutoff_day]
        4. vle_pre_course_clicks: Clicks in [-25, -1]
        5. vle_recent_clicks_14d: Clicks in [cutoff_day - 13, cutoff_day] (i.e. [22, 35])
        6. vle_velocity_ratio: (clicks[29, 35] / 7) / ((clicks[0, 28] / 29) + 1.0)
        7. days_since_last_vle_interaction: cutoff_day - max(date <= cutoff_day); sentinel 65 if 0 clicks
        8. has_no_vle_activity_by_cutoff: 1 if 0 clicks in [-25, cutoff_day], else 0
        9. vle_clicks_forumng: Clicks where activity_type == 'forumng' in [-25, cutoff_day]
        10. vle_clicks_oucontent: Clicks where activity_type == 'oucontent' in [-25, cutoff_day]
        11. vle_clicks_quiz: Clicks strictly where activity_type == 'quiz' in [-25, cutoff_day]

    Args:
        cohort_df: Eligible cohort DataFrame with canonical keys.
        student_vle_df: Raw or normalized student_vle interaction log table.
        vle_df: vle reference table mapping id_site to activity_type.
        cutoff_day: Relative day cutoff (default: 35).

    Returns:
        DataFrame: Canonical keys + 11 VLE features.
    """
    if cutoff_day != 35:
        raise ValueError(
            f"Unsupported cutoff_day={cutoff_day}. The Day-35 early warning feature engineering pipeline "
            f"strictly specifies and enforces cutoff_day=35."
        )

    canonical_grain = ["code_module", "code_presentation", "id_student"]

    # Base cohort scaffold to guarantee 100% preservation of grain and rows
    base = cohort_df[canonical_grain].copy()

    # 1. Enforce strict temporal cutoff on interaction logs
    # Observation window: Day -25 through Day 35 inclusive
    vle_window = student_vle_df[
        (student_vle_df["date"] >= -25) & (student_vle_df["date"] <= cutoff_day)
    ].copy()

    if vle_window.empty:
        # All students have zero activity
        base["vle_total_clicks_window"] = 0
        base["vle_active_days_window"] = 0
        base["vle_unique_sites_accessed"] = 0
        base["vle_pre_course_clicks"] = 0
        base["vle_recent_clicks_14d"] = 0
        base["vle_velocity_ratio"] = 0.0
        base["days_since_last_vle_interaction"] = 65
        base["has_no_vle_activity_by_cutoff"] = 1
        base["vle_clicks_forumng"] = 0
        base["vle_clicks_oucontent"] = 0
        base["vle_clicks_quiz"] = 0
        for col in [
            "vle_total_clicks_window",
            "vle_active_days_window",
            "vle_unique_sites_accessed",
            "vle_pre_course_clicks",
            "vle_recent_clicks_14d",
            "days_since_last_vle_interaction",
            "has_no_vle_activity_by_cutoff",
            "vle_clicks_forumng",
            "vle_clicks_oucontent",
            "vle_clicks_quiz",
        ]:
            base[col] = base[col].astype("Int64")
        base["vle_velocity_ratio"] = base["vle_velocity_ratio"].astype("float64")
        return base

    # 2. Window-level base aggregations
    base_agg = (
        vle_window.groupby(canonical_grain, observed=True)
        .agg(
            vle_total_clicks_window=("sum_click", "sum"),
            vle_active_days_window=("date", "nunique"),
            vle_unique_sites_accessed=("id_site", "nunique"),
            max_vle_date=("date", "max"),
        )
        .reset_index()
    )

    # 3. Pre-course window [-25, -1]
    precourse_mask = vle_window["date"] < 0
    precourse_agg = (
        vle_window[precourse_mask]
        .groupby(canonical_grain, observed=True)["sum_click"]
        .sum()
        .reset_index(name="vle_pre_course_clicks")
    )

    # 4. Recent window [22, 35] (14 inclusive days: [cutoff_day - 13, cutoff_day])
    recent_start = cutoff_day - 13  # 35 - 13 = 22
    recent_mask = (vle_window["date"] >= recent_start) & (vle_window["date"] <= cutoff_day)
    recent_agg = (
        vle_window[recent_mask]
        .groupby(canonical_grain, observed=True)["sum_click"]
        .sum()
        .reset_index(name="vle_recent_clicks_14d")
    )

    # 5. Velocity window components:
    # Numerator: clicks in [29, 35] (7 days: [cutoff_day - 6, cutoff_day]) / 7.0
    # Denominator: (clicks in [0, 28] (29 days: [0, cutoff_day - 7]) / 29.0) + 1.0
    vel_num_start = cutoff_day - 6  # 35 - 6 = 29
    vel_den_end = cutoff_day - 7    # 35 - 7 = 28
    vel_num_mask = (vle_window["date"] >= vel_num_start) & (vle_window["date"] <= cutoff_day)
    vel_den_mask = (vle_window["date"] >= 0) & (vle_window["date"] <= vel_den_end)

    vel_num_agg = (
        vle_window[vel_num_mask]
        .groupby(canonical_grain, observed=True)["sum_click"]
        .sum()
        .reset_index(name="clicks_29_35")
    )
    vel_den_agg = (
        vle_window[vel_den_mask]
        .groupby(canonical_grain, observed=True)["sum_click"]
        .sum()
        .reset_index(name="clicks_0_28")
    )

    # 6. Activity type specific aggregations: forumng, oucontent, quiz
    # Merge vle_df metadata to obtain activity_type
    vle_typed = vle_window.merge(
        vle_df[["id_site", "activity_type"]].drop_duplicates(subset=["id_site"]),
        on="id_site",
        how="left",
    )

    forum_mask = vle_typed["activity_type"] == "forumng"
    forum_agg = (
        vle_typed[forum_mask]
        .groupby(canonical_grain, observed=True)["sum_click"]
        .sum()
        .reset_index(name="vle_clicks_forumng")
    )

    oucontent_mask = vle_typed["activity_type"] == "oucontent"
    oucontent_agg = (
        vle_typed[oucontent_mask]
        .groupby(canonical_grain, observed=True)["sum_click"]
        .sum()
        .reset_index(name="vle_clicks_oucontent")
    )

    quiz_mask = vle_typed["activity_type"] == "quiz"
    quiz_agg = (
        vle_typed[quiz_mask]
        .groupby(canonical_grain, observed=True)["sum_click"]
        .sum()
        .reset_index(name="vle_clicks_quiz")
    )

    # 7. Merge all feature aggregations onto base cohort
    merged = base.merge(base_agg, on=canonical_grain, how="left")
    merged = merged.merge(precourse_agg, on=canonical_grain, how="left")
    merged = merged.merge(recent_agg, on=canonical_grain, how="left")
    merged = merged.merge(vel_num_agg, on=canonical_grain, how="left")
    merged = merged.merge(vel_den_agg, on=canonical_grain, how="left")
    merged = merged.merge(forum_agg, on=canonical_grain, how="left")
    merged = merged.merge(oucontent_agg, on=canonical_grain, how="left")
    merged = merged.merge(quiz_agg, on=canonical_grain, how="left")

    # 8. Clean up missing values and calculate recency & velocity
    # Total clicks and activity flags
    merged["vle_total_clicks_window"] = merged["vle_total_clicks_window"].fillna(0).astype("Int64")
    merged["vle_active_days_window"] = merged["vle_active_days_window"].fillna(0).astype("Int64")
    merged["vle_unique_sites_accessed"] = merged["vle_unique_sites_accessed"].fillna(0).astype("Int64")
    merged["vle_pre_course_clicks"] = merged["vle_pre_course_clicks"].fillna(0).astype("Int64")
    merged["vle_recent_clicks_14d"] = merged["vle_recent_clicks_14d"].fillna(0).astype("Int64")
    merged["vle_clicks_forumng"] = merged["vle_clicks_forumng"].fillna(0).astype("Int64")
    merged["vle_clicks_oucontent"] = merged["vle_clicks_oucontent"].fillna(0).astype("Int64")
    merged["vle_clicks_quiz"] = merged["vle_clicks_quiz"].fillna(0).astype("Int64")

    # Companion indicator for zero engagement
    has_activity = merged["vle_total_clicks_window"] > 0
    merged["has_no_vle_activity_by_cutoff"] = (~has_activity).astype(int).astype("Int64")

    # Recency with sentinel 65:
    # 35 - max_vle_date for active students, exactly 65 for inactive students
    days_since = np.where(
        merged["max_vle_date"].notna(),
        cutoff_day - merged["max_vle_date"],
        65,
    )
    merged["days_since_last_vle_interaction"] = pd.Series(days_since, index=merged.index).astype("Int64")

    # Velocity ratio:
    # (clicks[29, 35] / 7) / ((clicks[0, 28] / 29) + 1.0)
    c_num = merged["clicks_29_35"].fillna(0).astype(float)
    c_den = merged["clicks_0_28"].fillna(0).astype(float)
    num_rate = c_num / 7.0
    den_rate = (c_den / 29.0) + 1.0
    velocity = (num_rate / den_rate).round(4)
    merged["vle_velocity_ratio"] = pd.Series(velocity, index=merged.index, dtype="float64")

    # Drop temporary intermediate columns
    drop_cols = ["max_vle_date", "clicks_29_35", "clicks_0_28"]
    merged.drop(columns=drop_cols, inplace=True)

    return merged
