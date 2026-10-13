"""VLE digital engagement extraction, trajectory analysis, and pacing metrics."""
import numpy as np
import pandas as pd


def extract_cutoff_vle_features(
    cohort_df: pd.DataFrame,
    student_vle_df: pd.DataFrame,
    vle_df: pd.DataFrame | None = None,
    cutoff_day: int = 35,
) -> pd.DataFrame:
    """Extract digital engagement metrics up to the prediction cutoff.

    Calculates interaction intensity, active days, breadth of resource usage,
    pre-course vs recent interaction volumes, click velocity, and recency.

    Args:
        cohort_df: Eligible cohort DataFrame with canonical keys.
        student_vle_df: student_vle table containing interaction logs.
        vle_df: Optional vle reference table for activity type segmentation.
        cutoff_day: Relative day cutoff (default: 35).

    Returns:
        DataFrame: Cohort rows with added VLE engagement features.
    """
    grain = ["code_module", "code_presentation", "id_student"]

    # Filter interaction logs strictly up to cutoff
    vle_cutoff = student_vle_df[
        (student_vle_df["date"] <= cutoff_day) & (student_vle_df["date"] >= -25)
    ].copy()

    # Pre-aggregate basic metrics
    agg_dict = {
        "vle_total_clicks": ("sum_click", "sum"),
        "vle_active_days": ("date", "nunique"),
        "vle_distinct_sites": ("id_site", "nunique"),
        "vle_last_interaction_day": ("date", "max"),
    }

    base_agg = vle_cutoff.groupby(grain).agg(**agg_dict).reset_index()

    # Window-specific metrics: Pre-course [-25, -1], Early [0, 21], Recent [22, 35]
    precourse_mask = vle_cutoff["date"] < 0
    recent_mask = (vle_cutoff["date"] >= 22) & (vle_cutoff["date"] <= cutoff_day)

    precourse_agg = (
        vle_cutoff[precourse_mask]
        .groupby(grain)["sum_click"]
        .sum()
        .reset_index(name="vle_precourse_clicks")
    )

    recent_agg = (
        vle_cutoff[recent_mask]
        .groupby(grain)["sum_click"]
        .sum()
        .reset_index(name="vle_recent_clicks")
    )

    # Activity type specific metrics if vle_df provided
    quiz_agg = None
    forum_agg = None
    if vle_df is not None:
        vle_with_type = vle_cutoff.merge(
            vle_df[["id_site", "activity_type"]],
            on="id_site",
            how="left",
        )
        quiz_mask = vle_with_type["activity_type"] == "quiz"
        forum_mask = vle_with_type["activity_type"] == "forumng"

        quiz_agg = (
            vle_with_type[quiz_mask]
            .groupby(grain)["sum_click"]
            .sum()
            .reset_index(name="vle_quiz_clicks")
        )
        forum_agg = (
            vle_with_type[forum_mask]
            .groupby(grain)["sum_click"]
            .sum()
            .reset_index(name="vle_forum_clicks")
        )

    # Merge into cohort
    merged = cohort_df.merge(base_agg, on=grain, how="left")
    merged = merged.merge(precourse_agg, on=grain, how="left")
    merged = merged.merge(recent_agg, on=grain, how="left")

    if quiz_agg is not None:
        merged = merged.merge(quiz_agg, on=grain, how="left")
        merged["vle_quiz_clicks"] = merged["vle_quiz_clicks"].fillna(0).astype("Int64")

    if forum_agg is not None:
        merged = merged.merge(forum_agg, on=grain, how="left")
        merged["vle_forum_clicks"] = merged["vle_forum_clicks"].fillna(0).astype("Int64")

    # Handle missing/zero engagement
    merged["vle_total_clicks"] = merged["vle_total_clicks"].fillna(0).astype("Int64")
    merged["vle_active_days"] = merged["vle_active_days"].fillna(0).astype("Int64")
    merged["vle_distinct_sites"] = merged["vle_distinct_sites"].fillna(0).astype("Int64")
    merged["vle_precourse_clicks"] = merged["vle_precourse_clicks"].fillna(0).astype("Int64")
    merged["vle_recent_clicks"] = merged["vle_recent_clicks"].fillna(0).astype("Int64")

    # Inactivity representation: sentinel 65 if no clicks recorded in [-25, 35]
    has_activity = merged["vle_last_interaction_day"].notna()
    days_since = np.where(
        has_activity,
        cutoff_day - merged["vle_last_interaction_day"],
        65,
    )
    merged["days_since_last_vle"] = pd.Series(days_since, index=merged.index).astype("Int64")
    merged.drop(columns=["vle_last_interaction_day"], inplace=True)

    # Click velocity ratio: (recent_clicks / 14) / (total_clicks / 61)
    # Total observation window is 61 days (Day -25 through Day 35 inclusive)
    recent_rate = merged["vle_recent_clicks"].astype(float) / 14.0
    overall_rate = merged["vle_total_clicks"].astype(float) / 61.0
    velocity = np.where(
        overall_rate > 0,
        recent_rate / overall_rate,
        0.0,
    )
    merged["vle_click_velocity_ratio"] = pd.Series(velocity, index=merged.index).round(4)

    return merged


def compute_weekly_vle_trajectory(
    cohort_df: pd.DataFrame,
    student_vle_df: pd.DataFrame,
    cutoff_day: int = 35,
) -> pd.DataFrame:
    """Compute weekly average click trajectories from Week -3 to Week 5 stratified by outcome.

    Week assignment:
        Week -3: [-25, -15]
        Week -2: [-14, -8]
        Week -1: [-7, -1]
        Week 0:  [0, 6]
        Week 1:  [7, 13]
        Week 2:  [14, 20]
        Week 3:  [21, 27]
        Week 4:  [28, 34]
        Week 5:  [35, 35]

    Returns:
        DataFrame: Table of week index, week label, outcome group, mean clicks, median clicks.
    """
    grain = ["code_module", "code_presentation", "id_student"]

    # Filter interaction logs to [-25, cutoff_day]
    vle_filtered = student_vle_df[
        (student_vle_df["date"] >= -25) & (student_vle_df["date"] <= cutoff_day)
    ].copy()

    # Assign week bin
    conditions = [
        vle_filtered["date"] < -14,
        (vle_filtered["date"] >= -14) & (vle_filtered["date"] < -7),
        (vle_filtered["date"] >= -7) & (vle_filtered["date"] < 0),
        (vle_filtered["date"] >= 0) & (vle_filtered["date"] < 7),
        (vle_filtered["date"] >= 7) & (vle_filtered["date"] < 14),
        (vle_filtered["date"] >= 14) & (vle_filtered["date"] < 21),
        (vle_filtered["date"] >= 21) & (vle_filtered["date"] < 28),
        (vle_filtered["date"] >= 28) & (vle_filtered["date"] < 35),
        vle_filtered["date"] == 35,
    ]
    week_names = ["W-3", "W-2", "W-1", "W0", "W1", "W2", "W3", "W4", "W5"]
    week_nums = [-3, -2, -1, 0, 1, 2, 3, 4, 5]

    vle_filtered["week_num"] = np.select(conditions, week_nums, default=np.nan)
    vle_filtered["week_name"] = np.select(conditions, week_names, default="Other")

    # Aggregate by student and week
    student_week_clicks = (
        vle_filtered.groupby(grain + ["week_num", "week_name"])["sum_click"]
        .sum()
        .reset_index(name="weekly_clicks")
    )

    # Cross join cohort with all 9 weeks so students with 0 clicks in a week have 0
    all_weeks_df = pd.DataFrame({
        "week_num": week_nums,
        "week_name": week_names,
    })
    cohort_keys = cohort_df[grain + ["withdrawn_after_cutoff"]].copy()

    grid = cohort_keys.merge(all_weeks_df, how="cross")
    merged_grid = grid.merge(
        student_week_clicks,
        on=grain + ["week_num", "week_name"],
        how="left",
    )
    merged_grid["weekly_clicks"] = merged_grid["weekly_clicks"].fillna(0)

    # Summarize by outcome and week
    trajectory = (
        merged_grid.groupby(["week_num", "week_name", "withdrawn_after_cutoff"])
        .agg(
            mean_clicks=("weekly_clicks", "mean"),
            median_clicks=("weekly_clicks", "median"),
            q75_clicks=("weekly_clicks", lambda s: float(s.quantile(0.75))),
            q25_clicks=("weekly_clicks", lambda s: float(s.quantile(0.25))),
        )
        .reset_index()
    )

    trajectory["mean_clicks"] = trajectory["mean_clicks"].round(2)
    trajectory["outcome_label"] = trajectory["withdrawn_after_cutoff"].map(
        {0: "Retained (Completers)", 1: "Withdrawn After Day 35"}
    )

    return trajectory.sort_values(by=["week_num", "withdrawn_after_cutoff"]).reset_index(drop=True)
