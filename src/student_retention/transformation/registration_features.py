"""Registration timing feature extraction for Day 35 cutoff."""
import numpy as np
import pandas as pd


def extract_registration_features(
    cohort_df: pd.DataFrame,
    student_registration_df: pd.DataFrame | None = None,
    cutoff_day: int = 35,
) -> pd.DataFrame:
    """Extract registration lead time and late registration indicators for cohort.

    Sign Convention:
        registration_lead_time_days = -date_registration
        - Positive for advance registration prior to Day 0 (e.g., date -30 -> lead time +30 days).
        - Zero for registration on Day 0.
        - Negative for late registration after Day 0 (e.g., date +10 -> lead time -10 days).

    Late Registration Indicator:
        is_late_registration = 1 if 0 < date_registration <= cutoff_day, 0 if date_registration <= 0.
        Evaluates to NaN if date_registration is missing/null (under Policy B sensitivity).

    Args:
        cohort_df: Eligible cohort DataFrame containing canonical keys.
        student_registration_df: Optional student_registration table if date_registration not in cohort_df.
        cutoff_day: Day cutoff for eligible late registration (default: 35).

    Returns:
        DataFrame with canonical keys, 'registration_lead_time_days', and 'is_late_registration'.
    """
    if cutoff_day != 35:
        raise ValueError(
            f"Unsupported cutoff_day={cutoff_day}. The Day-35 early warning feature engineering pipeline "
            f"strictly specifies and enforces cutoff_day=35."
        )

    canonical_grain = ["code_module", "code_presentation", "id_student"]

    df = cohort_df[canonical_grain].copy()

    if "date_registration" in cohort_df.columns:
        reg_dates = cohort_df["date_registration"]
    elif student_registration_df is not None:
        merged = df.merge(
            student_registration_df[canonical_grain + ["date_registration"]],
            on=canonical_grain,
            how="left",
        )
        reg_dates = merged["date_registration"]
    else:
        raise ValueError(
            "date_registration must be present in cohort_df or student_registration_df provided"
        )

    # Lead time: -date_registration
    lead_time = -reg_dates.astype("float64")

    # Late registration: 1 if 0 < date <= cutoff_day, 0 if date <= 0, NaN if date is null
    is_late = pd.Series(np.nan, index=df.index, dtype="float64")
    not_null_mask = reg_dates.notna()

    is_late[not_null_mask] = np.where(
        (reg_dates[not_null_mask] > 0) & (reg_dates[not_null_mask] <= cutoff_day),
        1.0,
        0.0,
    )

    df["registration_lead_time_days"] = lead_time
    df["is_late_registration"] = is_late

    return df
