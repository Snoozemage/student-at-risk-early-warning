"""Cohort population, demographic distribution, and presentation-level variation analysis."""
from typing import Any

import pandas as pd


def summarize_cohort_demographics(cohort_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize overall cohort size, target prevalence, and demographic subgroup counts.

    Args:
        cohort_df: Analysis-ready cohort DataFrame (Policy C or B).

    Returns:
        dict: Detailed demographic breakdown and withdrawal rates by subgroup.
    """
    total = len(cohort_df)
    pos = int((cohort_df["withdrawn_after_cutoff"] == 1).sum())
    neg = int((cohort_df["withdrawn_after_cutoff"] == 0).sum())
    base_rate = round(pos / total * 100, 4) if total > 0 else 0.0

    demographic_cols = [
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]

    subgroup_summaries = {}
    for col in demographic_cols:
        if col in cohort_df.columns:
            counts = cohort_df.groupby(col, observed=True).agg(
                total_count=("withdrawn_after_cutoff", "count"),
                withdrawn_count=("withdrawn_after_cutoff", lambda s: int((s == 1).sum())),
            ).reset_index()
            counts["withdrawal_rate_pct"] = (
                counts["withdrawn_count"] / counts["total_count"] * 100
            ).round(2)
            counts["pct_of_cohort"] = (
                counts["total_count"] / total * 100
            ).round(2)
            subgroup_summaries[col] = counts.to_dict(orient="records")

    return {
        "total_records": total,
        "positive_cases": pos,
        "negative_cases": neg,
        "base_rate_pct": base_rate,
        "demographics": subgroup_summaries,
    }


def summarize_presentation_withdrawal_rates(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Compute eligible student counts, withdrawal counts, and rates for each module-presentation.

    Reports per-offering metrics across all 22 module-presentations, tagging offering season (B vs J)
    and module identifier. In the OULAD dataset specification (Kuzilek et al., 2017), modules CCC, DDD,
    EEE, FFF are classified under Science, Technology, Engineering, and Mathematics (STEM), while modules
    AAA, BBB, GGG are categorized under Social Sciences.
    """
    stem_modules = {"CCC", "DDD", "EEE", "FFF"}

    grain = ["code_module", "code_presentation"]
    summary = cohort_df.groupby(grain, observed=True).agg(
        eligible_students=("withdrawn_after_cutoff", "count"),
        withdrawn_after_cutoff=("withdrawn_after_cutoff", lambda s: int((s == 1).sum())),
        retained_students=("withdrawn_after_cutoff", lambda s: int((s == 0).sum())),
        mean_studied_credits=("studied_credits", "mean"),
        mean_prev_attempts=("num_of_prev_attempts", "mean"),
    ).reset_index()

    summary["withdrawal_rate_pct"] = (
        summary["withdrawn_after_cutoff"] / summary["eligible_students"] * 100
    ).round(2)
    summary["season"] = summary["code_presentation"].str[-1]  # 'B' or 'J'
    summary["offering"] = summary["code_module"] + " " + summary["code_presentation"]
    summary["faculty"] = summary["code_module"].apply(
        lambda m: "STEM" if m in stem_modules else "Social Sciences"
    )

    return summary.sort_values(by=["code_module", "code_presentation"]).reset_index(drop=True)


def compare_policy_sensitivities(
    policy_c_df: pd.DataFrame,
    policy_b_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compare cohort sizes, withdrawal rates, and delta between Policy C and Policy B."""
    nc = len(policy_c_df)
    nb = len(policy_b_df)
    pos_c = int((policy_c_df["withdrawn_after_cutoff"] == 1).sum())
    pos_b = int((policy_b_df["withdrawn_after_cutoff"] == 1).sum())

    rate_c = round(pos_c / nc * 100, 4) if nc > 0 else 0.0
    rate_b = round(pos_b / nb * 100, 4) if nb > 0 else 0.0

    return {
        "policy_c_eligible": nc,
        "policy_c_positive": pos_c,
        "policy_c_base_rate_pct": rate_c,
        "policy_b_eligible": nb,
        "policy_b_positive": pos_b,
        "policy_b_base_rate_pct": rate_b,
        "delta_eligible": nb - nc,
        "delta_positive": pos_b - pos_c,
        "delta_base_rate_pct": round(rate_b - rate_c, 4),
    }
