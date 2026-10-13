"""Statistical test helpers and effect size computations for EDA."""
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def compute_cohens_d(group1: pd.Series | np.ndarray, group2: pd.Series | np.ndarray) -> float:
    """Compute Cohen's d effect size between two independent groups.

    Formula:
        d = (mean1 - mean2) / s_pooled
        s_pooled = sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2))

    Returns:
        float: Standardized mean difference (Cohen's d).
    """
    s1 = pd.Series(group1).dropna().to_numpy(dtype=float)
    s2 = pd.Series(group2).dropna().to_numpy(dtype=float)

    n1, n2 = len(s1), len(s2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = np.mean(s1), np.mean(s2)
    var1, var2 = np.var(s1, ddof=1), np.var(s2, ddof=1)

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return 0.0

    return float((mean1 - mean2) / np.sqrt(pooled_var))


def compute_cliffs_delta(group1: pd.Series | np.ndarray, group2: pd.Series | np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size between two independent groups.

    Formula:
        delta = (2 * U / (n1 * n2)) - 1
        where U is the Mann-Whitney U statistic for group1.

    Cliff's delta ranges from -1 to 1:
        |delta| < 0.147: negligible
        0.147 <= |delta| < 0.330: small
        0.330 <= |delta| < 0.474: medium
        |delta| >= 0.474: large

    Returns:
        float: Cliff's delta effect size.
    """
    s1 = pd.Series(group1).dropna().to_numpy(dtype=float)
    s2 = pd.Series(group2).dropna().to_numpy(dtype=float)

    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0:
        return 0.0

    u_stat, _ = stats.mannwhitneyu(s1, s2, alternative="two-sided")
    delta = (2.0 * u_stat / (n1 * n2)) - 1.0
    return float(delta)


def compute_odds_ratio(
    exposed_cases: int,
    exposed_non_cases: int | None = None,
    unexposed_cases: int | None = None,
    unexposed_non_cases: int | None = None,
    exposed_controls: int | None = None,
    unexposed_controls: int | None = None,
) -> dict[str, Any]:
    """Compute Odds Ratio with 95% Confidence Interval using Haldane-Anscombe correction if zero cells exist.

    Args:
        exposed_cases: A (event in exposed)
        exposed_non_cases: B (non-event in exposed / controls)
        unexposed_cases: C (event in unexposed)
        unexposed_non_cases: D (non-event in unexposed / controls)
        exposed_controls: Alias for exposed_non_cases
        unexposed_controls: Alias for unexposed_non_cases

    Returns:
        dict: {'odds_ratio', 'ci_lower', 'ci_upper', 'p_value', 'a', 'b', 'c', 'd'}
    """
    b_val = exposed_non_cases if exposed_non_cases is not None else (exposed_controls if exposed_controls is not None else 0)
    d_val = unexposed_non_cases if unexposed_non_cases is not None else (unexposed_controls if unexposed_controls is not None else 0)
    c_val = unexposed_cases if unexposed_cases is not None else 0

    a, b, c, d = float(exposed_cases), float(b_val), float(c_val), float(d_val)

    # Haldane-Anscombe correction if any cell is 0
    if a == 0 or b == 0 or c == 0 or d == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    odds_ratio = (a * d) / (b * c)
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt((1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d))

    ci_lower = float(np.exp(log_or - 1.96 * se_log_or))
    ci_upper = float(np.exp(log_or + 1.96 * se_log_or))

    # Fisher exact test
    table = [[int(exposed_cases), int(b_val)], [int(c_val), int(d_val)]]
    _, p_val = stats.fisher_exact(table)

    return {
        "odds_ratio": round(float(odds_ratio), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": float(p_val),
        "a": int(exposed_cases),
        "b": int(b_val),
        "c": int(c_val),
        "d": int(d_val),
    }


def compare_numeric_distributions(
    df: pd.DataFrame,
    outcome_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compare numeric features between positive (1) and negative (0) outcome groups.

    Computes median, mean, IQR, Mann-Whitney U test statistic, p-value, Cohen's d, and Cliff's delta.
    """
    rows = []
    g0 = df[df[outcome_col] == 0]
    g1 = df[df[outcome_col] == 1]

    for col in feature_cols:
        if col not in df.columns:
            continue

        v0 = g0[col].dropna()
        v1 = g1[col].dropna()

        if len(v0) == 0 or len(v1) == 0:
            continue

        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(v1, v0, alternative="two-sided")
        cohens_d = compute_cohens_d(v1, v0)
        cliffs_delta = compute_cliffs_delta(v1, v0)

        rows.append({
            "feature": col,
            "n_retained": len(v0),
            "n_withdrawn": len(v1),
            "retained_mean": round(float(v0.mean()), 2),
            "retained_median": round(float(v0.median()), 2),
            "retained_iqr": round(float(v0.quantile(0.75) - v0.quantile(0.25)), 2),
            "withdrawn_mean": round(float(v1.mean()), 2),
            "withdrawn_median": round(float(v1.median()), 2),
            "withdrawn_iqr": round(float(v1.quantile(0.75) - v1.quantile(0.25)), 2),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "cohens_d": round(cohens_d, 4),
            "cliffs_delta": round(cliffs_delta, 4),
        })

    return pd.DataFrame(rows)


def compare_missingness_by_outcome(
    df: pd.DataFrame,
    outcome_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compare missingness proportions between outcome groups with Fisher exact tests."""
    rows = []
    g0 = df[df[outcome_col] == 0]
    g1 = df[df[outcome_col] == 1]

    n0 = len(g0)
    n1 = len(g1)

    for col in feature_cols:
        if col not in df.columns:
            continue

        m0 = int(g0[col].isna().sum())
        m1 = int(g1[col].isna().sum())

        pct0 = round(m0 / n0 * 100, 2) if n0 > 0 else 0.0
        pct1 = round(m1 / n1 * 100, 2) if n1 > 0 else 0.0

        table = [[m1, n1 - m1], [m0, n0 - m0]]
        _, p_val = stats.fisher_exact(table)

        rows.append({
            "column": col,
            "retained_missing_count": m0,
            "retained_missing_pct": pct0,
            "withdrawn_missing_count": m1,
            "withdrawn_missing_pct": pct1,
            "delta_pct": round(pct1 - pct0, 2),
            "p_value": float(p_val),
        })

    return pd.DataFrame(rows)
