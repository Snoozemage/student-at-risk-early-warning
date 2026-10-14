"""Publication-quality visualization and plotting utilities for reports and dashboard."""

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from student_retention.paths import TABLES_DIR

logger = logging.getLogger(__name__)

# Modern theme styling
COLORS = {
    "primary": "#1E3A8A",      # Deep blue
    "secondary": "#0D9488",    # Teal
    "urgent": "#DC2626",       # Red
    "watch": "#D97706",        # Amber
    "standard": "#2563EB",     # Blue
    "neutral": "#64748B",      # Slate
    "light_bg": "#F8FAFC",     # Light gray/slate
    "border": "#CBD5E1",       # Border slate
}


def create_tier_distribution_chart(
    tier_counts: dict[str, int],
) -> plt.Figure:
    """Create a clean bar chart showing student distribution across priority tiers."""
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)

    tiers = ["Tier 1 (Urgent)", "Tier 2 (Watch)", "Tier 3 (Standard)"]
    counts = [tier_counts.get(t, 0) for t in tiers]
    colors = [COLORS["urgent"], COLORS["watch"], COLORS["standard"]]

    bars = ax.barh(tiers, counts, color=colors, height=0.55, edgecolor="#334155", linewidth=0.8)

    ax.set_xlabel("Number of Student Enrollment Records", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Caseload Distribution by Provisional Priority Tier (K=500 per Term)", fontsize=11, fontweight="bold", pad=12, color="#0F172A")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    max_count = max(counts) if counts else 100
    ax.set_xlim(0, max_count * 1.25 + 10)

    # Annotate bars
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        pct = (w / sum(counts) * 100) if sum(counts) > 0 else 0
        ax.text(w + (max_count * 0.02), y, f"{int(w):,} ({pct:.1f}%)", va="center", ha="left", fontsize=9, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    return fig


def create_weekly_activity_chart(
    activity_csv_path: Path | None = None,
) -> plt.Figure:
    """Create longitudinal line chart of weekly VLE activity across outcome groups."""
    path = activity_csv_path or (TABLES_DIR / "weekly_activity_curves.csv")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    if not path.exists():
        ax.text(0.5, 0.5, "Weekly activity data not available", ha="center", va="center")
        return fig

    df = pd.read_csv(path)
    # Pivot or extract curves
    if "study_week" in df.columns:
        weeks = df["study_week"].tolist()
        if "mean_clicks_withdrawn" in df.columns and "mean_clicks_retained" in df.columns:
            ax.plot(weeks, df["mean_clicks_retained"], marker="o", color=COLORS["secondary"], label="Completers (Pass/Dist/Fail)", linewidth=2.2)
            ax.plot(weeks, df["mean_clicks_withdrawn"], marker="s", color=COLORS["urgent"], label="Withdrawn (> Day 35)", linewidth=2.2, linestyle="--")

    ax.axvline(x=5, color="#64748B", linestyle=":", linewidth=1.5, label="Day 35 Cutoff (Week 5)")
    ax.set_xlabel("Study Week Relative to Course Start", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Mean VLE Clicks per Student", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("VLE Digital Engagement Trajectory up to Day 35 Cutoff", fontsize=11, fontweight="bold", pad=12, color="#0F172A")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9)

    plt.tight_layout()
    return fig


def create_subgroup_disparity_chart(
    audit_df: pd.DataFrame,
    attribute: str = "gender",
    metric_col: str = "disparate_impact_ratio",
) -> plt.Figure:
    """Create a horizontal bar chart of disparate impact ratios for a demographic attribute."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    sub_df = audit_df[audit_df["protected_attribute"] == attribute].copy()
    if "is_statistically_stable" not in sub_df.columns:
        ax.text(0.5, 0.5, "Subgroup stability is unavailable; parity chart omitted", ha="center", va="center")
        return fig
    stable = sub_df["is_statistically_stable"].astype(str).str.lower().isin(["true", "1"])
    sub_df = sub_df.loc[stable].copy()
    if metric_col not in sub_df.columns:
        ax.text(0.5, 0.5, f"Metric not available: {metric_col}", ha="center", va="center")
        return fig
    sub_df[metric_col] = pd.to_numeric(sub_df[metric_col], errors="coerce")
    sub_df = sub_df[sub_df[metric_col].map(lambda value: pd.notna(value) and math.isfinite(float(value)))].copy()
    if sub_df.empty:
        ax.text(0.5, 0.5, f"No stable, defined {metric_col} values for attribute: {attribute}", ha="center", va="center")
        return fig

    # Sort by group
    sub_df = sub_df.sort_values(by="subgroup", ascending=False)
    groups = sub_df["subgroup"].tolist()
    ratios = sub_df[metric_col].tolist()

    # Color by 4/5ths threshold (0.80 - 1.25)
    bar_colors = []
    for r in ratios:
        if r < 0.80 or r > 1.25:
            bar_colors.append(COLORS["urgent"])
        else:
            bar_colors.append(COLORS["secondary"])

    bars = ax.barh(groups, ratios, color=bar_colors, height=0.55, edgecolor="#334155", linewidth=0.8)

    # Four-fifths reference bands
    ax.axvline(x=1.0, color="#1E293B", linestyle="-", linewidth=1.2, label="Parity (1.00)")
    ax.axvline(x=0.80, color=COLORS["urgent"], linestyle="--", linewidth=1.2, label="4/5ths Rule Lower (0.80)")
    ax.axvline(x=1.25, color=COLORS["urgent"], linestyle=":", linewidth=1.2, label="4/5ths Rule Upper (1.25)")

    ax.set_xlabel("Disparate Impact Ratio (Selection Rate vs Reference)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title(f"Stable Subgroup Parity Audit: {attribute.replace('_', ' ').title()}", fontsize=11, fontweight="bold", pad=12, color="#0F172A")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=8, loc="lower right")

    max_r = max(max(ratios), 1.35) if ratios else 1.5
    ax.set_xlim(0, max_r * 1.15)

    for bar, r in zip(bars, ratios):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2.0
        ax.text(w + 0.02, y, f"{r:.3f}", va="center", ha="left", fontsize=9, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    return fig


def create_top_k_enrichment_chart(top_k_df: pd.DataFrame) -> plt.Figure:
    """Create a pooled-test diagnostic chart comparing Precision@K to prevalence."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    valid_df = top_k_df.copy()
    if "evaluation_scope" in valid_df.columns:
        valid_df = valid_df[valid_df["evaluation_scope"] == "pooled_test_cohort"]
    if "presentation_code" in valid_df.columns:
        valid_df = valid_df[valid_df["presentation_code"] == "ALL_POOLED"]
    if "model_key" in valid_df.columns:
        valid_df = valid_df[valid_df["model_key"] == "random_forest"]
    if "is_valid_capacity" in valid_df.columns:
        valid_df = valid_df[valid_df["is_valid_capacity"].astype(str).str.lower().isin(["true", "1"])]
    if "random_precision_baseline" not in valid_df.columns and {"total_positives", "n_samples"}.issubset(valid_df.columns):
        n = pd.to_numeric(valid_df["n_samples"], errors="coerce")
        positives = pd.to_numeric(valid_df["total_positives"], errors="coerce")
        valid_df["random_precision_baseline"] = positives / n.where(n > 0)
    if {"k", "precision_at_k", "random_precision_baseline"}.issubset(valid_df.columns):
        for column in ("k", "precision_at_k", "random_precision_baseline"):
            valid_df[column] = pd.to_numeric(valid_df[column], errors="coerce")
        valid_df = valid_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["k", "precision_at_k", "random_precision_baseline"])
    else:
        valid_df = pd.DataFrame()
    if valid_df.empty:
        ax.text(0.5, 0.5, "No valid capacity data available", ha="center", va="center")
        return fig

    k_vals = valid_df["k"].tolist()
    prec_vals = (valid_df["precision_at_k"] * 100).tolist()
    base_vals = (valid_df["random_precision_baseline"] * 100).tolist()

    x = np.arange(len(k_vals))
    width = 0.35

    rects1 = ax.bar(x - width / 2, prec_vals, width, label="Model Precision@K", color=COLORS["primary"], edgecolor="#1E293B")
    rects2 = ax.bar(x + width / 2, base_vals, width, label="Baseline (Cohort Prior)", color=COLORS["neutral"], edgecolor="#1E293B")

    ax.set_xlabel("Advisor Capacity Limit (K)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Precision (% True At-Risk Identified)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Pooled Random-Test Top-K Diagnostic vs Cohort Prevalence", fontsize=11, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_vals], fontweight="bold")
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add value labels
    for bar in rects1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for bar in rects2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#475569")

    max_val = max(prec_vals + base_vals) if prec_vals else 50
    ax.set_ylim(0, max_val * 1.25 + 5)

    plt.tight_layout()
    return fig
