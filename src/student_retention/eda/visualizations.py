"""Publication-quality visualization generation for EDA findings."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plotting_theme():
    """Configure modern, publication-ready seaborn/matplotlib style."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    })


def plot_presentation_withdrawal_rates(
    presentation_df: pd.DataFrame,
    save_path: Path | str,
) -> None:
    """Plot withdrawal rate variation across all 22 module-presentations."""
    setup_plotting_theme()
    fig, ax = plt.subplots(figsize=(12, 9))

    df_plot = presentation_df.copy()
    if "offering" not in df_plot.columns:
        if "code_module" in df_plot.columns and "code_presentation" in df_plot.columns:
            df_plot["offering"] = df_plot["code_module"] + " " + df_plot["code_presentation"]
        elif "presentation" in df_plot.columns:
            df_plot["offering"] = df_plot["presentation"]
        else:
            df_plot["offering"] = df_plot["code_presentation"]

    df_sorted = df_plot.sort_values("withdrawal_rate_pct", ascending=True).reset_index(drop=True)

    if "season" not in df_sorted.columns and "code_presentation" in df_sorted.columns:
        df_sorted["season"] = df_sorted["code_presentation"].str[-1]

    if "season" in df_sorted.columns:
        season_map = {
            "B": "February (B)",
            "J": "October (J)",
            "February": "February (B)",
            "October": "October (J)",
            "February (B)": "February (B)",
            "October (J)": "October (J)",
        }
        df_sorted["season_label"] = df_sorted["season"].map(season_map).fillna(df_sorted["season"])
        unique_labels = df_sorted["season_label"].unique()
        base_palette = {"February (B)": "#d62728", "October (J)": "#1f77b4"}
        if all(label in base_palette for label in unique_labels):
            palette = base_palette
        else:
            palette = "deep"
        hue_col = "season_label"
    else:
        palette = None
        hue_col = None

    sns.barplot(
        data=df_sorted,
        x="withdrawal_rate_pct",
        y="offering",
        hue=hue_col,
        palette=palette,
        ax=ax,
    )

    # Annotate bar values
    for p in ax.patches:
        width = p.get_width()
        if width > 0 and not np.isnan(width):
            ax.annotate(
                f"{width:.1f}%",
                (width + 0.3, p.get_y() + p.get_height() / 2),
                ha="left",
                va="center",
                fontsize=8.5,
                color="#333333",
            )

    ax.set_title("Withdrawal Rates Across All 22 Course Offerings (Post-Day 35)")
    ax.set_xlabel("Withdrawal Rate (%)")
    ax.set_ylabel("Course Offering (Module-Presentation)")
    ax.set_xlim(0, max(df_sorted["withdrawal_rate_pct"]) * 1.15)
    if hue_col:
        ax.legend(title="Presentation Season", loc="lower right")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_vle_clicks_distribution(
    cohort_metrics_df: pd.DataFrame,
    save_path: Path | str,
) -> None:
    """Plot distribution of total clicks up to Day 35 on log10 scale by outcome."""
    setup_plotting_theme()
    fig, ax = plt.subplots(figsize=(10, 5))

    df = cohort_metrics_df.copy()
    df["log10_clicks"] = np.log10(df["vle_total_clicks"].astype(float) + 1.0)
    df["outcome_label"] = df["withdrawn_after_cutoff"].map(
        {0: "Retained (Completers)", 1: "Withdrawn After Day 35"}
    )

    palette = {"Retained (Completers)": "#1f77b4", "Withdrawn After Day 35": "#d62728"}

    sns.kdeplot(
        data=df,
        x="log10_clicks",
        hue="outcome_label",
        common_norm=False,
        fill=True,
        alpha=0.35,
        palette=palette,
        ax=ax,
    )

    # Median lines
    med_ret = df[df["withdrawn_after_cutoff"] == 0]["log10_clicks"].median()
    med_wd = df[df["withdrawn_after_cutoff"] == 1]["log10_clicks"].median()

    ax.axvline(med_ret, color="#1f77b4", linestyle="--", linewidth=1.5, label=f"Retained Median (10^{med_ret:.2f})")
    ax.axvline(med_wd, color="#d62728", linestyle="--", linewidth=1.5, label=f"Withdrawn Median (10^{med_wd:.2f})")

    ax.set_title("Early VLE Interaction Intensity: Total Clicks by Day 35 (Log10 Scale)")
    ax.set_xlabel("Log10(Total Clicks + 1)")
    ax.set_ylabel("Kernel Density")
    ax.legend(loc="upper left")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_weekly_activity_curves(
    weekly_trajectory_df: pd.DataFrame,
    save_path: Path | str,
) -> None:
    """Plot average weekly clicks from Week -3 to Week 5 by outcome group."""
    setup_plotting_theme()
    fig, ax = plt.subplots(figsize=(10, 5))

    palette = {"Retained (Completers)": "#1f77b4", "Withdrawn After Day 35": "#d62728"}

    sns.lineplot(
        data=weekly_trajectory_df,
        x="week_name",
        y="mean_clicks",
        hue="outcome_label",
        style="outcome_label",
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8,
        palette=palette,
        ax=ax,
    )

    ax.axvline(x="W0", color="gray", linestyle=":", alpha=0.7, label="Module Start (Day 0)")
    ax.set_title("Digital Engagement Trajectory: Mean Weekly Clicks (Orientation to Cutoff)")
    ax.set_xlabel("Study Week (Relative to Module Start Day 0)")
    ax.set_ylabel("Mean Weekly Clicks per Student")
    ax.legend(loc="upper left")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_registration_lead_time(
    cohort_df: pd.DataFrame,
    save_path: Path | str,
) -> None:
    """Plot registration lead time distribution comparing outcome groups."""
    setup_plotting_theme()
    fig, ax = plt.subplots(figsize=(10, 5))

    df = cohort_df.copy()
    # Registration lead time in days before start (positive days before start)
    # date_registration is negative for pre-start registrations (-30 = 30 days before start)
    lead_time = -df["date_registration"].astype(float)
    df["lead_time_days"] = lead_time
    df["outcome_label"] = df["withdrawn_after_cutoff"].map(
        {0: "Retained (Completers)", 1: "Withdrawn After Day 35"}
    )

    palette = {"Retained (Completers)": "#1f77b4", "Withdrawn After Day 35": "#d62728"}

    # Filter extreme outliers for clear visualization (< 250 days)
    filtered = df[(df["lead_time_days"] >= -35) & (df["lead_time_days"] <= 200)]

    sns.boxplot(
        data=filtered,
        x="outcome_label",
        y="lead_time_days",
        hue="outcome_label",
        palette=palette,
        legend=False,
        ax=ax,
        width=0.4,
    )

    ax.set_title("Registration Lead Time Distribution (Days Prior to Course Start)")
    ax.set_xlabel("Student Outcome Group")
    ax.set_ylabel("Lead Time (Days Before Start)")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_assessment_submission_progress(
    cohort_metrics_df: pd.DataFrame,
    save_path: Path | str,
) -> None:
    """Plot assessment submission rates and early missed assignment proportions."""
    setup_plotting_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    subset = cohort_metrics_df[cohort_metrics_df["assessments_due_count"] > 0].copy()
    subset["outcome_label"] = subset["withdrawn_after_cutoff"].map(
        {0: "Retained", 1: "Withdrawn"}
    )

    palette = {"Retained": "#1f77b4", "Withdrawn": "#d62728"}

    # Left: Submission rate distribution
    sns.barplot(
        data=subset,
        x="outcome_label",
        y="submission_rate",
        hue="outcome_label",
        palette=palette,
        legend=False,
        ax=ax1,
        capsize=0.1,
    )
    ax1.set_title("Mean Submission Rate for Due Assessments")
    ax1.set_xlabel("Outcome Group")
    ax1.set_ylabel("Proportion Submitted (<= Day 35)")
    ax1.set_ylim(0, 1.05)

    # Right: Percentage of students missing early assessment
    pct_missed = (
        subset.groupby("outcome_label")["has_missed_due_assessment"]
        .apply(lambda s: float(s.mean() * 100))
        .reset_index(name="pct_missed")
    )
    sns.barplot(
        data=pct_missed,
        x="outcome_label",
        y="pct_missed",
        hue="outcome_label",
        palette=palette,
        legend=False,
        ax=ax2,
    )
    for p in ax2.patches:
        h = p.get_height()
        ax2.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h + 1), ha="center", fontsize=10)

    ax2.set_title("Students with Missed Due Assessment (%)")
    ax2.set_xlabel("Outcome Group")
    ax2.set_ylabel("Percentage of Cohort (%)")
    ax2.set_ylim(0, max(pct_missed["pct_missed"]) * 1.25)

    plt.suptitle("Early Coursework Pacing & Milestone Adherence (Assessments Due <= Day 35)")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_correlation_matrix(
    cohort_metrics_df: pd.DataFrame,
    feature_cols: list[str],
    save_path: Path | str,
) -> None:
    """Plot Spearman correlation heatmap between candidate early predictors and target."""
    setup_plotting_theme()
    fig, ax = plt.subplots(figsize=(9, 7))

    cols = [c for c in feature_cols if c in cohort_metrics_df.columns] + ["withdrawn_after_cutoff"]
    corr_matrix = cohort_metrics_df[cols].astype(float).corr(method="spearman")

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-0.4,
        vmax=0.4,
        cbar_kws={"label": "Spearman Correlation"},
        ax=ax,
    )

    ax.set_title("Spearman Rank Correlation: Early Features and Withdrawal Target")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
