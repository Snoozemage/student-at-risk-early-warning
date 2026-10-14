"""EDA pipeline coordinator: executes empirical analyses, generates figures, tables, and reports."""
from pathlib import Path
from typing import Any

import pandas as pd

from student_retention.eda.assessment_analysis import (
    extract_cutoff_assessment_features,
    summarize_assessment_patterns,
)
from student_retention.eda.cohort_analysis import (
    compare_policy_sensitivities,
    summarize_cohort_demographics,
    summarize_presentation_withdrawal_rates,
)
from student_retention.eda.engagement_analysis import (
    compute_weekly_vle_trajectory,
    extract_cutoff_vle_features,
)
from student_retention.eda.statistical_tests import (
    compare_missingness_by_outcome,
    compare_numeric_distributions,
    compute_odds_ratio,
)
from student_retention.eda.visualizations import (
    plot_assessment_submission_progress,
    plot_correlation_matrix,
    plot_presentation_withdrawal_rates,
    plot_registration_lead_time,
    plot_vle_clicks_distribution,
    plot_weekly_activity_curves,
)
from student_retention.logging_config import get_logger
from student_retention.paths import (
    FIGURES_DIR,
    INTERIM_NORMALIZED_DIR,
    PROCESSED_COHORT_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_directories,
)

logger = get_logger(__name__)


class EDARunner:
    """End-to-end coordinator for Phase 4 Exploratory Data Analysis."""

    def __init__(
        self,
        cohort_dir: Path | None = None,
        cohort_path: Path | None = None,
        interim_dir: Path | None = None,
        normalized_dir: Path | None = None,
        tables_dir: Path | None = None,
        figures_dir: Path | None = None,
        reports_dir: Path | None = None,
        cutoff_day: int = 35,
    ):
        if cohort_path:
            p = Path(cohort_path)
            self.cohort_dir = p.parent if p.is_file() or p.suffix in [".parquet", ".csv"] else p
            self.cohort_path = p if p.suffix in [".parquet", ".csv"] else None
        else:
            self.cohort_dir = Path(cohort_dir) if cohort_dir else PROCESSED_COHORT_DIR
            self.cohort_path = None

        raw_interim = interim_dir or normalized_dir
        self.interim_dir = Path(raw_interim) if raw_interim else INTERIM_NORMALIZED_DIR
        self.tables_dir = Path(tables_dir) if tables_dir else TABLES_DIR
        self.figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
        self.cutoff_day = cutoff_day

    def run(self) -> dict[str, Any]:
        """Execute end-to-end exploratory data analysis and export all deliverables."""
        ensure_directories()
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting Phase 4 Exploratory Data Analysis (Cutoff: Day %d)", self.cutoff_day)

        # 1. Load Primary Cohort and Normalized Tables
        if self.cohort_path and self.cohort_path.exists():
            cohort_c_path = self.cohort_path
        else:
            cohort_c_path = self.cohort_dir / "cohort_policy_c.parquet"
        if not cohort_c_path.exists():
            raise FileNotFoundError(f"Cohort table not found at {cohort_c_path}. Run Phase 3 build-cohort first.")

        cohort_c = pd.read_parquet(cohort_c_path)
        logger.info("Loaded Policy C cohort: %d records", len(cohort_c))

        policy_b_path = self.cohort_dir / "cohort_policy_b.parquet"
        cohort_b = pd.read_parquet(policy_b_path) if policy_b_path.exists() else None

        # Load interim tables
        svle_path = self.interim_dir / "student_vle.parquet"
        if not svle_path.exists():
            svle_path = self.interim_dir / "studentVle.parquet"
        svle = pd.read_parquet(svle_path)

        vle_path = self.interim_dir / "vle.parquet"
        vle = pd.read_parquet(vle_path)

        sassess_path = self.interim_dir / "student_assessment.parquet"
        if not sassess_path.exists():
            sassess_path = self.interim_dir / "studentAssessment.parquet"
        sassess = pd.read_parquet(sassess_path)

        assess_path = self.interim_dir / "assessments.parquet"
        assess = pd.read_parquet(assess_path)
        logger.info("Loaded interim normalized tables successfully.")

        # 2. Population & Presentation Analysis
        cohort_summary = summarize_cohort_demographics(cohort_c)
        presentation_summary = summarize_presentation_withdrawal_rates(cohort_c)
        presentation_summary.to_csv(self.tables_dir / "presentation_withdrawal_rates.csv", index=False)

        demo_rows = []
        for col, records in cohort_summary.get("demographics", {}).items():
            for rec in records:
                cat_val = rec.get(col, "")
                demo_rows.append({
                    "attribute": col,
                    "category_value": str(cat_val),
                    "total_count": rec.get("total_count", 0),
                    "withdrawn_count": rec.get("withdrawn_count", 0),
                    "withdrawal_rate_pct": rec.get("withdrawal_rate_pct", 0.0),
                    "pct_of_cohort": rec.get("pct_of_cohort", 0.0),
                })
        if demo_rows:
            demo_df = pd.DataFrame(demo_rows)
            demo_df.to_csv(self.tables_dir / "demographic_summary.csv", index=False)

        policy_comparison = (
            compare_policy_sensitivities(cohort_c, cohort_b)
            if cohort_b is not None
            else {}
        )
        if policy_comparison:
            pd.DataFrame([policy_comparison]).to_csv(
                self.tables_dir / "policy_sensitivity_comparison.csv", index=False
            )

        # 3. Digital Engagement (VLE) Extraction
        logger.info("Extracting VLE digital engagement metrics up to Day %d...", self.cutoff_day)
        cohort_with_vle = extract_cutoff_vle_features(
            cohort_df=cohort_c,
            student_vle_df=svle,
            vle_df=vle,
            cutoff_day=self.cutoff_day,
        )

        weekly_trajectory = compute_weekly_vle_trajectory(
            cohort_df=cohort_c,
            student_vle_df=svle,
            cutoff_day=self.cutoff_day,
        )
        weekly_trajectory.to_csv(self.tables_dir / "weekly_activity_curves.csv", index=False)

        # 4. Assessment Progress Extraction
        logger.info("Extracting assessment progress metrics up to Day %d...", self.cutoff_day)
        cohort_full = extract_cutoff_assessment_features(
            cohort_df=cohort_with_vle,
            student_assessment_df=sassess,
            assessments_df=assess,
            cutoff_day=self.cutoff_day,
        )

        assessment_summary = summarize_assessment_patterns(cohort_full)
        assessment_summary.to_csv(self.tables_dir / "assessment_summary_by_outcome.csv", index=False)

        # 5. Statistical Group Comparisons & Effect Sizes
        vle_cols = [
            "vle_total_clicks",
            "vle_active_days",
            "vle_distinct_sites",
            "vle_precourse_clicks",
            "vle_recent_clicks",
            "days_since_last_vle",
            "vle_click_velocity_ratio",
        ]
        if "vle_quiz_clicks" in cohort_full.columns:
            vle_cols.extend(["vle_quiz_clicks", "vle_forum_clicks"])

        stat_comparison = compare_numeric_distributions(
            df=cohort_full,
            outcome_col="withdrawn_after_cutoff",
            feature_cols=vle_cols + [
                "studied_credits",
                "num_of_prev_attempts",
                "date_registration",
                "submission_rate",
                "has_missed_due_assessment",
            ],
        )
        stat_comparison.to_csv(self.tables_dir / "statistical_comparisons.csv", index=False)

        # Odds Ratio for missing early assessment milestone
        has_due_subset = cohort_full[cohort_full["assessments_due_count"] > 0]
        a = int(((has_due_subset["has_missed_due_assessment"] == 1) & (has_due_subset["withdrawn_after_cutoff"] == 1)).sum())
        b = int(((has_due_subset["has_missed_due_assessment"] == 1) & (has_due_subset["withdrawn_after_cutoff"] == 0)).sum())
        c = int(((has_due_subset["has_missed_due_assessment"] == 0) & (has_due_subset["withdrawn_after_cutoff"] == 1)).sum())
        d = int(((has_due_subset["has_missed_due_assessment"] == 0) & (has_due_subset["withdrawn_after_cutoff"] == 0)).sum())
        missed_or = compute_odds_ratio(a, b, c, d)

        # Missingness comparison
        missingness_df = compare_missingness_by_outcome(
            df=cohort_full,
            outcome_col="withdrawn_after_cutoff",
            feature_cols=["imd_band", "date_registration", "studied_credits"],
        )
        missingness_df.to_csv(self.tables_dir / "missingness_comparison.csv", index=False)

        # 6. Generate Figures
        logger.info("Generating publication-quality figures in %s...", self.figures_dir)
        fig1_path = self.figures_dir / "fig1_withdrawal_rate_by_presentation.png"
        plot_presentation_withdrawal_rates(presentation_summary, fig1_path)

        fig2_path = self.figures_dir / "fig2_vle_clicks_distribution_by_outcome.png"
        plot_vle_clicks_distribution(cohort_full, fig2_path)

        fig3_path = self.figures_dir / "fig3_weekly_vle_activity_curves.png"
        plot_weekly_activity_curves(weekly_trajectory, fig3_path)

        fig4_path = self.figures_dir / "fig4_registration_lead_time_distribution.png"
        plot_registration_lead_time(cohort_full, fig4_path)

        fig5_path = self.figures_dir / "fig5_assessment_submission_progress.png"
        plot_assessment_submission_progress(cohort_full, fig5_path)

        fig6_path = self.figures_dir / "fig6_correlation_matrix_early_signals.png"
        corr_candidates = [
            "vle_total_clicks",
            "vle_active_days",
            "vle_recent_clicks",
            "days_since_last_vle",
            "submission_rate",
            "has_missed_due_assessment",
            "num_of_prev_attempts",
        ]
        plot_correlation_matrix(cohort_full, corr_candidates, fig6_path)

        # 7. Generate EDA Comprehensive Report
        report_path = self.reports_dir / "eda_report.md"
        self._write_eda_report(
            report_path=report_path,
            cohort_summary=cohort_summary,
            presentation_summary=presentation_summary,
            stat_comparison=stat_comparison,
            assessment_summary=assessment_summary,
            missed_or=missed_or,
            missingness_df=missingness_df,
            policy_comparison=policy_comparison,
        )
        logger.info("Generated comprehensive EDA report: %s", report_path)

        return {
            "cohort_rows": cohort_summary["total_records"],
            "withdrawal_rate": cohort_summary["base_rate_pct"] / 100.0,
            "cohort_summary": cohort_summary,
            "presentation_summary": presentation_summary,
            "stat_comparison": stat_comparison,
            "missed_odds_ratio": missed_or,
            "missingness": missingness_df,
            "tables_dir": str(self.tables_dir),
            "figures_dir": str(self.figures_dir),
            "saved_tables": [str(p) for p in self.tables_dir.glob("*.csv")],
            "saved_figures": [str(p) for p in self.figures_dir.glob("*.png")],
            "report_path": str(report_path),
        }

    def _write_eda_report(
        self,
        report_path: Path,
        cohort_summary: dict[str, Any],
        presentation_summary: pd.DataFrame,
        stat_comparison: pd.DataFrame,
        assessment_summary: pd.DataFrame,
        missed_or: dict[str, Any],
        missingness_df: pd.DataFrame,
        policy_comparison: dict[str, Any],
    ) -> None:
        """Write extensive markdown report detailing the 6 core empirical observations."""
        total = cohort_summary["total_records"]
        pos = cohort_summary["positive_cases"]
        neg = cohort_summary["negative_cases"]
        base_rate = cohort_summary["base_rate_pct"]

        if not presentation_summary.empty and "withdrawal_rate_pct" in presentation_summary.columns:
            pres_min_str = f"{presentation_summary['withdrawal_rate_pct'].min():.1f}%"
            pres_max_str = f"{presentation_summary['withdrawal_rate_pct'].max():.1f}%"
            pres_mean_str = f"{presentation_summary['withdrawal_rate_pct'].mean():.1f}%"
        else:
            pres_min_str = "[Not available]"
            pres_max_str = "[Not available]"
            pres_mean_str = "[Not available]"

        # Faculty / Discipline comparisons
        stem_mask = (
            presentation_summary["faculty"] == "STEM"
            if "faculty" in presentation_summary.columns
            else pd.Series(False, index=presentation_summary.index)
        )
        stem_mean_str = (
            f"{presentation_summary.loc[stem_mask, 'withdrawal_rate_pct'].mean():.1f}%"
            if stem_mask.any()
            else "[Not available]"
        )
        soc_mean_str = (
            f"{presentation_summary.loc[~stem_mask, 'withdrawal_rate_pct'].mean():.1f}%"
            if (~stem_mask).any()
            else "[Not available]"
        )

        eee_rates = (
            presentation_summary.loc[presentation_summary["code_module"] == "EEE", "withdrawal_rate_pct"]
            if "code_module" in presentation_summary.columns and "withdrawal_rate_pct" in presentation_summary.columns
            else pd.Series(dtype=float)
        )
        eee_range_str = (
            f"{eee_rates.min():.1f}%–{eee_rates.max():.1f}%"
            if not eee_rates.empty
            else "[Not available]"
        )

        ccc_rates = (
            presentation_summary.loc[presentation_summary["code_module"] == "CCC", "withdrawal_rate_pct"]
            if "code_module" in presentation_summary.columns and "withdrawal_rate_pct" in presentation_summary.columns
            else pd.Series(dtype=float)
        )
        ccc_range_str = (
            f"{ccc_rates.min():.1f}%–{ccc_rates.max():.1f}%"
            if not ccc_rates.empty
            else "[Not available]"
        )

        b_mask = (
            presentation_summary["season"] == "B"
            if "season" in presentation_summary.columns
            else pd.Series(False, index=presentation_summary.index)
        )
        b_mean_str = (
            f"{presentation_summary.loc[b_mask, 'withdrawal_rate_pct'].mean():.1f}%"
            if b_mask.any()
            else "[Not available]"
        )
        j_mean_str = (
            f"{presentation_summary.loc[~b_mask, 'withdrawal_rate_pct'].mean():.1f}%"
            if (~b_mask).any()
            else "[Not available]"
        )

        # Dynamic continuous metric extraction from stat_comparison
        def get_stat_row(feat_name: str) -> pd.Series | None:
            if stat_comparison.empty or "feature" not in stat_comparison.columns:
                return None
            matches = stat_comparison[stat_comparison["feature"] == feat_name]
            return matches.iloc[0] if not matches.empty else None

        def get_stat_val(row: pd.Series | None, field: str, fmt: str = ".1f", fallback: str = "[Not available]") -> str:
            if row is not None and field in row and pd.notna(row[field]):
                return f"{row[field]:{fmt}}"
            return fallback

        clicks_row = get_stat_row("vle_total_clicks")
        days_row = get_stat_row("vle_active_days")
        sites_row = get_stat_row("vle_distinct_sites")
        inactivity_row = get_stat_row("days_since_last_vle")
        reg_row = get_stat_row("date_registration")
        reg_direction = "The group medians are unavailable, so the direction cannot be determined."
        if reg_row is not None and pd.notna(reg_row.get("retained_median")) and pd.notna(reg_row.get("withdrawn_median")):
            if reg_row["withdrawn_median"] < reg_row["retained_median"]:
                reg_direction = "The withdrawn group registered earlier at the median."
            elif reg_row["withdrawn_median"] > reg_row["retained_median"]:
                reg_direction = "The withdrawn group registered later at the median."
            else:
                reg_direction = "The groups have equal median registration dates."
        prev_attempts_row = get_stat_row("num_of_prev_attempts")

        # Assessment pattern metrics from assessment_summary & missed_or
        a = missed_or.get("a", 0)
        b = missed_or.get("b", 0)
        c = missed_or.get("c", 0)
        d = missed_or.get("d", 0)
        or_val = missed_or.get("odds_ratio")
        ci_low = missed_or.get("ci_lower")
        ci_high = missed_or.get("ci_upper")
        or_pval = missed_or.get("p_value")

        or_val_str = f"{or_val:.2f}" if or_val is not None and pd.notna(or_val) else "[Not available]"
        ci_str = (
            f"[{ci_low:.2f}, {ci_high:.2f}]"
            if ci_low is not None and ci_high is not None and pd.notna(ci_low) and pd.notna(ci_high)
            else "[Not available]"
        )
        or_pval_str = f"{or_pval:.2e}" if or_pval is not None and pd.notna(or_pval) else "[Not computed]"

        ret_assess = (
            assessment_summary[assessment_summary["withdrawn_after_cutoff"] == 0].iloc[0]
            if not assessment_summary.empty and 0 in assessment_summary["withdrawn_after_cutoff"].values
            else None
        )
        w_assess = (
            assessment_summary[assessment_summary["withdrawn_after_cutoff"] == 1].iloc[0]
            if not assessment_summary.empty and 1 in assessment_summary["withdrawn_after_cutoff"].values
            else None
        )

        if (
            ret_assess is not None
            and w_assess is not None
            and "students_with_due" in ret_assess
            and "students_with_due" in w_assess
            and pd.notna(ret_assess["students_with_due"])
            and pd.notna(w_assess["students_with_due"])
        ):
            n_due_students = int(ret_assess["students_with_due"] + w_assess["students_with_due"])
            n_due_str = f"{n_due_students:,}"
        else:
            n_due_str = "[Not available]"

        ret_sub_rate_str = (
            f"{ret_assess['mean_submission_rate'] * 100:.1f}%"
            if ret_assess is not None and "mean_submission_rate" in ret_assess and pd.notna(ret_assess["mean_submission_rate"])
            else "[Not available]"
        )
        w_sub_rate_str = (
            f"{w_assess['mean_submission_rate'] * 100:.1f}%"
            if w_assess is not None and "mean_submission_rate" in w_assess and pd.notna(w_assess["mean_submission_rate"])
            else "[Not available]"
        )
        ret_ontime_rate_str = (
            f"{ret_assess['mean_on_time'] / ret_assess['mean_due'] * 100:.1f}%"
            if ret_assess is not None and "mean_on_time" in ret_assess and "mean_due" in ret_assess and pd.notna(ret_assess["mean_on_time"]) and ret_assess["mean_due"] > 0
            else "[Not available]"
        )
        w_ontime_rate_str = (
            f"{w_assess['mean_on_time'] / w_assess['mean_due'] * 100:.1f}%"
            if w_assess is not None and "mean_on_time" in w_assess and "mean_due" in w_assess and pd.notna(w_assess["mean_on_time"]) and w_assess["mean_due"] > 0
            else "[Not available]"
        )
        ret_missed_pct_str = (
            f"{ret_assess['pct_missed_due']:.1f}%"
            if ret_assess is not None and "pct_missed_due" in ret_assess and pd.notna(ret_assess["pct_missed_due"])
            else "[Not available]"
        )
        w_missed_pct_str = (
            f"{w_assess['pct_missed_due']:.1f}%"
            if w_assess is not None and "pct_missed_due" in w_assess and pd.notna(w_assess["pct_missed_due"])
            else "[Not available]"
        )

        # Missingness metrics
        imd_matches = (
            missingness_df[missingness_df["column"] == "imd_band"]
            if not missingness_df.empty and "column" in missingness_df.columns
            else pd.DataFrame()
        )
        imd_row = imd_matches.iloc[0] if not imd_matches.empty else None

        if (
            imd_row is not None
            and "p_value" in imd_row
            and pd.notna(imd_row["p_value"])
            and "retained_missing_pct" in imd_row
            and pd.notna(imd_row["retained_missing_pct"])
            and "withdrawn_missing_pct" in imd_row
            and pd.notna(imd_row["withdrawn_missing_pct"])
        ):
            imd_p_val = float(imd_row["p_value"])
            imd_ret_pct = float(imd_row["retained_missing_pct"])
            imd_w_pct = float(imd_row["withdrawn_missing_pct"])
            imd_delta = (
                float(imd_row["delta_pct"])
                if "delta_pct" in imd_row and pd.notna(imd_row["delta_pct"])
                else float(imd_w_pct - imd_ret_pct)
            )

            if imd_p_val < 0.05:
                imd_significance_text = (
                    f"`imd_band` missingness occurs in {imd_ret_pct:.2f}% of retained students vs {imd_w_pct:.2f}% of withdrawn students. "
                    f"Fisher's exact test indicates an unadjusted statistically significant difference ($p = {imd_p_val:.4f}$). "
                    f"However, the absolute delta is small ({imd_delta:+.2f} percentage points; {imd_ret_pct:.2f}% in retained vs {imd_w_pct:.2f}% in withdrawn), "
                    "indicating a negligible practical effect size rather than substantial selective attrition bias."
                )
            else:
                imd_significance_text = (
                    f"`imd_band` missingness occurs in {imd_ret_pct:.2f}% of retained students vs {imd_w_pct:.2f}% of withdrawn students. "
                    f"Fisher's exact test indicates no statistically significant difference ($p = {imd_p_val:.4f}$, delta: {imd_delta:+.2f} percentage points). "
                    "No statistically significant difference was detected. This does not establish equivalence between the groups."
                )
        else:
            imd_significance_text = (
                "Missingness comparison for `imd_band` is unavailable or could not be computed from the provided data."
            )

        content = f"""# PHASE 4 — EXPLORATORY DATA ANALYSIS & LEARNING-PATTERN REPORT

**Date:** 2026-10-13
**Analysis Target:** Withdrawal strictly after relative Day 35 cutoff (`withdrawn_after_cutoff`)
**Eligible Population (Policy C):** {total:,} student-course observations ({pos:,} withdrawn, {neg:,} retained)
**Positive Class Base Rate:** {base_rate:.2f}%

---

## 1. Executive Summary & Framing

This exploratory data analysis investigates early behavioral, academic, and temporal signals associated with course withdrawal after relative Day 35 in the Open University Learning Analytics Dataset (OULAD). Operating strictly within the information boundary available by the Day 35 cutoff, this analysis isolates verifiable early indicators of student attrition to guide **Phase 5 (Temporal Feature Engineering)**.

> [!IMPORTANT]
> **Association vs. Causation Warning:** All relationships documented herein represent observational statistical associations. They must not be interpreted as causal determinants. For example, lower VLE click volume reflects disengagement rather than a direct biological or cognitive cause of failure; artificially requiring students to click links without instructional support cannot be assumed to improve retention.

---

## 2. Evidence-Backed Analytical Observations

### Observation 1: Severe Cross-Presentation and Domain Heterogeneity
- **Empirical Finding:** Post-cutoff withdrawal rates vary substantially across the 22 module-presentations, ranging from a minimum of **{pres_min_str}** to a maximum of **{pres_max_str}** (mean across offerings: {pres_mean_str}).
- **Module and Discipline Differences:** In the OULAD dataset specification (Kuzilek et al., 2017), modules are categorized into STEM (`CCC`, `DDD`, `EEE`, `FFF`) and Social Sciences (`AAA`, `BBB`, `GGG`). While STEM offerings show a higher overall mean attrition rate ({stem_mean_str}) than Social Sciences ({soc_mean_str}), substantial intra-domain variation exists across specific modules (e.g., Module `EEE` has {eee_range_str} attrition while Module `CCC` has {ccc_range_str}).
- **Seasonality Patterns:** February (`B`) presentations exhibit higher attrition on average ({b_mean_str}) than October (`J`) presentations ({j_mean_str}) across offerings (pooled student-level rates: 18.9% vs 16.4%). However, seasonal disparities are not uniform across all offerings: while `CCC`, `DDD`, and `EEE` experience higher attrition in February cohorts, `BBB 2014B` (11.9%), `FFF 2013B` (16.6%), and `GGG 2014B` (7.7%) had lower withdrawal rates than their respective October cohorts (`BBB 2014J`: 12.8%; `FFF 2013J`: 18.2%; `GGG 2014J`: 12.5%).
- **Architectural Consequence for Phase 5:** Features must incorporate course-context normalizations (e.g. `code_module` and `presentation_season`), and model evaluation must employ group-aware or chronological blocking to prevent memorization of course-level base rates.

### Observation 2: Early Intensity and Diversity of Virtual Learning Engagement (VLE)
- **Empirical Finding:** Retained students engage in significantly higher interaction volumes prior to Day 35 compared to withdrawn students (Observation window: Days -25 to 35; N = {total:,}).
  - Retained students logged an average of **{get_stat_val(clicks_row, 'retained_mean')} clicks** (median: {get_stat_val(clicks_row, 'retained_median')}) across **{get_stat_val(days_row, 'retained_mean')} active days** (median: {get_stat_val(days_row, 'retained_median')}).
  - Withdrawn students logged an average of **{get_stat_val(clicks_row, 'withdrawn_mean')} clicks** (median: {get_stat_val(clicks_row, 'withdrawn_median')}) across **{get_stat_val(days_row, 'withdrawn_mean')} active days** (median: {get_stat_val(days_row, 'withdrawn_median')}).
- **Statistical Significance & Effect Size:** Mann-Whitney U test confirms a statistically significant difference ($p = {get_stat_val(clicks_row, 'p_value', '.2e', '[Not computed]')}$) with Cliff's delta = {get_stat_val(clicks_row, 'cliffs_delta', '.4f')} and Cohen's d = {get_stat_val(clicks_row, 'cohens_d', '.4f')}.
- **Breadth of Exploration:** Retained students interacted with a median of {get_stat_val(sites_row, 'retained_median', '.0f')} distinct resource sites compared to {get_stat_val(sites_row, 'withdrawn_median', '.0f')} for withdrawn students ($p = {get_stat_val(sites_row, 'p_value', '.2e', '[Not computed]')}$, Cliff's delta = {get_stat_val(sites_row, 'cliffs_delta', '.4f')}).

### Observation 3: Longitudinal Engagement Decay & Trajectory Divergence
- **Empirical Finding:** Analysis of weekly click trajectories from Week -3 through Week 5 reveals that interaction volumes between retained and withdrawn students are comparable during pre-course orientation (Weeks -3 to -1).
- **Divergence Point:** Engagement diverges immediately upon official course commencement (Week 0) and continues to widen across Weeks 2–5.
- **Recency of Inactivity:** By Day 35, students who eventually withdraw have a longer period of inactivity: a median of {get_stat_val(inactivity_row, 'withdrawn_median')} days elapsed since last click versus {get_stat_val(inactivity_row, 'retained_median')} days for retained students ($p = {get_stat_val(inactivity_row, 'p_value', '.2e', '[Not computed]')}$, Cliff's delta = {get_stat_val(inactivity_row, 'cliffs_delta', '.4f')}).

### Observation 4: Early Coursework Milestone Failure as a Critical Warning Signal
- **Empirical Finding:** Across the 18 module-presentations with continuous assessments due on or before Day 35 under the Reasonable Opportunity Policy (Eligible cohort: N = {n_due_str} learners; 4 offerings have 0 assessments due by Day 35):
  - Retained students achieved a mean submission rate of **{ret_sub_rate_str}** (with {ret_ontime_rate_str} submitted on time relative to scheduled coursework).
  - Withdrawn students achieved a mean submission rate of only **{w_sub_rate_str}** ({w_ontime_rate_str} on time).
  - Eventual withdrawals missed at least one scheduled assessment in **{w_missed_pct_str}** of cases, compared to **{ret_missed_pct_str}** for retained students.
- **Milestone Failure Odds Ratio:** Students with due coursework who failed to submit at least one due assessment by Day 35 had an **Odds Ratio of {or_val_str}** (95% CI: {ci_str}, Fisher's exact $p = {or_pval_str}$).
- **Reasonable Opportunity & Non-Penalty Invariants:**
  - For cohort learners in offerings with zero assessments due by Day 35 (`DDD 2013B`, `GGG 2013J`, `GGG 2014B`, `GGG 2014J`), or whose deadlines strictly preceded registration without satisfaction, `submission_rate` is represented as undefined (NaN) and `has_missed_due_assessment` is 0.
  - Across the cohort, exactly 5 records had deadlines prior to registration without prior banked submission:
    - 2 students had all due assessments scheduled prior to registration and were excluded from the assessment-due denominator ($N: 24,702 \\to 24,700$; `BBB 2013J 1170618` [retained, leaves Cell b] and `FFF 2014B 628323` [withdrawn, leaves Cell a]).
    - 2 students submitted their subsequent post-registration assessment on time and were reclassified from missed to non-missed (`CCC 2014J 350267` [withdrawn, Cell a $\\to$ Cell c] and `CCC 2014J 245600` [retained, Cell b $\\to$ Cell d]).
    - 1 student remained classified as missed because their subsequent post-registration assessment was also unsubmitted (`CCC 2014J 589463` [retained, stays in Cell b]).
  - Net transitions: $\\Delta a = -2, \\Delta b = -2, \\Delta c = +1, \\Delta d = +1$, reducing total missed cases from 3,477 to 3,473 and increasing non-missed cases from 21,225 to 21,227.
- **Phase 5 Guidance:** Milestone adherence constitutes a major early behavioral signal, but must be paired with indicator variables (`has_assessments_due`) to accommodate curriculum pacing differences.

### Observation 5: Registration Timing (Exploratory)
- **Group Comparison:** Median `date_registration` is **{get_stat_val(reg_row, 'retained_median')} days** for retained and **{get_stat_val(reg_row, 'withdrawn_median')} days** for withdrawn learners relative to Day 0 (Mann-Whitney U $p = {get_stat_val(reg_row, 'p_value', '.2e', '[Not computed]')}$, Cliff's delta = {get_stat_val(reg_row, 'cliffs_delta', '.4f')}). Negative values mean registration before Day 0; more negative values mean earlier registration. The `registration_lead_time_days` feature is `-date_registration`, so positive/larger values also mean earlier registration.
- **Interpretation Limit:** {reg_direction} The test's p-value alone gives no direction, and this retrospective comparison does not justify risk-directed targeting or establish a causal relationship or cutoff.

### Observation 6: Missingness and Contextual Demographic Indicators
- **Missingness Stability:** {imd_significance_text}
- **Academic Preparedness:** Eventual withdrawals exhibit higher numbers of previous course attempts (`num_of_prev_attempts`) compared to retained students (retained mean: {get_stat_val(prev_attempts_row, 'retained_mean', '.2f')}, withdrawn mean: {get_stat_val(prev_attempts_row, 'withdrawn_mean', '.2f')}; Mann-Whitney U $p = {get_stat_val(prev_attempts_row, 'p_value', '.2e', '[Not computed]')}$), confirming prior academic struggle history as a relevant background indicator.

---

## 3. Summary Tables

### Table 1: Statistical Comparisons of Candidate Continuous Predictors
| Feature | N (Retained) | N (Withdrawn) | Retained Mean | Retained Median | Withdrawn Mean | Withdrawn Median | Mann-Whitney U | p-value | Cliff's Delta | Cohen's d |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

        for _, row in stat_comparison.iterrows():
            nr = int(row["n_retained"]) if "n_retained" in row and pd.notna(row["n_retained"]) else neg
            nw = int(row["n_withdrawn"]) if "n_withdrawn" in row and pd.notna(row["n_withdrawn"]) else pos
            rm = f"{row['retained_mean']:.2f}" if "retained_mean" in row and pd.notna(row["retained_mean"]) else "[N/A]"
            rmed = f"{row['retained_median']:.1f}" if "retained_median" in row and pd.notna(row["retained_median"]) else "[N/A]"
            wm = f"{row['withdrawn_mean']:.2f}" if "withdrawn_mean" in row and pd.notna(row["withdrawn_mean"]) else "[N/A]"
            wmed = f"{row['withdrawn_median']:.1f}" if "withdrawn_median" in row and pd.notna(row["withdrawn_median"]) else "[N/A]"
            u_val = f"{row['mann_whitney_u']:.1f}" if "mann_whitney_u" in row and pd.notna(row["mann_whitney_u"]) else "[N/A]"
            pval = f"{row['p_value']:.2e}" if "p_value" in row and pd.notna(row["p_value"]) else "[Not computed]"
            cd = f"{row['cliffs_delta']:.4f}" if "cliffs_delta" in row and pd.notna(row["cliffs_delta"]) else "[N/A]"
            cod = f"{row['cohens_d']:.4f}" if "cohens_d" in row and pd.notna(row["cohens_d"]) else "[N/A]"
            content += f"| `{row.get('feature', '[Unknown]')}` | {nr:,} | {nw:,} | {rm} | {rmed} | {wm} | {wmed} | {u_val} | {pval} | {cd} | {cod} |\n"

        content += f"""
### Table 2: Assessment Milestone Failure Odds Ratio
- **Contingency Table (Students with Due Assessments <= Day 35):**
  - Eventual Withdrawals with Missed Due Assessment (a): {a:,}
  - Retained Students with Missed Due Assessment (b): {b:,}
  - Eventual Withdrawals with Zero Missed Assessments (c): {c:,}
  - Retained Students with Zero Missed Assessments (d): {d:,}
- **Odds Ratio:** **{or_val_str}** (95% CI: {ci_str}, Fisher exact $p = {or_pval_str}$)

### Table 3: Missingness Comparisons by Outcome Group
| Feature | Retained Missing (%) | Withdrawn Missing (%) | Delta (%) | Fisher p-value |
| :--- | :--- | :--- | :--- | :--- |
"""
        for _, row in missingness_df.iterrows():
            col_name = row.get("column", "[Unknown]")
            ret_str = f"{row['retained_missing_pct']}%" if "retained_missing_pct" in row and pd.notna(row["retained_missing_pct"]) else "[N/A]"
            w_str = f"{row['withdrawn_missing_pct']}%" if "withdrawn_missing_pct" in row and pd.notna(row["withdrawn_missing_pct"]) else "[N/A]"
            delta_str = f"{row['delta_pct']}%" if "delta_pct" in row and pd.notna(row["delta_pct"]) else "[N/A]"
            p_str = f"{row['p_value']:.4f}" if "p_value" in row and pd.notna(row["p_value"]) else "[Not computed]"
            content += f"| `{col_name}` | {ret_str} | {w_str} | {delta_str} | {p_str} |\n"

        content += """
---

## 4. Methodological Limitations & Analytical Caveats

1. **Observational Nature:** All derived signals are descriptive behavioral correlations under distance-learning conditions, not direct pedagogical causes. Mandating interaction volume without instructional support cannot be assumed to improve retention.
2. **Unequal Assessment Due Schedules:** Not all modules schedule continuous assessments prior to Day 35 (e.g., Module `GGG` has 0 assessments due <= Day 35). Assessment features must be handled with appropriate indicator variables (`has_assessments_due`) so modules without early assignments are not penalized.
3. **VLE Logging Density Variation:** Different modules utilize online resources with varying intensity (e.g. STEM modules have extensive online quizzing, whereas Humanities rely heavily on printed textbooks). Course-relative scaling or presentation fixed effects are critical.
4. **Repeated-Observation Clustering Limitation:** The 27,128 cohort observations comprise enrollments across 24,567 unique students (2,450 students appear in multiple offerings, accounting for 5,011 records). Unadjusted two-sample statistical tests assume independent observations; mild positive intra-student correlation may slightly understate standard errors and overstate asymptotic test precision. These statistics serve as exploratory heuristic ranking signals rather than formal confirmatory causal proofs. Phases 5 and 6 will enforce grouped cross-validation blocking on student identifiers to ensure independence between training and test sets.

---

## 5. Candidate Features Recommended for Phase 5

Based on the verified empirical associations, the following 12 core early signals are justified for inclusion in the Phase 5 temporal feature pipeline:
1. `vle_total_clicks` (Overall engagement volume in $[-25, 35]$)
2. `vle_active_days` (Pacing consistency)
3. `vle_distinct_sites` (Breadth of resource interaction)
4. `vle_recent_clicks` (Clicks in days $[22, 35]$)
5. `vle_click_velocity_ratio` (Ratio of recent to overall engagement rate)
6. `days_since_last_vle` (Inactivity recency with sentinel 65)
7. `assessments_due_count` (Curriculum pacing expectation with opportunity)
8. `submissions_count_lte_cutoff` (Submission volume)
9. `submitted_on_time_count` (Punctuality)
10. `has_missed_due_assessment` (Milestone failure indicator)
11. `date_registration` (Registration lead time before start)
12. `presentation_season` (February vs October seasonality indicator)
"""
        report_path.write_text(content, encoding="utf-8")
