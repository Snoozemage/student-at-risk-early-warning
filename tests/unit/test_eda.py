"""Unit tests for Phase 4 Exploratory Data Analysis & Signal Discovery."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from student_retention.eda.assessment_analysis import (
    extract_cutoff_assessment_features,
)
from student_retention.eda.cohort_analysis import (
    compare_policy_sensitivities,
    summarize_cohort_demographics,
    summarize_presentation_withdrawal_rates,
)
from student_retention.eda.eda_runner import EDARunner
from student_retention.eda.engagement_analysis import (
    compute_weekly_vle_trajectory,
    extract_cutoff_vle_features,
)
from student_retention.eda.statistical_tests import (
    compare_missingness_by_outcome,
    compare_numeric_distributions,
    compute_cliffs_delta,
    compute_cohens_d,
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


def test_compute_cohens_d():
    """Verify Cohen's d effect size computation."""
    g1 = [10.0, 10.0, 10.0, 10.0]
    g2 = [10.0, 10.0, 10.0, 10.0]
    assert compute_cohens_d(g1, g2) == 0.0

    # Large difference
    g_low = [1, 2, 3, 4, 5]
    g_high = [10, 11, 12, 13, 14]
    d = compute_cohens_d(g_low, g_high)
    assert d < -3.0

    # Insufficient elements
    assert compute_cohens_d([1], [2]) == 0.0


def test_compute_cliffs_delta():
    """Verify Cliff's delta non-parametric effect size."""
    # Complete dominance: all g1 < all g2 -> -1.0
    g1 = [1, 2, 3]
    g2 = [10, 20, 30]
    assert compute_cliffs_delta(g1, g2) == -1.0

    # Reverse dominance: all g1 > all g2 -> 1.0
    assert compute_cliffs_delta(g2, g1) == 1.0

    # Identical
    assert compute_cliffs_delta([5, 5], [5, 5]) == 0.0


def test_compute_odds_ratio():
    """Verify odds ratio calculation with contingency table."""
    res = compute_odds_ratio(
        exposed_cases=20,
        exposed_controls=10,
        unexposed_cases=10,
        unexposed_controls=20,
    )
    assert res["odds_ratio"] > 3.0
    assert res["p_value"] < 0.05
    assert res["ci_lower"] > 1.0


def test_compare_numeric_distributions():
    """Verify comparative distribution stats between target groups."""
    df = pd.DataFrame({
        "clicks": [100, 120, 150, 200, 30, 40, 50, 60],
        "withdrawn_after_cutoff": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    res = compare_numeric_distributions(
        df=df,
        outcome_col="withdrawn_after_cutoff",
        feature_cols=["clicks"],
    )
    assert len(res) == 1
    row = res.iloc[0]
    assert row["retained_median"] > row["withdrawn_median"]
    assert row["cliffs_delta"] < 0
    assert row["p_value"] < 0.05


def test_compare_missingness_by_outcome():
    """Verify missingness rate comparison across outcomes."""
    df = pd.DataFrame({
        "imd_band": [10, 20, None, 40, None, None, None, 10],
        "withdrawn_after_cutoff": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    res = compare_missingness_by_outcome(
        df=df,
        outcome_col="withdrawn_after_cutoff",
        feature_cols=["imd_band"],
    )
    assert len(res) == 1
    row = res.iloc[0]
    assert row["retained_missing_pct"] == 25.0
    assert row["withdrawn_missing_pct"] == 75.0


def test_summarize_cohort_demographics():
    """Verify demographic summary table generation."""
    df = pd.DataFrame({
        "gender": ["M", "M", "F", "F"],
        "region": ["East", "East", "West", "West"],
        "highest_education": ["HE", "HE", "A", "A"],
        "imd_band": ["10-20%", "10-20%", "20-30%", "20-30%"],
        "age_band": ["0-35", "0-35", "35-55", "35-55"],
        "disability": ["N", "N", "N", "Y"],
        "withdrawn_after_cutoff": [0, 1, 0, 0],
    })
    res = summarize_cohort_demographics(df)
    assert res["total_records"] == 4
    assert res["positive_cases"] == 1
    assert "gender" in res["demographics"]


def test_summarize_presentation_withdrawal_rates():
    """Verify presentation-level breakdown and STEM tagging."""
    df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2014B", "2014B"],
        "studied_credits": [60, 60, 120, 120],
        "num_of_prev_attempts": [0, 0, 1, 0],
        "withdrawn_after_cutoff": [0, 1, 0, 0],
    })
    res = summarize_presentation_withdrawal_rates(df)
    assert len(res) == 2
    assert "faculty" in res.columns
    assert "season" in res.columns


def test_compare_policy_sensitivities():
    """Verify sensitivity comparison between Policy B and Policy C."""
    df_c = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [1, 2],
        "withdrawn_after_cutoff": [0, 1],
    })
    df_b = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3],
        "withdrawn_after_cutoff": [0, 1, 1],
    })
    res = compare_policy_sensitivities(df_c, df_b)
    assert res["policy_c_eligible"] == 2
    assert res["policy_b_eligible"] == 3


def test_extract_cutoff_vle_features_leakage_free():
    """Verify VLE feature extraction enforces cutoff and handles edge cases."""
    cohort_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [100, 200],
    })
    student_vle = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J"],
        "id_student": [100, 100, 100, 100],
        "id_site": [1, 2, 1, 3],
        "date": [-5, 10, 35, 40],
        "sum_click": [5, 10, 15, 100],
    })
    vle_df = pd.DataFrame({
        "id_site": [1, 2, 3],
        "code_module": ["AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "activity_type": ["resource", "forumng", "quiz"],
    })

    features = extract_cutoff_vle_features(
        cohort_df=cohort_df,
        student_vle_df=student_vle,
        vle_df=vle_df,
        cutoff_day=35,
    )
    assert len(features) == 2

    s100 = features[features["id_student"] == 100].iloc[0]
    assert s100["vle_total_clicks"] == 30
    assert s100["vle_active_days"] == 3
    assert s100["vle_distinct_sites"] == 2
    assert s100["vle_precourse_clicks"] == 5
    assert s100["vle_recent_clicks"] == 15
    assert s100["days_since_last_vle"] == 0

    s200 = features[features["id_student"] == 200].iloc[0]
    assert s200["vle_total_clicks"] == 0
    assert s200["vle_active_days"] == 0
    assert s200["days_since_last_vle"] == 65


def test_extract_cutoff_assessment_features():
    """Verify assessment milestone extraction enforces cutoff and handles due dates."""
    cohort_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [100, 200],
    })
    assessments_df = pd.DataFrame({
        "id_assessment": [10, 20],
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "assessment_type": ["TMA", "TMA"],
        "date": [25, 50],
        "weight": [10.0, 20.0],
    })
    student_assessment_df = pd.DataFrame({
        "id_student": [100],
        "id_assessment": [10],
        "date_submitted": [20],
        "score": [85.0],
    })

    features = extract_cutoff_assessment_features(
        cohort_df=cohort_df,
        assessments_df=assessments_df,
        student_assessment_df=student_assessment_df,
        cutoff_day=35,
    )
    assert len(features) == 2

    s100 = features[features["id_student"] == 100].iloc[0]
    assert s100["assessments_due_count"] == 1
    assert s100["submissions_count_lte_cutoff"] == 1
    assert s100["submitted_on_time_count"] == 1
    assert s100["unsubmitted_due_count"] == 0
    assert s100["has_missed_due_assessment"] == 0

    s200 = features[features["id_student"] == 200].iloc[0]
    assert s200["assessments_due_count"] == 1
    assert s200["submissions_count_lte_cutoff"] == 0
    assert s200["unsubmitted_due_count"] == 1
    assert s200["has_missed_due_assessment"] == 1


def test_compute_weekly_vle_trajectory():
    """Verify weekly trajectory computation across weeks."""
    cohort_df = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [100, 200],
        "withdrawn_after_cutoff": [0, 1],
    })
    student_vle = pd.DataFrame({
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [100, 200],
        "id_site": [1, 2],
        "date": [5, 5],
        "sum_click": [20, 5],
    })
    traj = compute_weekly_vle_trajectory(cohort_df, student_vle, cutoff_day=35)
    assert not traj.empty
    w1 = traj[traj["week_num"] == 1]
    assert len(w1) == 2


def test_visualizations_generate_figures(tmp_path: Path):
    """Verify visualization functions write valid image files."""
    rates_df = pd.DataFrame({
        "code_module": ["AAA", "BBB"],
        "code_presentation": ["2013J", "2013J"],
        "presentation": ["AAA 2013J", "BBB 2013J"],
        "faculty": ["Arts & Social Sciences", "STEM"],
        "season": ["October", "October"],
        "withdrawal_rate_pct": [15.0, 25.0],
        "eligible_students": [100, 200],
        "withdrawn_after_cutoff": [15, 50],
    })
    fig1 = tmp_path / "fig1.png"
    plot_presentation_withdrawal_rates(rates_df, fig1)
    assert fig1.exists() and fig1.stat().st_size > 0

    df_feat = pd.DataFrame({
        "vle_total_clicks": [0, 10, 50, 100, 200],
        "withdrawn_after_cutoff": [0, 1, 0, 1, 0],
    })
    fig2 = tmp_path / "fig2.png"
    plot_vle_clicks_distribution(df_feat, fig2)
    assert fig2.exists() and fig2.stat().st_size > 0

    traj = pd.DataFrame({
        "week_name": ["W1", "W1"],
        "week_num": [1, 1],
        "outcome_label": ["Retained (Completers)", "Withdrawn After Day 35"],
        "mean_clicks": [15.5, 8.2],
    })
    fig3 = tmp_path / "fig3.png"
    plot_weekly_activity_curves(traj, fig3)
    assert fig3.exists() and fig3.stat().st_size > 0

    df_reg = pd.DataFrame({
        "date_registration": [-60, -50, -30, -10],
        "withdrawn_after_cutoff": [0, 0, 1, 1],
    })
    fig4 = tmp_path / "fig4.png"
    plot_registration_lead_time(df_reg, fig4)
    assert fig4.exists() and fig4.stat().st_size > 0

    df_assess = pd.DataFrame({
        "assessments_due_count": [1, 1, 1],
        "submission_rate": [1.0, 0.0, 1.0],
        "has_missed_due_assessment": [0, 1, 0],
        "withdrawn_after_cutoff": [0, 1, 0],
    })
    fig5 = tmp_path / "fig5.png"
    plot_assessment_submission_progress(df_assess, fig5)
    assert fig5.exists() and fig5.stat().st_size > 0

    df_corr = pd.DataFrame({
        "vle_total_clicks": [10, 20, 30, 40],
        "vle_active_days": [2, 3, 5, 7],
        "withdrawn_after_cutoff": [1, 1, 0, 0],
    })
    fig6 = tmp_path / "fig6.png"
    plot_correlation_matrix(df_corr, ["vle_total_clicks", "vle_active_days"], fig6)
    assert fig6.exists() and fig6.stat().st_size > 0


def test_eda_runner_end_to_end_synthetic(tmp_path: Path):
    """Verify EDARunner executes end-to-end on synthetic datasets."""
    cohort_dir = tmp_path / "cohort"
    interim_dir = tmp_path / "interim"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    reports_dir = tmp_path / "reports"
    cohort_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)

    cohort_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3, 4],
        "gender": ["M", "F", "M", "F"],
        "region": ["East", "East", "West", "West"],
        "highest_education": ["HE", "HE", "A", "A"],
        "imd_band": ["0-10%", "10-20%", "20-30%", None],
        "age_band": ["0-35", "35-55", "0-35", "55<="],
        "num_of_prev_attempts": [0, 1, 0, 2],
        "studied_credits": [60, 60, 120, 120],
        "disability": ["N", "N", "N", "Y"],
        "date_registration": [-50, -40, -30, -20],
        "withdrawn_after_cutoff": [0, 1, 0, 1],
    })
    cohort_df.to_parquet(cohort_dir / "cohort_policy_c.parquet")

    student_vle = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [1, 2, 3],
        "id_site": [10, 20, 30],
        "date": [5, 10, 15],
        "sum_click": [20, 5, 30],
    })
    student_vle.to_parquet(interim_dir / "student_vle.parquet")

    vle = pd.DataFrame({
        "id_site": [10, 20, 30],
        "code_module": ["AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "activity_type": ["resource", "resource", "quiz"],
    })
    vle.to_parquet(interim_dir / "vle.parquet")

    assessments = pd.DataFrame({
        "id_assessment": [100, 200],
        "code_module": ["AAA", "BBB"],
        "code_presentation": ["2013J", "2013J"],
        "assessment_type": ["TMA", "TMA"],
        "date": [20, 30],
        "weight": [20.0, 20.0],
    })
    assessments.to_parquet(interim_dir / "assessments.parquet")

    student_assessment = pd.DataFrame({
        "id_student": [1, 3],
        "id_assessment": [100, 200],
        "date_submitted": [18, 28],
        "score": [80.0, 90.0],
    })
    student_assessment.to_parquet(interim_dir / "student_assessment.parquet")

    runner = EDARunner(
        cohort_dir=cohort_dir,
        interim_dir=interim_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
        cutoff_day=35,
    )
    res = runner.run()

    assert res["cohort_rows"] == 4
    assert (tables_dir / "demographic_summary.csv").exists()
    assert (tables_dir / "presentation_withdrawal_rates.csv").exists()
    assert (tables_dir / "statistical_comparisons.csv").exists()
    assert (figures_dir / "fig1_withdrawal_rate_by_presentation.png").exists()
    assert (reports_dir / "eda_report.md").exists()


def test_multi_course_assessment_isolation_canonical_grain():
    """Verify assessments are strictly isolated by (code_module, code_presentation, id_student).

    Regression test: A student enrolled in two different offerings must NOT have submissions
    from one offering transferred to the other.
    """
    cohort_df = pd.DataFrame({
        "code_module": ["AAA", "BBB"],
        "code_presentation": ["2013J", "2013J"],
        "id_student": [999, 999],
    })
    assessments_df = pd.DataFrame({
        "id_assessment": [10, 20],
        "code_module": ["AAA", "BBB"],
        "code_presentation": ["2013J", "2013J"],
        "assessment_type": ["TMA", "TMA"],
        "date": [20, 25],
        "weight": [20.0, 20.0],
    })
    # Student submitted for AAA, but NOT for BBB
    student_assessment_df = pd.DataFrame({
        "id_student": [999],
        "id_assessment": [10],
        "date_submitted": [18],
        "score": [85.0],
    })

    res = extract_cutoff_assessment_features(
        cohort_df=cohort_df,
        assessments_df=assessments_df,
        student_assessment_df=student_assessment_df,
        cutoff_day=35,
    )

    assert len(res) == 2, "Row count must match input cohort exactly"

    # Offering AAA: submitted
    row_aaa = res[(res["code_module"] == "AAA") & (res["id_student"] == 999)].iloc[0]
    assert row_aaa["assessments_due_count"] == 1
    assert row_aaa["submissions_count_lte_cutoff"] == 1
    assert row_aaa["unsubmitted_due_count"] == 0
    assert row_aaa["has_missed_due_assessment"] == 0
    assert row_aaa["submission_rate"] == 1.0
    assert row_aaa["has_assessments_due"] == 1

    # Offering BBB: NOT submitted (must not leak from AAA!)
    row_bbb = res[(res["code_module"] == "BBB") & (res["id_student"] == 999)].iloc[0]
    assert row_bbb["assessments_due_count"] == 1
    assert row_bbb["submissions_count_lte_cutoff"] == 0
    assert row_bbb["unsubmitted_due_count"] == 1
    assert row_bbb["has_missed_due_assessment"] == 1
    assert row_bbb["submission_rate"] == 0.0
    assert row_bbb["has_assessments_due"] == 1


def test_assessment_cutoff_boundary_and_leakage():
    """Verify assessments submitted after Day 35 do not count as submitted by cutoff."""
    cohort_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [101, 102, 103],
    })
    assessments_df = pd.DataFrame({
        "id_assessment": [10, 20],
        "code_module": ["AAA", "BBB"],
        "code_presentation": ["2013J", "2013J"],
        "assessment_type": ["TMA", "TMA"],
        "date": [25, 35],  # Assessment 20 is due exactly on Day 35
        "weight": [20.0, 20.0],
    })
    student_assessment_df = pd.DataFrame({
        "id_student": [101, 102, 103],
        "id_assessment": [10, 10, 20],
        "date_submitted": [35, 36, 35],  # 101 on cutoff day, 102 after cutoff, 103 on cutoff
        "score": [80.0, 80.0, 90.0],
    })

    res = extract_cutoff_assessment_features(
        cohort_df=cohort_df,
        assessments_df=assessments_df,
        student_assessment_df=student_assessment_df,
        cutoff_day=35,
    )

    assert len(res) == 3

    # Student 101: submitted on Day 35 (valid)
    s101 = res[res["id_student"] == 101].iloc[0]
    assert s101["submissions_count_lte_cutoff"] == 1
    assert s101["has_missed_due_assessment"] == 0
    assert s101["submission_rate"] == 1.0

    # Student 102: submitted on Day 36 (> Day 35, must NOT count by cutoff)
    s102 = res[res["id_student"] == 102].iloc[0]
    assert s102["submissions_count_lte_cutoff"] == 0
    assert s102["unsubmitted_due_count"] == 1
    assert s102["has_missed_due_assessment"] == 1
    assert s102["submission_rate"] == 0.0

    # Student 103: due exactly on Day 35 and submitted on Day 35
    s103 = res[res["id_student"] == 103].iloc[0]
    assert s103["assessments_due_count"] == 1
    assert s103["submissions_count_lte_cutoff"] == 1
    assert s103["submitted_on_time_count"] == 1
    assert s103["has_missed_due_assessment"] == 0


def test_assessment_zero_due_representation():
    """Verify zero due assessments result in NaN submission_rate and zero missed flag.

    Zero qualifying coursework must not be conflated with 100% submission rate.
    """
    cohort_df = pd.DataFrame({
        "code_module": ["GGG"],
        "code_presentation": ["2013J"],
        "id_student": [501],
    })
    # Only assessment in GGG is due on Day 60 (> Day 35)
    assessments_df = pd.DataFrame({
        "id_assessment": [99],
        "code_module": ["GGG"],
        "code_presentation": ["2013J"],
        "assessment_type": ["TMA"],
        "date": [60],
        "weight": [20.0],
    })
    student_assessment_df = pd.DataFrame({
        "id_student": [501],
        "id_assessment": [99],
        "date_submitted": [58],
        "score": [95.0],
    })

    res = extract_cutoff_assessment_features(
        cohort_df=cohort_df,
        assessments_df=assessments_df,
        student_assessment_df=student_assessment_df,
        cutoff_day=35,
    )

    assert len(res) == 1
    row = res.iloc[0]
    assert row["assessments_due_count"] == 0
    assert row["has_assessments_due"] == 0
    assert row["submissions_count_lte_cutoff"] == 0
    assert row["unsubmitted_due_count"] == 0
    assert row["has_missed_due_assessment"] == 0
    assert pd.isna(row["submission_rate"]), "submission_rate must be NaN, NOT 1.0"


def test_assessment_banked_credits_and_multiple_attempts():
    """Verify banked submissions satisfy requirements and multiple attempts do not multiply rows."""
    cohort_df = pd.DataFrame({
        "code_module": ["CCC", "CCC"],
        "code_presentation": ["2014B", "2014B"],
        "id_student": [301, 302],
    })
    assessments_df = pd.DataFrame({
        "id_assessment": [50],
        "code_module": ["CCC"],
        "code_presentation": ["2014B"],
        "assessment_type": ["CMA"],
        "date": [30],
        "weight": [10.0],
    })
    student_assessment_df = pd.DataFrame({
        "id_student": [301, 302, 302],  # 301 is banked, 302 has multiple submission records
        "id_assessment": [50, 50, 50],
        "date_submitted": [-1, 25, 28],
        "is_banked": [1, 0, 0],
        "score": [80.0, 70.0, 90.0],
    })

    res = extract_cutoff_assessment_features(
        cohort_df=cohort_df,
        assessments_df=assessments_df,
        student_assessment_df=student_assessment_df,
        cutoff_day=35,
    )

    assert len(res) == 2, "Row count must remain 2 without duplication"

    # 301: banked submission satisfies coursework
    s301 = res[res["id_student"] == 301].iloc[0]
    assert s301["banked_submissions_count"] == 1
    assert s301["unsubmitted_due_count"] == 0
    assert s301["has_missed_due_assessment"] == 0

    # 302: multiple attempts deduplicated to 1 assessment satisfied
    s302 = res[res["id_student"] == 302].iloc[0]
    assert s302["submissions_count_lte_cutoff"] == 1
    assert s302["has_missed_due_assessment"] == 0
    assert s302["submission_rate"] == 1.0


def test_plot_presentation_withdrawal_rates_offering_labels(tmp_path: Path):
    """Verify plot_presentation_withdrawal_rates constructs distinct offering labels."""
    rates_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB"],
        "code_presentation": ["2013J", "2014J", "2013J"],
        "season": ["J", "J", "J"],
        "withdrawal_rate_pct": [8.7, 9.4, 17.2],
        "eligible_students": [100, 100, 100],
        "withdrawn_after_cutoff": [8, 9, 17],
    })
    save_file = tmp_path / "test_offering_labels.png"
    plot_presentation_withdrawal_rates(rates_df, save_file)
    assert save_file.exists() and save_file.stat().st_size > 0


def test_assessment_counter_invariants_and_duplicate_deduplication():
    """Verify counter consistency: on_time + late == submissions_count (unique assessment units).

    Regression test: Duplicate attempts or resubmissions must never cause submitted_on_time_count
    or late_submissions_count to exceed submissions_count_lte_cutoff or inflate student metrics.
    """
    cohort_df = pd.DataFrame({
        "code_module": ["AAA", "AAA", "AAA"],
        "code_presentation": ["2013J", "2013J", "2013J"],
        "id_student": [701, 702, 703],
    })
    assessments_df = pd.DataFrame({
        "id_assessment": [10, 20],
        "code_module": ["AAA", "AAA"],
        "code_presentation": ["2013J", "2013J"],
        "assessment_type": ["TMA", "TMA"],
        "date": [20, 25],
        "weight": [10.0, 10.0],
    })
    student_assessment_df = pd.DataFrame({
        # Student 701: Assessment 10 has 1 on-time attempt (18) and 1 late attempt (22)
        # Student 702: Assessment 10 has 2 late attempts (22, 24)
        # Student 703: Assessment 10 has banked (is_banked=1) and current submission (18)
        "id_student": [701, 701, 702, 702, 703, 703],
        "id_assessment": [10, 10, 10, 10, 10, 10],
        "date_submitted": [18, 22, 22, 24, -1, 18],
        "is_banked": [0, 0, 0, 0, 1, 0],
        "score": [60.0, 85.0, 50.0, 70.0, 80.0, 80.0],
    })

    res = extract_cutoff_assessment_features(
        cohort_df=cohort_df,
        assessments_df=assessments_df,
        student_assessment_df=student_assessment_df,
        cutoff_day=35,
    )

    assert len(res) == 3

    # Student 701: 1 unique assessment submitted, 1 on-time, 0 late (satisfaction met on time)
    s701 = res[res["id_student"] == 701].iloc[0]
    assert s701["submissions_count_lte_cutoff"] == 1
    assert s701["submitted_on_time_count"] == 1
    assert s701["late_submissions_count_lte_cutoff"] == 0
    assert s701["submitted_on_time_count"] + s701["late_submissions_count_lte_cutoff"] == s701["submissions_count_lte_cutoff"]

    # Student 702: 1 unique assessment submitted, 0 on-time, 1 late
    s702 = res[res["id_student"] == 702].iloc[0]
    assert s702["submissions_count_lte_cutoff"] == 1
    assert s702["submitted_on_time_count"] == 0
    assert s702["late_submissions_count_lte_cutoff"] == 1
    assert s702["submitted_on_time_count"] + s702["late_submissions_count_lte_cutoff"] == s702["submissions_count_lte_cutoff"]

    # Student 703: 1 unique assessment satisfied across banked and current (no double counting)
    s703 = res[res["id_student"] == 703].iloc[0]
    assert s703["banked_submissions_count"] == 1
    assert s703["submissions_count_lte_cutoff"] == 1
    assert s703["unsubmitted_due_count"] == 1  # 2 due (10 and 20), assessment 10 satisfied -> 1 unsubmitted
    assert s703["has_missed_due_assessment"] == 1


def test_assessment_due_dates_relative_to_registration_opportunity():
    """Verify registration-opportunity policy does not penalize pre-registration deadlines."""
    cohort_df = pd.DataFrame({
        "code_module": ["BBB", "BBB", "BBB", "BBB"],
        "code_presentation": ["2013J", "2013J", "2013J", "2013J"],
        "id_student": [401, 402, 403, 404],
        "date_registration": [20, 20, -30, 20],  # 401, 402, 404 registered late (Day 20); 403 registered early (-30)
    })
    assessments_df = pd.DataFrame({
        "id_assessment": [10, 20],
        "code_module": ["BBB", "BBB"],
        "code_presentation": ["2013J", "2013J"],
        "assessment_type": ["TMA", "TMA"],
        "date": [15, 30],  # Assessment 10 due Day 15 (before Day 20); Assessment 20 due Day 30
        "weight": [10.0, 10.0],
    })
    student_assessment_df = pd.DataFrame({
        # 401: no submissions
        # 402: submitted assessment 10 on Day 25 (extension/catch-up)
        # 403: no submissions
        # 404: submitted assessment 20 on Day 28 (on-time for post-registration assessment)
        "id_student": [402, 404],
        "id_assessment": [10, 20],
        "date_submitted": [25, 28],
        "score": [80.0, 85.0],
    })

    res = extract_cutoff_assessment_features(
        cohort_df=cohort_df,
        assessments_df=assessments_df,
        student_assessment_df=student_assessment_df,
        cutoff_day=35,
        enforce_registration_opportunity=True,
    )

    assert len(res) == 4

    # 401: registered Day 20, assessment 10 due Day 15 (unsubmitted), assessment 20 due Day 30 (unsubmitted)
    # Under opportunity policy: assessment 10 not held against student, but assessment 20 is due
    s401 = res[res["id_student"] == 401].iloc[0]
    assert s401["assessments_due_count"] == 1  # Only assessment 20 was due after registration
    assert s401["submissions_count_lte_cutoff"] == 0
    assert s401["unsubmitted_due_count"] == 1
    assert s401["has_missed_due_assessment"] == 1

    # 402: registered Day 20, completed assessment 10 on Day 25 (extension), assessment 20 unsubmitted
    s402 = res[res["id_student"] == 402].iloc[0]
    assert s402["assessments_due_count"] == 2  # Both qualify because 10 was satisfied and 20 due post-registration
    assert s402["submissions_count_lte_cutoff"] == 1
    assert s402["unsubmitted_due_count"] == 1
    assert s402["has_missed_due_assessment"] == 1

    # 403: registered Day -30 (early), both assessments 10 and 20 qualify as due
    s403 = res[res["id_student"] == 403].iloc[0]
    assert s403["assessments_due_count"] == 2
    assert s403["unsubmitted_due_count"] == 2
    assert s403["has_missed_due_assessment"] == 1

    # 404: registered Day 20, assessment 10 due Day 15 (not submitted), assessment 20 due Day 30 (submitted on time)
    # Assessment 10 not held against student; assessment 20 satisfied on time -> 0 missed!
    s404 = res[res["id_student"] == 404].iloc[0]
    assert s404["assessments_due_count"] == 1
    assert s404["submissions_count_lte_cutoff"] == 1
    assert s404["unsubmitted_due_count"] == 0
    assert s404["has_missed_due_assessment"] == 0
    assert s404["submission_rate"] == 1.0


def test_eda_runner_report_fallbacks_and_missing_data(tmp_path: Path):
    """Verify report generation handles missing, uncomputed, or partial statistics defensibly.

    Ensures that when statistical tests or tables cannot be computed:
    1. Missingness does NOT fabricate p_value=1.0 or claim groups are balanced.
    2. Missing offerings do NOT fabricate hardcoded withdrawal rate ranges.
    3. Uncomputed effect sizes or statistics report '[Not available]' or '[Not computed]'.
    """
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    runner = EDARunner(reports_dir=reports_dir)

    cohort_summary = {
        "total_records": 1000,
        "positive_cases": 150,
        "negative_cases": 850,
        "base_rate_pct": 15.0,
    }

    # Test Case 1: Empty / missing tables & uncomputed p-values
    presentation_summary_empty = pd.DataFrame()
    stat_comparison_empty = pd.DataFrame()
    assessment_summary_empty = pd.DataFrame()
    missed_or_empty = {
        "odds_ratio": None,
        "p_value": None,
        "ci_lower": None,
        "ci_upper": None,
    }
    missingness_df_uncomputed = pd.DataFrame([
        {
            "column": "imd_band",
            "retained_missing_count": 0,
            "retained_missing_pct": float("nan"),
            "withdrawn_missing_count": 0,
            "withdrawn_missing_pct": float("nan"),
            "delta_pct": float("nan"),
            "p_value": float("nan"),
        }
    ])
    policy_comparison_empty = {}

    report_path = reports_dir / "eda_report.md"
    runner._write_eda_report(
        report_path=report_path,
        cohort_summary=cohort_summary,
        presentation_summary=presentation_summary_empty,
        stat_comparison=stat_comparison_empty,
        assessment_summary=assessment_summary_empty,
        missed_or=missed_or_empty,
        missingness_df=missingness_df_uncomputed,
        policy_comparison=policy_comparison_empty,
    )

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    # Verify no fabricated values
    assert "Missingness comparison for `imd_band` is unavailable or could not be computed from the provided data." in content
    assert "Fisher's exact test indicates no statistically significant difference ($p = 1.0000" not in content
    assert "confirming that missing socioeconomic status is balanced across outcome groups." not in content
    assert "11.9%–13.3%" not in content
    assert "26.0%–28.3%" not in content
    assert "[Not available]" in content
    assert "[Not computed]" in content

    # Test Case 2: Populated statistics with genuine values
    presentation_summary_valid = pd.DataFrame([
        {
            "code_module": "AAA",
            "code_presentation": "2013J",
            "withdrawal_rate_pct": 12.5,
            "season": "J",
            "faculty": "Social Sciences",
        },
        {
            "code_module": "CCC",
            "code_presentation": "2014B",
            "withdrawal_rate_pct": 28.3,
            "season": "B",
            "faculty": "STEM",
        },
    ])
    missingness_df_valid = pd.DataFrame([
        {
            "column": "imd_band",
            "retained_missing_count": 874,
            "retained_missing_pct": 3.90,
            "withdrawn_missing_count": 146,
            "withdrawn_missing_pct": 3.09,
            "delta_pct": -0.81,
            "p_value": 0.0079,
        }
    ])
    stat_comparison_valid = pd.DataFrame([
        {
            "feature": "vle_total_clicks",
            "retained_mean": 430.2,
            "retained_median": 275.0,
            "withdrawn_mean": 334.3,
            "withdrawn_median": 217.0,
            "p_value": 1.36e-33,
            "cliffs_delta": -0.1117,
            "cohens_d": -0.1948,
        }
    ])
    missed_or_valid = {
        "odds_ratio": 2.80,
        "ci_lower": 2.58,
        "ci_upper": 3.03,
        "p_value": 1.72e-132,
    }

    runner._write_eda_report(
        report_path=report_path,
        cohort_summary=cohort_summary,
        presentation_summary=presentation_summary_valid,
        stat_comparison=stat_comparison_valid,
        assessment_summary=assessment_summary_empty,
        missed_or=missed_or_valid,
        missingness_df=missingness_df_valid,
        policy_comparison=policy_comparison_empty,
    )

    valid_content = report_path.read_text(encoding="utf-8")
    assert "Fisher's exact test indicates an unadjusted statistically significant difference ($p = 0.0079$)" in valid_content
    assert "28.3%" in valid_content
    assert "430.2" in valid_content

    # Test Case 3: Non-significant missingness test (p >= 0.05) - must not claim groups are balanced
    missingness_df_nonsig = pd.DataFrame([
        {
            "column": "imd_band",
            "retained_missing_count": 874,
            "retained_missing_pct": 3.90,
            "withdrawn_missing_count": 180,
            "withdrawn_missing_pct": 3.82,
            "delta_pct": -0.08,
            "p_value": 0.65,
        }
    ])
    runner._write_eda_report(
        report_path=report_path,
        cohort_summary=cohort_summary,
        presentation_summary=presentation_summary_valid,
        stat_comparison=stat_comparison_valid,
        assessment_summary=assessment_summary_empty,
        missed_or=missed_or_valid,
        missingness_df=missingness_df_nonsig,
        policy_comparison=policy_comparison_empty,
    )
    nonsig_content = report_path.read_text(encoding="utf-8")
    assert "No statistically significant difference was detected. This does not establish equivalence between the groups." in nonsig_content
    assert "balanced across outcome groups" not in nonsig_content
    assert "multiple-comparison" not in nonsig_content
