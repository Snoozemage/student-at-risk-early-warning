"""Unit tests for Phase 8 reporting tables, charts, and deliverable generators."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from student_retention.reporting.charts import (
    create_subgroup_disparity_chart,
    create_tier_distribution_chart,
    create_top_k_enrichment_chart,
    create_weekly_activity_chart,
)
from student_retention.reporting.generate_report import (
    ReportingDataError,
    _format_exclusion_hierarchy,
    _format_subgroup_parity_findings,
    generate_all_deliverables,
    generate_executive_summary,
    generate_presentation_outline,
    generate_recommendations,
    generate_technical_report,
    validate_reporting_inputs,
)
from student_retention.reporting.tables import (
    count_stable_metric_warnings,
    df_to_markdown,
    format_calibration_table,
    format_model_comparison_table,
    format_subgroup_fairness_table,
    format_top_k_table,
    load_chronological_term_top_k,
    load_executive_kpis,
)


def test_load_executive_kpis_missing_sources_returns_none(tmp_path: Path):
    """Verify that missing source files produce None for empirical metrics, not hardcoded defaults."""
    empty_summary = tmp_path / "non_existent_summary.json"
    empty_csv = tmp_path / "non_existent.csv"

    kpis = load_executive_kpis(
        summary_path=empty_summary,
        test_metrics_path=empty_csv,
        top_k_path=empty_csv,
        queue_path=empty_csv,
        cohort_report_path=empty_summary,
        calibration_path=empty_csv,
    )

    # Static policy constants must be preserved
    assert kpis["cutoff_day"] == 35
    assert kpis["advisor_capacity_k"] == 500

    # Empirical metrics must be None, NOT hardcoded defaults (27128, 0.3101, etc.)
    assert kpis["cohort_size"] is None
    assert kpis["base_rate_pct"] is None
    assert kpis["test_cohort_size"] is None
    assert kpis["model_pr_auc"] is None
    assert kpis["dummy_pr_auc"] is None
    assert kpis["pr_auc_lift_pct"] is None
    assert kpis["precision_at_500"] is None
    assert kpis["precision_at_500_baseline"] is None
    assert kpis["enrichment_at_500"] is None
    assert kpis["queue_total_records"] is None
    assert kpis["queue_unique_students"] is None
    assert kpis["tier_1_urgent_count"] is None


def test_load_executive_kpis_dynamic_calculation(tmp_path: Path):
    """Verify that KPI derived metrics (PR-AUC lift, enrichment) are calculated dynamically from inputs."""
    # Write synthetic model comparison CSV
    test_metrics_csv = tmp_path / "model_comparison_test.csv"
    pd.DataFrame(
        [
            {"model_key": "dummy_baseline", "pr_auc": 0.20, "n_samples": 1000, "base_rate": 0.20},
            {"model_key": "random_forest", "pr_auc": 0.40, "n_samples": 1000, "base_rate": 0.20},
        ]
    ).to_csv(test_metrics_csv, index=False)

    # Write synthetic top_k evaluation CSV
    top_k_csv = tmp_path / "top_k_evaluation.csv"
    pd.DataFrame(
        [
            {
                "model_key": "random_forest",
                "k": 500,
                "precision_at_k": 0.50,
                "random_precision_baseline": 0.20,
            }
        ]
    ).to_csv(top_k_csv, index=False)

    kpis = load_executive_kpis(
        test_metrics_path=test_metrics_csv,
        top_k_path=top_k_csv,
        summary_path=tmp_path / "none.json",
        queue_path=tmp_path / "none.csv",
        cohort_report_path=tmp_path / "none.json",
        calibration_path=tmp_path / "none.csv",
    )

    # Model PR-AUC = 0.40, Dummy = 0.20 -> Lift = (0.40 - 0.20)/0.20 * 100 = 100.0%
    assert kpis["model_pr_auc"] == 0.40
    assert kpis["dummy_pr_auc"] == 0.20
    assert kpis["pr_auc_lift_pct"] == pytest.approx(100.0)

    # Precision@500 = 0.50, Baseline = 0.20 -> Enrichment = 0.50 / 0.20 = 2.50x
    assert kpis["precision_at_500"] == 0.50
    assert kpis["precision_at_500_baseline"] == 0.20
    assert kpis["enrichment_at_500"] == pytest.approx(2.50)

    # Test changing inputs dynamically updates derived lift
    pd.DataFrame(
        [
            {"model_key": "dummy_baseline", "pr_auc": 0.20, "n_samples": 1000, "base_rate": 0.20},
            {"model_key": "random_forest", "pr_auc": 0.50, "n_samples": 1000, "base_rate": 0.20},
        ]
    ).to_csv(test_metrics_csv, index=False)

    kpis2 = load_executive_kpis(
        test_metrics_path=test_metrics_csv,
        top_k_path=top_k_csv,
        summary_path=tmp_path / "none.json",
        queue_path=tmp_path / "none.csv",
        cohort_report_path=tmp_path / "none.json",
        calibration_path=tmp_path / "none.csv",
    )
    # Lift = (0.50 - 0.20)/0.20 * 100 = 150.0%
    assert kpis2["pr_auc_lift_pct"] == pytest.approx(150.0)


def test_validate_reporting_inputs_missing_raises(tmp_path: Path):
    """Verify that missing required reporting inputs raise an explicit FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_reporting_inputs(
            tables_dir=tmp_path / "empty_tables",
            predictions_dir=tmp_path / "empty_preds",
        )
    assert "Missing required empirical reporting input" in str(exc_info.value)
    assert "scripts/" in str(exc_info.value)


def test_validate_reporting_inputs_empty_file_raises(tmp_path: Path):
    """Verify that an empty required input file raises ReportingDataError."""
    # Create empty files
    t_dir = tmp_path / "tables"
    t_dir.mkdir(parents=True)
    p_dir = tmp_path / "preds"
    p_dir.mkdir(parents=True)

    (t_dir / "model_training_summary.json").touch()

    with pytest.raises(ReportingDataError) as exc_info:
        validate_reporting_inputs(tables_dir=t_dir, predictions_dir=p_dir)
    assert "empty" in str(exc_info.value).lower()


def test_generate_all_deliverables_missing_inputs_fails(tmp_path: Path):
    """Verify that deliverable generation refuses to run and fails cleanly when inputs are missing."""
    with pytest.raises(FileNotFoundError):
        generate_all_deliverables(
            deliverables_dir=tmp_path / "deliv",
            tables_dir=tmp_path / "empty_tables",
            predictions_dir=tmp_path / "empty_preds",
        )


def test_generate_deliverables_reflects_changed_fixture(tmp_path: Path):
    """Verify that deliverables reflect changed fixture numbers and do NOT produce hardcoded defaults."""
    t_dir = tmp_path / "tables"
    p_dir = tmp_path / "preds"
    q_dir = tmp_path / "quality"
    d_dir = tmp_path / "deliv"
    t_dir.mkdir(parents=True)
    p_dir.mkdir(parents=True)
    q_dir.mkdir(parents=True)

    # 1. Custom synthetic summary JSON with custom numbers
    summary_data = {
        "n_total": 12345,
        "n_train": 9876,
        "n_test": 2469,
        "features_used": ["f1", "f2"],
        "target": "withdrawn_after_cutoff",
    }
    with open(t_dir / "model_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f)

    # 2. Custom synthetic model comparison CSV
    pd.DataFrame(
        [
            {
                "model_key": "dummy_baseline",
                "pr_auc": 0.2000,
                "roc_auc": 0.5000,
                "precision": 0.2000,
                "recall": 1.0000,
                "f1": 0.3333,
                "brier_score": 0.1600,
                "n_samples": 2469,
                "base_rate": 0.2000,
                "decision_threshold": 0.01,
            },
            {
                "model_key": "random_forest",
                "pr_auc": 0.4500,
                "roc_auc": 0.7500,
                "precision": 0.4200,
                "recall": 0.4800,
                "f1": 0.4480,
                "brier_score": 0.1250,
                "n_samples": 2469,
                "base_rate": 0.2000,
                "decision_threshold": 0.28,
            },
        ]
    ).to_csv(t_dir / "model_comparison_test.csv", index=False)

    # 3. Custom synthetic top_k evaluation CSV
    pd.DataFrame(
        [
            {
                "model_key": "random_forest",
                "k": 500,
                "precision_at_k": 0.6000,
                "recall_at_k": 0.3500,
                "random_precision_baseline": 0.2000,
                "is_valid_capacity": True,
            }
        ]
    ).to_csv(t_dir / "top_k_evaluation.csv", index=False)

    # 4. Custom synthetic calibration CSV
    pd.DataFrame(
        [
            {
                "model_configuration": "Random Forest (Raw Probabilities)",
                "calibration_method": "none",
                "brier_score": 0.1250,
                "expected_calibration_error": 0.0200,
                "maximum_calibration_error": 0.1500,
                "min_probability": 0.01,
                "max_probability": 0.65,
                "mean_probability": 0.20,
            },
            {
                "model_configuration": "Random Forest (Isotonic Calibrated)",
                "calibration_method": "isotonic_group_kfold_cv",
                "brier_score": 0.1240,
                "expected_calibration_error": 0.0100,
                "maximum_calibration_error": 0.1000,
                "min_probability": 0.0,
                "max_probability": 0.85,
                "mean_probability": 0.20,
            },
        ]
    ).to_csv(t_dir / "calibration_comparison.csv", index=False)

    # 5. Custom synthetic fairness audit CSV
    pd.DataFrame(
        [
            {
                "protected_attribute": "gender",
                "subgroup": "F",
                "sample_size": 100,
                "selection_rate": 0.25,
                "disparate_impact_ratio": 0.85,
                "fpr_ratio": 1.10,
                "is_statistically_stable": True,
            }
        ]
    ).to_csv(t_dir / "support_queue_fairness_audit.csv", index=False)

    # 6. Custom synthetic queue CSV
    pd.DataFrame(
        [
            {
                "student_key": f"STU_{i}",
                "code_module": "AAA",
                "code_presentation": "2013J",
                "risk_probability": 0.75 if i == 0 else 0.30,
                "priority_tier": "Tier 1 (Urgent)" if i == 0 else "Tier 3 (Standard)",
                "support_reason_summary": "Missed assignment",
                "recommended_next_step": "Advisor call",
                "review_status": "Pending",
            }
            for i in range(10)
        ]
    ).to_csv(p_dir / "student_support_queue.csv", index=False)

    # 7. Custom cohort report
    with open(q_dir / "cohort_policy_c_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "reconciliation": {"eligible_cohort_records": 12345},
                "target_distribution": {"positive_base_rate_pct": 20.0},
            },
            f,
        )

    # Generate deliverables
    res = generate_all_deliverables(
        deliverables_dir=d_dir,
        tables_dir=t_dir,
        predictions_dir=p_dir,
        quality_reports_dir=q_dir,
    )

    exec_text = res["executive_summary"].read_text(encoding="utf-8")
    tech_text = res["technical_report"].read_text(encoding="utf-8")

    # Assert that custom fixture values are reflected in the generated deliverables
    assert "12,345" in exec_text
    assert "0.4500" in exec_text
    assert "+125.0%" in exec_text  # (0.45 - 0.20)/0.20 * 100 = 125.0%
    assert "3.00x" in exec_text  # 0.60 / 0.20 = 3.00x

    # Assert that old hardcoded defaults do NOT appear in the report
    assert "27,128" not in exec_text
    assert "0.3101" not in exec_text
    assert "84.5%" not in exec_text
    assert "2.23x" not in exec_text

    # Technical report asserts
    assert "12,345" in tech_text
    assert "2,469" in tech_text


def test_df_to_markdown_formatting():
    """Verify df_to_markdown formats tables correctly without external tabulate library."""
    df = pd.DataFrame(
        {
            "Model": ["RF", "LR"],
            "PR-AUC": ["0.3101", "0.2955"],
        }
    )
    md = df_to_markdown(df)
    assert "| Model | PR-AUC |" in md
    assert "| :--- | :--- |" in md
    assert "| RF | 0.3101 |" in md
    assert "| LR | 0.2955 |" in md


def test_format_calibration_table(tmp_path: Path):
    """Verify format_calibration_table formats columns cleanly."""
    cal_csv = tmp_path / "calibration_comparison.csv"
    pd.DataFrame(
        [
            {
                "model_configuration": "RF Raw",
                "calibration_method": "none",
                "brier_score": 0.13131,
                "expected_calibration_error": 0.01150,
                "maximum_calibration_error": 0.32556,
                "min_probability": 0.025,
                "max_probability": 0.5435,
                "mean_probability": 0.1739,
            }
        ]
    ).to_csv(cal_csv, index=False)

    df = format_calibration_table(cal_csv)
    assert "Model Configuration" in df.columns
    assert "ECE" in df.columns
    assert "MCE" in df.columns
    assert df["ECE"].iloc[0] == "0.01150"


def test_create_tier_distribution_chart():
    """Verify tier distribution chart generation produces valid Figure."""
    tier_counts = {
        "Tier 1 (Urgent)": 50,
        "Tier 2 (Watch)": 150,
        "Tier 3 (Standard)": 800,
    }
    fig = create_tier_distribution_chart(tier_counts)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_weekly_activity_chart():
    """Verify weekly activity chart generation produces valid Figure."""
    fig = create_weekly_activity_chart()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_subgroup_disparity_chart():
    """Verify subgroup disparity chart generation produces valid Figure."""
    mock_audit = pd.DataFrame(
        {
            "protected_attribute": ["gender", "gender"],
            "subgroup": ["M", "F"],
            "disparate_impact_ratio": [1.0, 0.65],
            "is_statistically_stable": [True, True],
        }
    )
    fig = create_subgroup_disparity_chart(mock_audit, attribute="gender")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_create_top_k_enrichment_chart():
    """Verify Top-K enrichment chart produces valid Figure."""
    mock_topk = pd.DataFrame(
        {
            "k": [100, 500],
            "precision_at_k": [0.45, 0.37],
            "random_precision_baseline": [0.17, 0.17],
            "is_valid_capacity": [True, True],
        }
    )
    fig = create_top_k_enrichment_chart(mock_topk)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_format_model_comparison_table_synthetic(tmp_path: Path):
    """Verify format_model_comparison_table formats synthetic metrics correctly."""
    m_path = tmp_path / "model_comparison_test.csv"
    pd.DataFrame(
        [
            {
                "model_key": "random_forest",
                "pr_auc": 0.3101,
                "roc_auc": 0.6797,
                "precision": 0.3170,
                "recall": 0.4374,
                "f1": 0.3677,
                "brier_score": 0.13131,
                "decision_threshold": 0.25,
            }
        ]
    ).to_csv(m_path, index=False)
    df = format_model_comparison_table(m_path)
    assert not df.empty
    assert "Model" in df.columns
    assert "PR-AUC" in df.columns
    assert df["PR-AUC"].iloc[0] == "0.3101"


def test_format_subgroup_fairness_table_synthetic(tmp_path: Path):
    """Verify format_subgroup_fairness_table formats synthetic audit data correctly."""
    audit_path = tmp_path / "support_queue_fairness_audit.csv"
    pd.DataFrame(
        [
            {
                "protected_attribute": "gender",
                "subgroup": "F",
                "sample_size": 100,
                "selection_rate": 0.25,
                "base_rate": 0.20,
                "recall": 0.30,
                "precision": 0.35,
                "disparate_impact_ratio": 0.85,
                "fpr_ratio": 1.10,
                "is_statistically_stable": True,
            }
        ]
    ).to_csv(audit_path, index=False)
    df = format_subgroup_fairness_table(audit_path=audit_path, filter_stable_only=True)
    assert not df.empty
    assert "selection_rate_pct" in df.columns
    assert df["selection_rate_pct"].iloc[0] == "25.00%"


def test_format_top_k_table_synthetic(tmp_path: Path):
    """Verify format_top_k_table formats synthetic top-k metrics correctly."""
    topk_path = tmp_path / "top_k_evaluation.csv"
    pd.DataFrame(
        [
            {
                "k": 500,
                "precision_at_k": 0.374,
                "recall_at_k": 0.205,
                "random_precision_baseline": 0.168,
            }
        ]
    ).to_csv(topk_path, index=False)
    df = format_top_k_table(topk_path)
    assert not df.empty
    assert "precision_at_k_pct" in df.columns
    assert df["precision_at_k_pct"].iloc[0] == "37.40%"


def test_format_top_k_zero_or_nonfinite_baseline_is_unavailable(tmp_path: Path):
    path = tmp_path / "top_k_zero_baseline.csv"
    pd.DataFrame(
        [
            {"k": 500, "precision_at_k": 0.5, "random_precision_baseline": 0.0},
            {"k": 250, "precision_at_k": 0.4, "random_precision_baseline": float("inf")},
        ]
    ).to_csv(path, index=False)
    result = format_top_k_table(path)
    assert result["enrichment_fmt"].tolist() == ["N/A", "N/A"]
    assert result["baseline_pct"].tolist() == ["0.00%", "N/A"]


def test_exclusion_hierarchy_minimal_report_no_hardcoded_leakage(tmp_path: Path):
    """Verify that when audit report only contains eligible_cohort_records, no hardcoded empirical defaults appear."""
    minimal_report = tmp_path / "cohort_policy_c_audit_report.json"
    with open(minimal_report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "reconciliation": {
                    "eligible_cohort_records": 12345
                }
            },
            f,
        )

    res = _format_exclusion_hierarchy(minimal_report)
    # The fixture's cohort count must appear
    assert "12,345" in res

    # Historical / hardcoded empirical numbers must NOT appear
    assert "32,593" not in res
    assert "27,128" not in res
    assert "83.23%" not in res
    assert "4,718" not in res
    assert "22,410" not in res


def test_exclusion_hierarchy_dynamic_percentage_derivation(tmp_path: Path):
    """Verify that percentage is derived only when both compatible numerator and denominator exist."""
    # Case A: both numerator and denominator present
    rep_a = tmp_path / "rep_a.json"
    with open(rep_a, "w", encoding="utf-8") as f:
        json.dump(
            {
                "reconciliation": {
                    "eligible_cohort_records": 1000,
                    "total_source_records": 2000,
                }
            },
            f,
        )
    res_a = _format_exclusion_hierarchy(rep_a)
    assert "1,000" in res_a
    assert "2,000" in res_a
    assert "50.00%" in res_a

    # Case B: total_source_records missing -> percentage omitted, does not use hardcoded total
    rep_b = tmp_path / "rep_b.json"
    with open(rep_b, "w", encoding="utf-8") as f:
        json.dump(
            {
                "reconciliation": {
                    "eligible_cohort_records": 1000,
                }
            },
            f,
        )
    res_b = _format_exclusion_hierarchy(rep_b)
    assert "1,000" in res_b
    assert "%" not in res_b
    assert "32,593" not in res_b


def test_load_executive_kpis_enrichment_never_substitutes_dummy_pr_auc(tmp_path: Path):
    """Do not substitute dummy PR-AUC or a different cohort's prevalence for test prevalence."""
    # Write model_comparison_test.csv with dummy PR-AUC = 0.15 but NO base_rate
    comp_csv = tmp_path / "model_comparison_test.csv"
    pd.DataFrame(
        [
            {"model_key": "dummy_baseline", "pr_auc": 0.1500, "n_samples": 1000},
            {"model_key": "random_forest", "pr_auc": 0.4500, "roc_auc": 0.7000, "n_samples": 1000},
        ]
    ).to_csv(comp_csv, index=False)

    # Top-K CSV has precision_at_k = 0.30, but NO random_precision_baseline, NO total_positives, NO base_rate
    top_k_csv = tmp_path / "top_k_evaluation.csv"
    pd.DataFrame(
        [
            {
                "model_key": "random_forest",
                "k": 500,
                "precision_at_k": 0.3000,
            }
        ]
    ).to_csv(top_k_csv, index=False)

    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(
        json.dumps({"reconciliation": {"eligible_cohort_records": 2000}, "target_distribution": {"positive_base_rate_pct": 20.0}}),
        encoding="utf-8",
    )

    kpis = load_executive_kpis(
        test_metrics_path=comp_csv,
        top_k_path=top_k_csv,
        summary_path=tmp_path / "none.json",
        queue_path=tmp_path / "none.csv",
        cohort_report_path=cohort_path,
        calibration_path=tmp_path / "none.csv",
    )

    # Dummy PR-AUC must be extracted
    assert kpis["dummy_pr_auc"] == 0.1500
    assert kpis["precision_at_500"] == 0.3000
    assert kpis["precision_at_500_baseline"] is None

    # Enrichment must be None, NOT 0.30 / 0.15 = 2.0x
    assert kpis["enrichment_at_500"] is None


def test_load_executive_kpis_enrichment_valid_baselines(tmp_path: Path):
    """Verify enrichment is derived from valid random precision baseline or prevalence."""
    # Case 1: Direct random_precision_baseline
    top_k_1 = tmp_path / "top_k_1.csv"
    pd.DataFrame(
        [
            {
                "model_key": "random_forest",
                "k": 500,
                "precision_at_k": 0.4000,
                "random_precision_baseline": 0.1000,
            }
        ]
    ).to_csv(top_k_1, index=False)

    kpis1 = load_executive_kpis(
        top_k_path=top_k_1,
        test_metrics_path=tmp_path / "none.csv",
        summary_path=tmp_path / "none.json",
        queue_path=tmp_path / "none.csv",
        cohort_report_path=tmp_path / "none.json",
        calibration_path=tmp_path / "none.csv",
    )
    assert kpis1["enrichment_at_500"] == pytest.approx(4.00)

    # Case 2: Derived from total_positives and n_samples
    top_k_2 = tmp_path / "top_k_2.csv"
    pd.DataFrame(
        [
            {
                "model_key": "random_forest",
                "k": 500,
                "precision_at_k": 0.5000,
                "total_positives": 250,
                "n_samples": 1000,
            }
        ]
    ).to_csv(top_k_2, index=False)

    kpis2 = load_executive_kpis(
        top_k_path=top_k_2,
        test_metrics_path=tmp_path / "none.csv",
        summary_path=tmp_path / "none.json",
        queue_path=tmp_path / "none.csv",
        cohort_report_path=tmp_path / "none.json",
        calibration_path=tmp_path / "none.csv",
    )
    # Prevalence = 250 / 1000 = 0.25 -> Enrichment = 0.50 / 0.25 = 2.00x
    assert kpis2["enrichment_at_500"] == pytest.approx(2.00)


def test_format_subgroup_fairness_missing_warning_columns_no_false_zeros(tmp_path: Path):
    """Verify that missing fairness warning columns produce None/Unavailable, never false zero warnings."""
    audit_csv = tmp_path / "incomplete_fairness_audit.csv"
    pd.DataFrame(
        [
            {
                "protected_attribute": "gender",
                "subgroup": "F",
                "sample_size": 200,
                "selection_rate": 0.25,
                "base_rate": 0.20,
                "recall": 0.30,
                "precision": 0.35,
                "is_statistically_stable": True,
            }
        ]
    ).to_csv(audit_csv, index=False)

    # In format_subgroup_fairness_table, warning columns must be None
    df = format_subgroup_fairness_table(audit_path=audit_csv)
    assert pd.isna(df["disparate_impact_warning"].iloc[0])
    assert pd.isna(df["elevated_fpr_warning"].iloc[0])

    # In _format_subgroup_parity_findings, it must explicitly state Unavailable
    findings_str = _format_subgroup_parity_findings(queue_audit_path=audit_csv)
    assert "Unavailable" in findings_str
    assert "0 subgroups (all groups within parity tolerance)" not in findings_str


def test_presentation_outline_dynamic_metrics(tmp_path: Path):
    """Verify that presentation outline uses dynamic metrics and contains zero hardcoded metrics."""
    out_file = tmp_path / "presentation_outline.md"
    kpis = {
        "cohort_size": 15000,
        "base_rate_pct": 22.50,
        "model_pr_auc": 0.4200,
        "model_roc_auc": 0.7450,
        "pr_auc_lift_pct": 110.0,
        "dummy_pr_auc": 0.2000,
        "precision_at_500": 0.5500,
        "precision_at_500_baseline": 0.2000,
        "enrichment_at_500": 2.75,
        "ece_raw": 0.0250,
        "ece_calibrated": 0.0125,
        "ece_reduction_pct": 50.0,
        "queue_total_records": 1000,
        "tier_1_urgent_count": 50,
        "tier_2_watch_count": 200,
        "tier_3_standard_count": 750,
    }

    generate_presentation_outline(kpis, output_path=out_file)
    text = out_file.read_text(encoding="utf-8")

    # Assert dynamic metrics are present
    assert "15,000" in text
    assert "22.50%" in text
    assert "0.4200" in text
    assert "ROC-AUC: 0.7450." in text
    assert "2.75x" in text
    assert "50.0% error reduction" in text

    # Assert historical hardcoded literals are NOT present
    assert "0.7202" not in text
    assert "2.23x" not in text
    assert "36.2%" not in text
    assert "0.0115" not in text
    assert "0.0073" not in text
    assert "Odds Ratio 3.23" not in text
    assert "4.1x" not in text
    assert "41 distinct" not in text


def test_validate_reporting_inputs_incompatible_cohort_raises(tmp_path: Path):
    """Verify that validate_reporting_inputs rejects incompatible cohort counts (total < eligible)."""
    t_dir = tmp_path / "tables"
    p_dir = tmp_path / "preds"
    q_dir = tmp_path / "quality"
    t_dir.mkdir(parents=True)
    p_dir.mkdir(parents=True)
    q_dir.mkdir(parents=True)

    # Populate valid dummy files for everything except cohort reconciliation
    with open(t_dir / "model_training_summary.json", "w") as f:
        json.dump({"n_total": 1000}, f)
    pd.DataFrame([{"model_key": "random_forest", "pr_auc": 0.35}]).to_csv(t_dir / "model_comparison_test.csv", index=False)
    pd.DataFrame([{"k": 500, "precision_at_k": 0.40, "random_precision_baseline": 0.20}]).to_csv(t_dir / "top_k_evaluation.csv", index=False)
    pd.DataFrame([{"calibration_method": "isotonic", "expected_calibration_error": 0.01}]).to_csv(t_dir / "calibration_comparison.csv", index=False)
    pd.DataFrame([{"protected_attribute": "gender", "subgroup": "F", "selection_rate": 0.2, "disparate_impact_ratio": 1.0, "fpr_ratio": 1.0, "is_statistically_stable": True}]).to_csv(t_dir / "support_queue_fairness_audit.csv", index=False)
    pd.DataFrame([{"student_key": "S1", "priority_tier": "Tier 1 (Urgent)", "risk_probability": 0.8}]).to_csv(p_dir / "student_support_queue.csv", index=False)

    # Incompatible cohort: total_source_records (500) < eligible_cohort_records (1000)
    with open(q_dir / "cohort_policy_c_audit_report.json", "w") as f:
        json.dump({"reconciliation": {"eligible_cohort_records": 1000, "total_source_records": 500}}, f)

    with pytest.raises(ReportingDataError) as exc_info:
        validate_reporting_inputs(tables_dir=t_dir, predictions_dir=p_dir, quality_reports_dir=q_dir)
    assert "Incompatible cohort reconciliation" in str(exc_info.value)


def test_executive_summary_missing_tier_counts_semantics(tmp_path: Path):
    """Test that missing tier counts remain unavailable and do not fabricate 0 cases or 0.0%."""
    dest = tmp_path / "executive_summary.md"
    kpis = {
        "cohort_size": 1000,
        "base_rate_pct": 20.0,
        "model_pr_auc": 0.40,
        "dummy_pr_auc": 0.20,
        "pr_auc_lift_pct": 100.0,
        "precision_at_500": 0.45,
        "enrichment_at_500": 2.25,
        "queue_total_records": None,
        "tier_1_urgent_count": None,
        "tier_2_watch_count": None,
        "tier_3_standard_count": None,
    }
    generate_executive_summary(kpis, dest, queue_audit_path=tmp_path / "nonexistent.csv")
    content = dest.read_text(encoding="utf-8")
    assert "Unavailable (queue data not loaded or evaluated)" in content
    assert "0 records (0.0% of demonstration queue)" not in content
    assert "Urgent Cases (Tier 1): 0 records" not in content


def test_generated_deliverables_distinguish_prediction_from_intervention(tmp_path: Path):
    """Generated prose must not claim inevitability or validated intervention timing/policy."""
    kpis = {"cohort_size": 1000}
    summary_path = tmp_path / "executive_summary.md"
    outline_path = tmp_path / "presentation_outline.md"
    recommendations_path = tmp_path / "recommendations.md"
    generate_executive_summary(kpis, summary_path, queue_audit_path=tmp_path / "missing.csv")
    generate_presentation_outline(kpis, outline_path, queue_audit_path=tmp_path / "missing.csv")
    generate_recommendations(kpis, recommendations_path)

    summary = summary_path.read_text(encoding="utf-8").lower()
    outline = outline_path.read_text(encoding="utf-8").lower()
    recommendations = recommendations_path.read_text(encoding="utf-8").lower()
    combined = f"{summary}\n{outline}\n{recommendations}"

    for unsupported in (
        "departure becomes inevitable",
        "unrecoverable academic failure",
        "often unrecoverable",
        "early warning at day 35 is actionable",
        "establish a rapid-response coursework extension protocol",
        "3-day sla",
        "7-day sla",
    ):
        assert unsupported not in combined

    assert "does not show that outreach at day 35 changes outcomes" in summary
    assert "not a validated service level" in recommendations
    assert "intervention effectiveness in preventing attrition requires prospective, randomized evaluation" in outline
    assert "proposed" in recommendations

    repo_root = Path(__file__).resolve().parents[2]
    notebook_text = "\n".join(
        (repo_root / "notebooks" / name).read_text(encoding="utf-8").lower()
        for name in ("08_fairness_and_threshold_analysis.ipynb", "09_business_recommendations.ipynb")
    )
    assert "actual intervention policy" not in notebook_text
    assert "validated operational threshold" not in notebook_text
    assert "sla enforcement" not in notebook_text
    assert "intervention effectiveness has not been established" in notebook_text


def test_executive_summary_measured_zero_tier_count(tmp_path: Path):
    """Test that an explicitly measured zero count remains 0 with valid percentage."""
    dest = tmp_path / "executive_summary.md"
    kpis = {
        "cohort_size": 1000,
        "base_rate_pct": 20.0,
        "model_pr_auc": 0.40,
        "dummy_pr_auc": 0.20,
        "pr_auc_lift_pct": 100.0,
        "precision_at_500": 0.45,
        "enrichment_at_500": 2.25,
        "queue_total_records": 500,
        "tier_1_urgent_count": 0,
        "tier_2_watch_count": 100,
        "tier_3_standard_count": 400,
    }
    generate_executive_summary(kpis, dest, queue_audit_path=tmp_path / "nonexistent.csv")
    content = dest.read_text(encoding="utf-8")
    assert "0 records (0.0% of demonstration queue)" in content
    assert "100 records (20.0% of demonstration queue)" in content
    assert "400 records (80.0% of demonstration queue)" in content


def test_executive_summary_missing_denominator_does_not_fabricate_percentage(tmp_path: Path):
    """Test that tier counts without valid queue denominator do not fabricate percentages."""
    dest = tmp_path / "executive_summary.md"
    kpis = {
        "cohort_size": 1000,
        "base_rate_pct": 20.0,
        "model_pr_auc": 0.40,
        "dummy_pr_auc": 0.20,
        "pr_auc_lift_pct": 100.0,
        "precision_at_500": 0.45,
        "enrichment_at_500": 2.25,
        "queue_total_records": None,
        "tier_1_urgent_count": 5,
        "tier_2_watch_count": 45,
        "tier_3_standard_count": 450,
    }
    generate_executive_summary(kpis, dest, queue_audit_path=tmp_path / "nonexistent.csv")
    content = dest.read_text(encoding="utf-8")
    assert "5 records (queue percentage unavailable)" in content
    assert "% of demonstration queue" not in content


def test_presentation_outline_missing_queue_does_not_claim_zero_cases(tmp_path: Path):
    """Test that presentation outline without queue data does not display 0 cases."""
    dest = tmp_path / "presentation_outline.md"
    kpis = {
        "cohort_size": 1000,
        "base_rate_pct": 20.0,
        "model_pr_auc": 0.40,
        "dummy_pr_auc": 0.20,
        "pr_auc_lift_pct": 100.0,
        "precision_at_500": 0.45,
        "enrichment_at_500": 2.25,
        "queue_total_records": None,
        "tier_1_urgent_count": None,
        "tier_2_watch_count": None,
        "tier_3_standard_count": None,
    }
    generate_presentation_outline(kpis, dest, queue_audit_path=tmp_path / "nonexistent.csv")
    content = dest.read_text(encoding="utf-8")
    assert "Queue data unavailable; operational triage distribution not evaluated" in content
    slide8_section = content.split("## Slide 8")[1].split("## Slide 9")[0]
    assert "0 cases" not in slide8_section


def test_recommendations_dynamic_statistical_evidence(tmp_path: Path):
    """Test that recommendations dynamically derives p-values and withdrawal rates."""
    stat_csv = tmp_path / "statistical_comparisons.csv"
    pd.DataFrame([
        {"feature": "has_missed_due_assessment", "p_value": 0.00345},
        {"feature": "days_since_last_vle", "p_value": 8.216120351754304e-36},
        {"feature": "date_registration", "p_value": 0.0123, "n_retained": 22410, "n_withdrawn": 4718, "retained_median": -52, "withdrawn_median": -58},
    ]).to_csv(stat_csv, index=False)

    wdr_csv = tmp_path / "presentation_withdrawal_rates.csv"
    pd.DataFrame([
        {"code_module": "CCC", "faculty": "STEM", "withdrawal_rate_pct": 32.5},
        {"code_module": "AAA", "faculty": "Social Sciences", "withdrawal_rate_pct": 11.5},
    ]).to_csv(wdr_csv, index=False)

    dest = tmp_path / "recommendations.md"
    generate_recommendations(
        output_path=dest,
        statistical_comparisons_path=stat_csv,
        presentation_withdrawal_rates_path=wdr_csv,
    )
    content = dest.read_text(encoding="utf-8")

    assert "p = 0.0034" in content or "p = 0.0035" in content or "p = 3.45e-03" in content
    assert "p = 0.0123" in content
    assert "withdrawn group registered earlier at the median (-58 vs -52 days" in content
    assert "does not justify targeting either group" in content
    assert "Affected Target Cohort:** All students may access onboarding" in content
    assert "Late-Enrolling Students" not in content
    assert "shorter registration lead times" not in content
    assert "p = 8.22e-36" in content
    assert "continuous comparison does not validate a 14-day threshold" in content
    assert "14+ days prior to Day 35" not in content
    assert "32.5%" in content
    assert "11.5%" in content

    # Fixed stale empirical literals are completely absent
    assert "10^{-100}" not in content
    assert "10^{-8}" not in content
    assert "20.5%" not in content
    assert "13.8%" not in content


def test_recommendations_omitted_statistical_evidence_handles_missing_cleanly(tmp_path: Path):
    """Test that recommendations without statistical tables omits stale numbers and states unavailable."""
    dest = tmp_path / "recommendations.md"
    generate_recommendations(
        output_path=dest,
        statistical_comparisons_path=tmp_path / "nonexistent_stat.csv",
        presentation_withdrawal_rates_path=tmp_path / "nonexistent_wdr.csv",
    )
    content = dest.read_text(encoding="utf-8")

    assert "statistical comparison metrics unavailable" in content
    assert "presentation withdrawal rate metrics unavailable" in content
    assert "does not validate a 14-day VLE inactivity threshold" in content
    assert "14+ days prior to Day 35" not in content
    assert "10^{-100}" not in content
    assert "10^{-8}" not in content
    assert "20.5%" not in content
    assert "13.8%" not in content


def test_fairness_guarantees_removed_and_dynamic_scope(tmp_path: Path):
    """Test that fairness reporting dynamically counts attributes/subgroups and removes 'ensures' guarantees."""
    dest_summary = tmp_path / "executive_summary.md"
    dest_outline = tmp_path / "presentation_outline.md"

    # 1. Missing fairness audit
    kpis = {"cohort_size": 1000}
    generate_executive_summary(kpis, dest_summary, queue_audit_path=tmp_path / "nonexistent.csv")
    generate_presentation_outline(kpis, dest_outline, queue_audit_path=tmp_path / "nonexistent.csv")

    summary_text = dest_summary.read_text(encoding="utf-8")
    outline_text = dest_outline.read_text(encoding="utf-8")

    assert "Fairness audit data is unavailable" in summary_text
    assert "Demographic subgroup audit data is unavailable" in outline_text
    assert "ensures no systemic under-servicing" not in summary_text
    assert "Audited 36 demographic subgroups across 6 protected characteristics" not in outline_text

    # 2. Synthetic audit with 3 attributes and 8 subgroups
    audit_csv = tmp_path / "support_queue_fairness_audit.csv"
    pd.DataFrame([
        {"protected_attribute": "gender", "subgroup": "F", "is_statistically_stable": True, "fpr_ratio": 1.1},
        {"protected_attribute": "gender", "subgroup": "M", "is_statistically_stable": True, "fpr_ratio": 1.0},
        {"protected_attribute": "age", "subgroup": "young", "is_statistically_stable": True, "fpr_ratio": 1.2},
        {"protected_attribute": "age", "subgroup": "mid", "is_statistically_stable": True, "fpr_ratio": 1.0},
        {"protected_attribute": "age", "subgroup": "senior", "is_statistically_stable": False, "fpr_ratio": 0.9},
        {"protected_attribute": "region", "subgroup": "north", "is_statistically_stable": True, "fpr_ratio": 1.6},
        {"protected_attribute": "region", "subgroup": "south", "is_statistically_stable": True, "fpr_ratio": 1.0},
        {"protected_attribute": "region", "subgroup": "east", "is_statistically_stable": True, "fpr_ratio": 1.0},
    ]).to_csv(audit_csv, index=False)

    generate_executive_summary(kpis, dest_summary, queue_audit_path=audit_csv)
    generate_presentation_outline(kpis, dest_outline, queue_audit_path=audit_csv)

    summary_text2 = dest_summary.read_text(encoding="utf-8")
    outline_text2 = dest_outline.read_text(encoding="utf-8")

    assert "evaluated across 3 protected demographic attributes" in summary_text2
    assert "8 subgroups total" in summary_text2
    assert "1 demographic subgroup exhibit elevated False Positive Ratios" in summary_text2
    assert "Audited 8 demographic subgroups across 3 protected characteristics" in outline_text2
    assert "1 stable subgroups exhibit elevated False Positive Ratios" in outline_text2
    assert "ensures no systemic under-servicing" not in summary_text2


def test_calibration_technically_accurate_wording(tmp_path: Path):
    """Test that calibration narratives avoid guarantees and handle missing metrics cleanly."""
    dest_summary = tmp_path / "executive_summary.md"
    dest_tech = tmp_path / "technical_report.md"
    dest_outline = tmp_path / "presentation_outline.md"

    # Missing calibration inputs
    kpis_missing = {"cohort_size": 1000}
    generate_executive_summary(kpis_missing, dest_summary)
    generate_technical_report(kpis_missing, dest_tech)
    generate_presentation_outline(kpis_missing, dest_outline)

    assert "Probability calibration diagnostics unavailable" in dest_summary.read_text(encoding="utf-8")
    assert "Test-set calibration diagnostics are unavailable; empirical probability alignment could not be verified." in dest_tech.read_text(encoding="utf-8")
    assert "Calibration diagnostics unavailable" in dest_outline.read_text(encoding="utf-8")
    assert "ensures true probabilities" not in dest_outline.read_text(encoding="utf-8")

    # Present calibration inputs
    kpis_present = {
        "cohort_size": 1000,
        "ece_raw": 0.020,
        "ece_calibrated": 0.010,
        "ece_reduction_pct": 50.0,
        "selected_calibration_method": "isotonic",
    }
    generate_executive_summary(kpis_present, dest_summary)
    generate_technical_report(kpis_present, dest_tech)
    generate_presentation_outline(kpis_present, dest_outline)

    assert "Isotonic calibration improved empirical probability alignment" in dest_summary.read_text(encoding="utf-8")
    assert "improved empirical probability alignment, achieving a **50.0% error reduction**" in dest_tech.read_text(encoding="utf-8")
    assert "Isotonic calibration improved empirical probability alignment" in dest_outline.read_text(encoding="utf-8")
    assert "ensures true probabilities" not in dest_outline.read_text(encoding="utf-8")


def test_conditional_claims_on_actual_metrics(tmp_path: Path):
    """Test that report language is strictly conditional on actual metric values."""
    dest_summary = tmp_path / "executive_summary.md"
    dest_tech = tmp_path / "technical_report.md"
    dest_outline = tmp_path / "presentation_outline.md"

    # 1. Missing metrics cannot produce strong predictive lift or double efficiency claims
    kpis_empty = {"cohort_size": 1000}
    generate_executive_summary(kpis_empty, dest_summary)
    generate_technical_report(kpis_empty, dest_tech)
    generate_presentation_outline(kpis_empty, dest_outline)

    s_text = dest_summary.read_text(encoding="utf-8")
    o_text = dest_outline.read_text(encoding="utf-8")
    t_text = dest_tech.read_text(encoding="utf-8")

    assert "strong predictive lift" not in s_text.lower()
    assert "strong predictive lift" not in t_text.lower()
    assert "more than double" not in s_text.lower()
    assert "double advisor triage efficiency" not in o_text.lower()
    assert "predictable and preventable" not in s_text.lower()
    assert "predictable and preventable" not in o_text.lower()
    assert "predictable and preventable" not in t_text.lower()
    assert "prospective" in s_text.lower() or "elevated risk" in s_text.lower()

    # 2. Enrichment = 1.4x: cannot claim double efficiency, must report actual factor
    kpis_14 = {
        "cohort_size": 1000,
        "dummy_pr_auc": 0.20,
        "model_pr_auc": 0.30,
        "pr_auc_lift_pct": 50.0,
        "enrichment_at_500": 1.40,
        "precision_at_500": 0.28,
            "precision_at_500_baseline": 0.20,
    }
    generate_executive_summary(kpis_14, dest_summary)
    generate_presentation_outline(kpis_14, dest_outline)
    s_text_14 = dest_summary.read_text(encoding="utf-8")
    o_text_14 = dest_outline.read_text(encoding="utf-8")

    assert "more than double" not in s_text_14.lower()
    assert "double advisor triage efficiency" not in o_text_14.lower()
    assert "1.40x relative to the test-split random-ranking baseline" in s_text_14
    assert "Pooled-test enrichment:** **1.40x** relative to random ranking" in o_text_14

    # 3. Enrichment = 2.45x: describes only pooled-test cohort-prior enrichment.
    kpis_24 = {
        "cohort_size": 1000,
        "dummy_pr_auc": 0.20,
        "model_pr_auc": 0.35,
        "pr_auc_lift_pct": 75.0,
        "enrichment_at_500": 2.45,
        "precision_at_500": 0.49,
            "precision_at_500_baseline": 0.20,
    }
    generate_executive_summary(kpis_24, dest_summary)
    generate_presentation_outline(kpis_24, dest_outline)
    s_text_24 = dest_summary.read_text(encoding="utf-8")
    o_text_24 = dest_outline.read_text(encoding="utf-8")

    assert "more than double the test-split random-ranking baseline in this pooled diagnostic" in s_text_24.lower()
    assert "more than double advisor triage efficiency" not in o_text_24.lower()
    assert "pooled-test enrichment" in o_text_24.lower()
    assert "2.45x" in s_text_24

    # 4. Negative PR-AUC lift is not described as a gain
    kpis_neg = {
        "cohort_size": 1000,
        "dummy_pr_auc": 0.25,
        "model_pr_auc": 0.20,
        "pr_auc_lift_pct": -20.0,
    }
    generate_executive_summary(kpis_neg, dest_summary)
    s_text_neg = dest_summary.read_text(encoding="utf-8")
    assert "substantial gain" not in s_text_neg.lower()
    assert "measured deficit" in s_text_neg.lower()
    assert "strong predictive lift" not in s_text_neg.lower()


def test_fairness_audit_completeness_and_stability(tmp_path: Path):
    """Test fairness audit completeness, handling missing stability and missing metrics."""
    dest_summary = tmp_path / "executive_summary.md"
    dest_outline = tmp_path / "presentation_outline.md"
    kpis = {"cohort_size": 1000}

    # 1. Valid artifact with stability and NO warnings
    audit_clean = tmp_path / "audit_clean.csv"
    pd.DataFrame([
        {"protected_attribute": "gender", "subgroup": "F", "is_statistically_stable": True, "fpr_ratio": 1.10, "disparate_impact_ratio": 1.05},
        {"protected_attribute": "gender", "subgroup": "M", "is_statistically_stable": True, "fpr_ratio": 1.00, "disparate_impact_ratio": 1.00},
    ]).to_csv(audit_clean, index=False)

    generate_executive_summary(kpis, dest_summary, queue_audit_path=audit_clean)
    generate_presentation_outline(kpis, dest_outline, queue_audit_path=audit_clean)
    s_clean = dest_summary.read_text(encoding="utf-8")
    o_clean = dest_outline.read_text(encoding="utf-8")

    assert "zero demographic subgroups exhibit elevated false positive ratios" in s_clean.lower()
    assert "zero stable subgroups exhibit elevated false positive ratios" in o_clean.lower()
    findings_clean = _format_subgroup_parity_findings(audit_clean)
    assert "0 stable subgroups exceeding FPR ratio 1.50" in findings_clean

    # 2. Valid artifact with ONE elevated FPR warning
    audit_warn = tmp_path / "audit_warn.csv"
    pd.DataFrame([
        {"protected_attribute": "gender", "subgroup": "F", "is_statistically_stable": True, "fpr_ratio": 1.65, "disparate_impact_ratio": 1.05},
        {"protected_attribute": "gender", "subgroup": "M", "is_statistically_stable": True, "fpr_ratio": 1.00, "disparate_impact_ratio": 1.00},
    ]).to_csv(audit_warn, index=False)

    generate_executive_summary(kpis, dest_summary, queue_audit_path=audit_warn)
    s_warn = dest_summary.read_text(encoding="utf-8")
    assert "1 demographic subgroup exhibit elevated False Positive Ratios" in s_warn

    # 3. Artifact missing is_statistically_stable: MUST be reported as incomplete audit
    audit_no_stability = tmp_path / "audit_no_stability.csv"
    pd.DataFrame([
        {"protected_attribute": "gender", "subgroup": "F", "fpr_ratio": 1.10, "disparate_impact_ratio": 1.05},
        {"protected_attribute": "gender", "subgroup": "M", "fpr_ratio": 1.00, "disparate_impact_ratio": 1.00},
    ]).to_csv(audit_no_stability, index=False)

    generate_executive_summary(kpis, dest_summary, queue_audit_path=audit_no_stability)
    generate_presentation_outline(kpis, dest_outline, queue_audit_path=audit_no_stability)
    s_no_stab = dest_summary.read_text(encoding="utf-8")
    o_no_stab = dest_outline.read_text(encoding="utf-8")

    # Incomplete cases must NEVER be reported as zero warnings among stable subgroups
    assert "0 demographic subgroups exhibit elevated False Positive Ratios" not in s_no_stab
    assert "0 stable subgroups exhibit elevated False Positive Ratios" not in o_no_stab
    assert "subgroup statistical stability information ('is_statistically_stable') is absent" in s_no_stab
    assert "subgroup statistical stability ('is_statistically_stable') is unverified" in o_no_stab.lower()

    findings_no_stab = _format_subgroup_parity_findings(audit_no_stability)
    assert "Incomplete audit (subgroup stability unverified)" in findings_no_stab
    assert "0 stable subgroups" not in findings_no_stab

    # format_subgroup_fairness_table with filter_stable_only must return empty df
    df_table_filtered = format_subgroup_fairness_table(audit_path=audit_no_stability, filter_stable_only=True)
    assert df_table_filtered.empty

    # 4. Artifact missing fpr_ratio
    audit_no_fpr = tmp_path / "audit_no_fpr.csv"
    pd.DataFrame([
        {"protected_attribute": "gender", "subgroup": "F", "is_statistically_stable": True, "disparate_impact_ratio": 1.05},
        {"protected_attribute": "gender", "subgroup": "M", "is_statistically_stable": True, "disparate_impact_ratio": 1.00},
    ]).to_csv(audit_no_fpr, index=False)

    generate_executive_summary(kpis, dest_summary, queue_audit_path=audit_no_fpr)
    s_no_fpr = dest_summary.read_text(encoding="utf-8")
    assert "0 demographic subgroups exhibit elevated False Positive Ratios" not in s_no_fpr
    assert "false positive ratio metrics ('fpr_ratio') are unavailable" in s_no_fpr

    findings_no_fpr = _format_subgroup_parity_findings(audit_no_fpr)
    assert "Unavailable (missing 'fpr_ratio')" in findings_no_fpr


def test_calibration_directional_semantics(tmp_path: Path):
    """Test calibration narratives reflect improvement, deterioration, no change, and raw ECE 0."""
    dest_summary = tmp_path / "executive_summary.md"
    dest_tech = tmp_path / "technical_report.md"

    # 1. Improvement: 0.02 to 0.01 -> 50% error reduction
    kpis_improve = {"cohort_size": 1000, "ece_raw": 0.02, "ece_calibrated": 0.01, "ece_reduction_pct": 50.0}
    generate_executive_summary(kpis_improve, dest_summary)
    generate_technical_report(kpis_improve, dest_tech)
    s_imp = dest_summary.read_text(encoding="utf-8")
    t_imp = dest_tech.read_text(encoding="utf-8")
    assert "50.0% error reduction" in s_imp
    assert "improved empirical probability alignment" in s_imp
    assert "improved empirical probability alignment, achieving a **50.0% error reduction**" in t_imp

    # 2. No change: 0.01 to 0.01
    kpis_nochange = {"cohort_size": 1000, "ece_raw": 0.01, "ece_calibrated": 0.01, "ece_reduction_pct": 0.0}
    generate_executive_summary(kpis_nochange, dest_summary)
    generate_technical_report(kpis_nochange, dest_tech)
    s_nc = dest_summary.read_text(encoding="utf-8")
    t_nc = dest_tech.read_text(encoding="utf-8")
    assert "no change" in s_nc
    assert "yielded no change in expected calibration error" in t_nc.lower()
    assert "improved empirical probability alignment" not in t_nc

    # 3. Deterioration: 0.01 to 0.02 -> 100% error increase
    kpis_worse = {"cohort_size": 1000, "ece_raw": 0.01, "ece_calibrated": 0.02, "ece_reduction_pct": -100.0}
    generate_executive_summary(kpis_worse, dest_summary)
    generate_technical_report(kpis_worse, dest_tech)
    s_worse = dest_summary.read_text(encoding="utf-8")
    t_worse = dest_tech.read_text(encoding="utf-8")
    assert "error increase" in s_worse
    assert "deteriorated empirical probability alignment" in s_worse
    assert "error increase" in t_worse
    assert "deteriorated empirical probability alignment" in t_worse
    assert "error reduction" not in s_worse
    assert "improved" not in t_worse

    # 4. Raw ECE == 0.0 -> undefined relative reduction
    cal_csv = tmp_path / "calibration_comparison.csv"
    pd.DataFrame([
        {"calibration_method": "uncalibrated", "expected_calibration_error": 0.0},
        {"calibration_method": "isotonic", "expected_calibration_error": 0.01},
    ]).to_csv(cal_csv, index=False)
    kpis_zero = load_executive_kpis(
        calibration_path=cal_csv,
        summary_path=tmp_path / "none.json",
        test_metrics_path=tmp_path / "none.csv",
        top_k_path=tmp_path / "none.csv",
        queue_path=tmp_path / "none.csv",
        cohort_report_path=tmp_path / "none.json",
    )
    assert kpis_zero["ece_raw"] == 0.0
    assert kpis_zero["ece_calibrated"] == 0.01
    assert kpis_zero["ece_reduction_pct"] is None  # Undefined denominator


def test_model_metric_identity_and_row_reordering(tmp_path: Path):
    """Test that model metric loading explicitly matches model key, independent of CSV row order."""
    m_csv = tmp_path / "model_comparison_test.csv"

    # Reorder test 1: dummy first, then rf_calibrated, then random_forest
    pd.DataFrame([
        {"model_key": "dummy_baseline", "pr_auc": 0.15, "n_samples": 1000, "base_rate": 0.15},
        {"model_key": "random_forest_calibrated", "pr_auc": 0.28, "n_samples": 1000, "base_rate": 0.15},
        {"model_key": "random_forest", "pr_auc": 0.35, "n_samples": 1000, "base_rate": 0.15},
    ]).to_csv(m_csv, index=False)

    kpis_order1 = load_executive_kpis(
        test_metrics_path=m_csv,
        summary_path=tmp_path / "none.json",
        top_k_path=tmp_path / "none.csv",
        queue_path=tmp_path / "none.csv",
        cohort_report_path=tmp_path / "none.json",
        calibration_path=tmp_path / "none.csv",
    )
    assert kpis_order1["model_pr_auc"] == 0.35
    assert kpis_order1["dummy_pr_auc"] == 0.15

    # Reorder test 2: random_forest first, then dummy, then rf_calibrated
    pd.DataFrame([
        {"model_key": "random_forest", "pr_auc": 0.35, "n_samples": 1000, "base_rate": 0.15},
        {"model_key": "dummy_baseline", "pr_auc": 0.15, "n_samples": 1000, "base_rate": 0.15},
        {"model_key": "random_forest_calibrated", "pr_auc": 0.28, "n_samples": 1000, "base_rate": 0.15},
    ]).to_csv(m_csv, index=False)

    kpis_order2 = load_executive_kpis(
        test_metrics_path=m_csv,
        summary_path=tmp_path / "none.json",
        top_k_path=tmp_path / "none.csv",
        queue_path=tmp_path / "none.csv",
        cohort_report_path=tmp_path / "none.json",
        calibration_path=tmp_path / "none.csv",
    )
    assert kpis_order2["model_pr_auc"] == 0.35
    assert kpis_order2["dummy_pr_auc"] == 0.15

    # Verify executive summary displays explicit model label
    dest = tmp_path / "executive_summary.md"
    generate_executive_summary(kpis_order2, dest)
    content = dest.read_text(encoding="utf-8")
    assert "Model PR-AUC (Random Forest)" in content
    assert "Calibrated RF" not in content


def test_top_k_helpers_keep_pooled_and_chronological_scopes_separate(tmp_path: Path):
    path = tmp_path / "top_k_evaluation.csv"
    pd.DataFrame(
        [
            {"evaluation_scope": "pooled_test_cohort", "presentation_code": "ALL_POOLED", "model_key": "random_forest", "k": 500, "precision_at_k": 0.37, "random_precision_baseline": 0.17, "is_valid_capacity": True},
            {"evaluation_scope": "pooled_test_cohort", "presentation_code": "ALL_POOLED", "model_key": "logistic_regression", "k": 500, "precision_at_k": 0.35, "random_precision_baseline": 0.17, "is_valid_capacity": True},
            {"evaluation_scope": "per_term", "presentation_code": "2014B", "model_key": "random_forest", "k": 500, "precision_at_k": 0.39, "random_precision_baseline": 0.18, "is_valid_capacity": True},
        ]
    ).to_csv(path, index=False)

    pooled = format_top_k_table(path)
    assert len(pooled) == 1
    assert pooled.iloc[0]["precision_at_k"] == pytest.approx(0.37)

    by_term_path = tmp_path / "top_k_evaluation_by_term.csv"
    pd.DataFrame(
        [
            {"evaluation_scope": "per_term", "cohort_partition": "chronological_full_presentation_2014", "model_key": "random_forest", "presentation_code": "2014B", "capacity_type": "absolute_k", "k": 500, "precision_at_k": 0.394, "n_samples": 6415, "unique_students_in_top_k": 479, "duplicate_student_slots": 21},
            {"evaluation_scope": "per_term", "cohort_partition": "chronological_full_presentation_2014", "model_key": "random_forest", "presentation_code": "2014J", "capacity_type": "absolute_k", "k": 500, "precision_at_k": 0.436, "n_samples": 9009, "unique_students_in_top_k": 481, "duplicate_student_slots": 19},
            {"evaluation_scope": "per_term", "cohort_partition": "random_test_partition_20pct", "model_key": "random_forest", "presentation_code": "2014B", "capacity_type": "absolute_k", "k": 500, "precision_at_k": 0.31, "n_samples": 1000, "unique_students_in_top_k": 400, "duplicate_student_slots": 100},
        ]
    ).to_csv(by_term_path, index=False)
    operational = load_chronological_term_top_k(by_term_path)
    assert operational["presentation_code"].tolist() == ["2014B", "2014J"]
    assert operational["precision_at_k"].tolist() == pytest.approx([0.394, 0.436])


def test_calibration_kpi_uses_selected_training_method(tmp_path: Path):
    cal_path = tmp_path / "calibration_comparison.csv"
    pd.DataFrame(
        [
            {"calibration_method": "none", "expected_calibration_error": 0.02},
            {"calibration_method": "isotonic_group_kfold_cv", "expected_calibration_error": 0.01},
            {"calibration_method": "sigmoid_group_kfold_cv", "expected_calibration_error": 0.015},
        ]
    ).to_csv(cal_path, index=False)
    diag_path = tmp_path / "calibration_diagnostics.json"
    diag_path.write_text(json.dumps({"training_calibration_selection": {"selected_method": "sigmoid"}}), encoding="utf-8")

    kpis = load_executive_kpis(
        summary_path=tmp_path / "none.json",
        test_metrics_path=tmp_path / "none.csv",
        top_k_path=tmp_path / "none.csv",
        queue_path=tmp_path / "none.csv",
        cohort_report_path=tmp_path / "none.json",
        calibration_path=cal_path,
        calibration_diagnostics_path=diag_path,
    )
    assert kpis["selected_calibration_method"] == "sigmoid"
    assert kpis["ece_calibrated"] == pytest.approx(0.015)

    dest = tmp_path / "executive_summary.md"
    generate_executive_summary(kpis, dest, queue_audit_path=tmp_path / "none.csv")
    content = dest.read_text(encoding="utf-8")
    assert "Sigmoid (Platt scaling) calibration improved empirical probability alignment" in content
    assert "Isotonic calibration improves" not in content


def test_stable_warning_counts_are_unavailable_for_undefined_metrics():
    stable_missing = pd.DataFrame({"is_statistically_stable": [True], "fpr_ratio": [float("nan")]})
    unstable_only = pd.DataFrame({"is_statistically_stable": [False], "fpr_ratio": [1.0]})
    complete = pd.DataFrame({"is_statistically_stable": [True, True], "fpr_ratio": [1.0, 1.6]})
    assert count_stable_metric_warnings(stable_missing, "fpr_ratio", upper=1.5) is None
    assert count_stable_metric_warnings(unstable_only, "fpr_ratio", upper=1.5) is None
    assert count_stable_metric_warnings(complete, "fpr_ratio", upper=1.5) == 1


def test_queue_kpis_do_not_turn_missing_keys_or_unknown_tiers_into_zero_counts(tmp_path: Path):
    queue_path = tmp_path / "student_support_queue.csv"
    pd.DataFrame(
        [
            {"student_key": "S1", "priority_tier": "Tier 1 (Urgent)"},
            {"student_key": None, "priority_tier": "Unmapped"},
        ]
    ).to_csv(queue_path, index=False)
    kpis = load_executive_kpis(
        summary_path=tmp_path / "none.json",
        test_metrics_path=tmp_path / "none.csv",
        top_k_path=tmp_path / "none.csv",
        queue_path=queue_path,
        cohort_report_path=tmp_path / "none.json",
        calibration_path=tmp_path / "none.csv",
    )
    assert kpis["queue_total_records"] == 2
    assert kpis["queue_unique_students"] is None
    assert kpis["queue_duplicate_slots"] is None
    assert kpis["tier_1_urgent_count"] is None
    assert kpis["tier_2_watch_count"] is None
    assert kpis["tier_3_standard_count"] is None


def test_subgroup_chart_does_not_plot_undefined_ratio_as_parity():
    audit = pd.DataFrame(
        {
            "protected_attribute": ["gender", "gender"],
            "subgroup": ["F", "M"],
            "disparate_impact_ratio": [float("nan"), 0.65],
            "is_statistically_stable": [True, True],
        }
    )
    fig = create_subgroup_disparity_chart(audit, attribute="gender")
    assert len(fig.axes[0].patches) == 1
    assert "F" not in [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    plt.close(fig)


def test_recommendations_do_not_invent_binary_registration_cutoff(tmp_path: Path):
    stat_path = tmp_path / "statistical_comparisons.csv"
    pd.DataFrame([{"feature": "date_registration", "p_value": 0.01}]).to_csv(stat_path, index=False)
    dest = tmp_path / "recommendations.md"
    generate_recommendations(output_path=dest, statistical_comparisons_path=stat_path)
    content = dest.read_text(encoding="utf-8")
    assert "no enrollment-timing cohort is justified" in content
    assert "registered after Day -14" not in content
    assert "Affected Target Cohort:** All students may access onboarding" in content


def test_registration_recommendation_direction_comes_from_group_medians_not_p_value(tmp_path: Path):
    stat_path = tmp_path / "statistical_comparisons.csv"
    dest = tmp_path / "recommendations.md"
    # Keep the p-value fixed while reversing the group medians: direction must follow medians,
    # and neither result may turn into risk-directed late-registration targeting.
    for retained, withdrawn, expected_direction in [
        (-52, -58, "withdrawn group registered earlier at the median"),
        (-58, -52, "withdrawn group registered later at the median"),
    ]:
        pd.DataFrame([{
            "feature": "date_registration",
            "p_value": 1e-9,
            "n_retained": 10,
            "n_withdrawn": 5,
            "retained_median": retained,
            "withdrawn_median": withdrawn,
        }]).to_csv(stat_path, index=False)
        generate_recommendations(output_path=dest, statistical_comparisons_path=stat_path)
        content = dest.read_text(encoding="utf-8")
        evidence = content.split("### Recommendation 3:", 1)[1].split("### Recommendation 4:", 1)[0]
        assert expected_direction in evidence
        assert "p-value does not determine direction" in evidence
        assert "does not justify targeting either group" in evidence
        assert "All students may access onboarding" in evidence
        assert "Affected Target Cohort:** Students with shorter registration lead times" not in evidence
        assert "Affected Target Cohort:** Students who registered late" not in evidence
