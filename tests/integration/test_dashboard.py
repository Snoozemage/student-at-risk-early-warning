"""Integration tests for Streamlit dashboard and pipeline execution."""

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.components import chart_rendering
from student_retention.paths import PROJECT_ROOT
from student_retention.pipeline.run_pipeline import run_pipeline

APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"


@pytest.fixture
def clean_outputs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Simulate a clean checkout environment without any generated outputs."""
    clean_dir = tmp_path / "clean_outputs"
    clean_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("STUDENT_RETENTION_OUTPUTS_DIR", str(clean_dir))
    return clean_dir


@pytest.fixture
def populated_outputs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated temporary outputs directory populated with minimal valid synthetic data."""
    pop_dir = tmp_path / "pop_outputs"
    t_dir = pop_dir / "tables"
    p_dir = pop_dir / "predictions"
    q_dir = pop_dir / "quality_reports"
    f_dir = pop_dir / "figures"
    d_dir = pop_dir / "deliverables"
    t_dir.mkdir(parents=True, exist_ok=True)
    p_dir.mkdir(parents=True, exist_ok=True)
    q_dir.mkdir(parents=True, exist_ok=True)
    f_dir.mkdir(parents=True, exist_ok=True)
    d_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("STUDENT_RETENTION_OUTPUTS_DIR", str(pop_dir))
    monkeypatch.setenv("STUDENT_RETENTION_DELIVERABLES_DIR", str(d_dir))

    # 1. Synthetic model_training_summary.json
    with open(t_dir / "model_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_total": 5000,
                "n_train": 4000,
                "n_test": 1000,
                "features_used": ["feature_a", "feature_b"],
                "target": "withdrawn_after_cutoff",
            },
            f,
        )

    # 2. Synthetic model_comparison_test.csv
    pd.DataFrame(
        [
            {
                "model_key": "dummy_baseline",
                "pr_auc": 0.1800,
                "roc_auc": 0.5000,
                "precision": 0.1800,
                "recall": 1.0000,
                "f1": 0.3051,
                "brier_score": 0.1476,
                "n_samples": 1000,
                "base_rate": 0.1800,
                "decision_threshold": 0.01,
            },
            {
                "model_key": "random_forest",
                "pr_auc": 0.3500,
                "roc_auc": 0.7100,
                "precision": 0.3400,
                "recall": 0.4000,
                "f1": 0.3676,
                "brier_score": 0.1300,
                "n_samples": 1000,
                "base_rate": 0.1800,
                "decision_threshold": 0.25,
            },
        ]
    ).to_csv(t_dir / "model_comparison_test.csv", index=False)

    # 3. Synthetic top_k_evaluation.csv
    pd.DataFrame(
        [
            {
                "model_key": "random_forest",
                "k": 500,
                "precision_at_k": 0.4000,
                "recall_at_k": 0.2500,
                "random_precision_baseline": 0.1800,
                "is_valid_capacity": True,
            }
        ]
    ).to_csv(t_dir / "top_k_evaluation.csv", index=False)

    # 4. Synthetic calibration_comparison.csv
    pd.DataFrame(
        [
            {
                "model_configuration": "Random Forest (Raw Probabilities)",
                "calibration_method": "none",
                "brier_score": 0.1300,
                "expected_calibration_error": 0.0120,
                "maximum_calibration_error": 0.2500,
                "min_probability": 0.02,
                "max_probability": 0.55,
                "mean_probability": 0.18,
            },
            {
                "model_configuration": "Random Forest (Isotonic Calibrated)",
                "calibration_method": "isotonic_group_kfold_cv",
                "brier_score": 0.1290,
                "expected_calibration_error": 0.0080,
                "maximum_calibration_error": 0.2000,
                "min_probability": 0.0,
                "max_probability": 0.80,
                "mean_probability": 0.18,
            },
        ]
    ).to_csv(t_dir / "calibration_comparison.csv", index=False)

    # 5. Synthetic support_queue_fairness_audit.csv
    pd.DataFrame(
        [
            {
                "protected_attribute": "gender",
                "subgroup": "F",
                "sample_size": 200,
                "base_rate": 0.18,
                "selection_rate": 0.25,
                "recall": 0.35,
                "precision": 0.38,
                "disparate_impact_ratio": 0.88,
                "fpr_ratio": 1.05,
                "is_statistically_stable": True,
            },
            {
                "protected_attribute": "gender",
                "subgroup": "M",
                "sample_size": 250,
                "base_rate": 0.18,
                "selection_rate": 0.28,
                "recall": 0.40,
                "precision": 0.39,
                "disparate_impact_ratio": 1.00,
                "fpr_ratio": 1.00,
                "is_statistically_stable": True,
            },
        ]
    ).to_csv(t_dir / "support_queue_fairness_audit.csv", index=False)

    # 6. Minimal valid student support queue dataset
    pd.DataFrame(
        [
            {
                "student_key": f"STU_TEST_{i}",
                "code_module": "AAA" if i < 3 else "BBB",
                "code_presentation": "2013J" if i < 3 else "2014B",
                "risk_probability": 0.78 if i == 0 else (0.55 if i == 1 else 0.25),
                "priority_tier": "Tier 1 (Urgent)" if i == 0 else ("Tier 2 (Watch)" if i == 1 else "Tier 3 (Standard)"),
                "support_reason_summary": "Low VLE activity in Week 4" if i == 0 else "Routine pacing",
                "recommended_next_step": "Direct 1-on-1 advisor outreach within 3 days" if i == 0 else "Automated check-in",
                "review_status": "Pending",
            }
            for i in range(5)
        ]
    ).to_csv(p_dir / "student_support_queue.csv", index=False)

    # 7. Cohort report
    with open(q_dir / "cohort_policy_c_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "reconciliation": {"eligible_cohort_records": 5000},
                "target_distribution": {"positive_base_rate_pct": 18.0},
            },
            f,
        )

    return pop_dir


def test_dashboard_clean_environment_missing_inputs(clean_outputs_dir: Path):
    """Verify that in a clean checkout without outputs, all pages render cleanly without unhandled exceptions."""
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    assert not app.exception
    assert "Operational Overview" in app.title[0].value
    assert len(app.metric) >= 5

    # Switch to Student Support Queue
    app.sidebar.radio[0].set_value("Student Support Queue").run(timeout=30)
    assert not app.exception
    assert "Student Support & Advising Triage Queue" in app.title[0].value
    # In clean environment, queue CSV does not exist -> informative message shown, 0 dataframes
    assert any("not available" in info.value.lower() for info in app.info)
    assert len(app.dataframe) == 0

    # Switch to Engagement Patterns
    app.sidebar.radio[0].set_value("Engagement Patterns").run(timeout=30)
    assert not app.exception

    # Switch to Model Performance
    app.sidebar.radio[0].set_value("Model Performance").run(timeout=30)
    assert not app.exception
    unavailable_messages = [info.value for info in app.info]
    assert any("PR curves figure not available" in message for message in unavailable_messages)
    assert any("ROC curves figure not available" in message for message in unavailable_messages)
    assert any("Calibration reliability curves figure not available" in message for message in unavailable_messages)
    assert any("Top-K visualization data not available" in message for message in unavailable_messages)

    # Switch to Fairness and Limitations
    app.sidebar.radio[0].set_value("Fairness and Limitations").run(timeout=30)
    assert not app.exception


def test_dashboard_populated_environment_queue_page(populated_outputs_dir: Path):
    """Verify that when queue data is present, Page 3 renders the table and widgets without crashing."""
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    assert not app.exception

    # Switch to Student Support Queue
    app.sidebar.radio[0].set_value("Student Support Queue").run(timeout=30)
    assert not app.exception
    assert "Student Support & Advising Triage Queue" in app.title[0].value

    # Dataframe must be rendered
    assert len(app.dataframe) >= 1

    # Student inspector selectbox must be present
    assert len(app.selectbox) >= 1
    rendered = "\n".join(element.value for element in [*app.markdown, *app.info])
    assert "Example Support Option (for advisor review)" in rendered
    assert "Demonstration response setting (not validated)" in rendered
    assert "Response Timeline (SLA)" not in rendered

    # Filters change the displayed cohort and the case inspector follows selection.
    app.sidebar.multiselect[1].set_value(["AAA"]).run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == "3"
    app.selectbox[0].set_value("STU_TEST_1").run(timeout=30)
    assert not app.exception
    assert any("STU_TEST_1" in item.value for item in app.markdown)

    # The status action is explicitly a simulation and reports the selected value.
    app.selectbox[1].set_value("Under Review").run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any("Status for student `STU_TEST_1` updated to: **Under Review**" in item.value for item in app.success)


def test_dashboard_populated_environment_overview(populated_outputs_dir: Path):
    """Verify that Page 1 renders metric cards and tier distribution from the populated fixture."""
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    assert not app.exception
    assert "Operational Overview" in app.title[0].value
    assert len(app.metric) >= 5

    # Metric should reflect the fixture's cohort size (5,000)
    metric_values = [m.value for m in app.metric]
    assert any("5,000" in str(v) for v in metric_values)
    rendered = "\n".join(element.value for element in app.markdown)
    assert "Proposed Human-Reviewed Workflow (Demonstration)" in rendered
    assert "not locally validated thresholds or service levels" in rendered
    assert "intervention effectiveness" in rendered


def test_dashboard_populated_environment_model_performance(populated_outputs_dir: Path):
    """Verify that Page 4 renders model comparison table and calibration metrics from fixture."""
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    app.sidebar.radio[0].set_value("Model Performance").run(timeout=30)
    assert not app.exception
    assert "Model Evaluation, Probability Calibration & Error Diagnostics" in app.title[0].value
    assert len(app.dataframe) >= 1
    rendered = "\n".join(element.value for element in app.markdown)
    assert "Pooled Top-K metrics below use the uncalibrated Random Forest ranking" in rendered
    assert "Chronological 2014 evaluation" in rendered
    assert "Probability Calibration (Isotonic Regression)" in rendered
    assert "enabling students with extreme disengagement to reach" not in rendered


def test_dashboard_embeds_available_curve_assets(populated_outputs_dir: Path, monkeypatch):
    """Verify model curve assets are rendered from their bytes without media URLs."""
    figures_dir = populated_outputs_dir / "figures"
    for filename in ("pr_curves.png", "roc_curves.png"):
        Image.new("RGB", (2, 2), color="white").save(figures_dir / filename)
    rendered_html = []
    monkeypatch.setattr(chart_rendering.st, "html", rendered_html.append)

    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    app.sidebar.radio[0].set_value("Model Performance").run(timeout=30)

    assert not app.exception
    assert len(rendered_html) >= 2
    assert all('src="data:image/png;base64,' in markup for markup in rendered_html)
    assert all("/media/" not in markup for markup in rendered_html)
    assert any('alt="Precision-recall curves' in markup for markup in rendered_html)
    assert any('alt="ROC curves' in markup for markup in rendered_html)


def test_dashboard_engagement_avoids_unvalidated_registration_cutoff(populated_outputs_dir: Path):
    pd.DataFrame(
        [{"feature": "date_registration", "p_value": 0.01, "retained_median": -52, "withdrawn_median": -58}]
    ).to_csv(populated_outputs_dir / "tables" / "statistical_comparisons.csv", index=False)
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    app.sidebar.radio[0].set_value("Engagement Patterns").run(timeout=30)
    assert not app.exception
    rendered = "\n".join(element.value for element in app.markdown)
    assert "continuous retrospective comparison" in rendered
    assert "retained **-52.0** days vs withdrawn **-58.0** days" in rendered
    assert "Completing students register earlier" not in rendered
    assert "within 14 days of launch" not in rendered


def test_dashboard_populated_environment_fairness(populated_outputs_dir: Path):
    """Verify that Page 5 renders demographic fairness table from fixture."""
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)
    app.sidebar.radio[0].set_value("Fairness and Limitations").run(timeout=30)
    assert not app.exception
    assert "Subgroup Fairness Auditing & Ethical Limitations" in app.title[0].value
    assert len(app.dataframe) >= 1
    rendered = "\n".join(element.value for element in app.markdown)
    assert "Held-Out Demonstration Queue Subgroup Audit" in rendered
    assert "does not show that a student received support" in rendered
    assert "Actual Triage Experience" not in rendered


def test_empirical_queue_file_integrity():
    """Verify that empirical student_support_queue.csv is intact and has 2,000 records."""
    real_queue_path = PROJECT_ROOT / "outputs" / "predictions" / "student_support_queue.csv"
    if real_queue_path.exists():
        df_real = pd.read_csv(real_queue_path)
        assert len(df_real) == 2000, f"Empirical queue truncated: observed {len(df_real)} rows instead of 2,000"
        assert "student_key" in df_real.columns
        assert "risk_probability" in df_real.columns
        assert "priority_tier" in df_real.columns


def test_pipeline_runner_generate_deliverables_populated(populated_outputs_dir: Path):
    """Verify master pipeline runner generates deliverables cleanly from valid result data."""
    success = run_pipeline(["generate_deliverables"])
    assert success is True


def test_pipeline_runner_generate_deliverables_clean_fails_cleanly(clean_outputs_dir: Path):
    """Verify master pipeline runner fails cleanly without crashing when empirical inputs are missing."""
    success = run_pipeline(["generate_deliverables"])
    assert success is False


def test_pipeline_runner_invalid_stage():
    """Verify master pipeline runner reports failure on unknown stage without crashing."""
    success = run_pipeline(["non_existent_stage_name"])
    assert success is False
