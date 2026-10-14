"""Unit tests for Phase 7 subgroup fairness auditing and demographic sensitivity analysis."""

import numpy as np
import pandas as pd
import pytest

from student_retention.fairness.subgroup_audit import (
    audit_all_protected_attributes,
    audit_operational_support_queue,
    audit_subgroup_metrics,
)


@pytest.fixture
def synthetic_fairness_df() -> pd.DataFrame:
    """Construct deterministic synthetic cohort for fairness metric testing."""
    np.random.seed(42)
    n = 200

    # Group A: 120 samples, 20% positive base rate, 25% selection rate
    # Group B: 70 samples, 40% positive base rate, 50% selection rate
    # Group C: 10 samples (small sample < 50), 30% base rate
    g_a = ["Group_A"] * 120
    g_b = ["Group_B"] * 70
    g_c = ["Group_C"] * 10
    groups = g_a + g_b + g_c

    # Labels
    y_true = np.zeros(n, dtype=int)
    y_true[0:24] = 1  # 24/120 = 20%
    y_true[120:148] = 1  # 28/70 = 40%
    y_true[190:193] = 1  # 3/10 = 30%

    # Predictions
    y_pred = np.zeros(n, dtype=int)
    y_pred[0:30] = 1  # 30/120 = 25%
    y_pred[120:155] = 1  # 35/70 = 50%
    y_pred[190:192] = 1  # 2/10 = 20%

    y_prob = np.where(y_pred == 1, 0.65, 0.15)

    df = pd.DataFrame(
        {
            "id_student": [1000 + i for i in range(n)],
            "code_module": ["AAA"] * n,
            "code_presentation": ["2014B"] * n,
            "demographic_test_group": groups,
            "withdrawn_after_cutoff": y_true,
            "predicted_label": y_pred,
            "risk_probability": y_prob,
        }
    )
    return df


def test_audit_subgroup_metrics_calculation(synthetic_fairness_df: pd.DataFrame):
    """Test exact mathematical correctness of rates, counts, and ratios."""
    audit_df = audit_subgroup_metrics(
        df=synthetic_fairness_df,
        attribute="demographic_test_group",
        reference_group="Group_A",
        target_col="withdrawn_after_cutoff",
        pred_col="predicted_label",
        prob_col="risk_probability",
        min_sample_size=50,
    )

    assert len(audit_df) == 3
    assert set(audit_df["subgroup"]) == {"Group_A", "Group_B", "Group_C"}

    # Inspect Group A (Reference group)
    row_a = audit_df[audit_df["subgroup"] == "Group_A"].iloc[0]
    assert row_a["sample_size"] == 120
    assert bool(row_a["is_reference_group"]) is True
    assert bool(row_a["is_statistically_stable"]) is True
    assert np.isclose(row_a["base_rate"], 24 / 120)
    assert np.isclose(row_a["selection_rate"], 30 / 120)
    assert np.isclose(row_a["disparate_impact_ratio"], 1.0)
    assert np.isclose(row_a["fpr_ratio"], 1.0)

    # Inspect Group B
    row_b = audit_df[audit_df["subgroup"] == "Group_B"].iloc[0]
    assert row_b["sample_size"] == 70
    assert bool(row_b["is_reference_group"]) is False
    assert bool(row_b["is_statistically_stable"]) is True
    assert np.isclose(row_b["selection_rate"], 35 / 70)  # 0.50
    # Disparate impact ratio = 0.50 / 0.25 = 2.0
    assert np.isclose(row_b["disparate_impact_ratio"], 2.0)
    assert "disparate_impact_warning" in row_b["fairness_warning_flag"]

    # Inspect Group C (N < 50)
    row_c = audit_df[audit_df["subgroup"] == "Group_C"].iloc[0]
    assert row_c["sample_size"] == 10
    assert bool(row_c["is_statistically_stable"]) is False
    assert "small_sample_warning_N_lt_50" in row_c["fairness_warning_flag"]


def test_audit_subgroup_metrics_defaults_reference_to_largest(synthetic_fairness_df: pd.DataFrame):
    """Verify that reference_group=None defaults cleanly to the largest subgroup."""
    audit_df = audit_subgroup_metrics(
        df=synthetic_fairness_df,
        attribute="demographic_test_group",
        reference_group=None,  # Should pick Group_A (N=120)
        target_col="withdrawn_after_cutoff",
        pred_col="predicted_label",
    )
    row_a = audit_df[audit_df["subgroup"] == "Group_A"].iloc[0]
    assert bool(row_a["is_reference_group"]) is True


def test_audit_all_protected_attributes_multi(synthetic_fairness_df: pd.DataFrame):
    """Test multi-attribute audit concatenation and error handling."""
    # Add a second demographic column
    synthetic_fairness_df["gender"] = ["M", "F"] * 100

    cfg = {
        "protected_attributes": {
            "demographic_test_group": {
                "field": "demographic_test_group",
                "reference_group": "Group_A",
            },
            "gender": {
                "field": "gender",
                "reference_group": "M",
            },
        },
        "subgroup_reporting_rules": {
            "min_subgroup_sample_size": 50,
            "tolerance_thresholds": {
                "disparate_impact_warning_band": [0.80, 1.25],
                "fpr_ratio_warning_threshold": 1.50,
            },
        },
    }

    audit_all = audit_all_protected_attributes(
        df=synthetic_fairness_df,
        fairness_config=cfg,
        target_col="withdrawn_after_cutoff",
        pred_col="predicted_label",
    )

    # 3 groups from demographic_test_group + 2 groups from gender = 5 rows
    assert len(audit_all) == 5
    assert set(audit_all["protected_attribute"]) == {"demographic_test_group", "gender"}


def test_audit_subgroup_metrics_missing_column_raises():
    """Verify that audit raises ValueError when required columns are absent."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Protected attribute column 'gender' not found"):
        audit_subgroup_metrics(df, attribute="gender")


def test_fairness_report_dynamic_enumerates_both_di_and_fpr_warnings(tmp_path):
    """Verify that generate_subgroup_fairness_report explicitly lists both DI and FPR ratio flags."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_fairness_module",
        "scripts/07_run_fairness_and_decision_support.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Synthetic audit dataframe with both DI and FPR flags
    audit_data = pd.DataFrame(
        [
            {
                "protected_attribute": "disability",
                "subgroup": "Y",
                "is_reference_group": False,
                "sample_size": 200,
                "base_rate": 0.35,
                "selection_rate": 0.40,
                "disparate_impact_ratio": 1.60,
                "fpr_ratio": 1.75,
                "false_positive_rate": 0.25,
                "is_statistically_stable": True,
                "fairness_warning_flag": "disparate_impact_warning; elevated_fpr_ratio_warning",
            },
            {
                "protected_attribute": "disability",
                "subgroup": "N",
                "is_reference_group": True,
                "sample_size": 800,
                "base_rate": 0.20,
                "selection_rate": 0.25,
                "disparate_impact_ratio": 1.00,
                "fpr_ratio": 1.00,
                "false_positive_rate": 0.14,
                "is_statistically_stable": True,
                "fairness_warning_flag": "within_tolerance",
            },
            {
                "protected_attribute": "gender",
                "subgroup": "F",
                "is_reference_group": False,
                "sample_size": 500,
                "base_rate": 0.15,
                "selection_rate": 0.15,
                "disparate_impact_ratio": 0.60,
                "fpr_ratio": 0.90,
                "false_positive_rate": 0.10,
                "is_statistically_stable": True,
                "fairness_warning_flag": "disparate_impact_warning",
            },
            {
                "protected_attribute": "gender",
                "subgroup": "M",
                "is_reference_group": True,
                "sample_size": 500,
                "base_rate": 0.25,
                "selection_rate": 0.25,
                "disparate_impact_ratio": 1.00,
                "fpr_ratio": 1.00,
                "false_positive_rate": 0.15,
                "is_statistically_stable": True,
                "fairness_warning_flag": "within_tolerance",
            },
        ]
    )

    sensitivity_mock = {
        "model_comparison": pd.DataFrame(
            [
                {
                    "model_configuration": "Primary Baseline",
                    "feature_count": 22,
                    "pr_auc": 0.38,
                    "roc_auc": 0.68,
                    "brier_score": 0.13,
                    "precision": 0.35,
                    "recall": 0.42,
                    "f1": 0.38,
                },
                {
                    "model_configuration": "Demographic-Aware",
                    "feature_count": 28,
                    "pr_auc": 0.38,
                    "roc_auc": 0.68,
                    "brier_score": 0.13,
                    "precision": 0.35,
                    "recall": 0.42,
                    "f1": 0.38,
                },
            ]
        ),
        "summary_findings": {
            "policy_recommendation": "Exclusion of demographic features upheld.",
        },
    }

    report_path = tmp_path / "fairness_report_test.md"
    mod.generate_subgroup_fairness_report(
        audit_df=audit_data,
        sensitivity_results=sensitivity_mock,
        operational_threshold=0.25,
        threshold_provenance="5-fold GroupKFold CV on 80% train split",
        output_path=report_path,
    )

    assert report_path.exists()
    report_content = report_path.read_text(encoding="utf-8")

    # Invariants:
    # 1. Mentions disparate impact warning section and lists flags
    assert "Disparate Impact Ratio Flags (Outside [0.80, 1.25])" in report_content
    assert "disability = Y" in report_content
    assert "gender = F" in report_content

    # 2. Mentions elevated FPR ratio flags and lists disability=Y
    assert "Elevated False Positive Rate Ratio Flags (> 1.50)" in report_content
    assert "FPR Ratio = 1.75" in report_content

    # 3. Contains explicit threshold provenance
    assert "Fixed Decision Threshold $\\theta = 0.25$" in report_content
    assert "5-fold GroupKFold CV on 80% train split" in report_content


def test_audit_operational_support_queue_basic():
    """Verify operational queue audit correctly measures queue inclusion rates and disparity ratios."""
    n = 200
    cohort_df = pd.DataFrame(
        {
            "id_student": [1000 + i for i in range(n)],
            "code_module": ["BBB"] * n,
            "code_presentation": ["2014J"] * n,
            "gender": ["M"] * 100 + ["F"] * 100,
            "withdrawn_after_cutoff": [1] * 20 + [0] * 80 + [1] * 25 + [0] * 75,
            "calibrated_risk_probability": np.random.uniform(0.1, 0.8, n),
        }
    )

    # Queue contains 40 M students and 20 F students
    queue_df = pd.DataFrame(
        {
            "student_key": [str(1000 + i) for i in range(40)] + [str(1100 + i) for i in range(20)],
            "code_module": ["BBB"] * 60,
            "code_presentation": ["2014J"] * 60,
            "priority_tier": ["Tier 1 (Urgent)"] * 60,
        }
    )

    cfg = {
        "protected_attributes": {
            "gender": {"field": "gender", "reference_group": "M"},
        },
        "subgroup_reporting_rules": {
            "min_subgroup_sample_size": 50,
            "tolerance_thresholds": {
                "disparate_impact_warning_band": [0.80, 1.25],
                "fpr_ratio_warning_threshold": 1.50,
            },
        },
    }

    queue_audit = audit_operational_support_queue(
        queue_df=queue_df,
        cohort_df=cohort_df,
        fairness_config=cfg,
        target_col="withdrawn_after_cutoff",
    )

    assert len(queue_audit) == 2
    assert (queue_audit["audit_scope"] == "operational_support_queue").all()
    assert (queue_audit["decision_policy"] == "top_500_per_presentation_calibrated").all()

    # M group (Reference): 40 / 100 = 40% selection rate
    row_m = queue_audit[queue_audit["subgroup"] == "M"].iloc[0]
    assert row_m["sample_size"] == 100
    assert row_m["selected_count"] == 40
    assert np.isclose(row_m["selection_rate"], 0.40)
    assert np.isclose(row_m["disparate_impact_ratio"], 1.0)

    # F group: 20 / 100 = 20% selection rate -> DIR = 0.20 / 0.40 = 0.50
    row_f = queue_audit[queue_audit["subgroup"] == "F"].iloc[0]
    assert row_f["sample_size"] == 100
    assert row_f["selected_count"] == 20
    assert np.isclose(row_f["selection_rate"], 0.20)
    assert np.isclose(row_f["disparate_impact_ratio"], 0.50)
    assert "disparate_impact_warning" in row_f["fairness_warning_flag"]

    # Verify custom policy metadata provenance
    custom_audit = audit_operational_support_queue(
        queue_df=queue_df,
        cohort_df=cohort_df,
        fairness_config=cfg,
        target_col="withdrawn_after_cutoff",
        decision_policy="custom_test_policy_k50",
    )
    assert (custom_audit["decision_policy"] == "custom_test_policy_k50").all()


def test_audit_operational_support_queue_missing_columns_raise():
    """Verify that audit_operational_support_queue enforces required column presence."""
    q_df = pd.DataFrame({"code_module": ["A"], "code_presentation": ["2014B"]})
    c_df = pd.DataFrame({"id_student": [1], "code_module": ["A"], "code_presentation": ["2014B"], "withdrawn_after_cutoff": [0]})

    with pytest.raises(ValueError, match="Required column 'student_key' missing from queue DataFrame"):
        audit_operational_support_queue(q_df, c_df)
