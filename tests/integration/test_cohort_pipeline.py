"""Integration tests for CohortBuilder on empirical and synthetic OULAD datasets."""
import json
from pathlib import Path

import pandas as pd
import pytest

from student_retention.paths import DATA_RAW_DIR
from student_retention.transformation.build_cohort import CohortBuilder

# Parquet persistence and tests require the pyarrow engine
pytest.importorskip("pyarrow", reason="pyarrow is required for Parquet integration tests")


@pytest.mark.skipif(
    not (DATA_RAW_DIR / "studentInfo.csv").exists(),
    reason="Empirical OULAD raw dataset not available in execution environment",
)
def test_empirical_cohort_policy_c():
    """Verify exact empirical counts for Primary Policy C against real OULAD records."""
    builder = CohortBuilder(cutoff_day=35, policy="policy_c")
    info_df, reg_df = builder.load_inputs()
    results = builder.build_cohort(info_df, reg_df)

    cohort_df = results["cohort_df"]
    exclusions_df = results["exclusions_df"]
    audit = results["audit_report"]

    reconciliation = audit["reconciliation"]
    target_dist = audit["target_distribution"]
    excl_steps = audit["exclusion_precedence_hierarchy"]

    # 1. Total records
    assert len(exclusions_df) == 5465
    assert reconciliation["raw_student_info_records"] == 32593
    assert reconciliation["raw_student_registration_records"] == 32593
    assert reconciliation["total_source_records"] == 32593
    assert reconciliation["total_excluded_records"] == 5465
    assert reconciliation["eligible_cohort_records"] == 27128
    assert reconciliation["is_reconciled"] is True

    # 2. Target distribution
    assert target_dist["positive_cases"] == 4718
    assert target_dist["negative_cases"] == 22410
    assert abs(target_dist["positive_base_rate_pct"] - 17.3916) < 0.001

    # 3. Negative class breakdown
    neg_breakdown = target_dist["negative_class_breakdown"]
    assert neg_breakdown.get("Pass") == 12355
    assert neg_breakdown.get("Distinction") == 3024
    assert neg_breakdown.get("Fail") == 7031

    # 4. Exclusion precedence step counts
    assert excl_steps["step_1"]["count"] == 5349
    assert excl_steps["step_2"]["count"] == 15
    assert excl_steps["step_3"]["count"] == 7
    assert excl_steps["step_4"]["count"] == 93
    assert excl_steps["step_5"]["count"] == 1

    # 5. Schema and grain invariants
    grain = ["code_module", "code_presentation", "id_student"]
    assert len(cohort_df) == len(cohort_df.drop_duplicates(subset=grain))
    assert cohort_df["withdrawn_after_cutoff"].isna().sum() == 0
    assert set(cohort_df["withdrawn_after_cutoff"].unique()) == {0, 1}


@pytest.mark.skipif(
    not (DATA_RAW_DIR / "studentInfo.csv").exists(),
    reason="Empirical OULAD raw dataset not available in execution environment",
)
def test_empirical_cohort_policy_b_sensitivity():
    """Verify exact empirical counts for Sensitivity Policy B and delta against Policy C."""
    builder_b = CohortBuilder(cutoff_day=35, policy="policy_b")
    info_df, reg_df = builder_b.load_inputs()
    results_b = builder_b.build_cohort(info_df, reg_df)

    reconciliation_b = results_b["audit_report"]["reconciliation"]
    target_dist_b = results_b["audit_report"]["target_distribution"]

    assert reconciliation_b["total_source_records"] == 32593
    assert reconciliation_b["total_excluded_records"] == 5457
    assert reconciliation_b["eligible_cohort_records"] == 27136
    assert target_dist_b["positive_cases"] == 4719
    assert target_dist_b["negative_cases"] == 22417
    assert abs(target_dist_b["positive_base_rate_pct"] - 17.3902) < 0.001


def test_cohort_builder_run_persists_artifacts(tmp_path):
    """Verify that CohortBuilder.run persists parquet, csv, and audit report json using synthetic inputs."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    output_dir = tmp_path / "outputs"

    # Create synthetic minimal raw input CSVs
    (input_dir / "courses.csv").write_text(
        "code_module,code_presentation,module_presentation_length\n"
        "AAA,2013J,268\n"
        "BBB,2013J,268\n",
        encoding="utf-8",
    )
    (input_dir / "studentInfo.csv").write_text(
        "code_module,code_presentation,id_student,gender,region,highest_education,imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result\n"
        "AAA,2013J,101,M,East Anglian Region,HE Qualification,90-100%,55<=,0,240,N,Pass\n"
        "BBB,2013J,102,F,Scotland,HE Qualification,20-30%,35-55,0,60,N,Withdrawn\n",
        encoding="utf-8",
    )
    (input_dir / "studentRegistration.csv").write_text(
        "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
        "AAA,2013J,101,-30,?\n"
        "BBB,2013J,102,-20,100\n",
        encoding="utf-8",
    )

    builder = CohortBuilder(cutoff_day=35, policy="policy_c", data_dir=input_dir)
    results = builder.run(save_outputs=True, output_dir=output_dir)

    saved_paths = results["saved_paths"]
    for path_str in saved_paths.values():
        assert Path(path_str).exists()

    # Validate parquet readable via pyarrow
    loaded_parquet = pd.read_parquet(saved_paths["cohort_parquet"])
    assert len(loaded_parquet) == 2

    with open(saved_paths["audit_report_json"], encoding="utf-8") as f:
        report = json.load(f)
    assert Path(saved_paths["audit_report_json"]).parent == output_dir
    assert report["policy"] == "policy_c"
    assert report["reconciliation"]["is_reconciled"] is True
    assert report["reconciliation"]["eligible_cohort_records"] == 2
