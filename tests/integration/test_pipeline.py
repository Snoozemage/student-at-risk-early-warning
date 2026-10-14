"""Integration tests for end-to-end cleaning pipeline and quality report generation."""
import importlib
import json

import pandas as pd

from student_retention.cleaning.cleaning_pipeline import CleaningPipeline
from student_retention.validation.report_generator import generate_quality_report


def test_audit_sources_stage_uses_manifest_writer(tmp_path, monkeypatch):
    """The pipeline passes its target path to the manifest helper that persists JSON."""
    pipeline_module = importlib.import_module("student_retention.pipeline.run_pipeline")
    manifest_module = importlib.import_module("student_retention.ingestion.manifest")
    output_dir = tmp_path / "quality_reports"
    monkeypatch.setattr("student_retention.paths.QUALITY_REPORTS_DIR", output_dir)
    calls = []

    def write_manifest(output_path):
        calls.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"tables": {}}', encoding="utf-8")
        return {"tables": {}}

    monkeypatch.setattr(manifest_module, "generate_source_manifest", write_manifest)

    assert pipeline_module.run_stage_audit_sources() is True
    assert calls == [output_dir / "source_inventory.json"]
    assert json.loads(calls[0].read_text(encoding="utf-8")) == {"tables": {}}


def test_clean_and_validate_stage_uses_cleaning_and_quality_apis(monkeypatch):
    """Stage 2 passes normalized tables and audits to the canonical quality validator."""
    pipeline_module = importlib.import_module("student_retention.pipeline.run_pipeline")
    cleaning_module = importlib.import_module("student_retention.cleaning.cleaning_pipeline")
    quality_module = importlib.import_module("student_retention.validation.report_generator")
    tables = {"courses": pd.DataFrame({"code_module": ["AAA"]})}
    reports = {
        "courses": {
            "missing_report": {"is_valid": True},
            "identifier_report": {"is_valid": True},
        }
    }
    cleaning_calls = []
    quality_calls = []

    def clean():
        cleaning_calls.append(True)
        return {"tables": tables, "reports": reports}

    def validate(**kwargs):
        quality_calls.append(kwargs)
        return {"status": "PASSED"}

    monkeypatch.setattr(cleaning_module, "run_cleaning_pipeline", clean)
    monkeypatch.setattr(quality_module, "generate_quality_report", validate)

    assert pipeline_module.run_stage_clean_and_validate() is True
    assert cleaning_calls == [True]
    assert quality_calls == [
        {
            "tables": tables,
            "missing_reports": {"courses": {"is_valid": True}},
            "identifier_reports": {"courses": {"is_valid": True}},
        }
    ]


def test_train_models_stage_reads_feature_matrix_and_calls_trainer(tmp_path, monkeypatch):
    """Stage 6 uses the canonical trainer on the generated Policy C feature matrix."""
    pipeline_module = importlib.import_module("student_retention.pipeline.run_pipeline")
    training_module = importlib.import_module("student_retention.modeling.train")
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    feature_path = features_dir / "feature_matrix_policy_c.parquet"
    expected = pd.DataFrame({"marker": [1]})
    expected.to_parquet(feature_path, index=False)
    monkeypatch.setattr("student_retention.paths.PROCESSED_FEATURES_DIR", features_dir)
    calls = []

    def train(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(training_module, "train_and_evaluate_all_models", train)

    assert pipeline_module.run_stage_train_models() is True
    pd.testing.assert_frame_equal(calls[0]["feature_df"], expected)
    assert calls[0]["save_artifacts"] is True
    assert calls[0]["random_state"] == 42


def test_eda_stage_uses_supported_runner_arguments(monkeypatch):
    """Stage 4 constructs the current EDA runner with its supported Day-35 argument."""
    pipeline_module = importlib.import_module("student_retention.pipeline.run_pipeline")
    eda_module = importlib.import_module("student_retention.eda.eda_runner")
    calls = []

    class FakeRunner:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def run(self):
            calls.append(("run",))

    monkeypatch.setattr(eda_module, "EDARunner", FakeRunner)

    assert pipeline_module.run_stage_run_eda() is True
    assert calls == [("init", {"cutoff_day": 35}), ("run",)]


def test_feature_stage_uses_supported_builder_api(monkeypatch):
    """Stage 5 calls FeatureTableBuilder with the canonical Policy C setting."""
    pipeline_module = importlib.import_module("student_retention.pipeline.run_pipeline")
    feature_module = importlib.import_module("student_retention.transformation.build_feature_table")
    calls = []

    class FakeBuilder:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def run_pipeline(self, **kwargs):
            calls.append(("run_pipeline", kwargs))

    monkeypatch.setattr(feature_module, "FeatureTableBuilder", FakeBuilder)

    assert pipeline_module.run_stage_build_features() is True
    assert calls == [
        ("init", {"cutoff_day": 35}),
        ("run_pipeline", {"policy_version": "Policy C"}),
    ]


def test_pipeline_end_to_end_parquet(sample_tables, tmp_path):
    interim_dir = tmp_path / "normalized"
    pipeline = CleaningPipeline(output_dir=interim_dir)

    results = pipeline.run(tables=sample_tables, save_interim=True, save_format="parquet")

    assert results["status"] == "success"
    assert len(results["tables"]) == 7

    # Check parquet files on disk
    for table_name in sample_tables:
        expected_file = interim_dir / f"{table_name}.parquet"
        assert expected_file.exists(), f"File {expected_file} was not written"
        df_read = pd.read_parquet(expected_file)
        assert len(df_read) == len(sample_tables[table_name])

    # Generate quality report
    report_file = tmp_path / "reports" / "data_quality_report.json"
    report = generate_quality_report(
        tables=results["tables"],
        missing_reports={t: r["missing_report"] for t, r in results["reports"].items()},
        identifier_reports={t: r["identifier_report"] for t, r in results["reports"].items()},
        output_path=report_file,
    )

    assert report["status"] == "PASSED"
    assert report_file.exists()

    with open(report_file, encoding="utf-8") as f:
        loaded_report = json.load(f)
    assert loaded_report["status"] == "PASSED"
    assert loaded_report["summary"]["tables_audited"] == 7
    assert loaded_report["summary"]["total_violations"] == 0


def test_pipeline_csv_format(sample_tables, tmp_path):
    interim_dir = tmp_path / "normalized_csv"
    pipeline = CleaningPipeline(output_dir=interim_dir)

    results = pipeline.run(tables=sample_tables, save_interim=True, save_format="csv")

    assert results["status"] == "success"
    for table_name in sample_tables:
        csv_file = interim_dir / f"{table_name}.csv"
        assert csv_file.exists()
        df_read = pd.read_csv(csv_file)
        assert len(df_read) == len(sample_tables[table_name])
