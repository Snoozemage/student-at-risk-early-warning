"""Integration tests for end-to-end cleaning pipeline and quality report generation."""
import json

import pandas as pd

from student_retention.cleaning.cleaning_pipeline import CleaningPipeline
from student_retention.validation.report_generator import generate_quality_report


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
