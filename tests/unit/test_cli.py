"""Unit tests for student-retention CLI entry point."""
from unittest.mock import MagicMock, patch

from student_retention.cli import build_parser, main


def test_cli_help():
    parser = build_parser()
    assert parser.prog == "student-retention"
    # main with no arguments prints help and returns 0
    ret = main([])
    assert ret == 0


def test_cli_audit_dispatch(tmp_path):
    with patch("student_retention.cli.SourceManifest") as mock_manifest_cls:
        mock_instance = MagicMock()
        mock_instance.generate.return_value = {"total_raw_rows": 100}
        mock_instance.save.return_value = tmp_path / "inventory.json"
        mock_manifest_cls.return_value = mock_instance

        ret = main(["audit", "--output", str(tmp_path / "inventory.json")])
        assert ret == 0
        mock_instance.generate.assert_called_once()
        mock_instance.save.assert_called_once_with(
            output_path=tmp_path / "inventory.json",
            manifest_data=mock_instance.generate.return_value,
        )


def test_cli_clean_and_validate_dispatch(tmp_path):
    with (
        patch("student_retention.cli.CleaningPipeline") as mock_pipeline_cls,
        patch("student_retention.cli.generate_quality_report") as mock_report_fn,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "tables": {},
            "reports": {},
        }
        mock_pipeline_cls.return_value = mock_pipeline
        mock_report_fn.return_value = {
            "status": "PASSED",
            "all_violations": [],
        }

        ret = main(["clean-and-validate", "--report-path", str(tmp_path / "report.json")])
        assert ret == 0
        mock_pipeline.run.assert_called_once()
        mock_report_fn.assert_called_once()


def test_cli_build_cohort_dispatch(tmp_path):
    with patch("student_retention.cli.CohortBuilder") as mock_builder_cls:
        mock_builder = MagicMock()
        mock_builder.run.return_value = {
            "cohort": MagicMock(),
            "exclusions": MagicMock(),
            "audit_report": {
                "reconciliation": {
                    "total_source_records": 32593,
                    "total_excluded_records": 5465,
                    "pct_excluded": 16.77,
                    "eligible_cohort_records": 27128,
                    "pct_eligible": 83.23,
                },
                "target_distribution": {
                    "positive_cases": 4718,
                    "positive_base_rate_pct": 17.39,
                    "negative_cases": 22410,
                },
            },
            "saved_paths": {
                "cohort_parquet": str(tmp_path / "cohort.parquet"),
            },
        }
        mock_builder_cls.return_value = mock_builder

        ret = main([
            "build-cohort",
            "--policy", "policy_c",
            "--cutoff", "35",
            "--output-dir", str(tmp_path),
        ])
        assert ret == 0
        mock_builder_cls.assert_called_once()
        mock_builder.run.assert_called_once_with(
            save_outputs=True,
            output_dir=tmp_path,
        )


def test_cli_audit_fairness_dispatch(tmp_path):
    with (
        patch("student_retention.cli.run_audit_fairness") as mock_audit_fn,
    ):
        mock_audit_fn.return_value = 0
        ret = main(["audit-fairness", "--output-dir", str(tmp_path)])
        assert ret == 0
        mock_audit_fn.assert_called_once()


def test_cli_generate_support_queue_dispatch(tmp_path):
    with (
        patch("student_retention.cli.run_generate_support_queue") as mock_queue_fn,
    ):
        mock_queue_fn.return_value = 0
        ret = main(["generate-support-queue", "--output-path", str(tmp_path / "queue.csv")])
        assert ret == 0
        mock_queue_fn.assert_called_once()
