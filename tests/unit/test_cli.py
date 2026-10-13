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
