import json
from unittest.mock import patch

import pytest

from student_retention.ingestion.manifest import (
    SourceManifest,
    compute_sha256,
    generate_source_manifest,
)
from student_retention.ingestion.readers import (
    OULADReader,
    SourceNotFoundError,
    SourceRowMismatchError,
)


def test_reader_init(mock_raw_data_dir):
    reader = OULADReader(raw_data_dir=mock_raw_data_dir)
    assert reader.raw_data_dir == mock_raw_data_dir
    assert reader.missing_marker == "?"


def test_reader_get_table_path(mock_raw_data_dir):
    reader = OULADReader(raw_data_dir=mock_raw_data_dir)
    path = reader.get_table_path("courses")
    assert path.name == "courses.csv"
    assert path.exists()

    with pytest.raises(KeyError, match="Unknown table"):
        reader.get_table_path("non_existent_table")


def test_reader_read_table_success(mock_raw_data_dir):
    reader = OULADReader(raw_data_dir=mock_raw_data_dir)
    df = reader.read_table("courses", validate_counts=False)
    assert len(df) == 2
    assert "code_module" in df.columns
    assert "code_presentation" in df.columns


def test_reader_missing_file(tmp_path):
    reader = OULADReader(raw_data_dir=tmp_path)
    with pytest.raises(SourceNotFoundError, match="Source file not found"):
        reader.read_table("courses")


def test_reader_strict_row_mismatch(mock_raw_data_dir):
    reader = OULADReader(raw_data_dir=mock_raw_data_dir)
    # Sample courses has 2 rows, but sources.yaml expects 22
    with pytest.raises(SourceRowMismatchError, match="row count mismatch"):
        reader.read_table("courses", validate_counts=True, strict_counts=True)


def test_compute_sha256(tmp_path):
    test_file = tmp_path / "hello.txt"
    test_file.write_text("Hello World", encoding="utf-8")
    hash_val = compute_sha256(test_file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64


def test_source_manifest_generate_and_save(mock_raw_data_dir, tmp_path):
    manifest_gen = SourceManifest(raw_data_dir=mock_raw_data_dir)
    manifest = manifest_gen.generate()

    assert manifest["dataset_name"] == "Open University Learning Analytics Dataset (OULAD)"
    assert "courses" in manifest["tables"]
    assert manifest["tables"]["courses"]["row_count"] == 2

    out_file = tmp_path / "manifest.json"
    manifest_gen.save(out_file)
    assert out_file.exists()

    with open(out_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["tables"]["courses"]["row_count"] == 2


def test_source_manifest_single_pass_no_redundant_scans(mock_raw_data_dir, tmp_path):
    """Verify that SourceManifest.save does not redundantly re-run generate."""
    manifest_gen = SourceManifest(raw_data_dir=mock_raw_data_dir)
    with patch.object(manifest_gen, "generate", wraps=manifest_gen.generate) as spy_gen:
        manifest = manifest_gen.generate()
        assert spy_gen.call_count == 1

        out_file = tmp_path / "manifest.json"
        # Calling save with explicit manifest_data does not re-invoke generate
        manifest_gen.save(out_file, manifest_data=manifest)
        assert spy_gen.call_count == 1

        # Calling save without manifest_data uses cached manifest
        manifest_gen.save(out_file)
        assert spy_gen.call_count == 1


def test_generate_source_manifest_single_pass(mock_raw_data_dir, tmp_path):
    """Verify generate_source_manifest computes manifest exactly once and writes to disk."""
    out_file = tmp_path / "manifest.json"
    with patch.object(SourceManifest, "generate", autospec=True) as mock_gen:
        mock_gen.return_value = {
            "dataset_name": "Open University Learning Analytics Dataset (OULAD)",
            "tables": {},
            "total_raw_rows": 0,
            "total_raw_bytes": 0,
        }
        res = generate_source_manifest(raw_dir=mock_raw_data_dir, output_path=out_file)
        assert mock_gen.call_count == 1
        assert res["dataset_name"] == "Open University Learning Analytics Dataset (OULAD)"
        assert out_file.exists()
