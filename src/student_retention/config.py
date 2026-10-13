"""Configuration loading utilities."""
from pathlib import Path
from typing import Any

import yaml

from student_retention.paths import CONFIG_DIR


def load_yaml_config(filename: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        filename: Name of the config file (e.g. 'sources.yaml')
        config_dir: Directory containing configuration files

    Returns:
        Dictionary with parsed configuration data.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is empty.
    """
    file_path = config_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Configuration file is empty: {file_path}")

    return data


def get_project_config() -> dict[str, Any]:
    return load_yaml_config("project.yaml")


def get_sources_config() -> dict[str, Any]:
    return load_yaml_config("sources.yaml")


def get_schema_config() -> dict[str, Any]:
    return load_yaml_config("schema.yaml")


def get_quality_rules_config() -> dict[str, Any]:
    return load_yaml_config("quality_rules.yaml")


def get_feature_windows_config() -> dict[str, Any]:
    return load_yaml_config("feature_windows.yaml")


def get_model_config() -> dict[str, Any]:
    return load_yaml_config("model.yaml")


def get_evaluation_config() -> dict[str, Any]:
    return load_yaml_config("evaluation.yaml")


def get_fairness_config() -> dict[str, Any]:
    return load_yaml_config("fairness.yaml")


def get_reporting_config() -> dict[str, Any]:
    return load_yaml_config("reporting.yaml")
