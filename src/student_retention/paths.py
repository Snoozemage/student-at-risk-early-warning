"""Path management for Smart Student Retention & Early Warning System."""
import os
from pathlib import Path

# Repository root (two levels up from src/student_retention)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Configuration directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw" / "oulad"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Interim subdirectories
INTERIM_NORMALIZED_DIR = DATA_INTERIM_DIR / "normalized"
INTERIM_VALIDATED_DIR = DATA_INTERIM_DIR / "validated"

# Processed subdirectories
PROCESSED_COHORT_DIR = DATA_PROCESSED_DIR / "cohort"
PROCESSED_FEATURES_DIR = DATA_PROCESSED_DIR / "features"

# Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# Outputs directory resolution helpers (supports clean-environment test isolation)
def get_outputs_dir() -> Path:
    """Return root outputs directory, allowing override via STUDENT_RETENTION_OUTPUTS_DIR."""
    env_dir = os.environ.get("STUDENT_RETENTION_OUTPUTS_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return PROJECT_ROOT / "outputs"


def get_tables_dir() -> Path:
    """Return tables directory."""
    return get_outputs_dir() / "tables"


def get_predictions_dir() -> Path:
    """Return predictions directory."""
    return get_outputs_dir() / "predictions"


def get_figures_dir() -> Path:
    """Return figures directory."""
    return get_outputs_dir() / "figures"


def get_quality_reports_dir() -> Path:
    """Return quality reports directory."""
    return get_outputs_dir() / "quality_reports"


def get_reports_dir() -> Path:
    """Return reports directory."""
    return get_outputs_dir() / "reports"


def get_deliverables_dir() -> Path:
    """Return deliverables directory, allowing override via STUDENT_RETENTION_DELIVERABLES_DIR."""
    env_dir = os.environ.get("STUDENT_RETENTION_DELIVERABLES_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return PROJECT_ROOT / "deliverables"


# Outputs directories (module-level constants)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
QUALITY_REPORTS_DIR = OUTPUTS_DIR / "quality_reports"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Model artifact directories
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_BASELINE_DIR = MODELS_DIR / "baseline"

# Deliverables directory
DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"

def ensure_directories():
    """Ensure required runtime directories exist."""
    for directory in [
        DATA_INTERIM_DIR,
        INTERIM_NORMALIZED_DIR,
        INTERIM_VALIDATED_DIR,
        DATA_PROCESSED_DIR,
        PROCESSED_COHORT_DIR,
        PROCESSED_FEATURES_DIR,
        NOTEBOOKS_DIR,
        OUTPUTS_DIR,
        QUALITY_REPORTS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        PREDICTIONS_DIR,
        LOGS_DIR,
        MODELS_DIR,
        MODELS_BASELINE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
