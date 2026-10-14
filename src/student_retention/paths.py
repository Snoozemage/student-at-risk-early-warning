"""Path management for Smart Student Retention & Early Warning System."""
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

# Outputs directories
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
