"""Identifier validation and sanity checking."""
import pandas as pd

from student_retention.logging_config import get_logger

logger = get_logger(__name__)

VALID_MODULES = {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"}
VALID_PRESENTATIONS = {"2013B", "2013J", "2014B", "2014J"}


def validate_identifiers(table_name: str, df: pd.DataFrame) -> dict[str, any]:
    """Validate identifier fields for a given table.

    Checks:
    - Non-nullness of identifier fields.
    - Valid numeric formats for id_student, id_assessment, id_site.
    - Domain constraints for code_module and code_presentation.

    Returns:
        Dict with validation report: is_valid, errors, checked_rows.
    """
    errors: list[str] = []
    total_rows = len(df)

    # 1. code_module checks
    if "code_module" in df.columns:
        null_modules = df["code_module"].isna().sum()
        if null_modules > 0:
            errors.append(f"Found {null_modules} null code_module values in '{table_name}'")
        invalid_modules = set(df["code_module"].dropna().unique()) - VALID_MODULES
        if invalid_modules:
            errors.append(f"Found invalid code_module values in '{table_name}': {invalid_modules}")

    # 2. code_presentation checks
    if "code_presentation" in df.columns:
        null_pres = df["code_presentation"].isna().sum()
        if null_pres > 0:
            errors.append(f"Found {null_pres} null code_presentation values in '{table_name}'")
        invalid_pres = set(df["code_presentation"].dropna().unique()) - VALID_PRESENTATIONS
        if invalid_pres:
            errors.append(f"Found invalid code_presentation values in '{table_name}': {invalid_pres}")

    # 3. id_student checks
    if "id_student" in df.columns:
        null_students = df["id_student"].isna().sum()
        if null_students > 0:
            errors.append(f"Found {null_students} null id_student values in '{table_name}'")
        # Check integer parseability
        try:
            numeric_ids = pd.to_numeric(df["id_student"], errors="raise")
            if (numeric_ids <= 0).any():
                errors.append(f"Found non-positive id_student values in '{table_name}'")
        except (ValueError, TypeError) as e:
            errors.append(f"Non-numeric id_student values in '{table_name}': {e}")

    # 4. id_assessment checks
    if "id_assessment" in df.columns:
        null_assess = df["id_assessment"].isna().sum()
        if null_assess > 0:
            errors.append(f"Found {null_assess} null id_assessment values in '{table_name}'")
        try:
            numeric_ids = pd.to_numeric(df["id_assessment"], errors="raise")
            if (numeric_ids <= 0).any():
                errors.append(f"Found non-positive id_assessment values in '{table_name}'")
        except (ValueError, TypeError) as e:
            errors.append(f"Non-numeric id_assessment values in '{table_name}': {e}")

    # 5. id_site checks
    if "id_site" in df.columns:
        null_sites = df["id_site"].isna().sum()
        if null_sites > 0:
            errors.append(f"Found {null_sites} null id_site values in '{table_name}'")
        try:
            numeric_ids = pd.to_numeric(df["id_site"], errors="raise")
            if (numeric_ids <= 0).any():
                errors.append(f"Found non-positive id_site values in '{table_name}'")
        except (ValueError, TypeError) as e:
            errors.append(f"Non-numeric id_site values in '{table_name}': {e}")

    is_valid = len(errors) == 0
    if not is_valid:
        logger.error(f"Identifier validation failed for '{table_name}': {errors}")
    else:
        logger.info(f"Identifier validation passed for '{table_name}' ({total_rows:,} rows).")

    return {
        "table_name": table_name,
        "is_valid": is_valid,
        "checked_rows": total_rows,
        "errors": errors,
    }
