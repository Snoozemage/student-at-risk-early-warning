"""Feature selection and anti-leakage verification utilities."""
import pandas as pd

from student_retention.features.feature_definitions import (
    CANONICAL_GRAIN,
    CANONICAL_TARGET,
    DEMOGRAPHIC_AUDIT_FEATURES,
    FORBIDDEN_FEATURES,
    PRIMARY_FEATURE_ALLOWLIST,
)


def get_feature_allowlist() -> list[str]:
    """Return a copy of the locked 22-feature allowlist."""
    return list(PRIMARY_FEATURE_ALLOWLIST)


def select_primary_features(
    df: pd.DataFrame,
    include_target: bool = True,
    include_identifiers: bool = True,
) -> pd.DataFrame:
    """Select only the canonical identifiers, primary 22 features, and optional target.

    Args:
        df: Input DataFrame containing features.
        include_target: If True and target is present in df, include CANONICAL_TARGET.
        include_identifiers: If True, include canonical grain keys.

    Returns:
        DataFrame containing strictly the requested columns.

    Raises:
        ValueError: If any primary feature is missing from the input DataFrame.
    """
    missing = [f for f in PRIMARY_FEATURE_ALLOWLIST if f not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required primary features ({len(missing)}): {missing}"
        )

    cols = []
    if include_identifiers:
        for key in CANONICAL_GRAIN:
            if key in df.columns:
                cols.append(key)

    cols.extend(PRIMARY_FEATURE_ALLOWLIST)

    if include_target and CANONICAL_TARGET in df.columns:
        cols.append(CANONICAL_TARGET)

    # Deduplicate column names while preserving order
    deduped_cols = list(dict.fromkeys(cols))

    return df[deduped_cols].copy()


def select_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select canonical grain identifiers and demographic audit attributes.

    Args:
        df: Input DataFrame containing demographic fields.

    Returns:
        DataFrame containing grain keys and available demographic audit features.
    """
    cols = [col for col in CANONICAL_GRAIN if col in df.columns]
    for demo in DEMOGRAPHIC_AUDIT_FEATURES:
        if demo in df.columns:
            cols.append(demo)

    return df[cols].copy()


def verify_no_leakage_columns(
    columns: list[str] | set[str] | pd.Index,
    allow_target: bool = False,
) -> None:
    """Verify that no forbidden or target-derived columns are present in the feature list.

    Args:
        columns: List, set, or index of column names to check.
        allow_target: If True, CANONICAL_TARGET ('withdrawn_after_cutoff') is permitted.

    Raises:
        ValueError: If any forbidden column is detected.
    """
    col_set = set(columns)
    forbidden = set(FORBIDDEN_FEATURES)
    if allow_target:
        forbidden.discard(CANONICAL_TARGET)

    leakage = col_set.intersection(forbidden)
    if leakage:
        raise ValueError(
            f"Forbidden leakage columns detected in feature set: {sorted(leakage)}"
        )
