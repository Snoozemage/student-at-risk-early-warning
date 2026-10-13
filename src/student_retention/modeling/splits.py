"""Data splitting strategies for group-aware and chronological model evaluation."""
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from student_retention.logging_config import get_logger

logger = get_logger(__name__)


def verify_zero_group_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "id_student",
) -> None:
    """Verify that there is zero overlap between train and test groups.

    Args:
        train_df: Training partition.
        test_df: Test partition.
        group_col: Column name identifying groups (e.g. 'id_student').

    Raises:
        ValueError: If any group identifier appears in both train and test.
    """
    train_groups = set(train_df[group_col].dropna().unique())
    test_groups = set(test_df[group_col].dropna().unique())
    overlap = train_groups.intersection(test_groups)
    if len(overlap) > 0:
        raise ValueError(
            f"Group leakage detected! {len(overlap)} group keys appear in both train and test partitions "
            f"(sample overlapping keys: {list(overlap)[:5]})."
        )


def split_by_student_group(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    group_col: str = "id_student",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition dataset into train and test sets blocking on student group identifiers.

    All records belonging to the same student are assigned strictly to either the train
    or test partition, preventing over-optimistic evaluation from learner repetition.

    Args:
        df: Input feature table.
        test_size: Proportion of groups to allocate to the test partition (default: 0.20).
        random_state: Seed for reproducible random splitting (default: 42).
        group_col: Column name identifying student groups (default: 'id_student').

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' missing from input DataFrame.")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[group_col].values
    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Invariant verification: 100% disjoint student sets
    verify_zero_group_overlap(train_df, test_df, group_col=group_col)

    logger.info(
        "Group-aware split completed (%s): %d train rows (%d unique students), %d test rows (%d unique students)",
        group_col,
        len(train_df),
        train_df[group_col].nunique(),
        len(test_df),
        test_df[group_col].nunique(),
    )
    return train_df, test_df


def get_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "id_student",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate k-fold cross-validation train/validation index pairs blocking on groups.

    Args:
        df: Input training DataFrame.
        n_splits: Number of cross-validation folds (default: 5).
        group_col: Column name identifying student groups (default: 'id_student').

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: List of (train_indices, val_indices).
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' missing from input DataFrame.")

    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        train_groups = set(df.iloc[train_idx][group_col].unique())
        val_groups = set(df.iloc[val_idx][group_col].unique())
        overlap = train_groups.intersection(val_groups)
        if len(overlap) > 0:
            raise ValueError(
                f"Fold {fold_idx} group leakage: {len(overlap)} groups shared between train and validation."
            )
        splits.append((train_idx, val_idx))

    return splits


def split_chronological(
    df: pd.DataFrame,
    train_presentations: Sequence[str] | None = None,
    test_presentations: Sequence[str] | None = None,
    mode: str = "natural_operational",
    presentation_col: str = "code_presentation",
    group_col: str = "id_student",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition dataset chronologically by academic course presentations.

    Args:
        df: Input feature table.
        train_presentations: List of presentation codes for training (default: ['2013B', '2013J']).
        test_presentations: List of presentation codes for evaluation (default: ['2014B', '2014J']).
        mode: Splitting mode - 'natural_operational' (retains returning students in test set)
              or 'disjoint_purged' (purges returning students from test set).
        presentation_col: Column name containing presentation codes (default: 'code_presentation').
        group_col: Column name identifying student groups (default: 'id_student').

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if train_presentations is None:
        train_presentations = ["2013B", "2013J"]
    if test_presentations is None:
        test_presentations = ["2014B", "2014J"]

    if presentation_col not in df.columns:
        raise ValueError(f"Presentation column '{presentation_col}' missing from input DataFrame.")

    train_df = df[df[presentation_col].isin(train_presentations)].copy().reset_index(drop=True)
    test_df = df[df[presentation_col].isin(test_presentations)].copy().reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError(f"No records found matching train presentations: {train_presentations}")
    if len(test_df) == 0:
        raise ValueError(f"No records found matching test presentations: {test_presentations}")

    if mode == "disjoint_purged":
        train_students = set(train_df[group_col].unique())
        test_df = test_df[~test_df[group_col].isin(train_students)].copy().reset_index(drop=True)
        verify_zero_group_overlap(train_df, test_df, group_col=group_col)
    elif mode == "natural_operational":
        train_students = set(train_df[group_col].unique())
        test_students = set(test_df[group_col].unique())
        overlap = train_students.intersection(test_students)
        logger.info(
            "Natural operational chronological split: %d returning students (%.2f%% of test cohort)",
            len(overlap),
            len(overlap) / len(test_students) * 100 if len(test_students) > 0 else 0,
        )
    else:
        raise ValueError(
            f"Unknown chronological split mode '{mode}'. Expected 'natural_operational' or 'disjoint_purged'."
        )

    return train_df, test_df
