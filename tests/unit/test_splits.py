"""Unit tests for student group partitioning and chronological splits."""
import pandas as pd
import pytest

from student_retention.modeling.splits import (
    get_group_kfold_splits,
    split_by_student_group,
    split_chronological,
    verify_zero_group_overlap,
)


@pytest.fixture
def sample_student_df() -> pd.DataFrame:
    """Create synthetic student observations with multiple course enrollments."""
    records = []
    # 100 unique students, some with 1, 2, or 3 enrollments across presentations
    for student_id in range(1, 101):
        num_courses = 1 if student_id % 3 != 0 else (2 if student_id % 2 == 0 else 3)
        for c in range(num_courses):
            pres = "2013B" if c == 0 else ("2013J" if c == 1 else "2014J")
            records.append(
                {
                    "code_module": "AAA" if c % 2 == 0 else "BBB",
                    "code_presentation": pres,
                    "id_student": student_id,
                    "withdrawn_after_cutoff": 1 if (student_id + c) % 5 == 0 else 0,
                    "feature_1": float(student_id * 10 + c),
                }
            )
    return pd.DataFrame(records)


def test_verify_zero_group_overlap_passes(sample_student_df):
    """Ensure verification passes when train and test groups are strictly disjoint."""
    train_df = sample_student_df[sample_student_df["id_student"] <= 50]
    test_df = sample_student_df[sample_student_df["id_student"] > 50]
    # Should execute without raising
    verify_zero_group_overlap(train_df, test_df, group_col="id_student")


def test_verify_zero_group_overlap_fails_on_leakage(sample_student_df):
    """Ensure verification raises ValueError when an ID is present in both sets."""
    train_df = sample_student_df[sample_student_df["id_student"] <= 50]
    # Add a leaking student record from train into test
    test_df = sample_student_df[sample_student_df["id_student"] >= 50]
    with pytest.raises(ValueError, match="Group leakage detected"):
        verify_zero_group_overlap(train_df, test_df, group_col="id_student")


def test_split_by_student_group_disjointness(sample_student_df):
    """Test that split_by_student_group guarantees 100% disjoint groups."""
    train_df, test_df = split_by_student_group(
        sample_student_df, test_size=0.25, random_state=42, group_col="id_student"
    )

    assert len(train_df) + len(test_df) == len(sample_student_df)
    train_students = set(train_df["id_student"])
    test_students = set(test_df["id_student"])

    # Overlap must be strictly empty
    assert len(train_students.intersection(test_students)) == 0
    # Group size proportion should be roughly 75/25
    assert 0.20 <= len(test_students) / (len(train_students) + len(test_students)) <= 0.30


def test_get_group_kfold_splits_coverage_and_disjointness(sample_student_df):
    """Test 5-fold GroupKFold ensures disjoint folds and full single-fold validation coverage."""
    n_splits = 5
    splits = get_group_kfold_splits(sample_student_df, n_splits=n_splits, group_col="id_student")

    assert len(splits) == n_splits
    seen_val_indices = []

    for tr_idx, val_idx in splits:
        tr_df = sample_student_df.iloc[tr_idx]
        val_df = sample_student_df.iloc[val_idx]

        # Verify zero group overlap inside each fold
        verify_zero_group_overlap(tr_df, val_df, group_col="id_student")
        seen_val_indices.extend(val_idx)

    # Every sample must be in validation exactly once
    assert sorted(seen_val_indices) == list(range(len(sample_student_df)))


def test_split_chronological(sample_student_df):
    """Test chronological splits into train (2013) and test (2014) partitions."""
    train_df, test_op = split_chronological(sample_student_df, mode="natural_operational")
    _, test_purged = split_chronological(sample_student_df, mode="disjoint_purged")

    # Train presentations must be strictly 2013
    assert set(train_df["code_presentation"]).issubset({"2013B", "2013J"})
    # Test presentations must be strictly 2014
    assert set(test_op["code_presentation"]).issubset({"2014B", "2014J"})
    assert set(test_purged["code_presentation"]).issubset({"2014B", "2014J"})

    # Purged test partition must have strictly 0 overlap with train students
    verify_zero_group_overlap(train_df, test_purged, group_col="id_student")
    assert len(test_purged) <= len(test_op)


def test_chronological_threshold_selection_independent_of_2014_labels():
    """Regression test proving modifying 2014 test labels cannot change chronological thresholds.

    Demonstrates that:
    1. Chronological thresholds derived via GroupKFold CV on 2013 chrono_train are 100% invariant
       to any changes, perturbations, or flips in 2014 test labels (zero leakage).
    2. Primary random-split evaluation and chronological evaluation remain separate experimental designs.
    """
    import numpy as np

    from student_retention.features.feature_definitions import (
        CANONICAL_TARGET,
        PRIMARY_FEATURE_ALLOWLIST,
    )
    from student_retention.modeling.train import run_cross_validation_for_model

    np.random.seed(42)
    n = 300
    rows = []
    for i in range(n):
        student_id = 1000 + (i % 60)  # Multiple enrollments
        pres = np.random.choice(["2013B", "2013J", "2014B", "2014J"])
        row = {
            "id_student": student_id,
            "code_presentation": pres,
            "code_module": "BBB",
            "presentation_season": str(pres[-1]),
            CANONICAL_TARGET: int(np.random.binomial(1, 0.20)),
        }
        for feat in PRIMARY_FEATURE_ALLOWLIST:
            if feat not in ("code_module", "presentation_season"):
                row[feat] = float(np.random.randn())
        rows.append(row)

    df_original = pd.DataFrame(rows)

    # 1. Chronological training partition strictly from 2013
    chrono_train_orig, _ = split_chronological(df_original, mode="natural_operational")
    _, _, chrono_thresh_orig = run_cross_validation_for_model(
        model_key="logistic_regression",
        train_df=chrono_train_orig,
        n_splits=3,
        random_state=42,
    )

    # 2. Perturb/invert the 2014 test labels completely in the full dataset
    df_perturbed = df_original.copy()
    mask_2014 = df_perturbed["code_presentation"].isin(["2014B", "2014J"])
    df_perturbed.loc[mask_2014, CANONICAL_TARGET] = 1 - df_perturbed.loc[mask_2014, CANONICAL_TARGET]

    # Re-extract chronological train partition from perturbed full dataset
    chrono_train_perturbed, _ = split_chronological(df_perturbed, mode="natural_operational")
    _, _, chrono_thresh_perturbed = run_cross_validation_for_model(
        model_key="logistic_regression",
        train_df=chrono_train_perturbed,
        n_splits=3,
        random_state=42,
    )

    # Invariant: Chronological threshold MUST be completely identical regardless of 2014 labels
    assert chrono_thresh_orig == chrono_thresh_perturbed
