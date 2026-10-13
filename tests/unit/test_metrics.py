"""Unit tests for classification, ranking, and decision support metrics."""
import numpy as np

from student_retention.modeling.evaluate import (
    compute_classification_metrics,
    compute_top_k_metrics,
)


def test_compute_classification_metrics_perfect_predictions():
    """Test classification metrics on perfect binary predictions."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.20, 0.85, 0.90, 0.95])

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.50)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["true_positives"] == 3
    assert metrics["false_positives"] == 0
    assert metrics["true_negatives"] == 3
    assert metrics["false_negatives"] == 0
    assert metrics["brier_score"] < 0.05


def test_compute_classification_metrics_random_predictions():
    """Test classification metrics on noisy predictions."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.20, size=500)
    y_prob = np.random.uniform(0, 1, size=500)

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.50)

    assert 0.40 <= metrics["roc_auc"] <= 0.60
    assert 0.0 <= metrics["pr_auc"] <= 0.40
    assert metrics["n_samples"] == 500
    assert metrics["true_positives"] + metrics["false_negatives"] == np.sum(y_true)


def test_compute_top_k_metrics():
    """Test decision support Top-K prioritization metrics."""
    # 100 students, 20 positive withdrawals
    y_true = np.zeros(100, dtype=int)
    y_true[:15] = 1  # 15 positives at the top
    y_true[20:25] = 1  # remaining 5 positives after 5 false alarms

    # Predictor ranks from index 0 downwards
    y_prob = np.linspace(1.0, 0.0, 100)

    top_k_df = compute_top_k_metrics(
        y_true=y_true,
        y_prob=y_prob,
        k_values=(10, 20, 50),
        pct_values=(0.10, 0.20),
    )

    assert not top_k_df.empty
    # Top 10 should have 10 positives -> precision = 1.0
    top_10 = top_k_df[top_k_df["k"] == 10].iloc[0]
    assert top_10["precision_at_k"] == 1.0
    assert top_10["recall_at_k"] == 10 / 20  # 50% capture

    # Top 20: 15 positives out of top 20 -> precision = 15/20 = 0.75
    top_20 = top_k_df[top_k_df["k"] == 20].iloc[0]
    assert top_20["precision_at_k"] == 15 / 20
    assert top_20["recall_at_k"] == 15 / 20

    import itertools

    # Cumulative gain must be non-decreasing as capacity increases
    recalls = top_k_df[top_k_df["capacity_type"] == "absolute_k"]["recall_at_k"].tolist()
    assert all(x <= y for x, y in itertools.pairwise(recalls))


def test_compute_top_k_metrics_constant_scores():
    """Test that constant predicted probabilities (Dummy prior baseline) produce exact expected random yield.

    For constant predictions, Precision@K must identically equal the sample prevalence (base rate),
    and Recall@K must identically equal K / N, avoiding arbitrary sorting artifacts from numerical argsort.
    """
    n_samples = 200
    positives = 40  # 20% base rate
    y_true = np.zeros(n_samples, dtype=int)
    y_true[:positives] = 1

    # Constant probability for all students
    y_prob = np.full(n_samples, 0.20)

    top_k_df = compute_top_k_metrics(
        y_true=y_true,
        y_prob=y_prob,
        k_values=(10, 50, 100),
        pct_values=(0.05, 0.10, 0.25),
    )

    base_rate = positives / n_samples  # 0.20

    # Every absolute K slice must have precision equal to base rate and recall equal to k / n_samples
    for k in (10, 50, 100):
        row = top_k_df[(top_k_df["capacity_type"] == "absolute_k") & (top_k_df["k"] == k)].iloc[0]
        assert bool(row["is_valid_capacity"]) is True
        assert np.isclose(row["precision_at_k"], base_rate, atol=1e-6)
        assert np.isclose(row["recall_at_k"], k / n_samples, atol=1e-6)
        assert np.isclose(row["positives_captured"], k * base_rate, atol=1e-6)


def test_compute_top_k_metrics_tied_boundary():
    """Test analytical expectation when probabilities have tied scores at the K boundary."""
    # N=20 total: 5 scores at 0.90 (all 5 positive), 10 scores at 0.50 (4 positive, 6 negative), 5 scores at 0.10 (0 positive)
    # Total positives = 9
    y_true = np.array([1, 1, 1, 1, 1] + [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0], dtype=int)
    y_prob = np.array([0.90] * 5 + [0.50] * 10 + [0.10] * 5, dtype=float)

    # For K=10:
    # 5 items are strictly above 0.50 (all 5 positive)
    # The remaining 5 items are drawn from the 10 tied items with probability 4/10
    # Expected positives captured = 5 + 5 * (4/10) = 5 + 2 = 7.0
    # Expected Precision@10 = 7.0 / 10 = 0.70
    # Expected Recall@10 = 7.0 / 9 = 7/9
    top_k_df = compute_top_k_metrics(
        y_true=y_true,
        y_prob=y_prob,
        k_values=(10,),
    )

    row = top_k_df[(top_k_df["capacity_type"] == "absolute_k") & (top_k_df["k"] == 10)].iloc[0]
    assert np.isclose(row["positives_captured"], 7.0, atol=1e-6)
    assert np.isclose(row["precision_at_k"], 0.70, atol=1e-6)
    assert np.isclose(row["recall_at_k"], 7.0 / 9.0, atol=1e-6)


def test_compute_top_k_metrics_insufficient_sample_size():
    """Test that capacity K exceeding partition size N is marked invalid rather than falsely capped."""
    y_true = np.array([0, 1, 0, 1, 0])  # N=5
    y_prob = np.array([0.2, 0.8, 0.3, 0.9, 0.1])

    top_k_df = compute_top_k_metrics(
        y_true=y_true,
        y_prob=y_prob,
        k_values=(3, 10),
    )

    valid_row = top_k_df[top_k_df["k"] == 3].iloc[0]
    assert bool(valid_row["is_valid_capacity"]) is True
    assert not np.isnan(valid_row["precision_at_k"])

    invalid_row = top_k_df[top_k_df["k"] == 10].iloc[0]
    assert bool(invalid_row["is_valid_capacity"]) is False
    assert np.isnan(invalid_row["precision_at_k"])
    assert np.isnan(invalid_row["recall_at_k"])
    assert "smaller than requested capacity K=10" in invalid_row["notes"]


def test_compute_top_k_metrics_by_term():
    """Test disaggregated per-term Top-K metric calculation across pooled modules."""
    import pandas as pd

    df = pd.DataFrame({
        "code_presentation": ["2014B"] * 50 + ["2014J"] * 50,
        "withdrawn_after_cutoff": [1] * 10 + [0] * 40 + [1] * 15 + [0] * 35,
    })
    y_prob = np.random.uniform(0, 1, size=100)

    from student_retention.modeling.evaluate import (
        compute_top_k_metrics_by_term,
    )

    term_df = compute_top_k_metrics_by_term(
        df=df,
        y_prob=y_prob,
        term_col="code_presentation",
        target_col="withdrawn_after_cutoff",
        k_values=(10, 20),
        pct_values=(0.10,),
    )

    assert not term_df.empty
    assert set(term_df["presentation_code"].unique()) == {"2014B", "2014J"}
    assert all(term_df["evaluation_scope"] == "per_term")
    assert all(term_df["operational_unit"] == "academic_term_cross_module")


def test_compute_top_k_metrics_by_presentation():
    """Test backwards compatibility of compute_top_k_metrics_by_presentation."""
    import pandas as pd

    df = pd.DataFrame({
        "code_presentation": ["2014B"] * 50 + ["2014J"] * 50,
        "withdrawn_after_cutoff": [1] * 10 + [0] * 40 + [1] * 15 + [0] * 35,
    })
    y_prob = np.random.uniform(0, 1, size=100)

    from student_retention.modeling.evaluate import (
        compute_top_k_metrics_by_presentation,
    )

    pres_df = compute_top_k_metrics_by_presentation(
        df=df,
        y_prob=y_prob,
        presentation_col="code_presentation",
        target_col="withdrawn_after_cutoff",
        k_values=(10, 20),
        pct_values=(0.10,),
    )

    assert not pres_df.empty
    assert set(pres_df["presentation_code"].unique()) == {"2014B", "2014J"}
    assert all(pres_df["evaluation_scope"] == "per_term")


def test_compute_top_k_metrics_by_module_presentation():
    """Test disaggregated per-module-presentation Top-K metric calculation."""
    import pandas as pd

    df = pd.DataFrame({
        "code_module": ["AAA"] * 30 + ["BBB"] * 70,
        "code_presentation": ["2014B"] * 30 + ["2014B"] * 70,
        "withdrawn_after_cutoff": [1] * 5 + [0] * 25 + [1] * 15 + [0] * 55,
    })
    y_prob = np.random.uniform(0, 1, size=100)

    from student_retention.modeling.evaluate import (
        compute_top_k_metrics_by_module_presentation,
    )

    mod_pres_df = compute_top_k_metrics_by_module_presentation(
        df=df,
        y_prob=y_prob,
        module_col="code_module",
        presentation_col="code_presentation",
        target_col="withdrawn_after_cutoff",
        k_values=(10, 50),
        pct_values=(0.10,),
    )

    assert not mod_pres_df.empty
    assert set(mod_pres_df["presentation_code"].unique()) == {"AAA_2014B", "BBB_2014B"}
    assert set(mod_pres_df["code_module"].unique()) == {"AAA", "BBB"}
    assert set(mod_pres_df["code_presentation"].unique()) == {"2014B"}
    assert all(mod_pres_df["evaluation_scope"] == "per_module_presentation")
    assert all(mod_pres_df["operational_unit"] == "module_presentation_course_offering")

    # AAA has N=30, so K=50 must be marked invalid
    aaa_k50 = mod_pres_df[(mod_pres_df["presentation_code"] == "AAA_2014B") & (mod_pres_df["k"] == 50)].iloc[0]
    assert bool(aaa_k50["is_valid_capacity"]) is False
    assert np.isnan(aaa_k50["precision_at_k"])


def test_compute_top_k_metrics_by_term_unique_students():
    """Test that compute_top_k_metrics_by_term audits unique students and duplicate slots."""
    import pandas as pd

    from student_retention.modeling.evaluate import compute_top_k_metrics_by_term

    # 10 enrollment records across 8 unique students:
    # Student S1 has 2 enrollments (ranks 1 and 2), Student S2 has 2 enrollments (ranks 3 and 4)
    # Students S3..S8 each have 1 enrollment
    df = pd.DataFrame({
        "code_presentation": ["2014B"] * 10,
        "id_student": [101, 101, 102, 102, 103, 104, 105, 106, 107, 108],
        "code_module": ["AAA", "BBB", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"],
        "withdrawn_after_cutoff": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    })
    # Decreasing predicted probabilities aligning with the indices
    y_prob = np.array([0.95, 0.90, 0.85, 0.80, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05])

    res_df = compute_top_k_metrics_by_term(
        df=df,
        y_prob=y_prob,
        term_col="code_presentation",
        target_col="withdrawn_after_cutoff",
        student_col="id_student",
        k_values=(2, 4, 10),
        pct_values=(0.20,),
    )

    assert not res_df.empty
    assert all(res_df["queue_unit"] == "module_presentation_enrollment_record")
    assert all(res_df["operational_unit"] == "academic_term_cross_module")
    assert all(res_df["term_total_unique_students"] == 8)

    # K=2: S1 appears twice -> 1 unique student, 1 duplicate slot
    row_k2 = res_df[res_df["k"] == 2].iloc[0]
    assert row_k2["unique_students_in_top_k"] == 1
    assert row_k2["duplicate_student_slots"] == 1
    assert "1 unique students, 1 duplicate slots" in row_k2["notes"]

    # K=4: S1 (2 records) and S2 (2 records) -> 2 unique students, 2 duplicate slots
    row_k4 = res_df[res_df["k"] == 4].iloc[0]
    assert row_k4["unique_students_in_top_k"] == 2
    assert row_k4["duplicate_student_slots"] == 2
    assert "2 unique students, 2 duplicate slots" in row_k4["notes"]

    # K=10: All 10 records -> all 8 unique students, 2 duplicate slots
    row_k10 = res_df[res_df["k"] == 10].iloc[0]
    assert row_k10["unique_students_in_top_k"] == 8
    assert row_k10["duplicate_student_slots"] == 2


def test_compute_top_k_metrics_by_term_tie_boundary_invariance():
    """Test that tied scores at the K-th boundary have invariant, analytical unique student metrics.

    Verifies that when predicted probabilities contain ties crossing the K-th rank boundary,
    reordering input rows does NOT change unique_students_in_top_k or duplicate_student_slots,
    and the resulting count reflects the exact analytical expectation under uniform tie-breaking.
    """
    import pandas as pd

    from student_retention.modeling.evaluate import compute_top_k_metrics_by_term

    # 4 records with tied probabilities (0.50) across 3 students:
    # Student 101 has 2 enrollments, Student 102 has 1 enrollment, Student 103 has 1 enrollment.
    # At K=2: 2 records chosen from 4 tied records.
    # Analytical expectation:
    # P(101 chosen) = 1 - comb(2,2)/comb(4,2) = 1 - 1/6 = 5/6
    # P(102 chosen) = 1 - comb(3,2)/comb(4,2) = 1 - 3/6 = 1/2
    # P(103 chosen) = 1 - comb(3,2)/comb(4,2) = 1 - 3/6 = 1/2
    # E[unique_students] = 5/6 + 1/2 + 1/2 = 11/6 ≈ 1.8333
    # E[duplicate_slots] = 2 - 11/6 = 1/6 ≈ 0.1667

    # Order 1: consecutive enrollments for student 101
    df_order1 = pd.DataFrame({
        "code_presentation": ["2014B"] * 4,
        "id_student": [101, 101, 102, 103],
        "withdrawn_after_cutoff": [1, 0, 1, 0],
    })
    probs1 = np.array([0.50, 0.50, 0.50, 0.50])

    # Order 2: interleaved enrollments for student 101
    df_order2 = pd.DataFrame({
        "code_presentation": ["2014B"] * 4,
        "id_student": [101, 102, 101, 103],
        "withdrawn_after_cutoff": [1, 1, 0, 0],
    })
    probs2 = np.array([0.50, 0.50, 0.50, 0.50])

    # Order 3: reversed order
    df_order3 = pd.DataFrame({
        "code_presentation": ["2014B"] * 4,
        "id_student": [103, 102, 101, 101],
        "withdrawn_after_cutoff": [0, 1, 0, 1],
    })
    probs3 = np.array([0.50, 0.50, 0.50, 0.50])

    res1 = compute_top_k_metrics_by_term(df_order1, probs1, k_values=[2], pct_values=[])
    res2 = compute_top_k_metrics_by_term(df_order2, probs2, k_values=[2], pct_values=[])
    res3 = compute_top_k_metrics_by_term(df_order3, probs3, k_values=[2], pct_values=[])

    val1 = res1.iloc[0]["unique_students_in_top_k"]
    val2 = res2.iloc[0]["unique_students_in_top_k"]
    val3 = res3.iloc[0]["unique_students_in_top_k"]

    # Invariance across row orderings
    assert val1 == val2 == val3, f"Expected invariant metrics, but got {val1}, {val2}, {val3}"
    assert np.isclose(val1, 11 / 6, atol=1e-5), f"Expected 11/6 ≈ 1.8333, got {val1}"

    dup1 = res1.iloc[0]["duplicate_student_slots"]
    dup2 = res2.iloc[0]["duplicate_student_slots"]
    assert dup1 == dup2
    assert np.isclose(dup1, 1 / 6, atol=1e-5), f"Expected 1/6 ≈ 0.1667, got {dup1}"

    # Verify diagnostic note mentions uniform tie-breaking
    note = res1.iloc[0]["notes"]
    assert "expected unique students" in note
    assert "uniform tie-breaking" in note
