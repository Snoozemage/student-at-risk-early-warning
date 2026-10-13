"""Modeling package: split strategies, pipelines, baseline, training, evaluation, and interpretation."""
from student_retention.modeling.baseline import train_dummy_baseline
from student_retention.modeling.evaluate import (
    compute_classification_metrics,
    compute_top_k_metrics,
    compute_top_k_metrics_by_module_presentation,
    compute_top_k_metrics_by_presentation,
    compute_top_k_metrics_by_term,
    find_error_cases,
)
from student_retention.modeling.interpretation import (
    compute_permutation_importance,
    extract_logistic_regression_coefficients,
    extract_random_forest_importance,
)
from student_retention.modeling.pipelines import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_model_pipeline,
    create_preprocessor,
    get_feature_subsets,
)
from student_retention.modeling.predict import load_model, predict_withdrawal_risk
from student_retention.modeling.splits import (
    get_group_kfold_splits,
    split_by_student_group,
    split_chronological,
    verify_zero_group_overlap,
)
from student_retention.modeling.threshold import (
    find_optimal_threshold,
    scan_thresholds,
)
from student_retention.modeling.train import (
    MODEL_CONFIGS,
    run_cross_validation_for_model,
    train_and_evaluate_all_models,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "MODEL_CONFIGS",
    "NUMERIC_FEATURES",
    "build_model_pipeline",
    "compute_classification_metrics",
    "compute_permutation_importance",
    "compute_top_k_metrics",
    "compute_top_k_metrics_by_module_presentation",
    "compute_top_k_metrics_by_presentation",
    "compute_top_k_metrics_by_term",
    "create_preprocessor",
    "extract_logistic_regression_coefficients",
    "extract_random_forest_importance",
    "find_error_cases",
    "find_optimal_threshold",
    "get_feature_subsets",
    "get_group_kfold_splits",
    "load_model",
    "predict_withdrawal_risk",
    "run_cross_validation_for_model",
    "scan_thresholds",
    "split_by_student_group",
    "split_chronological",
    "train_and_evaluate_all_models",
    "train_dummy_baseline",
    "verify_zero_group_overlap",
]
