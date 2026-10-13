"""Model interpretation utilities: Logistic Regression coefficients and Permutation Importance."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from student_retention.logging_config import get_logger

logger = get_logger(__name__)


def get_feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    """Extract transformed feature names from a fitted Pipeline with ColumnTransformer.

    Args:
        pipeline: Fitted scikit-learn Pipeline.

    Returns:
        list[str]: Output feature names after imputation, scaling, and one-hot encoding.
    """
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline does not contain a step named 'preprocessor'.")

    try:
        names = list(preprocessor.get_feature_names_out())
        # Clean scikit-learn prefixes like 'num__' and 'cat__'
        cleaned_names = [
            n.replace("num__", "").replace("cat__", "") for n in names
        ]
        return cleaned_names
    except (AttributeError, ValueError) as e:
        logger.warning("Could not extract feature names via get_feature_names_out: %s", e)
        return []


def extract_logistic_regression_coefficients(
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Extract log-odds coefficients and Odds Ratios (exp(coef)) from a fitted Logistic Regression pipeline.

    Args:
        pipeline: Fitted Pipeline containing 'classifier' (LogisticRegression).

    Returns:
        pd.DataFrame: Table with columns [feature, coefficient_log_odds, odds_ratio, direction].
    """
    clf = pipeline.named_steps.get("classifier")
    if clf is None or not hasattr(clf, "coef_"):
        raise ValueError("Pipeline classifier is not a fitted linear model with 'coef_'.")

    coefs = clf.coef_[0]
    feature_names = get_feature_names_from_pipeline(pipeline)
    if len(feature_names) != len(coefs):
        feature_names = [f"feature_{i}" for i in range(len(coefs))]

    odds_ratios = np.exp(coefs)
    directions = [
        "Increases Risk" if c > 0 else "Protective (Reduces Risk)" for c in coefs
    ]

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient_log_odds": coefs,
            "odds_ratio": odds_ratios,
            "abs_coef": np.abs(coefs),
            "effect_direction": directions,
        }
    )

    df = df.sort_values(by="abs_coef", ascending=False).reset_index(drop=True)
    return df


def extract_random_forest_importance(
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Extract Gini impurity feature importances from a fitted Random Forest pipeline.

    Args:
        pipeline: Fitted Pipeline containing 'classifier' (RandomForestClassifier).

    Returns:
        pd.DataFrame: Table with columns [feature, impurity_importance, relative_rank].
    """
    clf = pipeline.named_steps.get("classifier")
    if clf is None or not hasattr(clf, "feature_importances_"):
        raise ValueError("Pipeline classifier does not expose 'feature_importances_'.")

    importances = clf.feature_importances_
    feature_names = get_feature_names_from_pipeline(pipeline)
    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "impurity_importance": importances,
        }
    )
    df = df.sort_values(by="impurity_importance", ascending=False).reset_index(drop=True)
    df["relative_rank"] = df.index + 1
    return df


def compute_permutation_importance(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scoring: str = "average_precision",
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance on out-of-fold or held-out validation data.

    Measures decrease in primary ranking score (PR-AUC / average precision) when
    each raw feature column is independently shuffled.

    Args:
        pipeline: Fitted pipeline.
        X_val: Validation feature table (raw pre-pipeline features).
        y_val: Validation ground truth labels.
        scoring: Scikit-learn metric string (default: 'average_precision').
        n_repeats: Number of shuffle repetitions.
        random_state: Reproducible seed.

    Returns:
        pd.DataFrame: Table with columns [feature, importance_mean, importance_std].
    """
    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df = pd.DataFrame(
        {
            "feature": list(X_val.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    df = df.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df
