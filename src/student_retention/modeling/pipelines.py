"""Scikit-learn pipeline definitions for leak-free Day-35 model training."""
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from student_retention.features.feature_definitions import PRIMARY_FEATURE_ALLOWLIST
from student_retention.logging_config import get_logger

logger = get_logger(__name__)

# Canonical division of the 22 primary allowlist features
CATEGORICAL_FEATURES: list[str] = ["code_module", "presentation_season"]
NUMERIC_FEATURES: list[str] = [
    f for f in PRIMARY_FEATURE_ALLOWLIST if f not in CATEGORICAL_FEATURES
]


def get_feature_subsets() -> tuple[list[str], list[str]]:
    """Return copies of the standard numeric and categorical feature lists.

    Returns:
        tuple[list[str], list[str]]: (numeric_features, categorical_features)
    """
    return list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)


def create_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Create a ColumnTransformer preprocessor fitted strictly within pipelines.

    Numeric branch:
        - Median imputation with missingness indicator (handles registration lead time etc.).
        - Standard scaling (zero mean, unit variance).
    Categorical branch:
        - Constant imputation ("MISSING") for unseen or missing categories.
        - One-hot encoding with handle_unknown="ignore" and dense array output.

    Args:
        numeric_features: List of numeric column names. Defaults to canonical NUMERIC_FEATURES.
        categorical_features: List of categorical column names. Defaults to canonical CATEGORICAL_FEATURES.

    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    if numeric_features is None:
        numeric_features = list(NUMERIC_FEATURES)
    if categorical_features is None:
        categorical_features = list(CATEGORICAL_FEATURES)

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def build_model_pipeline(
    model_name: str,
    class_weight: str | None = None,
    random_state: int = 42,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    **classifier_kwargs: Any,
) -> Pipeline:
    """Build a complete scikit-learn Pipeline with preprocessing and classifier.

    Supported model_name values:
        - 'dummy': DummyClassifier predicting prior class distribution.
        - 'logistic_regression': L2-regularized logistic regression.
        - 'random_forest': Random forest classifier (200 trees, max_depth=8).

    Args:
        model_name: Identifier for the algorithm.
        class_weight: Weighting strategy (e.g. 'balanced' or None).
        random_state: Reproducible seed.
        numeric_features: Optional custom numeric columns.
        categorical_features: Optional custom categorical columns.
        **classifier_kwargs: Additional keyword arguments forwarded to the classifier.

    Returns:
        Pipeline: Assembled leak-free training pipeline.

    Raises:
        ValueError: If model_name is unrecognized.
    """
    preprocessor = create_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    canonical_name = model_name.lower().replace("-", "_")

    if canonical_name in ("dummy", "dummy_baseline"):
        clf = DummyClassifier(strategy="prior", random_state=random_state)
    elif canonical_name in ("logistic_regression", "lr", "logit"):
        params: dict[str, Any] = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": class_weight,
            "random_state": random_state,
        }
        params.update(classifier_kwargs)
        clf = LogisticRegression(**params)
    elif canonical_name in ("random_forest", "rf"):
        params = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": -1,
        }
        params.update(classifier_kwargs)
        clf = RandomForestClassifier(**params)
    else:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Choose from: 'dummy', 'logistic_regression', 'random_forest'."
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    logger.debug(
        "Constructed pipeline for '%s' (class_weight=%s, random_state=%d)",
        canonical_name,
        class_weight,
        random_state,
    )
    return pipeline
