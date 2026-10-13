"""Baseline model wrappers and reference benchmark builders."""
import pandas as pd
from sklearn.pipeline import Pipeline

from student_retention.logging_config import get_logger
from student_retention.modeling.pipelines import build_model_pipeline

logger = get_logger(__name__)


def train_dummy_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Train a DummyClassifier baseline on training data.

    The dummy baseline predicts the positive class prior probability, serving as
    an empirical minimum performance baseline to verify that predictive models
    substantively outperform uninformative base rates.

    Args:
        X_train: Training feature table.
        y_train: Binary training labels.
        random_state: Seed for reproducibility.

    Returns:
        Pipeline: Fitted dummy baseline pipeline.
    """
    pipeline = build_model_pipeline(model_name="dummy", random_state=random_state)
    pipeline.fit(X_train, y_train)
    logger.info(
        "Fitted dummy baseline model (prior positive probability: %.4f on %d samples)",
        y_train.mean(),
        len(y_train),
    )
    return pipeline
