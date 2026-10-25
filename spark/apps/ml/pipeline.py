"""
==============================================================================
spark/apps/ml/pipeline.py
Unfitted Spark ML Pipeline definitions for feature assembly and scaling.

Stages:
  1. VectorAssembler: Combines configured feature columns into a dense vector (raw_features).
     Configured with handleInvalid="error" to ensure any unexpected null/NaN values fail loudly.
  2. StandardScaler: Normalizes feature vectors to zero-mean and unit variance (features).
     StandardScaler remains UNFITTED here; fit() must only be called on training split
     to prevent data leakage across validation or test partitions.
==============================================================================
"""

import logging
import os
import sys
from typing import List, Optional, Sequence, Union

from pyspark.ml import Pipeline
from pyspark.ml.base import Estimator, Transformer
from pyspark.ml.feature import StandardScaler, VectorAssembler

# Ensure repo root is available
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spark.apps.ml.features import FEATURE_COLUMNS

logger = logging.getLogger("crypto_ml_pipeline")

PipelineStage = Union[Estimator, Transformer]


def build_feature_stages(
    feature_cols: Optional[Sequence[str]] = None,
    raw_features_col: str = "raw_features",
    features_col: str = "features",
    with_mean: bool = True,
    with_std: bool = True,
) -> List[PipelineStage]:
    """
    Construct unfitted Spark ML stages: VectorAssembler -> StandardScaler.

    Args:
        feature_cols: Sequence of input feature column names.
            Defaults to FEATURE_COLUMNS if None.
        raw_features_col: Intermediate column name for the assembled raw feature vector.
        features_col: Final column name for the standardized feature vector.
        with_mean: Whether to center features with mean before scaling (default True).
        with_std: Whether to scale features to unit standard deviation (default True).

    Returns:
        List containing [VectorAssembler, StandardScaler] unfitted stages.

    Raises:
        ValueError: If feature_cols is empty.
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)
    if not cols:
        raise ValueError("Cannot build feature stages with empty feature column list.")

    assembler = VectorAssembler(
        inputCols=cols,
        outputCol=raw_features_col,
        handleInvalid="error",
    )

    scaler = StandardScaler(
        inputCol=raw_features_col,
        outputCol=features_col,
        withMean=with_mean,
        withStd=with_std,
    )

    return [assembler, scaler]


def build_feature_pipeline(
    feature_cols: Optional[Sequence[str]] = None,
    raw_features_col: str = "raw_features",
    features_col: str = "features",
    with_mean: bool = True,
    with_std: bool = True,
) -> Pipeline:
    """
    Construct an unfitted Spark ML Pipeline with feature assembly and scaling stages.

    Args:
        feature_cols: Sequence of input feature column names.
            Defaults to FEATURE_COLUMNS if None.
        raw_features_col: Intermediate column name for the assembled raw feature vector.
        features_col: Final column name for the standardized feature vector.
        with_mean: Whether to center features with mean before scaling (default True).
        with_std: Whether to scale features to unit standard deviation (default True).

    Returns:
        Unfitted Pipeline instance containing the feature preprocessing stages.
    """
    stages = build_feature_stages(
        feature_cols=feature_cols,
        raw_features_col=raw_features_col,
        features_col=features_col,
        with_mean=with_mean,
        with_std=with_std,
    )
    return Pipeline(stages=stages)
