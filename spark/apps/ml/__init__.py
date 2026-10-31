"""
Package initialization for Spark ML analytics modules.
"""

from spark.apps.ml.features import (
    BASE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    add_features,
    drop_null_features,
    get_feature_columns,
)
from spark.apps.ml.labels import (
    DEFAULT_HORIZONS,
    add_labels,
    compute_label_balance,
)
from spark.apps.ml.pipeline import (
    build_feature_pipeline,
    build_feature_stages,
)
from spark.apps.ml.spark_session import get_spark_session
from spark.apps.ml.splits import (
    compute_time_boundaries,
    make_time_splits,
    split_chronological,
    split_time_series,
)
from spark.apps.ml.train import (
    DEFAULT_PARAM_GRIDS,
    add_sample_weights,
    compute_balanced_weights,
    compute_majority_baseline,
    create_estimator,
    extract_feature_importances,
    format_results_table,
    run_training_job,
    save_model_artifacts,
    train_model,
)

__all__ = [
    "BASE_FEATURE_COLUMNS",
    "DEFAULT_HORIZONS",
    "DEFAULT_PARAM_GRIDS",
    "FEATURE_COLUMNS",
    "add_features",
    "add_labels",
    "add_sample_weights",
    "build_feature_pipeline",
    "build_feature_stages",
    "compute_balanced_weights",
    "compute_label_balance",
    "compute_majority_baseline",
    "compute_time_boundaries",
    "create_estimator",
    "drop_null_features",
    "extract_feature_importances",
    "format_results_table",
    "get_feature_columns",
    "get_spark_session",
    "make_time_splits",
    "run_training_job",
    "save_model_artifacts",
    "split_chronological",
    "split_time_series",
    "train_model",
]
