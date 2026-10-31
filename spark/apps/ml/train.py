"""
==============================================================================
spark/apps/ml/train.py
Production-style Spark MLlib model training with time-aware validation.

Guarantees & Constraints:
  1. Chronological splitting with gap-aware purge/embargo to prevent forward lookahead.
  2. Balanced sample weights calculated strictly on the training partition.
  3. Feature scaling (StandardScaler) fitted strictly on the training partition.
  4. Hyperparameter tuning on validation AUC (zero TrainValidationSplit/CrossValidator random leakage).
  5. Test split scored exactly once per selected model for honest evaluation.
  6. Comprehensive metadata, predictions, and pipeline model artifact persistence.
  7. Strict support for binary direction prediction across multiple forward horizons.
==============================================================================
"""

import argparse
from datetime import datetime
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pyspark
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.base import Estimator
from pyspark.ml.classification import (
    GBTClassificationModel,
    GBTClassifier,
    LogisticRegression,
    LogisticRegressionModel,
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType

# Ensure repo root is available
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spark.apps.ml.features import (
    FEATURE_COLUMNS,
    add_features,
    drop_null_features,
)
from spark.apps.ml.labels import MS_PER_MINUTE, add_labels
from spark.apps.ml.pipeline import build_feature_stages
from spark.apps.ml.spark_session import get_spark_session
from spark.apps.ml.splits import split_time_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("crypto_ml_train")

# Default explicit parameter grids for laptop-friendly time-aware validation
DEFAULT_PARAM_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "logreg": [
        {"regParam": 0.01, "elasticNetParam": 0.1},
        {"regParam": 0.1, "elasticNetParam": 0.5},
    ],
    "rf": [
        {"numTrees": 20, "maxDepth": 5},
    ],
    "gbt": [
        {"maxIter": 20, "maxDepth": 4, "stepSize": 0.1},
    ],
}


def get_git_commit_hash() -> str:
    """Retrieve current Git commit hash or fallback to unknown."""
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return commit_hash
    except Exception:
        return "unknown"


def compute_balanced_weights(df: DataFrame, label_col: str = "label") -> Dict[float, float]:
    """
    Compute balanced sample weights from the training split only:
      w_c = N / (2.0 * N_c)

    Args:
        df: Training DataFrame containing binary labels.
        label_col: Name of the label column.

    Returns:
        Mapping of float label (0.0, 1.0) to sample weight.
    """
    counts = df.groupBy(label_col).count().collect()
    total = sum(row["count"] for row in counts)
    weights: Dict[float, float] = {}
    for row in counts:
        lbl = float(row[label_col])
        cnt = row["count"]
        weights[lbl] = float(total) / (2.0 * float(cnt)) if cnt > 0 else 1.0

    if 0.0 not in weights:
        weights[0.0] = 1.0
    if 1.0 not in weights:
        weights[1.0] = 1.0
    return weights


def add_sample_weights(
    df: DataFrame,
    weights: Dict[float, float],
    label_col: str = "label",
    weight_col: str = "weight",
) -> DataFrame:
    """
    Attach precomputed sample weights to DataFrame rows based on label.

    Args:
        df: Input DataFrame.
        weights: Dictionary mapping label to sample weight.
        label_col: Label column name.
        weight_col: Target sample weight column name.

    Returns:
        DataFrame with weight column appended.
    """
    w0 = float(weights.get(0.0, 1.0))
    w1 = float(weights.get(1.0, 1.0))
    return df.withColumn(
        weight_col,
        F.when(F.col(label_col).isNull(), F.lit(None).cast(DoubleType()))
        .when(F.col(label_col) == 1.0, F.lit(w1))
        .otherwise(F.lit(w0))
        .cast(DoubleType()),
    )


def compute_majority_baseline(
    train_df: DataFrame,
    test_df: DataFrame,
    label_col: str = "label",
) -> Tuple[float, float]:
    """
    Compute majority class from train split and calculate its accuracy on test split.

    Args:
        train_df: Training DataFrame.
        test_df: Out-of-sample test DataFrame.
        label_col: Label column name.

    Returns:
        Tuple of (majority_class, majority_baseline_accuracy).
    """
    train_counts = train_df.groupBy(label_col).count().collect()
    n0 = next((r["count"] for r in train_counts if float(r[label_col]) == 0.0), 0)
    n1 = next((r["count"] for r in train_counts if float(r[label_col]) == 1.0), 0)
    majority_class = 1.0 if n1 >= n0 else 0.0

    test_total = test_df.count()
    if test_total == 0:
        return majority_class, 0.0

    test_majority_correct = test_df.filter(F.col(label_col) == majority_class).count()
    baseline_acc = float(test_majority_correct) / float(test_total)
    return majority_class, baseline_acc


def extract_feature_importances(
    fitted_pipeline: PipelineModel,
    feature_names: Sequence[str],
) -> Dict[str, float]:
    """
    Extract feature coefficients (LogisticRegression) or importances (RF/GBT).

    Args:
        fitted_pipeline: Fitted PipelineModel.
        feature_names: Ordered list of feature column names.

    Returns:
        Mapping of feature name to importance / coefficient.
    """
    model_stage = fitted_pipeline.stages[-1]
    result: Dict[str, float] = {}

    if isinstance(model_stage, LogisticRegressionModel):
        coef_arr = model_stage.coefficients.toArray()
        for name, coef in zip(feature_names, coef_arr):
            result[name] = float(coef)
        result["_intercept"] = float(model_stage.intercept)
    elif isinstance(model_stage, (RandomForestClassificationModel, GBTClassificationModel)):
        imp_arr = model_stage.featureImportances.toArray()
        for name, imp in zip(feature_names, imp_arr):
            result[name] = float(imp)
    return result


def create_estimator(
    model_name: str,
    params: Dict[str, Any],
    seed: int,
    features_col: str = "features",
    label_col: str = "label",
    weight_col: str = "weight",
) -> Estimator:
    """
    Instantiate an MLlib classification estimator with the specified hyperparameter configuration.

    Args:
        model_name: Identifier ('logreg', 'rf', 'gbt').
        params: Hyperparameter dictionary.
        seed: Random seed.
        features_col: Feature vector column name.
        label_col: Target label column name.
        weight_col: Sample weight column name.

    Returns:
        Configured unfitted Estimator.
    """
    m = model_name.lower().strip()
    if m == "logreg":
        return LogisticRegression(
            featuresCol=features_col,
            labelCol=label_col,
            weightCol=weight_col,
            maxIter=params.get("maxIter", 50),
            regParam=params.get("regParam", 0.01),
            elasticNetParam=params.get("elasticNetParam", 0.1),
        )
    elif m == "rf":
        return RandomForestClassifier(
            featuresCol=features_col,
            labelCol=label_col,
            weightCol=weight_col,
            seed=seed,
            numTrees=params.get("numTrees", 20),
            maxDepth=params.get("maxDepth", 5),
        )
    elif m == "gbt":
        return GBTClassifier(
            featuresCol=features_col,
            labelCol=label_col,
            weightCol=weight_col,
            seed=seed,
            maxIter=params.get("maxIter", 20),
            maxDepth=params.get("maxDepth", 4),
            stepSize=params.get("stepSize", 0.1),
        )
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: 'logreg', 'rf', 'gbt'")


def train_model(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    model_name: str,
    feature_cols: Sequence[str],
    param_grid: Optional[List[Dict[str, Any]]] = None,
    seed: int = 42,
    label_col: str = "label",
    weight_col: str = "weight",
) -> Dict[str, Any]:
    """
    Train a candidate model using chronological validation AUC tuning and score test split once.

    Pipeline stages (VectorAssembler -> StandardScaler -> Estimator) are fit strictly on train_df.

    Args:
        train_df: Training partition.
        val_df: Validation partition.
        test_df: Test partition.
        model_name: Model identifier ('logreg', 'rf', 'gbt').
        feature_cols: Sequence of input feature columns.
        param_grid: Optional explicit list of hyperparameter dicts.
        seed: Deterministic seed.
        label_col: Label column name.
        weight_col: Sample weight column name.

    Returns:
        Dictionary containing best model, hyperparameters, validation and test metrics.
    """
    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRIDS.get(model_name.lower(), [{}])
    if not grid:
        grid = [{}]

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy",
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="f1",
    )

    best_val_auc = -1.0
    best_pipe_model: Optional[PipelineModel] = None
    best_params: Dict[str, Any] = {}

    start_time = time.time()

    # Explicit grid search evaluated on validation split
    for candidate_params in grid:
        estimator = create_estimator(
            model_name=model_name,
            params=candidate_params,
            seed=seed,
            features_col="features",
            label_col=label_col,
            weight_col=weight_col,
        )

        stages = build_feature_stages(
            feature_cols=feature_cols,
            raw_features_col="raw_features",
            features_col="features",
        ) + [estimator]

        pipeline = Pipeline(stages=stages)
        # FIT THE WHOLE PIPELINE ON TRAIN SPLIT ONLY
        fitted_pipeline = pipeline.fit(train_df)

        val_preds = fitted_pipeline.transform(val_df)
        val_auc = float(evaluator_auc.evaluate(val_preds))

        logger.info(
            "Model: %s | Params: %s | Val AUC: %.4f",
            model_name,
            candidate_params,
            val_auc,
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_pipe_model = fitted_pipeline
            best_params = candidate_params

    training_time_seconds = time.time() - start_time

    if best_pipe_model is None:
        raise RuntimeError(f"Training failed to produce a valid model for {model_name}.")

    # Score TEST split exactly once with the best validated model
    test_preds = best_pipe_model.transform(test_df)
    test_auc = float(evaluator_auc.evaluate(test_preds))
    test_acc = float(evaluator_acc.evaluate(test_preds))
    test_f1 = float(evaluator_f1.evaluate(test_preds))

    majority_class, majority_baseline_acc = compute_majority_baseline(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
    )

    feature_importances = extract_feature_importances(
        fitted_pipeline=best_pipe_model,
        feature_names=feature_cols,
    )

    return {
        "model_name": model_name,
        "best_params": best_params,
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_baseline_acc,
        "training_time_seconds": training_time_seconds,
        "fitted_pipeline": best_pipe_model,
        "test_predictions": test_preds,
        "feature_importance": feature_importances,
    }


def save_model_artifacts(
    output_dir: str,
    horizon: int,
    model_name: str,
    fitted_pipeline: PipelineModel,
    test_predictions: DataFrame,
    metadata: Dict[str, Any],
    feature_weights: Dict[str, float],
) -> str:
    """
    Persist the fitted pipeline model, metadata, test predictions, and feature importance.

    Artifact Structure:
      {output_dir}/horizon_{horizon}m/{model_name}/
        ├── pipeline_model/
        ├── metadata.json
        ├── feature_importance.json
        └── test_predictions.parquet

    Returns:
        Path to the model directory.
    """
    model_dir = os.path.join(output_dir, f"horizon_{horizon}m", model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 1. PipelineModel
    pipe_path = os.path.join(model_dir, "pipeline_model")
    fitted_pipeline.write().overwrite().save(pipe_path)

    # 2. Metadata JSON
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 3. Feature Importance JSON
    feat_path = os.path.join(model_dir, "feature_importance.json")
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_weights, f, indent=2)

    # 4. Test Predictions Parquet
    pred_path = os.path.join(model_dir, "test_predictions.parquet")
    save_preds_df = test_predictions
    if "symbol" not in save_preds_df.columns:
        save_preds_df = save_preds_df.withColumn("symbol", F.lit("UNKNOWN").cast(StringType()))
    if "open_time" not in save_preds_df.columns:
        save_preds_df = save_preds_df.withColumn("open_time", F.lit(0).cast(LongType()))

    pred_cols = ["symbol", "open_time", "label", "prediction", "probability"]
    save_preds_df.select(*pred_cols).write.mode("overwrite").parquet(pred_path)

    logger.info("Artifacts successfully saved to: %s", model_dir)
    return model_dir


def format_results_table(results: List[Dict[str, Any]]) -> str:
    """Format training results as a compact summary table."""
    headers = [
        "Horizon",
        "Model",
        "Val AUC",
        "Test AUC",
        "Test Acc",
        "Majority Base",
        "Test F1",
        "Train Time (s)",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                f"{r['horizon']}m",
                r["model_name"],
                f"{r['val_auc']:.4f}",
                f"{r['test_auc']:.4f}",
                f"{r['test_accuracy']:.4f}",
                f"{r['majority_baseline_accuracy']:.4f}",
                f"{r['test_f1']:.4f}",
                f"{r['training_time_seconds']:.1f}s",
            ]
        )

    # Column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(val))

    def make_row(items: List[str]) -> str:
        return " | ".join(item.ljust(col_widths[i]) for i, item in enumerate(items))

    sep_line = "-+-".join("-" * w for w in col_widths)
    header_line = make_row(headers)
    table_lines = [header_line, sep_line]
    for row in rows:
        table_lines.append(make_row(row))

    return "\n".join(table_lines)


def run_training_job(
    input_path: str = "data/sample_klines.parquet",
    symbols: Optional[Sequence[str]] = None,
    horizons: Sequence[int] = (5, 15, 60),
    models: Sequence[str] = ("logreg", "rf", "gbt"),
    output_dir: str = "data/models",
    seed: int = 42,
    exclude_features: Optional[Sequence[str]] = None,
    sample_fraction: Optional[float] = None,
    conservative_purge: bool = False,
    param_grid_override: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    quick: bool = False,
    spark: Optional[SparkSession] = None,
) -> List[Dict[str, Any]]:
    """
    Execute full cryptocurrency ML model training and validation pipeline.

    Args:
        input_path: Path to klines parquet dataset.
        symbols: Optional subset of symbols to train on. Defaults to all.
        horizons: Sequence of lookahead horizons in minutes.
        models: Sequence of model types ('logreg', 'rf', 'gbt').
        output_dir: Destination path for persistent model artifacts.
        seed: Random seed.
        exclude_features: Features to omit from FEATURE_COLUMNS for ablations.
        sample_fraction: Deterministic downsampling fraction (0.0 < fraction <= 1.0).
        conservative_purge: If True, uses max(horizons) for all purge embargoes.
        param_grid_override: Optional custom parameter grid dict.
        quick: If True, restricts grids to a single hyperparameter configuration.
        spark: Existing SparkSession or None to create one.

    Returns:
        List of structured result dictionaries per (horizon, model).
    """
    spark_session = spark or get_spark_session(app_name="CryptoMLTrainingJob")

    logger.info("Loading klines dataset from: %s", input_path)
    df = spark_session.read.parquet(input_path)

    # Filter symbols if requested
    if symbols is not None and len(symbols) > 0 and symbols != ["all"]:
        symbol_list = [s.strip().upper() for s in symbols]
        df = df.filter(F.col("symbol").isin(symbol_list))
        logger.info("Filtered dataset to symbols: %s", symbol_list)

    # Optional deterministic downsampling
    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
        logger.info("Sampled dataset with fraction %.4f (seed=%d)", sample_fraction, seed)

    # Active feature set determination
    excluded = set(exclude_features or [])
    active_features = [f for f in FEATURE_COLUMNS if f not in excluded]
    if not active_features:
        raise ValueError("Cannot train with empty feature list. All features excluded.")
    logger.info("Active features count: %d (excluded %d)", len(active_features), len(excluded))

    # Precompute causal features once across the entire dataset
    logger.info("Computing gap-safe scale-free technical features...")
    df_with_features = add_features(df, drop_nulls=False)

    git_hash = get_git_commit_hash()
    pyspark_ver = str(pyspark.__version__)
    max_horizon = max(horizons)

    aggregated_results: List[Dict[str, Any]] = []

    for h in horizons:
        logger.info("==================================================================")
        logger.info("Processing Horizon: %dm", h)
        logger.info("==================================================================")

        target_col = f"target_direction_{h}m"
        # 1. Add forward labels for horizon h
        df_labeled = add_labels(df_with_features, horizons=[h], drop_null_labels=False)
        df_labeled = df_labeled.withColumn("label", F.col(target_col).cast(DoubleType()))

        # 2. Drop rows with null features (warm-up) or null labels
        df_clean = drop_null_features(df_labeled, feature_cols=active_features)
        df_clean = df_clean.filter(F.col("label").isNotNull() & ~F.isnan(F.col("label")))

        # 3. Chronological split with purge
        purge_horizon = max_horizon if conservative_purge else h
        train_df, val_df, test_df, split_info = split_time_series(
            df=df_clean,
            horizon_minutes=purge_horizon,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            time_col="open_time",
            label_col="label",
            feature_cols=active_features,
            drop_nulls=True,
        )

        # 4. Balanced sample weights calculated strictly on TRAIN split
        weights = compute_balanced_weights(train_df, label_col="label")
        logger.info("Horizon %dm train balanced weights: %s", h, weights)

        train_df = add_sample_weights(train_df, weights, label_col="label", weight_col="weight").cache()
        val_df = add_sample_weights(val_df, weights, label_col="label", weight_col="weight").cache()
        test_df = add_sample_weights(test_df, weights, label_col="label", weight_col="weight").cache()

        for model_name in models:
            m_name = model_name.lower().strip()
            logger.info("Training %s for horizon %dm...", m_name, h)

            if param_grid_override and m_name in param_grid_override:
                m_grid = param_grid_override[m_name]
            else:
                m_grid = DEFAULT_PARAM_GRIDS.get(m_name, [{}])

            if quick and len(m_grid) > 1:
                m_grid = [m_grid[0]]

            model_result = train_model(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                model_name=m_name,
                feature_cols=active_features,
                param_grid=m_grid,
                seed=seed,
                label_col="label",
                weight_col="weight",
            )

            # Metadata creation
            metadata = {
                "feature_list": list(active_features),
                "horizon": int(h),
                "model_name": m_name,
                "split_boundaries_and_row_counts": split_info,
                "purge_setting": {
                    "horizon_minutes": purge_horizon,
                    "window_ms": purge_horizon * MS_PER_MINUTE,
                },
                "hyperparameters": model_result["best_params"],
                "metrics": {
                    "val_auc": float(model_result["best_val_auc"]),
                    "test_auc": float(model_result["test_auc"]),
                    "test_accuracy": float(model_result["test_accuracy"]),
                    "test_f1": float(model_result["test_f1"]),
                    "majority_baseline_accuracy": float(model_result["majority_baseline_accuracy"]),
                    "training_time_seconds": float(model_result["training_time_seconds"]),
                },
                "seed": int(seed),
                "git_commit_hash": str(git_hash),
                "pyspark_version": pyspark_ver,
                "created_at_utc": datetime.utcnow().isoformat() + "Z",
            }

            # Save artifacts
            save_model_artifacts(
                output_dir=output_dir,
                horizon=h,
                model_name=m_name,
                fitted_pipeline=model_result["fitted_pipeline"],
                test_predictions=model_result["test_predictions"],
                metadata=metadata,
                feature_weights=model_result["feature_importance"],
            )

            summary_item = {
                "horizon": h,
                "model_name": m_name,
                "val_auc": model_result["best_val_auc"],
                "test_auc": model_result["test_auc"],
                "test_accuracy": model_result["test_accuracy"],
                "majority_baseline_accuracy": model_result["majority_baseline_accuracy"],
                "test_f1": model_result["test_f1"],
                "training_time_seconds": model_result["training_time_seconds"],
                "best_params": model_result["best_params"],
            }
            aggregated_results.append(summary_item)

        train_df.unpersist()
        val_df.unpersist()
        test_df.unpersist()

    print("\n" + "=" * 90)
    print("CRYPTO PRICE DIRECTION TRAINING RESULTS SUMMARY")
    print("=" * 90)
    print(format_results_table(aggregated_results))
    print("=" * 90 + "\n")

    return aggregated_results


def main() -> None:
    """CLI Entrypoint for training cryptocurrency ML models."""
    parser = argparse.ArgumentParser(description="Cryptocurrency MLlib Training Job")
    parser.add_argument(
        "--input",
        default=os.environ.get("CRYPTO_INPUT_PATH", "data/sample_klines.parquet"),
        help="Path to input klines parquet dataset",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols to include (default: all)",
    )
    parser.add_argument(
        "--horizons",
        default="5,15,60",
        help="Comma-separated forward horizons in minutes (default: 5,15,60)",
    )
    parser.add_argument(
        "--models",
        default="logreg,rf,gbt",
        help="Comma-separated model algorithms: logreg,rf,gbt (default: logreg,rf,gbt)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("CRYPTO_OUTPUT_DIR", "data/models"),
        help="Destination directory for persistent model artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("CRYPTO_SEED", "42")),
        help="Fixed random seed for deterministic training (default: 42)",
    )
    parser.add_argument(
        "--exclude-features",
        default=None,
        help="Comma-separated list of features to exclude (for ablations)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Optional deterministic subsampling fraction between 0.0 and 1.0",
    )
    parser.add_argument(
        "--conservative-purge",
        action="store_true",
        help="Use max(horizons) for all split embargoes",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast with 1 hyperparameter point per model",
    )

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    models = [m.strip().lower() for m in args.models.split(",")]
    exclude_features = [f.strip() for f in args.exclude_features.split(",")] if args.exclude_features else None

    run_training_job(
        input_path=args.input,
        symbols=symbols,
        horizons=horizons,
        models=models,
        output_dir=args.output_dir,
        seed=args.seed,
        exclude_features=exclude_features,
        sample_fraction=args.sample_fraction,
        conservative_purge=args.conservative_purge,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()

