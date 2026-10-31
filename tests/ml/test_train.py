"""
==============================================================================
tests/ml/test_train.py
Unit tests for Spark MLlib training with time-aware validation.

Validates:
  1. Scaler leakage prevention: StandardScaler fitted strictly on train split statistics.
  2. Leakage canary:
     (a) Planted learnable signal yields test AUC > 0.80.
     (b) Pure random noise label yields test AUC within 0.50 +/- 0.07.
  3. Model persistence & metadata completeness:
     - Saved PipelineModel reloads and produces identical predictions.
     - metadata.json contains all required keys.
     - test_predictions.parquet contains required columns.
  4. Balanced class weight calculation strictly on train split.
  5. Majority baseline accuracy computation.
  6. Multi-model execution (LogisticRegression, RandomForest, GBT).
==============================================================================
"""

import json
import os
import random
import sys
from typing import List

import numpy as np
import pytest
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# Ensure repo root is available
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spark.apps.ml.labels import MS_PER_MINUTE
from spark.apps.ml.pipeline import build_feature_stages
from spark.apps.ml.spark_session import get_spark_session
from spark.apps.ml.splits import split_time_series
from spark.apps.ml.train import (
    add_sample_weights,
    compute_balanced_weights,
    compute_majority_baseline,
    save_model_artifacts,
    train_model,
)


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Session-scoped SparkSession configured for fast local testing."""
    session = get_spark_session(
        app_name="TestTrainSuite",
        master="local[2]",
        log_level="ERROR",
    )
    return session


class TestTrainPipeline:
    """Comprehensive test suite for MLlib training pipeline."""

    def test_scaler_no_leakage(self, spark: SparkSession) -> None:
        """
        Verify StandardScaler mean/std match the TRAIN split, not the full dataset.

        Train rows have feature mean ~10.0; Val/Test rows have feature mean ~1000.0.
        Full data mean is ~300.0+.
        """
        base_time = 1_700_000_000_000
        rows = []
        for i in range(100):
            t = base_time + (i * MS_PER_MINUTE)
            # First 70 rows (train): values centered at 10.0
            # Last 30 rows (val/test): values centered at 1000.0
            val = 10.0 + (i * 0.1) if i < 70 else 1000.0 + (i * 0.1)
            lbl = float(i % 2)
            rows.append((t, "BTCUSDT", float(val), lbl))

        schema = StructType(
            [
                StructField("open_time", LongType(), False),
                StructField("symbol", StringType(), False),
                StructField("f1", DoubleType(), False),
                StructField("label", DoubleType(), False),
            ]
        )
        df = spark.createDataFrame(rows, schema=schema)

        train_df, val_df, test_df, _ = split_time_series(
            df=df,
            horizon_minutes=0,  # 0 purge for exact boundary alignment
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            label_col="label",
        )

        train_rows = train_df.select("f1").collect()
        train_mean = np.mean([r["f1"] for r in train_rows])
        full_mean = np.mean([r["f1"] for r in df.select("f1").collect()])

        # Build feature pipeline: VectorAssembler -> StandardScaler
        stages = build_feature_stages(feature_cols=["f1"], raw_features_col="raw_f", features_col="features")
        pipeline = Pipeline(stages=stages)

        # Fit strictly on train_df
        fitted_pipe = pipeline.fit(train_df)
        fitted_scaler = fitted_pipe.stages[1]  # StandardScalerModel

        scaler_mean = fitted_scaler.mean.toArray()[0]

        # Scaler mean must match train mean, NOT full data mean
        assert np.isclose(scaler_mean, train_mean, atol=1e-4)
        assert not np.isclose(scaler_mean, full_mean, atol=50.0)

    def test_leakage_canary_planted_signal(self, spark: SparkSession) -> None:
        """
        Canary part (a): Planted learnable signal must yield test AUC clearly above 0.80.
        """
        rng = random.Random(42)
        base_time = 1_700_000_000_000
        rows = []
        for i in range(300):
            t = base_time + (i * MS_PER_MINUTE)
            f1 = rng.gauss(0.0, 1.0)
            # Strong signal: label correlates with f1 sign
            lbl = 1.0 if f1 > 0.0 else 0.0
            rows.append((t, "BTCUSDT", f1, lbl))

        schema = StructType(
            [
                StructField("open_time", LongType(), False),
                StructField("symbol", StringType(), False),
                StructField("f1", DoubleType(), False),
                StructField("label", DoubleType(), False),
            ]
        )
        df = spark.createDataFrame(rows, schema=schema)

        train_df, val_df, test_df, _ = split_time_series(
            df=df,
            horizon_minutes=5,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            label_col="label",
        )

        weights = compute_balanced_weights(train_df, label_col="label")
        train_df = add_sample_weights(train_df, weights)
        val_df = add_sample_weights(val_df, weights)
        test_df = add_sample_weights(test_df, weights)

        res = train_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name="logreg",
            feature_cols=["f1"],
            param_grid=[{"regParam": 0.01, "elasticNetParam": 0.0}],
            seed=42,
        )

        test_auc = res["test_auc"]
        # Planted signal must achieve high test AUC (> 0.80)
        assert test_auc > 0.80, f"Planted signal failed: test AUC={test_auc:.4f} <= 0.80"

    def test_leakage_canary_pure_noise(self, spark: SparkSession) -> None:
        """
        Canary part (b): Pure random noise label must yield test AUC within 0.50 +/- 0.07.
        If noise scores high, the pipeline leaks future or test data.
        """
        rng = random.Random(42)
        base_time = 1_700_000_000_000
        rows = []
        for i in range(600):
            t = base_time + (i * MS_PER_MINUTE)
            f1 = rng.gauss(0.0, 1.0)
            f2 = rng.gauss(0.0, 1.0)
            # Pure independent Bernoulli noise
            lbl = float(rng.choice([0, 1]))
            rows.append((t, "BTCUSDT", f1, f2, lbl))

        schema = StructType(
            [
                StructField("open_time", LongType(), False),
                StructField("symbol", StringType(), False),
                StructField("f1", DoubleType(), False),
                StructField("f2", DoubleType(), False),
                StructField("label", DoubleType(), False),
            ]
        )
        df = spark.createDataFrame(rows, schema=schema)

        train_df, val_df, test_df, _ = split_time_series(
            df=df,
            horizon_minutes=5,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            label_col="label",
        )

        weights = compute_balanced_weights(train_df, label_col="label")
        train_df = add_sample_weights(train_df, weights)
        val_df = add_sample_weights(val_df, weights)
        test_df = add_sample_weights(test_df, weights)

        res = train_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name="logreg",
            feature_cols=["f1", "f2"],
            param_grid=[{"regParam": 0.1, "elasticNetParam": 0.0}],
            seed=42,
        )

        test_auc = res["test_auc"]
        # Honest un-leaked evaluation on pure noise must be within 0.50 +/- 0.07
        assert (
            0.43 <= test_auc <= 0.57
        ), f"Leakage canary triggered! Pure noise produced non-chance AUC={test_auc:.4f} (expected 0.43..0.57)"

    def test_majority_baseline_computation(self, spark: SparkSession) -> None:
        """Verify majority-class baseline accuracy calculated from train split."""
        # Train: 70 ones, 30 zeros -> majority class is 1.0
        train_data = [(1.0,)] * 70 + [(0.0,)] * 30
        train_df = spark.createDataFrame(train_data, ["label"])

        # Test: 40 ones, 60 zeros -> accuracy of always predicting 1.0 on test is 40/100 = 0.40
        test_data = [(1.0,)] * 40 + [(0.0,)] * 60
        test_df = spark.createDataFrame(test_data, ["label"])

        majority_class, baseline_acc = compute_majority_baseline(train_df, test_df, label_col="label")
        assert majority_class == 1.0
        assert np.isclose(baseline_acc, 0.40)

    def test_model_persistence_and_metadata(self, spark: SparkSession, tmp_path) -> None:
        """
        Verify saved model reloads with identical predictions and metadata.json contains all keys.
        """
        rng = random.Random(123)
        base_time = 1_700_000_000_000
        rows = []
        for i in range(150):
            t = base_time + (i * MS_PER_MINUTE)
            f1 = float(i * 0.1)
            lbl = float(i % 2)
            close = 50000.0 + i
            fut_ret = 0.002 * (1.0 if lbl == 1.0 else -1.0)
            rows.append((t, "BTCUSDT", f1, close, fut_ret, lbl))

        schema = StructType(
            [
                StructField("open_time", LongType(), False),
                StructField("symbol", StringType(), False),
                StructField("f1", DoubleType(), False),
                StructField("close", DoubleType(), False),
                StructField("future_return_5m", DoubleType(), False),
                StructField("label", DoubleType(), False),
            ]
        )
        df = spark.createDataFrame(rows, schema=schema)

        train_df, val_df, test_df, split_info = split_time_series(
            df=df,
            horizon_minutes=5,
            label_col="label",
        )

        weights = compute_balanced_weights(train_df)
        train_df = add_sample_weights(train_df, weights)
        val_df = add_sample_weights(val_df, weights)
        test_df = add_sample_weights(test_df, weights)

        res = train_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name="logreg",
            feature_cols=["f1"],
            param_grid=[{"regParam": 0.05, "elasticNetParam": 0.0}],
            seed=42,
        )

        out_dir = str(tmp_path / "models")
        meta = {
            "feature_list": ["f1"],
            "horizon": 5,
            "model_name": "logreg",
            "split_boundaries_and_row_counts": split_info,
            "purge_setting": {"horizon_minutes": 5, "window_ms": 300000},
            "hyperparameters": res["best_params"],
            "metrics": {
                "val_auc": res["best_val_auc"],
                "test_auc": res["test_auc"],
                "test_accuracy": res["test_accuracy"],
                "test_f1": res["test_f1"],
                "majority_baseline_accuracy": res["majority_baseline_accuracy"],
                "training_time_seconds": res["training_time_seconds"],
            },
            "seed": 42,
            "git_commit_hash": "test_hash_123",
            "pyspark_version": str(spark.version),
        }

        saved_path = save_model_artifacts(
            output_dir=out_dir,
            horizon=5,
            model_name="logreg",
            fitted_pipeline=res["fitted_pipeline"],
            test_predictions=res["test_predictions"],
            metadata=meta,
            feature_weights=res["feature_importance"],
        )

        # 1. Check metadata.json
        meta_file = os.path.join(saved_path, "metadata.json")
        assert os.path.exists(meta_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            loaded_meta = json.load(f)

        required_keys = [
            "prediction_schema_version",
            "feature_list",
            "horizon",
            "model_name",
            "split_boundaries_and_row_counts",
            "purge_setting",
            "hyperparameters",
            "metrics",
            "seed",
            "git_commit_hash",
            "pyspark_version",
        ]
        for k in required_keys:
            assert k in loaded_meta, f"Missing required metadata key: {k}"
        assert loaded_meta["prediction_schema_version"] == "1.0.0"

        # 2. Check test_predictions.parquet
        pred_file = os.path.join(saved_path, "test_predictions.parquet")
        assert os.path.exists(pred_file)
        pred_df = spark.read.parquet(pred_file)
        expected_cols = [
            "symbol",
            "open_time",
            "open_time_dt",
            "horizon_min",
            "model_name",
            "label",
            "prediction",
            "p_up",
            "current_close",
            "future_return",
        ]
        assert pred_df.columns == expected_cols
        assert pred_df.count() == test_df.count()

        # 3. Reload PipelineModel and verify predictions match exactly
        pipe_path = os.path.join(saved_path, "pipeline_model")
        reloaded_model = PipelineModel.load(pipe_path)
        reloaded_preds = reloaded_model.transform(test_df)

        orig_preds_list = [r["prediction"] for r in res["test_predictions"].select("prediction").collect()]
        reloaded_preds_list = [r["prediction"] for r in reloaded_preds.select("prediction").collect()]
        assert orig_preds_list == reloaded_preds_list

    def test_prediction_schema_contract_and_p_up_range(self, spark: SparkSession, tmp_path) -> None:
        """
        Verify the saved test-prediction Parquet matches the common model-agnostic schema exactly:
          1. Columns: symbol, open_time, open_time_dt, horizon_min, model_name, label, prediction,
             p_up, current_close, future_return.
          2. No Vector columns exist in the output.
          3. p_up is bounded in [0.0, 1.0].
          4. future_return is strictly NOT NULL for scored rows.
        """
        base_time = 1_700_000_000_000
        rows = []
        for i in range(120):
            t = base_time + (i * MS_PER_MINUTE)
            f1 = float(i % 5)
            lbl = float(i % 2)
            close = 60000.0 + (i * 10)
            fut_ret = 0.0015 * (1.0 if lbl == 1.0 else -1.0)
            rows.append((t, "ETHUSDT", f1, close, fut_ret, lbl))

        schema = StructType(
            [
                StructField("open_time", LongType(), False),
                StructField("symbol", StringType(), False),
                StructField("f1", DoubleType(), False),
                StructField("close", DoubleType(), False),
                StructField("future_return_15m", DoubleType(), False),
                StructField("label", DoubleType(), False),
            ]
        )
        df = spark.createDataFrame(rows, schema=schema)
        train_df, val_df, test_df, split_info = split_time_series(df=df, horizon_minutes=15, label_col="label")

        weights = compute_balanced_weights(train_df)
        train_df = add_sample_weights(train_df, weights)
        val_df = add_sample_weights(val_df, weights)
        test_df = add_sample_weights(test_df, weights)

        res = train_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name="rf",
            feature_cols=["f1"],
            param_grid=[{"numTrees": 5, "maxDepth": 3}],
            seed=42,
        )

        out_dir = str(tmp_path / "schema_check")
        saved_path = save_model_artifacts(
            output_dir=out_dir,
            horizon=15,
            model_name="rf",
            fitted_pipeline=res["fitted_pipeline"],
            test_predictions=res["test_predictions"],
            metadata={
                "prediction_schema_version": "1.0.0",
                "feature_list": ["f1"],
                "horizon": 15,
                "model_name": "rf",
            },
            feature_weights=res["feature_importance"],
        )

        pred_file = os.path.join(saved_path, "test_predictions.parquet")
        assert os.path.exists(pred_file)
        pred_df = spark.read.parquet(pred_file)

        # 1. Exact columns match
        expected_cols = [
            "symbol",
            "open_time",
            "open_time_dt",
            "horizon_min",
            "model_name",
            "label",
            "prediction",
            "p_up",
            "current_close",
            "future_return",
        ]
        assert pred_df.columns == expected_cols

        # 2. No vector columns
        for field in pred_df.schema.fields:
            type_str = field.dataType.simpleString()
            assert "vector" not in type_str.lower(), f"Vector column detected in output: {field.name}"

        # 3. p_up in [0.0, 1.0]
        p_up_stats = pred_df.select(F.min("p_up").alias("min_p"), F.max("p_up").alias("max_p")).collect()[0]
        assert p_up_stats["min_p"] >= 0.0, f"p_up min is below 0: {p_up_stats['min_p']}"
        assert p_up_stats["max_p"] <= 1.0, f"p_up max is above 1: {p_up_stats['max_p']}"

        # 4. future_return is strictly NOT NULL
        null_returns = pred_df.filter(F.col("future_return").isNull()).count()
        assert null_returns == 0, f"Found {null_returns} null future_return rows in scored predictions"

    def test_tree_models_fast_execution(self, spark: SparkSession) -> None:
        """Verify RandomForest and GBT models train cleanly without failure."""
        rng = random.Random(99)
        base_time = 1_700_000_000_000
        rows = []
        for i in range(120):
            t = base_time + (i * MS_PER_MINUTE)
            f1 = rng.gauss(0.0, 1.0)
            f2 = rng.gauss(0.0, 1.0)
            lbl = float(i % 2)
            rows.append((t, "BTCUSDT", f1, f2, lbl))

        schema = StructType(
            [
                StructField("open_time", LongType(), False),
                StructField("symbol", StringType(), False),
                StructField("f1", DoubleType(), False),
                StructField("f2", DoubleType(), False),
                StructField("label", DoubleType(), False),
            ]
        )
        df = spark.createDataFrame(rows, schema=schema)

        train_df, val_df, test_df, _ = split_time_series(df=df, horizon_minutes=5, label_col="label")
        weights = compute_balanced_weights(train_df)
        train_df = add_sample_weights(train_df, weights)
        val_df = add_sample_weights(val_df, weights)
        test_df = add_sample_weights(test_df, weights)

        # Test RandomForest
        rf_res = train_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name="rf",
            feature_cols=["f1", "f2"],
            param_grid=[{"numTrees": 5, "maxDepth": 3}],
            seed=42,
        )
        assert "test_auc" in rf_res
        assert "f1" in rf_res["feature_importance"]

        # Test GBT
        gbt_res = train_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name="gbt",
            feature_cols=["f1", "f2"],
            param_grid=[{"maxIter": 5, "maxDepth": 3, "stepSize": 0.1}],
            seed=42,
        )
        assert "test_auc" in gbt_res
        assert "f1" in gbt_res["feature_importance"]

