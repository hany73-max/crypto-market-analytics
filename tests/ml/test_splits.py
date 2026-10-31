"""
==============================================================================
tests/ml/test_splits.py
Unit tests for chronological splitting with label purge and embargo.

Validates:
  1. Strict chronological boundary order (min_time < val_start < test_start).
  2. Purge / embargo window isolation (no train row reaches val_start;
     no val row reaches test_start).
  3. Zero row overlap across train, val, and test sets.
  4. Shared global boundaries across multiple cryptocurrency symbols.
  5. Strict null exclusion (null labels and null features are dropped, never cast).
  6. Error handling for invalid split configurations.
==============================================================================
"""

import os
import sys
from typing import List, Tuple

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
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
from spark.apps.ml.spark_session import get_spark_session
from spark.apps.ml.splits import (
    compute_time_boundaries,
    split_chronological,
    split_time_series,
)


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Session-scoped SparkSession configured for fast local testing."""
    session = get_spark_session(
        app_name="TestSplitsSuite",
        master="local[2]",
        log_level="ERROR",
    )
    return session


def make_synthetic_klines(
    spark: SparkSession,
    symbols: List[str] = ["BTCUSDT"],
    num_minutes: int = 100,
    base_time: int = 1_700_000_000_000,
    with_labels: bool = True,
    with_features: bool = True,
) -> DataFrame:
    """Helper to generate deterministic synthetic klines for splitting tests."""
    rows = []
    for sym in symbols:
        for i in range(num_minutes):
            t = base_time + (i * MS_PER_MINUTE)
            close = 100.0 + (i * 0.1)
            label = int(i % 2) if with_labels else None
            f1 = float(i * 0.01) if with_features else None
            f2 = float(i * 0.02) if with_features else None
            rows.append((t, sym, close, label, f1, f2))

    schema = StructType(
        [
            StructField("open_time", LongType(), nullable=False),
            StructField("symbol", StringType(), nullable=False),
            StructField("close", DoubleType(), nullable=False),
            StructField("label", IntegerType(), nullable=True),
            StructField("feature_1", DoubleType(), nullable=True),
            StructField("feature_2", DoubleType(), nullable=True),
        ]
    )
    return spark.createDataFrame(rows, schema=schema)


class TestTimeSplits:
    """Test suite for chronological splitting and purge behavior."""

    def test_boundaries_chronological(self, spark: SparkSession) -> None:
        """Verify time boundaries are strictly chronological and proportional."""
        df = make_synthetic_klines(spark, symbols=["BTCUSDT"], num_minutes=100)
        val_start, test_start = compute_time_boundaries(df, train_ratio=0.70, val_ratio=0.15)

        min_time = df.agg(F.min("open_time")).collect()[0][0]
        max_time = df.agg(F.max("open_time")).collect()[0][0]

        assert min_time < val_start < test_start <= max_time
        # In a 100-minute linear sequence, 70% cutoff is at index ~70 and 85% at ~85
        expected_val = min_time + (69 * MS_PER_MINUTE)
        expected_test = min_time + (84 * MS_PER_MINUTE)
        assert abs(val_start - expected_val) <= 2 * MS_PER_MINUTE
        assert abs(test_start - expected_test) <= 2 * MS_PER_MINUTE

    def test_purge_embargo_isolation(self, spark: SparkSession) -> None:
        """
        Verify purge removes any row whose forward label window reaches the next partition.

        Train: open_time + h * 60,000 < val_start
        Val:   open_time + h * 60,000 < test_start
        """
        h = 5  # 5 minutes horizon
        df = make_synthetic_klines(spark, symbols=["BTCUSDT"], num_minutes=100)

        train_df, val_df, test_df, info = split_time_series(
            df=df,
            horizon_minutes=h,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        val_start = info["val_start"]
        test_start = info["test_start"]
        purge_ms = h * MS_PER_MINUTE

        # Train rows forward window must not reach val_start
        train_max_open = train_df.agg(F.max("open_time")).collect()[0][0]
        assert train_max_open + purge_ms < val_start
        assert train_max_open < info["train_end"]

        # Val rows forward window must not reach test_start
        val_min_open = val_df.agg(F.min("open_time")).collect()[0][0]
        val_max_open = val_df.agg(F.max("open_time")).collect()[0][0]
        assert val_min_open >= val_start
        assert val_max_open + purge_ms < test_start
        assert val_max_open < info["val_end"]

        # Test rows must start at or after test_start
        test_min_open = test_df.agg(F.min("open_time")).collect()[0][0]
        assert test_min_open >= test_start

        # Check embargo gap
        assert (val_start - train_max_open) > purge_ms
        assert (test_start - val_max_open) > purge_ms

    def test_no_row_overlap(self, spark: SparkSession) -> None:
        """Verify no row appears in multiple splits."""
        df = make_synthetic_klines(spark, symbols=["BTCUSDT"], num_minutes=100)

        train_df, val_df, test_df, _ = split_time_series(df=df, horizon_minutes=15)

        # Cross joins/inner joins on open_time must be completely empty
        train_val_overlap = train_df.join(val_df, on=["symbol", "open_time"]).count()
        train_test_overlap = train_df.join(test_df, on=["symbol", "open_time"]).count()
        val_test_overlap = val_df.join(test_df, on=["symbol", "open_time"]).count()

        assert train_val_overlap == 0
        assert train_test_overlap == 0
        assert val_test_overlap == 0

    def test_multi_symbol_shared_boundaries(self, spark: SparkSession) -> None:
        """Verify multiple symbols share identical global time boundaries."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        df = make_synthetic_klines(spark, symbols=symbols, num_minutes=120)

        train_df, val_df, test_df, info = split_time_series(
            df=df,
            horizon_minutes=10,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        # Both symbols must be present in all 3 splits
        for s in symbols:
            s_train = train_df.filter(F.col("symbol") == s)
            s_val = val_df.filter(F.col("symbol") == s)
            s_test = test_df.filter(F.col("symbol") == s)

            assert s_train.count() > 0
            assert s_val.count() > 0
            assert s_test.count() > 0

            # Each symbol's max open_time in train must respect the global boundary
            s_train_max = s_train.agg(F.max("open_time")).collect()[0][0]
            assert s_train_max < info["train_end"]

            s_val_min = s_val.agg(F.min("open_time")).collect()[0][0]
            s_val_max = s_val.agg(F.max("open_time")).collect()[0][0]
            assert s_val_min >= info["val_start"]
            assert s_val_max < info["val_end"]

            s_test_min = s_test.agg(F.min("open_time")).collect()[0][0]
            assert s_test_min >= info["test_start"]

    def test_null_handling_exclusion(self, spark: SparkSession) -> None:
        """
        Verify rows with null label or null feature are excluded and never become class 0.
        """
        base_time = 1_700_000_000_000
        # Create dataset with 30 rows:
        # Rows 0-9: label 1, feature 1.0
        # Rows 10-19: label 0, feature 2.0
        # Rows 20-24: label NULL, feature 3.0
        # Rows 25-29: label 1, feature NULL
        rows = []
        for i in range(30):
            t = base_time + (i * MS_PER_MINUTE)
            if i < 10:
                lbl, f1 = 1, 1.0
            elif i < 20:
                lbl, f1 = 0, 2.0
            elif i < 25:
                lbl, f1 = None, 3.0
            else:
                lbl, f1 = 1, None
            rows.append((t, "BTCUSDT", 100.0, lbl, f1))

        schema = StructType(
            [
                StructField("open_time", LongType(), nullable=False),
                StructField("symbol", StringType(), nullable=False),
                StructField("close", DoubleType(), nullable=False),
                StructField("label", IntegerType(), nullable=True),
                StructField("f1", DoubleType(), nullable=True),
            ]
        )
        df = spark.createDataFrame(rows, schema=schema)

        train_df, val_df, test_df, info = split_time_series(
            df=df,
            horizon_minutes=0,  # no purge to test pure null filtering
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            label_col="label",
            feature_cols=["f1"],
            drop_nulls=True,
        )

        all_splits = train_df.union(val_df).union(test_df)

        # Total surviving rows must be exactly 20 (10 of class 1, 10 of class 0)
        assert all_splits.count() == 20
        # Zero null labels
        assert all_splits.filter(F.col("label").isNull()).count() == 0
        # Zero null features
        assert all_splits.filter(F.col("f1").isNull()).count() == 0

        # Class counts: exactly 10 class 1 and 10 class 0 (nulls did not become 0)
        class_0_count = all_splits.filter(F.col("label") == 0).count()
        class_1_count = all_splits.filter(F.col("label") == 1).count()
        assert class_0_count == 10
        assert class_1_count == 10

    def test_invalid_configurations(self, spark: SparkSession) -> None:
        """Verify descriptive ValueErrors on invalid inputs."""
        df = make_synthetic_klines(spark, num_minutes=50)

        # Ratios do not sum to 1.0
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            split_time_series(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)

        # Negative horizon
        with pytest.raises(ValueError, match="horizon_minutes must be non-negative"):
            split_time_series(df, horizon_minutes=-5)

        # Missing time column
        df_no_time = df.drop("open_time")
        with pytest.raises(ValueError, match="Timestamp column 'open_time' not found"):
            split_time_series(df_no_time)

