"""
==============================================================================
tests/ml/test_labels.py
Unit tests for gap-safe forward return and target direction labeling.

Validates:
  - Normal continuous klines (binary targets & return formulas).
  - Missing-minute gaps in the middle (labels crossing gaps must be NULL).
  - Symbol partition boundaries (strict isolation, no cross-symbol leakage).
  - Explicit last-h rows behavior (null-out vs drop).
  - Three-class direction labeling via return threshold (epsilon).
  - Configurable horizons (default (5, 15, 60) and custom lookaheads).
  - Zero dependency on wall-clock time columns.
  - Class balance computation functionality.
==============================================================================
"""

import math
import os
import sys
from typing import List, Tuple

import pytest
from pyspark.sql import DataFrame, SparkSession
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

from spark.apps.ml.labels import (
    BINARY_DOWN_OR_FLAT,
    BINARY_UP,
    DEFAULT_HORIZONS,
    MS_PER_MINUTE,
    THREE_CLASS_DOWN,
    THREE_CLASS_FLAT,
    THREE_CLASS_UP,
    add_labels,
    compute_label_balance,
    format_balance_table,
)
from spark.apps.ml.spark_session import get_spark_session


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Session-scoped SparkSession configured for fast local testing."""
    session = get_spark_session(
        app_name="TestLabelsSuite",
        master="local[2]",
        log_level="ERROR",
    )
    yield session
    session.stop()


def make_kline_df(
    spark: SparkSession,
    rows: List[Tuple[int, str, float]],
) -> DataFrame:
    """
    Helper to construct minimal synthetic kline DataFrames.
    rows: List of (open_time_ms, symbol, close_price)
    """
    schema = StructType([
        StructField("open_time", LongType(), False),
        StructField("symbol", StringType(), False),
        StructField("close", DoubleType(), False),
    ])
    return spark.createDataFrame(rows, schema=schema)


class TestLabelsPipeline:
    """Test suite for gap-safe label generation."""

    def test_normal_case_binary(self, spark: SparkSession):
        """Verify standard continuous sequence: returns and binary direction (1 if >0 else 0)."""
        # 6 consecutive minutes, prices: 100, 105, 102, 110, 108, 120
        # For h=2:
        # i=0 (t=0m, p=100) -> future is i=2 (t=2m, p=102) -> ret = (102-100)/100 = +0.02 -> UP (1)
        # i=1 (t=1m, p=105) -> future is i=3 (t=3m, p=110) -> ret = (110-105)/105 = +0.047619 -> UP (1)
        # i=2 (t=2m, p=102) -> future is i=4 (t=4m, p=108) -> ret = (108-102)/102 = +0.058823 -> UP (1)
        # i=3 (t=3m, p=110) -> future is i=5 (t=5m, p=120) -> ret = (120-110)/110 = +0.090909 -> UP (1)
        # i=4 (t=4m, p=108) -> last 2 rows -> NULL
        # i=5 (t=5m, p=120) -> last 2 rows -> NULL
        base_time = 1_700_000_000_000
        rows = [
            (base_time + 0 * MS_PER_MINUTE, "BTCUSDT", 100.0),
            (base_time + 1 * MS_PER_MINUTE, "BTCUSDT", 105.0),
            (base_time + 2 * MS_PER_MINUTE, "BTCUSDT", 102.0),
            (base_time + 3 * MS_PER_MINUTE, "BTCUSDT", 110.0),
            (base_time + 4 * MS_PER_MINUTE, "BTCUSDT", 108.0),
            (base_time + 5 * MS_PER_MINUTE, "BTCUSDT", 120.0),
        ]
        df = make_kline_df(spark, rows)
        labeled = add_labels(df, horizons=[2], drop_null_labels=False)

        res = labeled.orderBy("open_time").collect()
        assert len(res) == 6

        # Row 0: +0.02 -> 1
        assert math.isclose(res[0]["future_return_2m"], 0.02, rel_tol=1e-5)
        assert res[0]["target_direction_2m"] == BINARY_UP

        # Row 1: (110-105)/105 -> 1
        assert math.isclose(res[1]["future_return_2m"], (110.0 - 105.0) / 105.0, rel_tol=1e-5)
        assert res[1]["target_direction_2m"] == BINARY_UP

        # Negative return check: let's add a negative return row
        rows_down = [
            (base_time + 0 * MS_PER_MINUTE, "BTCUSDT", 100.0),
            (base_time + 1 * MS_PER_MINUTE, "BTCUSDT", 95.0),
            (base_time + 2 * MS_PER_MINUTE, "BTCUSDT", 90.0),  # ret from 0 to 2 is -0.10
        ]
        df_down = make_kline_df(spark, rows_down)
        labeled_down = add_labels(df_down, horizons=[2], drop_null_labels=False).orderBy("open_time").collect()
        assert math.isclose(labeled_down[0]["future_return_2m"], -0.10, rel_tol=1e-5)
        assert labeled_down[0]["target_direction_2m"] == BINARY_DOWN_OR_FLAT

        # Zero return check (flat is class 0 in binary mode)
        rows_flat = [
            (base_time + 0 * MS_PER_MINUTE, "BTCUSDT", 100.0),
            (base_time + 1 * MS_PER_MINUTE, "BTCUSDT", 105.0),
            (base_time + 2 * MS_PER_MINUTE, "BTCUSDT", 100.0),  # ret from 0 to 2 is 0.0
        ]
        df_flat = make_kline_df(spark, rows_flat)
        labeled_flat = add_labels(df_flat, horizons=[2], drop_null_labels=False).orderBy("open_time").collect()
        assert math.isclose(labeled_flat[0]["future_return_2m"], 0.0, abs_tol=1e-5)
        assert labeled_flat[0]["target_direction_2m"] == BINARY_DOWN_OR_FLAT

    def test_missing_minute_gap_in_middle(self, spark: SparkSession):
        """
        Verify that labels crossing a missing-minute gap evaluate to NULL.
        Timestamps: 0, 1, 2, 4, 5, 6, 7 (Minute 3 is MISSING!).
        For horizon h=2:
          - row 0 (t=0m): lead(2) is row 2 (t=2m). Diff=2m == 2*60s -> VALID.
          - row 1 (t=1m): lead(2) is row 3 (t=4m). Diff=3m != 2*60s -> CROSSES GAP -> MUST BE NULL!
          - row 2 (t=2m): lead(2) is row 4 (t=5m). Diff=3m != 2*60s -> CROSSES GAP -> MUST BE NULL!
          - row 3 (t=4m): lead(2) is row 5 (t=6m). Diff=2m == 2*60s -> VALID.
          - row 4 (t=5m): lead(2) is row 6 (t=7m). Diff=2m == 2*60s -> VALID.
          - row 5 (t=6m): last 2 rows -> MUST BE NULL!
          - row 6 (t=7m): last 2 rows -> MUST BE NULL!
        """
        base_time = 1_700_000_000_000
        minutes = [0, 1, 2, 4, 5, 6, 7]  # minute 3 is skipped
        prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0]
        rows = [
            (base_time + m * MS_PER_MINUTE, "BTCUSDT", p)
            for m, p in zip(minutes, prices)
        ]

        df = make_kline_df(spark, rows)
        labeled = add_labels(df, horizons=[2], drop_null_labels=False).orderBy("open_time").collect()

        assert len(labeled) == 7

        # Row 0: valid (diff = 2m)
        assert labeled[0]["future_return_2m"] is not None
        assert labeled[0]["target_direction_2m"] is not None
        assert math.isclose(labeled[0]["future_return_2m"], (104.0 - 100.0) / 100.0, rel_tol=1e-5)

        # Row 1: crosses gap (t=1m to t=4m, diff=3m) -> NULL
        assert labeled[1]["future_return_2m"] is None
        assert labeled[1]["target_direction_2m"] is None

        # Row 2: crosses gap (t=2m to t=5m, diff=3m) -> NULL
        assert labeled[2]["future_return_2m"] is None
        assert labeled[2]["target_direction_2m"] is None

        # Row 3: valid (t=4m to t=6m, diff=2m)
        assert labeled[3]["future_return_2m"] is not None
        assert labeled[3]["target_direction_2m"] is not None
        assert math.isclose(labeled[3]["future_return_2m"], (110.0 - 106.0) / 106.0, rel_tol=1e-5)

        # Row 4: valid (t=5m to t=7m, diff=2m)
        assert labeled[4]["future_return_2m"] is not None
        assert labeled[4]["target_direction_2m"] is not None
        assert math.isclose(labeled[4]["future_return_2m"], (112.0 - 108.0) / 108.0, rel_tol=1e-5)

        # Row 5 & 6: last-h rows -> NULL
        assert labeled[5]["future_return_2m"] is None
        assert labeled[5]["target_direction_2m"] is None
        assert labeled[6]["future_return_2m"] is None
        assert labeled[6]["target_direction_2m"] is None

    def test_symbol_boundaries_no_leakage(self, spark: SparkSession):
        """
        Verify that window lead() strictly partitions by symbol.
        The last rows of BTCUSDT must NEVER leak into ETHUSDT.
        """
        base_time = 1_700_000_000_000
        # 4 rows of BTC, 4 rows of ETH, h=2
        rows = [
            # BTC rows
            (base_time + 0 * MS_PER_MINUTE, "BTCUSDT", 100.0),
            (base_time + 1 * MS_PER_MINUTE, "BTCUSDT", 110.0),
            (base_time + 2 * MS_PER_MINUTE, "BTCUSDT", 120.0),
            (base_time + 3 * MS_PER_MINUTE, "BTCUSDT", 130.0),
            # ETH rows
            (base_time + 0 * MS_PER_MINUTE, "ETHUSDT", 2000.0),
            (base_time + 1 * MS_PER_MINUTE, "ETHUSDT", 2100.0),
            (base_time + 2 * MS_PER_MINUTE, "ETHUSDT", 2200.0),
            (base_time + 3 * MS_PER_MINUTE, "ETHUSDT", 2300.0),
        ]

        df = make_kline_df(spark, rows)
        labeled = add_labels(df, horizons=[2], drop_null_labels=False)

        btc_res = labeled.filter("symbol = 'BTCUSDT'").orderBy("open_time").collect()
        eth_res = labeled.filter("symbol = 'ETHUSDT'").orderBy("open_time").collect()

        assert len(btc_res) == 4
        assert len(eth_res) == 4

        # BTC: rows 0 and 1 have valid futures within BTC
        assert math.isclose(btc_res[0]["future_return_2m"], (120.0 - 100.0) / 100.0, rel_tol=1e-5)
        assert math.isclose(btc_res[1]["future_return_2m"], (130.0 - 110.0) / 110.0, rel_tol=1e-5)

        # BTC: rows 2 and 3 are the last 2 rows of BTC. They MUST be NULL and not leak into ETH prices!
        assert btc_res[2]["future_return_2m"] is None
        assert btc_res[2]["target_direction_2m"] is None
        assert btc_res[3]["future_return_2m"] is None
        assert btc_res[3]["target_direction_2m"] is None

        # ETH: rows 0 and 1 have valid futures within ETH
        assert math.isclose(eth_res[0]["future_return_2m"], (2200.0 - 2000.0) / 2000.0, rel_tol=1e-5)
        assert math.isclose(eth_res[1]["future_return_2m"], (2300.0 - 2100.0) / 2100.0, rel_tol=1e-5)

        # ETH: rows 2 and 3 are last 2 rows of ETH -> NULL
        assert eth_res[2]["future_return_2m"] is None
        assert eth_res[2]["target_direction_2m"] is None
        assert eth_res[3]["future_return_2m"] is None
        assert eth_res[3]["target_direction_2m"] is None

    def test_last_h_rows_null_vs_drop(self, spark: SparkSession):
        """Verify behavior for drop_null_labels=False (null-out) vs drop_null_labels=True (drop)."""
        base_time = 1_700_000_000_000
        # 10 continuous rows, horizon=3
        rows = [
            (base_time + i * MS_PER_MINUTE, "BTCUSDT", 100.0 + i)
            for i in range(10)
        ]
        df = make_kline_df(spark, rows)

        # Case 1: drop_null_labels=False (default: preserve all rows, null out last h)
        df_nulls = add_labels(df, horizons=[3], drop_null_labels=False)
        res_nulls = df_nulls.orderBy("open_time").collect()
        assert len(res_nulls) == 10
        # First 7 rows are not null
        for i in range(7):
            assert res_nulls[i]["future_return_3m"] is not None
            assert res_nulls[i]["target_direction_3m"] is not None
        # Last 3 rows are null
        for i in range(7, 10):
            assert res_nulls[i]["future_return_3m"] is None
            assert res_nulls[i]["target_direction_3m"] is None

        # Case 2: drop_null_labels=True (drops all rows where any target is null)
        df_dropped = add_labels(df, horizons=[3], drop_null_labels=True)
        res_dropped = df_dropped.orderBy("open_time").collect()
        assert len(res_dropped) == 7
        for r in res_dropped:
            assert r["future_return_3m"] is not None
            assert r["target_direction_3m"] is not None

    def test_epsilon_three_class_behavior(self, spark: SparkSession):
        """
        Verify 3-class target behavior with return threshold epsilon.
          - return > epsilon: class 2 (UP)
          - return < -epsilon: class 0 (DOWN)
          - -epsilon <= return <= epsilon: class 1 (FLAT)
        """
        base_time = 1_700_000_000_000
        epsilon = 0.01  # 1% threshold
        # For h=1:
        # i=0: 100 -> 102: ret = +0.02 (> 0.01) -> UP (2)
        # i=1: 102 -> 102.5: ret = +0.0049 (<= 0.01) -> FLAT (1)
        # i=2: 102.5 -> 102.5: ret = 0.0000 -> FLAT (1)
        # i=3: 102.5 -> 102.0: ret = -0.00488 (> -0.01) -> FLAT (1)
        # i=4: 102.0 -> 99.0: ret = -0.0294 (< -0.01) -> DOWN (0)
        # i=5: 99.0 (last row, h=1) -> NULL
        prices = [100.0, 102.0, 102.5, 102.5, 102.0, 99.0]
        rows = [
            (base_time + i * MS_PER_MINUTE, "BTCUSDT", prices[i])
            for i in range(len(prices))
        ]
        df = make_kline_df(spark, rows)
        labeled = add_labels(df, horizons=[1], epsilon=epsilon, drop_null_labels=False).orderBy("open_time").collect()

        assert len(labeled) == 6
        assert labeled[0]["target_direction_1m"] == THREE_CLASS_UP       # +2.0%
        assert labeled[1]["target_direction_1m"] == THREE_CLASS_FLAT     # +0.49%
        assert labeled[2]["target_direction_1m"] == THREE_CLASS_FLAT     # 0.0%
        assert labeled[3]["target_direction_1m"] == THREE_CLASS_FLAT     # -0.49%
        assert labeled[4]["target_direction_1m"] == THREE_CLASS_DOWN     # -2.94%
        assert labeled[5]["target_direction_1m"] is None                 # Last row

    def test_horizon_config(self, spark: SparkSession):
        """Verify configurable horizons: custom list, default list, and input validation."""
        base_time = 1_700_000_000_000
        rows = [
            (base_time + i * MS_PER_MINUTE, "BTCUSDT", 100.0 + i)
            for i in range(70)
        ]
        df = make_kline_df(spark, rows)

        # Default horizons: (5, 15, 60)
        labeled_default = add_labels(df)
        for h in DEFAULT_HORIZONS:
            assert f"future_return_{h}m" in labeled_default.columns
            assert f"target_direction_{h}m" in labeled_default.columns

        # Custom horizons: [1, 3, 10]
        labeled_custom = add_labels(df, horizons=[1, 3, 10])
        for h in [1, 3, 10]:
            assert f"future_return_{h}m" in labeled_custom.columns
            assert f"target_direction_{h}m" in labeled_custom.columns
        assert "future_return_5m" not in labeled_custom.columns

        # Invalid horizons validation
        with pytest.raises(ValueError, match="positive integer"):
            add_labels(df, horizons=[0])

        with pytest.raises(ValueError, match="positive integer"):
            add_labels(df, horizons=[-5])

        with pytest.raises(ValueError, match="at least one"):
            add_labels(df, horizons=[])

        # Invalid epsilon validation
        with pytest.raises(ValueError, match="cannot be negative"):
            add_labels(df, epsilon=-0.01)

    def test_no_wall_clock_dependency(self, spark: SparkSession):
        """Verify that missing wall-clock columns (ingested_at, kline_date, etc.) do not affect labels."""
        base_time = 1_700_000_000_000
        # DataFrame strictly contains ONLY open_time, symbol, close
        rows = [
            (base_time + i * MS_PER_MINUTE, "BTCUSDT", 100.0 + (i % 3))
            for i in range(10)
        ]
        df = make_kline_df(spark, rows)
        assert set(df.columns) == {"open_time", "symbol", "close"}

        labeled = add_labels(df, horizons=[2])
        assert "future_return_2m" in labeled.columns
        assert "target_direction_2m" in labeled.columns
        assert labeled.count() == 10

    def test_compute_label_balance(self, spark: SparkSession):
        """Verify compute_label_balance and format_balance_table."""
        base_time = 1_700_000_000_000
        # 10 rows: 5 up, 3 down, 2 nulls (h=2)
        # prices: 100, 105, 110, 115, 120, 110, 100, 90, 80, 70
        prices = [100.0, 105.0, 110.0, 115.0, 120.0, 110.0, 100.0, 90.0, 80.0, 70.0]
        rows = [
            (base_time + i * MS_PER_MINUTE, "BTCUSDT", prices[i])
            for i in range(len(prices))
        ]
        df = make_kline_df(spark, rows)
        balance = compute_label_balance(df, horizons=[2])

        assert "BTCUSDT" in balance
        assert "2m" in balance["BTCUSDT"]
        b = balance["BTCUSDT"]["2m"]
        assert b["total_rows"] == 10
        assert b["valid_rows"] == 8
        assert b["null_rows"] == 2
        # UP returns:
        # i=0: 100 -> 110 (+10) UP
        # i=1: 105 -> 115 (+10) UP
        # i=2: 110 -> 120 (+10) UP
        # i=3: 115 -> 110 (-5) DOWN
        # i=4: 120 -> 100 (-20) DOWN
        # i=5: 110 -> 90 (-20) DOWN
        # i=6: 100 -> 80 (-20) DOWN
        # i=7: 90 -> 70 (-20) DOWN
        # i=8, 9: NULL
        assert b["up_count"] == 3
        assert b["down_count"] == 5
        assert math.isclose(b["up_share"], 3 / 8, rel_tol=1e-5)

        # Check table formatting produces non-empty string
        table_str = format_balance_table(balance)
        assert "BTCUSDT" in table_str
        assert "37.50%" in table_str
