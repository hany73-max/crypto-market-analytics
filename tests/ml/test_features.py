"""
==============================================================================
tests/ml/test_features.py
Unit tests for gap-safe, scale-free technical feature engineering and pipeline.

Validates:
  1. Strict causality (no future leakage): mutating prices after time t leaves
     features at and before t identical.
  2. Numerical accuracy: returns, close_to_sma, taker_buy_ratio, and parkinson_vol
     checked against hand-computed / pandas values.
  3. Gap safety: rolling / lag windows spanning missing minutes evaluate to NULL.
  4. Warm-up handling: initial rows remain NULL and are cleanly filtered by drop helper.
  5. Division-by-zero guards: zero volume, flat candles (high == low), zero stddev.
  6. Forbidden columns check: FEATURE_COLUMNS contains no label, raw price, or metadata cols.
  7. VectorAssembler & Pipeline: unfitted stages, handles clean data, fails loudly on NULLs.
  8. Scale-free invariance: scaling prices by 1000 yields identical feature values.
  9. Optional BTC market returns: joins correctly when enabled, disabled by default.
==============================================================================
"""

import math
import os
import sys
from typing import List, Tuple

import pytest
from pyspark.ml.linalg import Vector
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

from spark.apps.ml.features import (
    BASE_FEATURE_COLUMNS,
    BTC_MARKET_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    PARKINSON_FACTOR,
    add_features,
    drop_null_features,
    get_ema_weights,
    get_feature_columns,
)
from spark.apps.ml.pipeline import (
    build_feature_pipeline,
    build_feature_stages,
)
from spark.apps.ml.spark_session import get_spark_session

KLINE_SCHEMA = StructType(
    [
        StructField("open_time", LongType(), False),
        StructField("symbol", StringType(), False),
        StructField("open", DoubleType(), False),
        StructField("high", DoubleType(), False),
        StructField("low", DoubleType(), False),
        StructField("close", DoubleType(), False),
        StructField("volume", DoubleType(), False),
        StructField("number_of_trades", LongType(), False),
        StructField("taker_buy_base_volume", DoubleType(), False),
    ]
)


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Session-scoped SparkSession configured for fast local testing."""
    session = get_spark_session(
        app_name="TestFeaturesSuite",
        master="local[2]",
        log_level="ERROR",
    )
    session.conf.set("spark.sql.shuffle.partitions", "2")
    session.createDataFrame([(1,)], ["id"]).count()
    yield session
    session.stop()


def make_continuous_klines(
    symbol: str,
    n_rows: int,
    start_time_ms: int = 1_700_000_000_000,
    base_price: float = 100.0,
    step: float = 0.5,
) -> List[Tuple]:
    """Generate n_rows of synthetic continuous 1-minute klines."""
    rows = []
    price = base_price
    for i in range(n_rows):
        t = start_time_ms + i * 60_000
        o = price
        c = price + step
        h = max(o, c) + 0.2
        l = min(o, c) - 0.2
        vol = 10.0 + (i % 5)
        trades = 100 + (i % 10)
        taker = vol * 0.6
        rows.append((t, symbol, float(o), float(h), float(l), float(c), float(vol), int(trades), float(taker)))
        price = c
    return rows


# ==============================================================================
# 1. Causality (Zero Future Leakage)
# ==============================================================================
def test_feature_causality(spark: SparkSession) -> None:
    """
    Assert modifying prices after time t does NOT alter features at or before time t.
    """
    n_rows = 70
    split_idx = 45
    rows_orig = make_continuous_klines("BTCUSDT", n_rows)
    df_orig = spark.createDataFrame(rows_orig, schema=KLINE_SCHEMA)

    # Mutate prices strictly after split_idx
    cutoff_time = rows_orig[split_idx][0]
    rows_mutated = []
    for r in rows_orig:
        t, sym, o, h, l, c, vol, trades, taker = r
        if t > cutoff_time:
            # Drastic price and volume change in future
            rows_mutated.append((t, sym, o * 5.0, h * 5.0, l * 5.0, c * 5.0, vol * 2.0, trades * 2, taker * 2.0))
        else:
            rows_mutated.append(r)

    df_mutated = spark.createDataFrame(rows_mutated, schema=KLINE_SCHEMA)

    rows_feat_orig = (
        add_features(df_orig).filter(f"open_time <= {cutoff_time}").sort("open_time").collect()
    )
    rows_feat_mutated = (
        add_features(df_mutated).filter(f"open_time <= {cutoff_time}").sort("open_time").collect()
    )

    assert len(rows_feat_orig) == len(rows_feat_mutated)

    for r_orig, r_mut in zip(rows_feat_orig, rows_feat_mutated):
        for col_name in FEATURE_COLUMNS:
            val_orig = r_orig[col_name]
            val_mut = r_mut[col_name]
            if val_orig is None:
                assert val_mut is None, f"Leakage: {col_name} was None in orig but {val_mut} in mutated at t={r_orig['open_time']}"
            else:
                assert val_mut is not None, f"Leakage: {col_name} was {val_orig} in orig but None in mutated at t={r_orig['open_time']}"
                assert math.isclose(val_orig, val_mut, rel_tol=1e-5, abs_tol=1e-7), (
                    f"Leakage in feature {col_name}: orig={val_orig}, mutated={val_mut} at t={r_orig['open_time']}"
                )


# ==============================================================================
# 2. Values Checked Against Hand-Computed / Pandas Numbers
# ==============================================================================
def test_feature_exact_values(spark: SparkSession) -> None:
    """
    Verify numerical precision of returns, close_to_sma, taker_buy_ratio, and parkinson_vol.
    """
    rows = []
    start_t = 1_700_000_000_000
    prices = [100.0 + i for i in range(25)]
    for i, p in enumerate(prices):
        t = start_t + i * 60_000
        h = p * 1.02
        l = p * 0.98
        vol = 100.0
        trades = 50
        taker = 40.0
        rows.append((t, "BTCUSDT", float(p), float(h), float(l), float(p), float(vol), int(trades), float(taker)))

    df = spark.createDataFrame(rows, schema=KLINE_SCHEMA)
    feat_rows = add_features(df).sort("open_time").collect()

    # 1m return check at row 1 (index 1)
    # p[1] = 101, p[0] = 100 -> return = 1/100 = 0.01
    assert math.isclose(feat_rows[1]["price_return_1m"], 0.01, rel_tol=1e-6)

    # 5m return check at row 5 (index 5)
    # p[5] = 105, p[0] = 100 -> return = 5/100 = 0.05
    assert math.isclose(feat_rows[5]["price_return_5m"], 0.05, rel_tol=1e-6)

    # taker_buy_ratio check
    # 40 / 100 = 0.4
    assert math.isclose(feat_rows[0]["taker_buy_ratio"], 0.4, rel_tol=1e-6)

    # close_to_sma_7 check at row 6 (7 rows: 100..106, avg = 103, close = 106)
    expected_sma_7 = sum(prices[0:7]) / 7.0
    expected_ratio_7 = (prices[6] / expected_sma_7) - 1.0
    assert math.isclose(feat_rows[6]["close_to_sma_7"], expected_ratio_7, rel_tol=1e-6)

    # Parkinson vol check over 20 rows (index 19)
    # Each row has h/l = 1.02 / 0.98
    hl_ratio = 1.02 / 0.98
    expected_parkinson = math.sqrt((math.log(hl_ratio) ** 2) / (4.0 * math.log(2.0)))
    assert math.isclose(feat_rows[19]["parkinson_vol_20"], expected_parkinson, rel_tol=1e-6)


# ==============================================================================
# 3. Gap Safety
# ==============================================================================
def test_feature_gap_safety(spark: SparkSession) -> None:
    """
    Assert rolling / lag features crossing missing-minute gaps evaluate strictly to NULL.
    """
    start_t = 1_700_000_000_000
    # Rows 0..10: continuous
    # Missing minute 11
    # Rows 12..35: continuous
    rows = []
    for i in list(range(11)) + list(range(12, 36)):
        t = start_t + i * 60_000
        p = 100.0 + i
        rows.append((t, "BTCUSDT", float(p), float(p + 1), float(p - 1), float(p), 10.0, 5, 5.0))

    df = spark.createDataFrame(rows, schema=KLINE_SCHEMA)
    feat_rows = add_features(df).sort("open_time").collect()

    # Minute 12 is at index 11
    # Its 1m return compares minute 12 to minute 10 (delta 120,000 ms != 60,000 ms) -> NULL
    assert feat_rows[11]["price_return_1m"] is None

    # Minute 13 is at index 12 (delta 60,000 ms from minute 12) -> valid
    assert feat_rows[12]["price_return_1m"] is not None

    # 5m return crossing the gap (minute 14, index 13) compares to minute 9 -> NULL
    assert feat_rows[13]["price_return_5m"] is None

    # 20-row rolling features covering the gap (e.g. index 19) must be NULL
    assert feat_rows[19]["bb_percent_b"] is None
    assert feat_rows[19]["realized_vol_20"] is None
    assert feat_rows[19]["parkinson_vol_20"] is None


# ==============================================================================
# 4. Warm-up Rows & Drop Helper
# ==============================================================================
def test_warmup_rows_and_drop_helper(spark: SparkSession) -> None:
    """
    Assert warm-up rows are NULL and drop_null_features properly removes them.
    """
    n_rows = 70
    rows = make_continuous_klines("BTCUSDT", n_rows)
    df = spark.createDataFrame(rows, schema=KLINE_SCHEMA)

    df_feat = add_features(df, drop_nulls=False)
    feat_rows = df_feat.sort("open_time").collect()

    # First 60 rows (indices 0..59) must be NULL for 60m return and realized vol
    for idx in range(60):
        assert feat_rows[idx]["price_return_60m"] is None
        assert feat_rows[idx]["realized_vol_60"] is None

    # Row 60 must be non-null
    assert feat_rows[60]["price_return_60m"] is not None
    assert feat_rows[60]["realized_vol_60"] is not None

    # Verify drop helper
    clean_df = drop_null_features(df_feat)
    clean_rows = clean_df.collect()
    assert len(clean_rows) == 10

    for r in clean_rows:
        for col_name in FEATURE_COLUMNS:
            assert r[col_name] is not None, f"Found null in {col_name} after drop_null_features"


# ==============================================================================
# 5. Division-by-Zero Guards
# ==============================================================================
def test_division_by_zero_guards(spark: SparkSession) -> None:
    """
    Assert volume=0, high==low, and constant prices do not produce NaN or Inf.
    """
    start_t = 1_700_000_000_000
    rows = []
    for i in range(30):
        t = start_t + i * 60_000
        p = 100.0
        rows.append((t, "BTCUSDT", p, p, p, p, 0.0, 0, 0.0))

    df = spark.createDataFrame(rows, schema=KLINE_SCHEMA)
    feat_rows = add_features(df).sort("open_time").collect()

    # At row index 25 (after warm-up):
    r = feat_rows[25]
    assert r["taker_buy_ratio"] == 0.0
    assert r["volume_ratio_20"] == 0.0
    assert r["trade_intensity_20"] == 0.0
    assert r["parkinson_vol_20"] == 0.0
    assert r["candle_range_ratio"] == 0.0
    assert r["bb_percent_b"] == 0.5
    assert r["bb_bandwidth"] == 0.0

    # Ensure no values in valid rows are NaN or Inf
    for row in feat_rows[20:]:
        for col_name in [
            "taker_buy_ratio",
            "volume_ratio_20",
            "trade_intensity_20",
            "parkinson_vol_20",
            "bb_percent_b",
            "bb_bandwidth",
        ]:
            val = row[col_name]
            assert val is not None
            assert not math.isnan(val)
            assert not math.isinf(val)


# ==============================================================================
# 6. Forbidden Columns Check
# ==============================================================================
def test_no_forbidden_columns() -> None:
    """
    Assert FEATURE_COLUMNS contains NO forbidden labels, raw price levels, or metadata.
    """
    forbidden_exact = {
        "open",
        "high",
        "low",
        "close",
        "data_source",
        "ingested_at",
        "kline_date",
        "open_time_dt",
        "close_time_dt",
    }

    for col_name in FEATURE_COLUMNS:
        assert not col_name.startswith("future_"), f"Forbidden future col in features: {col_name}"
        assert not col_name.startswith("target_"), f"Forbidden target col in features: {col_name}"
        assert col_name not in forbidden_exact, f"Forbidden exact col in features: {col_name}"
        assert col_name not in {"sma_7", "sma_20", "sma_50", "bb_upper", "bb_lower", "rolling_std_20"}


# ==============================================================================
# 7. Scale-Free Invariance
# ==============================================================================
def test_scale_free_invariance(spark: SparkSession) -> None:
    """
    Assert scaling all price columns by 1000 yields identical price-derived features.
    """
    n_rows = 70
    rows_orig = make_continuous_klines("BTCUSDT", n_rows, base_price=50.0, step=0.25)
    rows_scaled = []
    scale = 1000.0
    for r in rows_orig:
        t, sym, o, h, l, c, vol, trades, taker = r
        rows_scaled.append((t, sym, o * scale, h * scale, l * scale, c * scale, vol, trades, taker))

    df_orig = spark.createDataFrame(rows_orig, schema=KLINE_SCHEMA)
    df_scaled = spark.createDataFrame(rows_scaled, schema=KLINE_SCHEMA)

    clean_orig = drop_null_features(add_features(df_orig)).sort("open_time").collect()
    clean_scaled = drop_null_features(add_features(df_scaled)).sort("open_time").collect()

    price_derived_cols = [
        "price_return_1m",
        "price_return_5m",
        "price_return_15m",
        "price_return_60m",
        "close_to_sma_7",
        "close_to_sma_20",
        "close_to_sma_50",
        "bb_percent_b",
        "bb_bandwidth",
        "realized_vol_20",
        "realized_vol_60",
        "parkinson_vol_20",
        "candle_body_ratio",
        "candle_range_ratio",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
    ]

    for r_orig, r_scaled in zip(clean_orig, clean_scaled):
        for col_name in price_derived_cols:
            vo = r_orig[col_name]
            vs = r_scaled[col_name]
            assert math.isclose(vo, vs, rel_tol=1e-5, abs_tol=1e-7), (
                f"Feature {col_name} scaled differently: orig={vo}, scaled={vs}"
            )


# ==============================================================================
# 8. VectorAssembler and Pipeline Stages
# ==============================================================================
def test_pipeline_assembly_and_loud_null_failure(spark: SparkSession) -> None:
    """
    Assert VectorAssembler dimension equals len(FEATURE_COLUMNS), handles clean data,
    and fails loudly on uncleaned data containing NULLs.
    """
    n_rows = 70
    rows = make_continuous_klines("BTCUSDT", n_rows)
    df = spark.createDataFrame(rows, schema=KLINE_SCHEMA)
    df_feat = add_features(df, drop_nulls=False)

    pipeline = build_feature_pipeline()
    assembler = pipeline.getStages()[0]
    scaler = pipeline.getStages()[1]

    assert len(assembler.getInputCols()) == len(FEATURE_COLUMNS)
    assert assembler.getHandleInvalid() == "error"

    # Pipeline transform must FAIL loudly on uncleaned data with nulls
    with pytest.raises(Exception):
        assembler.transform(df_feat).collect()

    # Cleaned DataFrame must succeed and output correct vector dimension
    clean_df = drop_null_features(df_feat)
    assembled_df = assembler.transform(clean_df)
    first_vec: Vector = assembled_df.select("raw_features").first()["raw_features"]
    assert first_vec.size == len(FEATURE_COLUMNS)

    # Scaler can be fit on clean data
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)
    scaled_vec: Vector = scaled_df.select("features").first()["features"]
    assert scaled_vec.size == len(FEATURE_COLUMNS)


# ==============================================================================
# 9. Optional BTC Market-Return Feature Flag
# ==============================================================================
def test_optional_btc_market_return(spark: SparkSession) -> None:
    """
    Assert BTC market returns join on exact open_time when enabled, and stay off by default.
    """
    n_rows = 30
    rows_btc = make_continuous_klines("BTCUSDT", n_rows, base_price=100.0)
    rows_eth = make_continuous_klines("ETHUSDT", n_rows, base_price=10.0)
    df = spark.createDataFrame(rows_btc + rows_eth, schema=KLINE_SCHEMA)

    # Default: OFF
    df_default = add_features(df, include_btc_market_return=False)
    assert "btc_return_5m" not in df_default.columns
    assert "btc_return_15m" not in df_default.columns

    # Enabled: ON
    df_with_btc = add_features(df, include_btc_market_return=True)
    assert "btc_return_5m" in df_with_btc.columns
    assert "btc_return_15m" in df_with_btc.columns

    rows_eth = df_with_btc.filter("symbol = 'ETHUSDT'").sort("open_time").collect()
    rows_btc = df_with_btc.filter("symbol = 'BTCUSDT'").sort("open_time").collect()

    # Check at index 10 (valid 5m return)
    expected_btc_ret_5m = rows_btc[10]["price_return_5m"]
    actual_joined_ret = rows_eth[10]["btc_return_5m"]
    assert math.isclose(expected_btc_ret_5m, actual_joined_ret, rel_tol=1e-6)
