"""
==============================================================================
spark/apps/ml/features.py
Gap-safe, scale-free technical feature engineering for cryptocurrency ML models.

Computes normalized time-series market features across pooled assets:
  - Multi-horizon price returns (1m, 5m, 15m, 60m)
  - Relative distance to moving averages (close_to_sma_7, 20, 50)
  - Bollinger Band metrics (bb_percent_b, bb_bandwidth)
  - Realized volatility of 1-minute returns (20, 60 periods)
  - Parkinson high-low volatility over 20 periods
  - Normalized candlestick shape metrics (body ratio, range ratio)
  - Normalized volume and trade intensity ratios (20 periods)
  - Taker buy base volume ratio
  - Cutler's 14-period Simple-Average Relative Strength Index (RSI-14)
  - MACD(12, 26, 9) normalized by close (line, signal, histogram)
  - Cyclic time-of-day features derived strictly from open_time (epoch ms)
  - Optional BTC market-return feature (default OFF)

Guarantees:
  - Zero forward leakage: strictly causal, past-only lookbacks.
  - Scale-free: dollar-denominated prices cancel out, enabling pooled training.
  - Gap-safe: windows crossing missing-minute gaps or partition boundaries evaluate to NULL.
  - Zero wall-clock dependencies: deterministic epoch ms calculations.
  - Warm-up periods remain explicit NULLs without arbitrary imputation.
==============================================================================
"""

import argparse
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType

# Ensure repo root is available for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spark.apps.ml.labels import MS_PER_MINUTE, add_labels
from spark.apps.ml.spark_session import get_spark_session

logger = logging.getLogger("crypto_ml_features")

# Canonical Scale-Free Feature Column Definitions
BASE_FEATURE_COLUMNS: Tuple[str, ...] = (
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
    "volume_ratio_20",
    "trade_intensity_20",
    "taker_buy_ratio",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "sin_minute_of_day",
    "cos_minute_of_day",
    "sin_day_of_week",
    "cos_day_of_week",
)

BTC_MARKET_FEATURE_COLUMNS: Tuple[str, ...] = (
    "btc_return_5m",
    "btc_return_15m",
)

# Default exposed feature list
FEATURE_COLUMNS: Tuple[str, ...] = BASE_FEATURE_COLUMNS

# Constant factor for Parkinson volatility: 4 * ln(2)
PARKINSON_FACTOR: float = 4.0 * math.log(2.0)


def get_ema_weights(period: int) -> List[float]:
    """
    Compute normalized fixed-lookback exponential moving average weights.

    In a standard recursive EMA, weights decay as (1 - alpha)^k where alpha = 2 / (period + 1).
    For a deterministic, causal, fixed-lookback window of length N:
        w_k = (1 - alpha)^k  for k in [0, N - 1]
        c_k = w_k / sum_{j=0}^{N-1} w_j

    Args:
        period: Lookback window length N (positive integer).

    Returns:
        List of normalized float weights summing to 1.0.
    """
    alpha = 2.0 / (period + 1.0)
    weights = [(1.0 - alpha) ** k for k in range(period)]
    total = sum(weights)
    return [w / total for w in weights]


def get_feature_columns(include_btc_market_return: bool = False) -> Tuple[str, ...]:
    """
    Return the tuple of active feature column names based on configuration.

    Args:
        include_btc_market_return: If True, appends BTC market returns (5m, 15m).

    Returns:
        Tuple of feature column names.
    """
    if include_btc_market_return:
        return BASE_FEATURE_COLUMNS + BTC_MARKET_FEATURE_COLUMNS
    return BASE_FEATURE_COLUMNS


def add_features(
    df: DataFrame,
    include_btc_market_return: bool = False,
    drop_nulls: bool = False,
) -> DataFrame:
    """
    Compute gap-safe, scale-free technical features from canonical cryptocurrency klines.

    Mathematical Specifications:
      1. Returns:
         - price_return_{n}m = (close - lag(close, n)) / lag(close, n)
         - Valid if open_time - lag(open_time, n) == n * 60,000 ms, else NULL.
         - Evaluated for n in {1, 5, 15, 60}.
      2. Distance to Moving Averages:
         - close_to_sma_{n} = (close / sma_{n}) - 1.0
         - sma_{n} = avg(close) over rowsBetween(-(n-1), 0).
         - Valid if open_time - lag(open_time, n-1) == (n-1) * 60,000 ms and sma_{n} > 0, else NULL.
         - Evaluated for n in {7, 20, 50}.
      3. Bollinger Bands:
         - sma_20 = avg(close) over 20 rows.
         - rolling_std_20 = stddev(close) over 20 rows.
         - bb_upper = sma_20 + 2 * rolling_std_20; bb_lower = sma_20 - 2 * rolling_std_20.
         - bb_percent_b = (close - bb_lower) / (bb_upper - bb_lower), guarded to 0.5 when stddev == 0.
         - bb_bandwidth = (bb_upper - bb_lower) / sma_20, guarded to 0.0 when sma_20 <= 0.
         - Valid if open_time - lag(open_time, 19) == 19 * 60,000 ms, else NULL.
      4. Realized Volatility:
         - Rolling sample standard deviation of 1-minute returns over 20 and 60 rows.
         - realized_vol_20 valid if open_time - lag(open_time, 20) == 20 * 60,000 ms.
         - realized_vol_60 valid if open_time - lag(open_time, 60) == 60 * 60,000 ms.
      5. Parkinson High-Low Volatility:
         - parkinson_vol_20 = sqrt( avg( ln(high / low)^2 ) / (4 * ln(2)) ) over 20 rows.
         - Guarded against non-positive prices and high == low.
         - Valid if open_time - lag(open_time, 19) == 19 * 60,000 ms, else NULL.
      6. Candlestick Shape:
         - candle_body_ratio = (close - open) / open
         - candle_range_ratio = (high - low) / close
         - Single-candle, scale-free metrics guarded against zero denominators.
      7. Volume & Trade Intensity:
         - volume_ratio_20 = volume / avg(volume) over 20 rows, guarded to 0.0 on zero avg.
         - trade_intensity_20 = number_of_trades / avg(number_of_trades) over 20 rows.
         - Valid if open_time - lag(open_time, 19) == 19 * 60,000 ms, else NULL.
      8. Taker Buy Ratio:
         - taker_buy_ratio = taker_buy_base_volume / volume, guarded to 0.0 when volume <= 0.
      9. Cutler's 14-period Simple-Average RSI:
         - avg_gain = avg(max(close - lag(close, 1), 0)) over 14 periods.
         - avg_loss = avg(max(lag(close, 1) - close, 0)) over 14 periods.
         - rsi_14 = 100 * avg_gain / (avg_gain + avg_loss), guarded to 50.0 when both are zero.
         - Valid if open_time - lag(open_time, 14) == 14 * 60,000 ms, else NULL.
      10. Normalized MACD(12, 26, 9):
          - Fast EMA (12): fixed-lookback weighted sum of close over 12 lags.
          - Slow EMA (26): fixed-lookback weighted sum of close over 26 lags.
          - macd_line = (fast_ema - slow_ema) / close. Valid if continuous over 26 rows.
          - macd_signal = 9-period fixed-lookback weighted sum of macd_line.
          - macd_hist = macd_line - macd_signal.
          - Both signal and hist valid if continuous over 34 rows (33 lags).
      11. Cyclic Time-of-Day Features:
          - minute_of_day = (open_time / 60,000) % 1440
          - day_of_week = ((open_time / 86,400,000) + 3) % 7 (0=Monday ... 6=Sunday)
          - sin/cos of minute_of_day (period 1440) and day_of_week (period 7).
          - Derived strictly from open_time epoch ms; zero wall-clock dependency.
      12. Optional BTC Market Returns:
          - If include_btc_market_return=True, joins BTCUSDT 5m and 15m returns on exact open_time.

    Args:
        df: Input DataFrame containing kline columns.
        include_btc_market_return: If True, joins BTC returns on exact open_time.
        drop_nulls: If True, removes rows containing NULL in any active feature column.

    Returns:
        Augmented DataFrame with engineered scale-free feature columns.

    Raises:
        ValueError: If required columns are missing from the input DataFrame.
    """
    required_cols = {
        "symbol",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "number_of_trades",
        "taker_buy_base_volume",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame missing required feature columns: {sorted(missing_cols)}")

    w = Window.partitionBy("symbol").orderBy("open_time")
    w_7 = w.rowsBetween(-6, 0)
    w_14 = w.rowsBetween(-13, 0)
    w_20 = w.rowsBetween(-19, 0)
    w_50 = w.rowsBetween(-49, 0)
    w_60 = w.rowsBetween(-59, 0)

    # -------------------------------------------------------------------------
    # 1. Multi-Horizon Returns (1m, 5m, 15m, 60m)
    # -------------------------------------------------------------------------
    # Gap checks: open_time - lag(open_time, n) == n * 60,000 ms
    is_cont_1 = (F.col("open_time") - F.lag("open_time", 1).over(w)) == F.lit(1 * MS_PER_MINUTE)
    is_cont_5 = (F.col("open_time") - F.lag("open_time", 5).over(w)) == F.lit(5 * MS_PER_MINUTE)
    is_cont_15 = (F.col("open_time") - F.lag("open_time", 15).over(w)) == F.lit(15 * MS_PER_MINUTE)
    is_cont_60 = (F.col("open_time") - F.lag("open_time", 60).over(w)) == F.lit(60 * MS_PER_MINUTE)

    lag_c1 = F.lag("close", 1).over(w)
    lag_c5 = F.lag("close", 5).over(w)
    lag_c15 = F.lag("close", 15).over(w)
    lag_c60 = F.lag("close", 60).over(w)

    ret_1m = F.when(is_cont_1 & (lag_c1 > 0.0), (F.col("close") - lag_c1) / lag_c1).otherwise(
        F.lit(None).cast(DoubleType())
    )
    ret_5m = F.when(is_cont_5 & (lag_c5 > 0.0), (F.col("close") - lag_c5) / lag_c5).otherwise(
        F.lit(None).cast(DoubleType())
    )
    ret_15m = F.when(is_cont_15 & (lag_c15 > 0.0), (F.col("close") - lag_c15) / lag_c15).otherwise(
        F.lit(None).cast(DoubleType())
    )
    ret_60m = F.when(is_cont_60 & (lag_c60 > 0.0), (F.col("close") - lag_c60) / lag_c60).otherwise(
        F.lit(None).cast(DoubleType())
    )

    res = (
        df.withColumn("price_return_1m", ret_1m)
        .withColumn("price_return_5m", ret_5m)
        .withColumn("price_return_15m", ret_15m)
        .withColumn("price_return_60m", ret_60m)
    )

    # -------------------------------------------------------------------------
    # 2. Distance to Moving Averages (close_to_sma_7, 20, 50)
    # -------------------------------------------------------------------------
    is_cont_7_rows = (F.col("open_time") - F.lag("open_time", 6).over(w)) == F.lit(6 * MS_PER_MINUTE)
    is_cont_20_rows = (F.col("open_time") - F.lag("open_time", 19).over(w)) == F.lit(19 * MS_PER_MINUTE)
    is_cont_50_rows = (F.col("open_time") - F.lag("open_time", 49).over(w)) == F.lit(49 * MS_PER_MINUTE)

    raw_sma_7 = F.avg("close").over(w_7)
    raw_sma_20 = F.avg("close").over(w_20)
    raw_sma_50 = F.avg("close").over(w_50)

    res = (
        res.withColumn("_temp_sma_7", raw_sma_7)
        .withColumn("_temp_sma_20", raw_sma_20)
        .withColumn("_temp_sma_50", raw_sma_50)
        .withColumn(
            "close_to_sma_7",
            F.when(is_cont_7_rows & (F.col("_temp_sma_7") > 0.0), (F.col("close") / F.col("_temp_sma_7")) - 1.0).otherwise(
                F.lit(None).cast(DoubleType())
            ),
        )
        .withColumn(
            "close_to_sma_20",
            F.when(
                is_cont_20_rows & (F.col("_temp_sma_20") > 0.0), (F.col("close") / F.col("_temp_sma_20")) - 1.0
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
        .withColumn(
            "close_to_sma_50",
            F.when(
                is_cont_50_rows & (F.col("_temp_sma_50") > 0.0), (F.col("close") / F.col("_temp_sma_50")) - 1.0
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
    )

    # -------------------------------------------------------------------------
    # 3. Bollinger Bands (bb_percent_b, bb_bandwidth)
    # -------------------------------------------------------------------------
    raw_std_20 = F.stddev("close").over(w_20)
    res = (
        res.withColumn("_temp_std_20", raw_std_20)
        .withColumn(
            "_temp_bb_upper",
            F.col("_temp_sma_20") + (F.lit(2.0) * F.coalesce(F.col("_temp_std_20"), F.lit(0.0))),
        )
        .withColumn(
            "_temp_bb_lower",
            F.col("_temp_sma_20") - (F.lit(2.0) * F.coalesce(F.col("_temp_std_20"), F.lit(0.0))),
        )
        .withColumn(
            "bb_percent_b",
            F.when(
                is_cont_20_rows,
                F.when(
                    (F.col("_temp_bb_upper") - F.col("_temp_bb_lower")) > 0.0,
                    (F.col("close") - F.col("_temp_bb_lower")) / (F.col("_temp_bb_upper") - F.col("_temp_bb_lower")),
                ).otherwise(F.lit(0.5)),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
        .withColumn(
            "bb_bandwidth",
            F.when(
                is_cont_20_rows & (F.col("_temp_sma_20") > 0.0),
                (F.col("_temp_bb_upper") - F.col("_temp_bb_lower")) / F.col("_temp_sma_20"),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
    )

    # -------------------------------------------------------------------------
    # 4. Realized Volatility of 1m Returns (20 and 60 periods)
    # -------------------------------------------------------------------------
    # 20 periods of 1m returns span 21 klines: open_time - lag(open_time, 20) == 20 * 60,000 ms
    # 60 periods span 61 klines: open_time - lag(open_time, 60) == 60 * 60,000 ms
    is_cont_ret_20 = (F.col("open_time") - F.lag("open_time", 20).over(w)) == F.lit(20 * MS_PER_MINUTE)
    is_cont_ret_60 = (F.col("open_time") - F.lag("open_time", 60).over(w)) == F.lit(60 * MS_PER_MINUTE)

    res = res.withColumn(
        "realized_vol_20",
        F.when(is_cont_ret_20, F.stddev("price_return_1m").over(w_20)).otherwise(F.lit(None).cast(DoubleType())),
    ).withColumn(
        "realized_vol_60",
        F.when(is_cont_ret_60, F.stddev("price_return_1m").over(w_60)).otherwise(F.lit(None).cast(DoubleType())),
    )

    # -------------------------------------------------------------------------
    # 5. Parkinson High-Low Volatility (20 periods)
    # -------------------------------------------------------------------------
    # sqrt( avg( ln(high / low)^2 ) / (4 * ln(2)) )
    log_hl = F.when(
        (F.col("low") > 0.0) & (F.col("high") >= F.col("low")),
        F.log(F.col("high") / F.col("low")),
    ).otherwise(F.lit(0.0))
    log_hl_sq = F.pow(log_hl, 2)
    avg_log_sq_20 = F.avg(log_hl_sq).over(w_20)

    res = res.withColumn(
        "parkinson_vol_20",
        F.when(
            is_cont_20_rows,
            F.sqrt(avg_log_sq_20 / F.lit(PARKINSON_FACTOR)),
        ).otherwise(F.lit(None).cast(DoubleType())),
    )

    # -------------------------------------------------------------------------
    # 6. Candlestick Shape (Scale-Free)
    # -------------------------------------------------------------------------
    res = res.withColumn(
        "candle_body_ratio",
        F.when(F.col("open") > 0.0, (F.col("close") - F.col("open")) / F.col("open")).otherwise(F.lit(0.0)),
    ).withColumn(
        "candle_range_ratio",
        F.when(F.col("close") > 0.0, (F.col("high") - F.col("low")) / F.col("close")).otherwise(F.lit(0.0)),
    )

    # -------------------------------------------------------------------------
    # 7. Volume Ratio & Trade Intensity (20 periods)
    # -------------------------------------------------------------------------
    avg_vol_20 = F.avg("volume").over(w_20)
    avg_trades_20 = F.avg("number_of_trades").over(w_20)

    res = (
        res.withColumn("_temp_avg_vol_20", avg_vol_20)
        .withColumn("_temp_avg_trades_20", avg_trades_20)
        .withColumn(
            "volume_ratio_20",
            F.when(
                is_cont_20_rows,
                F.when(F.col("_temp_avg_vol_20") > 0.0, F.col("volume") / F.col("_temp_avg_vol_20")).otherwise(
                    F.lit(0.0)
                ),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
        .withColumn(
            "trade_intensity_20",
            F.when(
                is_cont_20_rows,
                F.when(
                    F.col("_temp_avg_trades_20") > 0.0,
                    F.col("number_of_trades") / F.col("_temp_avg_trades_20"),
                ).otherwise(F.lit(0.0)),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
    )

    # -------------------------------------------------------------------------
    # 8. Taker Buy Base Volume Ratio
    # -------------------------------------------------------------------------
    res = res.withColumn(
        "taker_buy_ratio",
        F.when(
            F.col("volume") > 0.0,
            F.col("taker_buy_base_volume") / F.col("volume"),
        ).otherwise(F.lit(0.0)),
    )

    # -------------------------------------------------------------------------
    # 9. Cutler's 14-period Simple-Average RSI (RSI-14)
    # -------------------------------------------------------------------------
    # 14 periods of 1-minute price changes span 15 klines: open_time - lag(open_time, 14) == 14 * 60,000 ms
    is_cont_14_periods = (F.col("open_time") - F.lag("open_time", 14).over(w)) == F.lit(14 * MS_PER_MINUTE)
    diff = F.col("close") - F.lag("close", 1).over(w)
    gain = F.when(diff > 0.0, diff).otherwise(F.lit(0.0))
    loss = F.when(diff < 0.0, -diff).otherwise(F.lit(0.0))

    res = (
        res.withColumn("_temp_gain", gain)
        .withColumn("_temp_loss", loss)
        .withColumn("_temp_avg_gain_14", F.avg("_temp_gain").over(w_14))
        .withColumn("_temp_avg_loss_14", F.avg("_temp_loss").over(w_14))
        .withColumn(
            "rsi_14",
            F.when(
                is_cont_14_periods,
                F.when(
                    (F.col("_temp_avg_gain_14") + F.col("_temp_avg_loss_14")) > 0.0,
                    F.lit(100.0)
                    * F.col("_temp_avg_gain_14")
                    / (F.col("_temp_avg_gain_14") + F.col("_temp_avg_loss_14")),
                ).otherwise(F.lit(50.0)),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
    )

    # -------------------------------------------------------------------------
    # 10. Normalized MACD(12, 26, 9)
    # -------------------------------------------------------------------------
    w12 = get_ema_weights(12)
    w26 = get_ema_weights(26)
    w9 = get_ema_weights(9)

    fast_ema_expr = sum(weight * F.lag("close", k).over(w) for k, weight in enumerate(w12))
    slow_ema_expr = sum(weight * F.lag("close", k).over(w) for k, weight in enumerate(w26))

    is_cont_26_rows = (F.col("open_time") - F.lag("open_time", 25).over(w)) == F.lit(25 * MS_PER_MINUTE)
    is_cont_34_rows = (F.col("open_time") - F.lag("open_time", 33).over(w)) == F.lit(33 * MS_PER_MINUTE)

    res = (
        res.withColumn("_temp_fast_ema", fast_ema_expr)
        .withColumn("_temp_slow_ema", slow_ema_expr)
        .withColumn(
            "macd_line",
            F.when(
                is_cont_26_rows & (F.col("close") > 0.0),
                (F.col("_temp_fast_ema") - F.col("_temp_slow_ema")) / F.col("close"),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
    )

    w_9 = w.rowsBetween(-8, 0)
    macd_signal_expr = F.avg("macd_line").over(w_9)

    res = (
        res.withColumn(
            "macd_signal",
            F.when(is_cont_34_rows, macd_signal_expr).otherwise(F.lit(None).cast(DoubleType())),
        )
        .withColumn(
            "macd_hist",
            F.when(
                is_cont_34_rows & F.col("macd_line").isNotNull() & F.col("macd_signal").isNotNull(),
                F.col("macd_line") - F.col("macd_signal"),
            ).otherwise(F.lit(None).cast(DoubleType())),
        )
    )

    # -------------------------------------------------------------------------
    # 11. Cyclic Time-of-Day Features
    # -------------------------------------------------------------------------
    # Minute of day in [0, 1439]
    minute_of_day = ((F.col("open_time") / F.lit(MS_PER_MINUTE)) % F.lit(1440)).cast(LongType())
    # Epoch 1970-01-01 was Thursday (index 3 if Monday is 0)
    day_of_week = (
        (F.floor(F.col("open_time") / F.lit(24 * 60 * MS_PER_MINUTE)) + F.lit(3)) % F.lit(7)
    ).cast(LongType())

    angle_minute = F.lit(2.0 * math.pi) * minute_of_day / F.lit(1440.0)
    angle_dow = F.lit(2.0 * math.pi) * day_of_week / F.lit(7.0)

    res = (
        res.withColumn("sin_minute_of_day", F.sin(angle_minute))
        .withColumn("cos_minute_of_day", F.cos(angle_minute))
        .withColumn("sin_day_of_week", F.sin(angle_dow))
        .withColumn("cos_day_of_week", F.cos(angle_dow))
    )

    # -------------------------------------------------------------------------
    # 12. Optional BTC Market-Return Feature
    # -------------------------------------------------------------------------
    if include_btc_market_return:
        btc_returns = (
            res.filter(F.col("symbol") == "BTCUSDT")
            .select(
                F.col("open_time").alias("_btc_open_time"),
                F.col("price_return_5m").alias("btc_return_5m"),
                F.col("price_return_15m").alias("btc_return_15m"),
            )
        )
        res = res.join(btc_returns, res.open_time == btc_returns._btc_open_time, how="left").drop(
            "_btc_open_time"
        )

    # Drop all intermediate calculation columns
    temp_cols = [c for c in res.columns if c.startswith("_temp_")]
    if temp_cols:
        res = res.drop(*temp_cols)

    # Filter out null features if requested
    if drop_nulls:
        active_features = get_feature_columns(include_btc_market_return)
        res = drop_null_features(res, feature_cols=active_features)

    return res


def drop_null_features(
    df: DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> DataFrame:
    """
    Drop rows where any of the specified feature columns are NULL or NaN.

    Args:
        df: Input DataFrame containing computed features.
        feature_cols: Sequence of feature column names. Defaults to FEATURE_COLUMNS.

    Returns:
        Filtered DataFrame containing only complete rows.
    """
    cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)
    condition = None
    for c in cols:
        col_cond = F.col(c).isNotNull() & (~F.isnan(F.col(c)))
        condition = col_cond if condition is None else (condition & col_cond)

    if condition is not None:
        return df.filter(condition)
    return df


def inspect_features_cli(
    parquet_path: str = "data/sample_klines.parquet",
    sample_limit: Optional[int] = None,
) -> None:
    """
    Load sample dataset, compute features and labels, and display statistics.
    """
    spark = get_spark_session(app_name="InspectFeaturesCLI", log_level="ERROR")

    resolved_path = (
        parquet_path
        if os.path.isabs(parquet_path)
        else os.path.join(REPO_ROOT, parquet_path.replace("/", os.sep))
    )

    if not os.path.exists(resolved_path):
        print(f"Dataset path does not exist: {resolved_path}")
        return

    print("=" * 80)
    print(f"Loading klines from: {resolved_path}")
    raw_df = spark.read.parquet(resolved_path)
    total_raw_rows = raw_df.count()
    symbols = [r["symbol"] for r in raw_df.select("symbol").distinct().collect()]
    print(f"Total raw rows: {total_raw_rows} across symbols: {symbols}")

    print("\nComputing scale-free features and forward labels...")
    df_feat = add_features(raw_df, include_btc_market_return=False, drop_nulls=False)
    df_all = add_labels(df_feat, drop_null_labels=False)

    print(f"\nFeature count: {len(FEATURE_COLUMNS)}")
    print("=" * 80)
    print(f"{'Feature Name':<25} | {'Null Count':<12} | {'Null Pct':<10}")
    print("-" * 80)

    # Aggregations for null counts
    null_exprs = [F.count(F.when(F.col(c).isNull() | F.isnan(F.col(c)), 1)).alias(c) for c in FEATURE_COLUMNS]
    null_counts_row = df_all.agg(*null_exprs).collect()[0]

    for c in FEATURE_COLUMNS:
        n_null = null_counts_row[c]
        pct_null = (n_null / total_raw_rows) * 100.0
        print(f"{c:<25} | {n_null:<12} | {pct_null:6.2f}%")

    print("=" * 80)
    cleaned_df = drop_null_features(df_all)
    clean_rows = cleaned_df.count()
    dropped_rows = total_raw_rows - clean_rows
    print(f"Rows after dropping null warm-up features: {clean_rows} (dropped: {dropped_rows} rows)")

    print("\nFeature Summary Statistics (Cleaned Data):")
    print("=" * 80)
    print(f"{'Feature':<22} | {'Mean':<12} | {'StdDev':<12} | {'Min':<12} | {'Max':<12}")
    print("-" * 80)

    stat_exprs = []
    for c in FEATURE_COLUMNS:
        stat_exprs.extend(
            [
                F.mean(F.col(c)).alias(f"{c}_mean"),
                F.stddev(F.col(c)).alias(f"{c}_std"),
                F.min(F.col(c)).alias(f"{c}_min"),
                F.max(F.col(c)).alias(f"{c}_max"),
            ]
        )

    stats_row = cleaned_df.agg(*stat_exprs).collect()[0]
    for c in FEATURE_COLUMNS:
        mean_val = stats_row[f"{c}_mean"]
        std_val = stats_row[f"{c}_std"]
        min_val = stats_row[f"{c}_min"]
        max_val = stats_row[f"{c}_max"]

        m_str = f"{mean_val:11.6f}" if mean_val is not None else "        N/A"
        s_str = f"{std_val:11.6f}" if std_val is not None else "        N/A"
        min_str = f"{min_val:11.6f}" if min_val is not None else "        N/A"
        max_str = f"{max_val:11.6f}" if max_val is not None else "        N/A"

        print(f"{c:<22} | {m_str} | {s_str} | {min_str} | {max_str}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cryptocurrency ML Feature Inspection")
    parser.add_argument(
        "--input",
        default="data/sample_klines.parquet",
        help="Path to klines parquet dataset",
    )
    args = parser.parse_args()
    inspect_features_cli(parquet_path=args.input)
