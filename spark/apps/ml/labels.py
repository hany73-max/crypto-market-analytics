"""
==============================================================================
spark/apps/ml/labels.py
Gap-safe label generation for cryptocurrency price direction prediction.

Computes multi-horizon forward returns and direction classification targets:
  - Validates temporal continuity using lead(open_time, h).
  - Explicitly nulls out rows crossing data gaps or partition boundaries.
  - Supports configurable binary (UP / DOWN-FLAT) and 3-class (UP / FLAT / DOWN).
  - Preserves deterministic chronological behavior with zero wall-clock dependencies.
==============================================================================
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

# Ensure repo root is available for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spark.apps.ml.spark_session import get_spark_session

logger = logging.getLogger("crypto_ml_labels")

# Constants
MS_PER_MINUTE: int = 60_000
DEFAULT_HORIZONS: Tuple[int, ...] = (5, 15, 60)

# Binary classification targets (epsilon is None or 0.0)
BINARY_DOWN_OR_FLAT: int = 0
BINARY_UP: int = 1

# Three-class classification targets (when epsilon > 0.0)
THREE_CLASS_DOWN: int = 0
THREE_CLASS_FLAT: int = 1
THREE_CLASS_UP: int = 2


def add_labels(
    df: DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    epsilon: Optional[float] = None,
    drop_null_labels: bool = False,
) -> DataFrame:
    """
    Compute gap-safe forward returns and target directions for cryptocurrency klines.

    For each specified horizon h (in minutes):
      1. Validates gap continuity: verifies lead(open_time, h) - open_time == h * 60,000 ms.
      2. If continuous and valid, computes price return: (lead(close, h) - close) / close.
         Otherwise, future_return_{h}m is set to NULL.
      3. Assigns target_direction_{h}m:
         - Binary mode (epsilon is None or 0.0):
             1 (UP) if future_return > 0, else 0 (DOWN/FLAT).
         - Three-class mode (epsilon > 0.0):
             2 (UP) if future_return > epsilon,
             0 (DOWN) if future_return < -epsilon,
             1 (FLAT) if -epsilon <= future_return <= epsilon.
         - NULL if future_return is NULL (e.g. crossing gaps or last h rows).
      4. If drop_null_labels is True, filters out rows where any target is NULL.

    Args:
        df: Input PySpark DataFrame containing at least 'symbol', 'open_time', and 'close'.
        horizons: Sequence of integer forward lookaheads in minutes (default: 5, 15, 60).
        epsilon: Optional return threshold for flat class. If None or 0.0, uses binary mode.
        drop_null_labels: If True, drops rows where any target_direction_{h}m is null.
            If False (default), preserves rows with explicit nulls.

    Returns:
        Augmented DataFrame with future_return_{h}m and target_direction_{h}m columns.

    Raises:
        ValueError: If required columns are missing, horizons are invalid, or epsilon is negative.
    """
    required_cols = {"symbol", "open_time", "close"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame missing required columns: {sorted(missing_cols)}")

    if not horizons:
        raise ValueError("Horizons sequence must contain at least one positive integer horizon.")

    for h in horizons:
        if not isinstance(h, int) or h <= 0:
            raise ValueError(f"Each horizon must be a positive integer in minutes, got: {h}")

    if epsilon is not None and epsilon < 0.0:
        raise ValueError(f"Epsilon cannot be negative, got: {epsilon}")

    w = Window.partitionBy("symbol").orderBy("open_time")
    result_df = df
    target_cols: List[str] = []

    for h in horizons:
        return_col_name = f"future_return_{h}m"
        target_col_name = f"target_direction_{h}m"
        target_cols.append(target_col_name)

        future_open_time = F.lead("open_time", h).over(w)
        future_close = F.lead("close", h).over(w)

        # Gap-safety check: future candle open_time must be exactly h * 60,000 ms ahead
        expected_diff_ms = h * MS_PER_MINUTE
        time_diff_ms = future_open_time - F.col("open_time")
        is_gap_free = (
            future_open_time.isNotNull()
            & (time_diff_ms == expected_diff_ms)
            & F.col("close").isNotNull()
            & (F.col("close") > 0.0)
        )

        raw_return = (future_close - F.col("close")) / F.col("close")
        return_expr = F.when(is_gap_free, raw_return).otherwise(F.lit(None).cast(DoubleType()))

        result_df = result_df.withColumn(return_col_name, return_expr)

        return_col = F.col(return_col_name)
        if epsilon is None or epsilon == 0.0:
            direction_expr = (
                F.when(return_col.isNull(), F.lit(None).cast(IntegerType()))
                .when(return_col > 0.0, F.lit(BINARY_UP))
                .otherwise(F.lit(BINARY_DOWN_OR_FLAT))
            )
        else:
            direction_expr = (
                F.when(return_col.isNull(), F.lit(None).cast(IntegerType()))
                .when(return_col > epsilon, F.lit(THREE_CLASS_UP))
                .when(return_col < -epsilon, F.lit(THREE_CLASS_DOWN))
                .otherwise(F.lit(THREE_CLASS_FLAT))
            )

        result_df = result_df.withColumn(target_col_name, direction_expr)

    if drop_null_labels:
        for t_col in target_cols:
            result_df = result_df.filter(F.col(t_col).isNotNull())

    return result_df


def compute_label_balance(
    df: DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    epsilon: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate class distribution and share of UP labels per symbol and horizon.

    Args:
        df: Input DataFrame (if target columns are missing, add_labels will be applied).
        horizons: Sequence of horizons to inspect.
        epsilon: Optional epsilon threshold for 3-class target calculation.

    Returns:
        Dict mapped by symbol and horizon containing row counts, class counts, and UP share.
    """
    needed_cols = [f"target_direction_{h}m" for h in horizons]
    if not all(col in df.columns for col in needed_cols):
        df = add_labels(df, horizons=horizons, epsilon=epsilon, drop_null_labels=False)

    is_three_class = epsilon is not None and epsilon > 0.0

    agg_exprs = [F.count(F.lit(1)).alias("total_rows")]
    for h in horizons:
        t_col = F.col(f"target_direction_{h}m")
        agg_exprs.extend([
            F.count(t_col).alias(f"valid_{h}m"),
            F.count(F.when(t_col.isNull(), 1)).alias(f"null_{h}m"),
        ])
        if is_three_class:
            agg_exprs.extend([
                F.count(F.when(t_col == THREE_CLASS_DOWN, 1)).alias(f"down_{h}m"),
                F.count(F.when(t_col == THREE_CLASS_FLAT, 1)).alias(f"flat_{h}m"),
                F.count(F.when(t_col == THREE_CLASS_UP, 1)).alias(f"up_{h}m"),
            ])
        else:
            agg_exprs.extend([
                F.count(F.when(t_col == BINARY_DOWN_OR_FLAT, 1)).alias(f"down_{h}m"),
                F.count(F.when(t_col == BINARY_UP, 1)).alias(f"up_{h}m"),
            ])

    grouped_df = df.groupBy("symbol").agg(*agg_exprs).orderBy("symbol")
    rows = grouped_df.collect()

    results: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        symbol = r["symbol"]
        total_rows = r["total_rows"]
        results[symbol] = {}

        for h in horizons:
            valid_cnt = r[f"valid_{h}m"]
            null_cnt = r[f"null_{h}m"]
            up_cnt = r[f"up_{h}m"]
            down_cnt = r[f"down_{h}m"]
            up_share = (up_cnt / valid_cnt) if valid_cnt > 0 else 0.0

            stats: Dict[str, Any] = {
                "total_rows": total_rows,
                "valid_rows": valid_cnt,
                "null_rows": null_cnt,
                "up_count": up_cnt,
                "down_count": down_cnt,
                "up_share": up_share,
            }

            if is_three_class:
                stats["flat_count"] = r[f"flat_{h}m"]
                stats["flat_share"] = (r[f"flat_{h}m"] / valid_cnt) if valid_cnt > 0 else 0.0
                stats["down_share"] = (down_cnt / valid_cnt) if valid_cnt > 0 else 0.0

            results[symbol][f"{h}m"] = stats

    return results


def format_balance_table(
    balance_data: Dict[str, Dict[str, Any]],
    epsilon: Optional[float] = None,
) -> str:
    """Format balance data into a human-readable summary table."""
    is_three_class = epsilon is not None and epsilon > 0.0
    mode_str = f"3-class (epsilon={epsilon})" if is_three_class else "Binary (UP / DOWN-FLAT)"

    lines = [
        "=" * 92,
        f"LABEL CLASS BALANCE SUMMARY - Mode: {mode_str}",
        "=" * 92,
    ]

    if is_three_class:
        header = f"{'Symbol':<10} {'Horizon':<8} {'Total':<8} {'Valid':<8} {'Null':<6} {'DOWN':<8} {'FLAT':<8} {'UP':<8} {'Share UP':<10}"
        lines.append(header)
        lines.append("-" * 92)
        for sym, horizons_dict in sorted(balance_data.items()):
            for h_label, s in sorted(horizons_dict.items(), key=lambda x: int(x[0].replace('m', ''))):
                row = (
                    f"{sym:<10} {h_label:<8} {s['total_rows']:<8} {s['valid_rows']:<8} {s['null_rows']:<6} "
                    f"{s['down_count']:<8} {s['flat_count']:<8} {s['up_count']:<8} {s['up_share'] * 100:>7.2f}%"
                )
                lines.append(row)
    else:
        header = f"{'Symbol':<10} {'Horizon':<8} {'Total':<8} {'Valid':<8} {'Null':<6} {'DOWN/FLAT':<11} {'UP':<8} {'Share UP':<10}"
        lines.append(header)
        lines.append("-" * 92)
        for sym, horizons_dict in sorted(balance_data.items()):
            for h_label, s in sorted(horizons_dict.items(), key=lambda x: int(x[0].replace('m', ''))):
                row = (
                    f"{sym:<10} {h_label:<8} {s['total_rows']:<8} {s['valid_rows']:<8} {s['null_rows']:<6} "
                    f"{s['down_count']:<11} {s['up_count']:<8} {s['up_share'] * 100:>7.2f}%"
                )
                lines.append(row)

    lines.append("=" * 92)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and inspect gap-safe ML labels.")
    parser.add_argument(
        "--input",
        default="data/sample_klines.parquet",
        help="Input Parquet file path (default: data/sample_klines.parquet)",
    )
    parser.add_argument(
        "--horizons",
        default="5,15,60",
        help="Comma-separated forward horizons in minutes (default: 5,15,60)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Flat return threshold for 3-class target (default: None, binary)",
    )
    parser.add_argument(
        "--drop-nulls",
        action="store_true",
        default=False,
        help="Drop rows where any target is null",
    )
    parser.add_argument(
        "--master",
        default=None,
        help="Spark master URL (default: from SPARK_MASTER or local[*])",
    )
    parser.add_argument(
        "--log-level",
        default="WARN",
        help="Spark log level (default: WARN)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    horizon_list = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    spark = get_spark_session(
        app_name="CryptoMLLabelGenerator",
        master=args.master,
        log_level=args.log_level,
    )

    try:
        resolved_input = os.path.abspath(args.input).replace("\\", "/")
        logger.info("Loading klines from: %s", resolved_input)
        df_raw = spark.read.parquet(resolved_input)

        logger.info("Generating labels for horizons %s (epsilon=%s)...", horizon_list, args.epsilon)
        df_labeled = add_labels(
            df_raw,
            horizons=horizon_list,
            epsilon=args.epsilon,
            drop_null_labels=args.drop_nulls,
        )

        balance = compute_label_balance(
            df_labeled,
            horizons=horizon_list,
            epsilon=args.epsilon,
        )

        summary_table = format_balance_table(balance, epsilon=args.epsilon)
        print("\n" + summary_table + "\n")

    finally:
        spark.stop()
        logger.info("Spark session cleanly stopped.")


if __name__ == "__main__":
    main()
