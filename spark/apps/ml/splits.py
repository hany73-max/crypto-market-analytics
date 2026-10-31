"""
==============================================================================
spark/apps/ml/splits.py
Chronological dataset splitting with gap-aware label purge and embargo.

Guarantees:
  1. Strict chronological ordering: Train < Val < Test based on epoch ms timestamps.
  2. Global boundaries: Time boundaries are computed globally across all symbols,
     ensuring synchronous evaluation windows without cross-asset alignment bias.
  3. Purge / Embargo: For forward horizon h (minutes), eliminates overlap:
       - Train excludes any row where open_time + h * 60,000 >= val_start.
       - Val excludes any row where open_time + h * 60,000 >= test_start.
  4. Robust null handling: Records with null labels or features are excluded
     cleanly rather than cast to default values.
  5. Zero randomness: Pure deterministic timestamp partitioning.
==============================================================================
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Ensure repo root is available
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spark.apps.ml.features import drop_null_features
from spark.apps.ml.labels import MS_PER_MINUTE

logger = logging.getLogger("crypto_ml_splits")


def compute_time_boundaries(
    df: DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    time_col: str = "open_time",
) -> Tuple[int, int]:
    """
    Compute global chronological boundaries for train/val/test splits.

    Time boundaries are calculated using exact empirical quantiles across all rows
    in the dataset, ensuring all symbols share identical temporal cutoffs.

    Args:
        df: Input DataFrame containing the timestamp column.
        train_ratio: Proportion of time range allocated to train split (e.g. 0.70).
        val_ratio: Proportion of time range allocated to validation split (e.g. 0.15).
        time_col: Name of the timestamp column in epoch milliseconds.

    Returns:
        Tuple of (val_start, test_start) timestamps in epoch milliseconds.

    Raises:
        ValueError: If time_col is missing, ratios are invalid, or dataset is empty.
    """
    if time_col not in df.columns:
        raise ValueError(f"Timestamp column '{time_col}' not found in DataFrame.")

    if train_ratio <= 0.0 or val_ratio <= 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. "
            "Ratios must be positive and their sum must be less than 1.0."
        )

    # Use approxQuantile with 0.0 relativeError for exact quantile computation
    quantiles = df.stat.approxQuantile(time_col, [train_ratio, train_ratio + val_ratio], 0.0)
    if not quantiles or len(quantiles) < 2:
        raise ValueError(f"Failed to calculate time quantiles on column '{time_col}'.")

    val_start = int(round(quantiles[0]))
    test_start = int(round(quantiles[1]))

    if val_start >= test_start:
        raise ValueError(
            f"Calculated invalid boundaries: val_start={val_start} >= test_start={test_start}. "
            "Ensure the dataset has sufficient temporal depth."
        )

    return val_start, test_start


def split_time_series(
    df: DataFrame,
    horizon_minutes: int = 60,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    time_col: str = "open_time",
    label_col: Optional[str] = None,
    feature_cols: Optional[Sequence[str]] = None,
    drop_nulls: bool = True,
) -> Tuple[DataFrame, DataFrame, DataFrame, Dict[str, Any]]:
    """
    Split time series DataFrame into chronological Train, Validation, and Test sets with purge.

    Purge Rule:
      A sample at time t predicting horizon h looks ahead to t + h * 60,000 ms.
      To prevent information leaking across boundaries:
        - Train excludes all rows where open_time + h * 60,000 >= val_start.
        - Validation excludes all rows where open_time + h * 60,000 >= test_start.
        - Test contains all rows where open_time >= test_start.

    Null Handling:
      Boundaries are computed strictly on the temporal index. If drop_nulls is True,
      rows with null/NaN values in label_col or feature_cols are explicitly removed
      prior to partitioning, ensuring no incomplete records are assigned default classes.

    Args:
        df: Input DataFrame.
        horizon_minutes: Forward lookahead horizon h in minutes used for purge embargo.
        train_ratio: Fraction of timeline allocated to train split (default: 0.70).
        val_ratio: Fraction of timeline allocated to validation split (default: 0.15).
        test_ratio: Fraction of timeline allocated to test split (default: 0.15).
        time_col: Timestamp column name in epoch milliseconds (default: 'open_time').
        label_col: Optional target label column to validate and filter nulls.
        feature_cols: Optional sequence of feature columns to validate and filter nulls.
        drop_nulls: Whether to drop rows with null label or features (default: True).

    Returns:
        Tuple containing:
          - train_df: Purged training DataFrame.
          - val_df: Purged validation DataFrame.
          - test_df: Out-of-sample test DataFrame.
          - split_info: Dictionary detailing boundaries, purge settings, and split counts.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio} (sum={total_ratio})"
        )

    if horizon_minutes < 0:
        raise ValueError(f"horizon_minutes must be non-negative, got: {horizon_minutes}")

    # Compute global boundaries on time only
    val_start, test_start = compute_time_boundaries(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        time_col=time_col,
    )

    clean_df = df
    if drop_nulls:
        target_label = label_col if label_col is not None else ("label" if "label" in clean_df.columns else None)
        if target_label is not None and target_label in clean_df.columns:
            clean_df = clean_df.filter(F.col(target_label).isNotNull() & ~F.isnan(F.col(target_label)))
        if feature_cols is not None:
            clean_df = drop_null_features(clean_df, feature_cols=feature_cols)

    purge_ms = int(horizon_minutes * MS_PER_MINUTE)
    train_end = val_start - purge_ms
    val_end = test_start - purge_ms

    train_condition = F.col(time_col) < train_end
    val_condition = (F.col(time_col) >= val_start) & (F.col(time_col) < val_end)
    test_condition = F.col(time_col) >= test_start

    train_df = clean_df.filter(train_condition)
    val_df = clean_df.filter(val_condition)
    test_df = clean_df.filter(test_condition)

    train_count = train_df.count()
    val_count = val_df.count()
    test_count = test_df.count()
    total_count = train_count + val_count + test_count

    split_info = {
        "val_start": val_start,
        "test_start": test_start,
        "train_end": train_end,
        "val_end": val_end,
        "purge_horizon_minutes": horizon_minutes,
        "purge_window_ms": purge_ms,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "total_count": total_count,
    }

    logger.info(
        "Chronological split completed: train=%d, val=%d, test=%d | val_start=%d, test_start=%d, purge=%dm",
        train_count,
        val_count,
        test_count,
        val_start,
        test_start,
        horizon_minutes,
    )

    return train_df, val_df, test_df, split_info


# Aliases for flexible import
split_chronological = split_time_series
make_time_splits = split_time_series

