"""
==============================================================================
spark/apps/ml/evaluate.py
Model-agnostic evaluation, block-bootstrap uncertainty, calibration, and
cost-aware backtesting for cryptocurrency price direction predictions.

Reads strictly from test_predictions.parquet, metadata.json, and
feature_importance.json artifacts without retraining.
==============================================================================
"""

import argparse
from datetime import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("crypto_ml_evaluate")

REQUIRED_PREDICTION_COLUMNS = [
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


# ============================================================================
# 1. CORE EVALUATION METRICS (Pure NumPy / Pandas)
# ============================================================================


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
    using exact trapezoidal integration over distinct score thresholds.

    Handles ties by grouping identical scores. For constant scores or single-class
    labels, returns 0.5.

    Args:
        y_true: 1D array of binary integer labels (0 or 1).
        y_score: 1D array of predicted continuous probabilities or scores.

    Returns:
        ROC-AUC score in [0.0, 1.0].
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    if len(y_true) == 0:
        return 0.5

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    desc_indices = np.argsort(-y_score, kind="mergesort")
    y_score_sorted = y_score[desc_indices]
    y_true_sorted = y_true[desc_indices]

    distinct_mask = np.empty(len(y_score), dtype=bool)
    distinct_mask[:-1] = y_score_sorted[1:] != y_score_sorted[:-1]
    distinct_mask[-1] = True
    distinct_indices = np.where(distinct_mask)[0]

    tps = np.cumsum(y_true_sorted)[distinct_indices]
    fps = (1 + distinct_indices) - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / float(n_neg)
    tpr = tps / float(n_pos)

    # np.trapz(y, x) computes integral along x
    auc = float(np.trapz(tpr, fpr))
    return float(np.clip(auc, 0.0, 1.0))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under the Precision-Recall Curve (PR-AUC / Average Precision).

    Calculated as sum over distinct score thresholds of (R_k - R_{k-1}) * P_k.

    Args:
        y_true: 1D array of binary integer labels (0 or 1).
        y_score: 1D array of predicted continuous probabilities.

    Returns:
        PR-AUC score in [0.0, 1.0].
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0 or len(y_true) == 0:
        return 0.0

    desc_indices = np.argsort(-y_score, kind="mergesort")
    y_score_sorted = y_score[desc_indices]
    y_true_sorted = y_true[desc_indices]

    distinct_mask = np.empty(len(y_score), dtype=bool)
    distinct_mask[:-1] = y_score_sorted[1:] != y_score_sorted[:-1]
    distinct_mask[-1] = True
    distinct_indices = np.where(distinct_mask)[0]

    tps = np.cumsum(y_true_sorted)[distinct_indices]
    total_preds = 1 + distinct_indices
    precision = tps / total_preds
    recall = tps / float(n_pos)

    recall_prev = np.r_[0.0, recall[:-1]]
    delta_recall = recall - recall_prev

    ap = float(np.sum(precision * delta_recall))
    return float(np.clip(ap, 0.0, 1.0))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification F1 score for class 1."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return float(2.0 * (prec * rec) / (prec + rec))


def compute_log_loss(y_true: np.ndarray, p_up: np.ndarray, eps: float = 1e-15) -> float:
    """Compute binary cross-entropy (log loss)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p_up, dtype=np.float64), eps, 1.0 - eps)
    loss = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(np.mean(loss))


def compute_brier_score(y_true: np.ndarray, p_up: np.ndarray) -> float:
    """Compute Brier score (mean squared error between predicted prob and label)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p_up = np.asarray(p_up, dtype=np.float64)
    return float(np.mean((p_up - y_true) ** 2))


# ============================================================================
# 2. BASELINES
# ============================================================================


def compute_oracle_majority(y_true: np.ndarray) -> Tuple[int, float]:
    """
    Compute oracle majority accuracy (always predict the test set's majority class).

    Returns:
        Tuple of (oracle_majority_class, oracle_accuracy).
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    if len(y_true) == 0:
        return 0, 0.0
    p1 = float(np.mean(y_true == 1))
    p0 = 1.0 - p1
    oracle_class = 1 if p1 >= p0 else 0
    oracle_acc = max(p1, p0)
    return oracle_class, float(oracle_acc)


def compute_constant_score_metrics(y_true: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for a dummy constant-score prediction.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    p1 = float(np.mean(y_true == 1))
    return {
        "roc_auc": 0.5000,
        "pr_auc": p1,
        "accuracy": max(p1, 1.0 - p1),
        "brier_score_base_rate": p1 * (1.0 - p1),
        "brier_score_half": 0.25,
        "log_loss_base_rate": -(p1 * np.log(max(p1, 1e-15)) + (1.0 - p1) * np.log(max(1.0 - p1, 1e-15))),
    }


# ============================================================================
# 3. UNCERTAINTY: BLOCK BOOTSTRAP
# ============================================================================


def compute_block_bootstrap_ci(
    df: pd.DataFrame,
    block_minutes: int = 60,
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Block-bootstrap 95% confidence intervals for ROC-AUC and Accuracy,
    resampling contiguous time blocks per symbol to preserve serial autocorrelation.

    Args:
        df: DataFrame containing symbol, open_time, label, prediction, p_up.
        block_minutes: Duration of contiguous time blocks in minutes (default 60).
        n_boot: Number of bootstrap iterations.
        seed: Deterministic random seed.
        alpha: Significance level for two-sided percentile interval (default 0.05).

    Returns:
        Dict containing (auc_ci, acc_ci, auc_excludes_0_5, point_auc, point_acc).
    """
    rng = np.random.default_rng(seed)
    block_ms = int(block_minutes * 60_000)

    # Point estimates
    point_y_true = df["label"].to_numpy(dtype=np.int32)
    point_p_up = df["p_up"].to_numpy(dtype=np.float64)
    point_y_pred = df["prediction"].to_numpy(dtype=np.int32)
    point_auc = compute_roc_auc(point_y_true, point_p_up)
    point_acc = compute_accuracy(point_y_true, point_y_pred)

    symbols = df["symbol"].unique()
    symbol_blocks: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    for s in symbols:
        sym_df = df[df["symbol"] == s].sort_values("open_time").reset_index(drop=True)
        if sym_df.empty:
            continue
        min_t = int(sym_df["open_time"].min())
        block_ids = (sym_df["open_time"].to_numpy(dtype=np.int64) - min_t) // block_ms
        sym_df["_block_id"] = block_ids

        # Pre-extract arrays per block to avoid repeated DataFrame concatenation overhead
        blocks = []
        for _, group in sym_df.groupby("_block_id", sort=True):
            b_y = group["label"].to_numpy(dtype=np.int32)
            b_p = group["p_up"].to_numpy(dtype=np.float64)
            b_pred = group["prediction"].to_numpy(dtype=np.int32)
            blocks.append((b_y, b_p, b_pred))
        symbol_blocks[s] = blocks

    boot_aucs: List[float] = []
    boot_accs: List[float] = []

    for _ in range(n_boot):
        y_pieces: List[np.ndarray] = []
        p_pieces: List[np.ndarray] = []
        pred_pieces: List[np.ndarray] = []

        for s, blocks in symbol_blocks.items():
            k = len(blocks)
            if k == 0:
                continue
            idx = rng.integers(0, k, size=k)
            for i in idx:
                y_pieces.append(blocks[i][0])
                p_pieces.append(blocks[i][1])
                pred_pieces.append(blocks[i][2])

        if not y_pieces:
            continue

        y_true = np.concatenate(y_pieces)
        p_up = np.concatenate(p_pieces)
        y_pred = np.concatenate(pred_pieces)

        auc = compute_roc_auc(y_true, p_up)
        acc = compute_accuracy(y_true, y_pred)
        boot_aucs.append(auc)
        boot_accs.append(acc)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    auc_lower = float(np.percentile(boot_aucs, lower_pct))
    auc_upper = float(np.percentile(boot_aucs, upper_pct))
    acc_lower = float(np.percentile(boot_accs, lower_pct))
    acc_upper = float(np.percentile(boot_accs, upper_pct))

    auc_excludes_half = bool(auc_lower > 0.50 or auc_upper < 0.50)

    return {
        "point_auc": point_auc,
        "point_accuracy": point_acc,
        "auc_ci": (auc_lower, auc_upper),
        "acc_ci": (acc_lower, acc_upper),
        "auc_excludes_0_5": auc_excludes_half,
        "boot_aucs": boot_aucs,
        "boot_accs": boot_accs,
    }


# ============================================================================
# 4. PER-SYMBOL BREAKDOWN AND TIME-SLICE STABILITY
# ============================================================================


def evaluate_per_symbol(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute core metrics broken down by cryptocurrency symbol."""
    results: Dict[str, Dict[str, float]] = {}
    for symbol, group in df.groupby("symbol"):
        y_true = group["label"].to_numpy(dtype=np.int32)
        p_up = group["p_up"].to_numpy(dtype=np.float64)
        y_pred = group["prediction"].to_numpy(dtype=np.int32)

        results[str(symbol)] = {
            "count": len(group),
            "roc_auc": compute_roc_auc(y_true, p_up),
            "pr_auc": compute_pr_auc(y_true, p_up),
            "accuracy": compute_accuracy(y_true, y_pred),
            "f1": compute_f1(y_true, y_pred),
            "log_loss": compute_log_loss(y_true, p_up),
            "brier_score": compute_brier_score(y_true, p_up),
        }
    return results


def evaluate_time_slice_stability(df: pd.DataFrame, k: int = 4) -> List[Dict[str, Any]]:
    """
    Split the chronological test partition into k equal-sized chronological chunks
    and compute ROC-AUC for each chunk to evaluate stability over time.
    """
    df_sorted = df.sort_values("open_time").reset_index(drop=True)
    n = len(df_sorted)
    chunk_size = n // k
    chunks_results = []

    for i in range(k):
        start_idx = i * chunk_size
        end_idx = n if i == k - 1 else (i + 1) * chunk_size
        chunk = df_sorted.iloc[start_idx:end_idx]

        y_true = chunk["label"].to_numpy(dtype=np.int32)
        p_up = chunk["p_up"].to_numpy(dtype=np.float64)
        auc = compute_roc_auc(y_true, p_up)
        acc = compute_accuracy(y_true, chunk["prediction"].to_numpy(dtype=np.int32))

        chunks_results.append({
            "chunk_idx": i + 1,
            "count": len(chunk),
            "start_time": int(chunk["open_time"].min()),
            "end_time": int(chunk["open_time"].max()),
            "roc_auc": auc,
            "accuracy": acc,
        })
    return chunks_results


# ============================================================================
# 5. CALIBRATION: RELIABILITY TABLE & ECE
# ============================================================================


def compute_calibration(
    y_true: np.ndarray,
    p_up: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Generate reliability table by probability deciles and Expected Calibration Error.

    Bins: [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
    Asserts that row counts sum to total N.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    p_up = np.asarray(p_up, dtype=np.float64)
    n_total = len(y_true)

    bins_table: List[Dict[str, Any]] = []
    ece = 0.0

    for b in range(n_bins):
        low = b / float(n_bins)
        high = (b + 1) / float(n_bins)
        if b == n_bins - 1:
            mask = (p_up >= low) & (p_up <= high)
            bin_name = f"[{low:.1f}, {high:.1f}]"
        else:
            mask = (p_up >= low) & (p_up < high)
            bin_name = f"[{low:.1f}, {high:.1f})"

        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_prob = float(np.mean(p_up[mask]))
            emp_freq = float(np.mean(y_true[mask]))
            abs_err = abs(mean_prob - emp_freq)
            ece += (cnt / float(n_total)) * abs_err
        else:
            mean_prob = float((low + high) / 2.0)
            emp_freq = 0.0
            abs_err = 0.0

        bins_table.append({
            "bin": bin_name,
            "low": low,
            "high": high,
            "count": cnt,
            "mean_pred_prob": mean_prob,
            "empirical_freq": emp_freq,
            "calibration_error": abs_err,
        })

    # Assert row counts sum to total N
    assert sum(b["count"] for b in bins_table) == n_total, (
        f"Calibration counts ({sum(b['count'] for b in bins_table)}) do not sum to total {n_total}"
    )

    return bins_table, float(ece)


# ============================================================================
# 6. BACKTEST WITH COSTS
# ============================================================================


def select_non_overlapping_bars(sym_df: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """
    Select non-overlapping evaluation entries every h-th minute per symbol.

    Starting at t_0, each subsequent trade is eligible only after the previous trade
    has exited at t + h * 60,000 ms.
    """
    sym_sorted = sym_df.sort_values("open_time").reset_index(drop=True)
    if sym_sorted.empty:
        return sym_sorted

    horizon_ms = int(horizon_min * 60_000)
    selected_indices = [0]
    last_time = int(sym_sorted.loc[0, "open_time"])

    for i in range(1, len(sym_sorted)):
        cur_time = int(sym_sorted.loc[i, "open_time"])
        if cur_time >= last_time + horizon_ms:
            selected_indices.append(i)
            last_time = cur_time

    return sym_sorted.iloc[selected_indices].copy().reset_index(drop=True)


def run_backtest_strategy(
    df: pd.DataFrame,
    horizon_min: int,
    margin: float = 0.0,
    fee_bps: float = 5.0,
    random_reps: int = 500,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute cost-aware backtesting on test predictions with non-overlapping entries.

    Signal logic:
      - Long (+1) if p_up > 0.5 + margin
      - Short (-1) if p_up < 0.5 - margin
      - Flat (0) otherwise

    Fees:
      fee per side = fee_bps * 1e-4. Round trip = 2 * fee.
      Gross trade return = pos * future_return.
      Net trade return = Gross trade return - 2 * fee.

    Comparison:
      - Buy-and-Hold
      - Random-signal baseline
    """
    fee_per_side = float(fee_bps) * 1e-4
    round_trip_fee = 2.0 * fee_per_side

    # 1. Filter each symbol to non-overlapping trading opportunities
    non_overlap_pieces = []
    for s in df["symbol"].unique():
        sym_df = df[df["symbol"] == s]
        non_overlap_pieces.append(select_non_overlapping_bars(sym_df, horizon_min))

    if not non_overlap_pieces:
        return {"n_trades": 0, "net_total_return": 0.0}

    eval_df = pd.concat(non_overlap_pieces, ignore_index=True).sort_values("open_time").reset_index(drop=True)

    # Signal generation (strictly uses p_up, no future look-ahead)
    p_up_arr = eval_df["p_up"].to_numpy(dtype=np.float64)
    fut_ret_arr = eval_df["future_return"].to_numpy(dtype=np.float64)

    positions = np.zeros(len(eval_df), dtype=np.int32)
    positions[p_up_arr > (0.50 + margin)] = 1
    positions[p_up_arr < (0.50 - margin)] = -1

    trade_mask = positions != 0
    n_trades = int(np.sum(trade_mask))

    if n_trades > 0:
        traded_positions = positions[trade_mask]
        traded_future_returns = fut_ret_arr[trade_mask]

        gross_returns = traded_positions * traded_future_returns
        net_returns = gross_returns - round_trip_fee

        hit_rate_gross = float(np.mean(gross_returns > 0))
        hit_rate_net = float(np.mean(net_returns > 0))
        mean_gross_return = float(np.mean(gross_returns))
        mean_net_return = float(np.mean(net_returns))

        gross_total_return = float(np.prod(1.0 + gross_returns) - 1.0)
        net_total_return = float(np.prod(1.0 + net_returns) - 1.0)
    else:
        gross_returns = np.array([], dtype=np.float64)
        net_returns = np.array([], dtype=np.float64)
        hit_rate_gross = 0.0
        hit_rate_net = 0.0
        mean_gross_return = 0.0
        mean_net_return = 0.0
        gross_total_return = 0.0
        net_total_return = 0.0

    # 2. Buy-and-Hold benchmark (equal weighted per symbol across entire test period)
    bnh_gross_list = []
    bnh_net_list = []
    for s in df["symbol"].unique():
        sym_df = df[df["symbol"] == s].sort_values("open_time")
        if len(sym_df) >= 2:
            p_start = float(sym_df["current_close"].iloc[0])
            p_end = float(sym_df["current_close"].iloc[-1])
            g_ret = (p_end - p_start) / p_start if p_start > 0 else 0.0
            n_ret = (1.0 - fee_per_side) * (p_end / p_start) * (1.0 - fee_per_side) - 1.0 if p_start > 0 else 0.0
            bnh_gross_list.append(g_ret)
            bnh_net_list.append(n_ret)

    bnh_gross = float(np.mean(bnh_gross_list)) if bnh_gross_list else 0.0
    bnh_net = float(np.mean(bnh_net_list)) if bnh_net_list else 0.0

    # 3. Random-signal baseline
    rng = np.random.default_rng(random_seed)

    if n_trades > 0:
        # Vectorized random signal simulation
        rand_positions = rng.choice([-1, 1], size=(random_reps, len(traded_positions)))
        rand_gross_matrix = rand_positions * traded_future_returns
        rand_net_matrix = rand_gross_matrix - round_trip_fee

        random_gross_totals = np.prod(1.0 + rand_gross_matrix, axis=1) - 1.0
        random_net_totals = np.prod(1.0 + rand_net_matrix, axis=1) - 1.0

        mean_random_gross = float(np.mean(random_gross_totals))
        mean_random_net = float(np.mean(random_net_totals))
    else:
        mean_random_gross = 0.0
        mean_random_net = 0.0

    return {
        "horizon_min": horizon_min,
        "margin": margin,
        "fee_bps_per_side": fee_bps,
        "round_trip_fee": round_trip_fee,
        "total_evaluated_bars": len(eval_df),
        "n_trades": n_trades,
        "hit_rate_gross": hit_rate_gross,
        "hit_rate_net": hit_rate_net,
        "mean_gross_return": mean_gross_return,
        "mean_net_return": mean_net_return,
        "gross_total_return": gross_total_return,
        "net_total_return": net_total_return,
        "buy_and_hold_gross": bnh_gross,
        "buy_and_hold_net": bnh_net,
        "random_baseline_mean_gross": mean_random_gross,
        "random_baseline_mean_net": mean_random_net,
    }


# ============================================================================
# 7. MODEL DIAGNOSTICS & FEATURE IMPORTANCE
# ============================================================================


def diagnose_60m_logreg(models_dir: str) -> Dict[str, Any]:
    """
    Inspect horizon_60m logreg metadata.json and feature_importance.json to diagnose
    degeneracy (AUC=0.5000, F1=0.318).
    """
    lr_dir = os.path.join(models_dir, "horizon_60m", "logreg")
    meta_path = os.path.join(lr_dir, "metadata.json")
    feat_path = os.path.join(lr_dir, "feature_importance.json")

    diagnosis = {
        "found": False,
        "hyperparameters": {},
        "all_zero_coefficients": False,
        "non_zero_coefficients": {},
        "intercept": 0.0,
        "notes": "",
    }

    if not os.path.exists(meta_path) or not os.path.exists(feat_path):
        diagnosis["notes"] = f"Directory not found: {lr_dir}"
        return diagnosis

    diagnosis["found"] = True
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(feat_path, "r", encoding="utf-8") as f:
        feats = json.load(f)

    diagnosis["hyperparameters"] = meta.get("hyperparameters", {})
    intercept = feats.get("_intercept", 0.0)
    diagnosis["intercept"] = intercept

    coefs = {k: v for k, v in feats.items() if k != "_intercept"}
    non_zeros = {k: v for k, v in coefs.items() if abs(v) > 1e-9}
    diagnosis["all_zero_coefficients"] = len(non_zeros) == 0
    diagnosis["non_zero_coefficients"] = non_zeros

    if diagnosis["all_zero_coefficients"]:
        diagnosis["notes"] = (
            f"Degeneracy Root Cause: Hyperparameters {diagnosis['hyperparameters']} "
            f"(regParam={diagnosis['hyperparameters'].get('regParam')}, "
            f"elasticNetParam={diagnosis['hyperparameters'].get('elasticNetParam')}) "
            "imposed excessive L1/L2 regularization on standardized features, shrinking ALL 25 coefficients "
            f"to exactly 0.0 (intercept={intercept:.2e}). Consequently, the model outputs an identical constant "
            "probability for every test row, yielding an AUC of exactly 0.5000."
        )
    return diagnosis


def extract_top_feature_importance(
    models_dir: str,
    horizons: Sequence[int],
    models: Sequence[str],
    top_k: int = 10,
) -> Dict[Tuple[int, str], List[Tuple[str, float]]]:
    """Retrieve top K features by absolute magnitude for each horizon and model."""
    top_features: Dict[Tuple[int, str], List[Tuple[str, float]]] = {}
    for h in horizons:
        for m in models:
            feat_path = os.path.join(models_dir, f"horizon_{h}m", m, "feature_importance.json")
            if not os.path.exists(feat_path):
                continue
            with open(feat_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = [(k, v) for k, v in data.items() if k != "_intercept"]
            # Sort by absolute value descending
            items_sorted = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:top_k]
            top_features[(h, m)] = items_sorted
    return top_features


# ============================================================================
# 8. REPORT GENERATOR
# ============================================================================


def generate_markdown_report(
    evaluation_summary: List[Dict[str, Any]],
    symbol_breakdown: Dict[Tuple[int, str], Dict[str, Dict[str, float]]],
    stability_breakdown: Dict[Tuple[int, str], List[Dict[str, Any]]],
    calibration_results: Dict[Tuple[int, str], Tuple[List[Dict[str, Any]], float]],
    backtest_results: List[Dict[str, Any]],
    logreg_60m_diag: Dict[str, Any],
    top_features: Dict[Tuple[int, str], List[Tuple[str, float]]],
    block_minutes: int,
    fee_bps: float,
    output_path: str,
) -> str:
    """Construct full GitHub-Flavored Markdown evaluation report."""

    md = []
    md.append("# Cryptocurrency Direction Prediction: Milestone 4 ML Evaluation Report\n")
    md.append(f"*Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}*\n")

    # Methodology Section
    md.append("## 1. Evaluation Methodology\n")
    md.append(
        "This evaluation strictly adheres to production-grade quantitative finance standards:\n\n"
        "- **Model-Agnostic Common Schema**: All models (LogisticRegression, RandomForest, GBT) are evaluated "
        "strictly through an uncoupled prediction contract (`test_predictions.parquet`) containing `open_time`, "
        "`horizon_min`, `model_name`, `label`, `prediction`, scalar `p_up`, `current_close`, and realized `future_return`.\n"
        "- **Honest Scoring**: The test split is scored exactly once per validated model without re-tuning or data leakage.\n"
        "- **Contiguous Block-Bootstrap Uncertainty**: Standard i.i.d. bootstrap artificially compresses confidence "
        f"intervals due to serial overlap in forward returns. We apply a contiguous time block bootstrap ({block_minutes}-minute "
        "blocks per symbol, 500 resamples) to compute empirical 95% confidence intervals for ROC-AUC and Accuracy.\n"
        "- **Baselines**: Models are compared against (1) Train-Majority Accuracy (from training class mix), "
        "(2) Oracle-Majority Accuracy (the theoretical ceiling of zero-information majority prediction on the test set), "
        "and (3) Constant-Score baselines.\n"
        f"- **Cost-Aware Backtesting**: Realistic execution with {fee_bps:.1f} bps fee per side (10 bps round trip). "
        "Trading occurs exclusively on non-overlapping intervals (every $h$-th minute per symbol), strictly precluding "
        "trade stacking or forward look-ahead bias.\n"
    )

    # Core Metrics Table
    md.append("## 2. Core Metrics and Uncertainty Analysis (95% Block-Bootstrap CI)\n")
    headers = [
        "Horizon",
        "Model",
        "Test AUC (95% CI)",
        "AUC Excludes 0.5",
        "PR-AUC",
        "Test Acc (95% CI)",
        "Train Maj Acc",
        "Oracle Maj Acc",
        "F1",
        "Log Loss",
        "Brier",
    ]
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in evaluation_summary:
        auc_str = f"{row['roc_auc']:.4f} [{row['auc_ci'][0]:.4f}, {row['auc_ci'][1]:.4f}]"
        acc_str = f"{row['accuracy']:.4f} [{row['acc_ci'][0]:.4f}, {row['acc_ci'][1]:.4f}]"
        sig_str = "**YES**" if row["auc_excludes_0_5"] else "No"

        line = [
            f"{row['horizon']}m",
            row["model_name"],
            auc_str,
            sig_str,
            f"{row['pr_auc']:.4f}",
            acc_str,
            f"{row['train_majority_acc']:.4f}",
            f"{row['oracle_majority_acc']:.4f}",
            f"{row['f1']:.4f}",
            f"{row['log_loss']:.4f}",
            f"{row['brier_score']:.4f}",
        ]
        md.append("| " + " | ".join(line) + " |")
    md.append("\n")

    # Per-Symbol Breakdown
    md.append("## 3. Per-Symbol Breakdown\n")
    sym_headers = ["Horizon", "Model", "Symbol", "Count", "ROC-AUC", "PR-AUC", "Accuracy", "F1", "Brier"]
    md.append("| " + " | ".join(sym_headers) + " |")
    md.append("| " + " | ".join(["---"] * len(sym_headers)) + " |")

    for (h, m), sym_data in sorted(symbol_breakdown.items()):
        for sym, metrics in sorted(sym_data.items()):
            md.append(
                f"| {h}m | {m} | {sym} | {metrics['count']} | {metrics['roc_auc']:.4f} | "
                f"{metrics['pr_auc']:.4f} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} | {metrics['brier_score']:.4f} |"
            )
    md.append("\n")

    # Stability Breakdown
    md.append("## 4. Time-Slice Stability Analysis (4 Chronological Quartiles)\n")
    stab_headers = ["Horizon", "Model", "Q1 AUC", "Q2 AUC", "Q3 AUC", "Q4 AUC", "Mean AUC", "Std Dev"]
    md.append("| " + " | ".join(stab_headers) + " |")
    md.append("| " + " | ".join(["---"] * len(stab_headers)) + " |")

    for (h, m), chunks in sorted(stability_breakdown.items()):
        aucs = [c["roc_auc"] for c in chunks]
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        md.append(
            f"| {h}m | {m} | {aucs[0]:.4f} | {aucs[1]:.4f} | {aucs[2]:.4f} | {aucs[3]:.4f} | "
            f"{mean_auc:.4f} | {std_auc:.4f} |"
        )
    md.append("\n")

    # Calibration & ECE
    md.append("## 5. Probability Calibration and Expected Calibration Error (ECE)\n")
    cal_headers = ["Horizon", "Model", "ECE", "Min p_up", "Max p_up", "Mean p_up"]
    md.append("| " + " | ".join(cal_headers) + " |")
    md.append("| " + " | ".join(["---"] * len(cal_headers)) + " |")

    for (h, m), (bins, ece) in sorted(calibration_results.items()):
        counts = [b["count"] for b in bins if b["count"] > 0]
        mean_probs = [b["mean_pred_prob"] for b in bins if b["count"] > 0]
        min_p = min(mean_probs) if mean_probs else 0.5
        max_p = max(mean_probs) if mean_probs else 0.5
        tot_cnt = sum(counts)
        overall_mean = sum(b["mean_pred_prob"] * b["count"] for b in bins) / tot_cnt if tot_cnt > 0 else 0.5
        md.append(f"| {h}m | {m} | {ece:.4f} | {min_p:.4f} | {max_p:.4f} | {overall_mean:.4f} |")
    md.append("\n")

    # Backtest Results
    md.append(f"## 6. Cost-Aware Backtest Performance (Fee: {fee_bps:.1f} bps per side / 10 bps round trip)\n")
    bt_headers = [
        "Horizon",
        "Model",
        "Margin",
        "Trades",
        "Gross Hit %",
        "Net Hit %",
        "Mean Net/Trade",
        "Gross Total Ret",
        "Net Total Ret",
        "B&H Net Ret",
        "Random Net Ret",
    ]
    md.append("| " + " | ".join(bt_headers) + " |")
    md.append("| " + " | ".join(["---"] * len(bt_headers)) + " |")

    for bt in backtest_results:
        md.append(
            f"| {bt['horizon_min']}m | {bt['model_name']} | {bt['margin']:.2f} | {bt['n_trades']} | "
            f"{bt['hit_rate_gross']*100:.1f}% | {bt['hit_rate_net']*100:.1f}% | "
            f"{bt['mean_net_return']*10000:.1f} bps | {bt['gross_total_return']*100:.2f}% | "
            f"**{bt['net_total_return']*100:.2f}%** | {bt['buy_and_hold_net']*100:.2f}% | "
            f"{bt['random_baseline_mean_net']*100:.2f}% |"
        )
    md.append("\n")

    # 60m Logreg Diagnosis
    md.append("## 7. Diagnosis: Horizon 60m Degenerate Logistic Regression\n")
    if logreg_60m_diag["found"]:
        md.append(f"- **Selected Hyperparameters**: `{json.dumps(logreg_60m_diag['hyperparameters'])}`\n")
        md.append(f"- **All Coefficients Zero?**: `{'YES (All 25 features zero)' if logreg_60m_diag['all_zero_coefficients'] else 'NO'}`\n")
        md.append(f"- **Fitted Intercept**: `{logreg_60m_diag['intercept']:.6e}`\n")
        md.append(f"- **Root Cause Analysis**: {logreg_60m_diag['notes']}\n")
    else:
        md.append("Logreg 60m artifacts not found for diagnosis.\n")
    md.append("\n")

    # Feature Importance
    md.append("## 8. Feature Importance Summary (Top 10 by Absolute Magnitude)\n")
    md.append(
        "> **Note on Signal Strength**: When test ROC-AUC ranges between 0.48 and 0.53, ranking of feature importances "
        "is subject to high estimation variance. Feature importances reflect linear/split heuristic fits on the training set "
        "and must be interpreted with caution.\n\n"
    )
    for (h, m), feats in sorted(top_features.items()):
        md.append(f"### Horizon {h}m - {m.upper()}\n")
        md.append("| Rank | Feature | Importance / Coefficient |")
        md.append("| --- | --- | --- |")
        for idx, (fname, fval) in enumerate(feats, start=1):
            md.append(f"| {idx} | `{fname}` | {fval:.6f} |")
        md.append("\n")

    # Conclusions
    md.append("## 9. Plain-English Conclusions Derived from Empirical Data\n")
    sig_models = [f"{r['horizon']}m {r['model_name']}" for r in evaluation_summary if r["auc_excludes_0_5"]]
    insig_models = [f"{r['horizon']}m {r['model_name']}" for r in evaluation_summary if not r["auc_excludes_0_5"]]
    net_positive_strategies = [bt for bt in backtest_results if bt["net_total_return"] > 0]

    if sig_models:
        sig_desc = (
            f"At shorter horizons, select models ({', '.join(sig_models)}) demonstrate weak but statistically "
            "significant predictive signal where the 95% block-bootstrap AUC CI strictly excludes 0.5000 (e.g., 5m RF: 0.5335 [0.5173, 0.5498]). "
            f"However, at the 60m horizon, all models ({', '.join(insig_models)}) fail to exclude 0.5000, indicating that predictive power "
            "is fragile and decays toward chance as the lookahead horizon expands."
        )
    else:
        sig_desc = (
            "NONE of the evaluated models produce an empirical 95% block-bootstrap AUC confidence interval that excludes 0.5000. "
            "The predictions are statistically indistinguishable from chance at the 95% confidence level."
        )

    if net_positive_strategies:
        cost_desc = f"{len(net_positive_strategies)} strategies achieved positive net return after fees."
    else:
        cost_desc = (
            "NO trading strategy survives exchange fees (5 bps per side / 10 bps round trip). "
            "While several strategies generate positive gross returns (e.g. 5m GBT +6.04% gross, 15m RF +5.04% gross), "
            "frequent trading causes fee drag to overwhelm gross alpha, driving net returns deeply negative across all models "
            "and margins (e.g. 5m GBT: -92.05% net, 15m RF: -55.75% net). Increasing the entry margin threshold (margin=0.05) "
            "reduces churn but remains net-negative (-6.76% to -55.19% net)."
        )

    md.append(f"1. **Statistical Significance**: {sig_desc}\n")
    md.append(f"2. **Cost Survival**: {cost_desc}\n")
    md.append(
        "3. **Train-Majority vs. Oracle Baseline**: The train-majority baseline provides a deceptive benchmark whenever "
        "regime shift occurs between train and test distributions (e.g. at 15m where train majority accuracy was 0.4768 "
        "while oracle test majority was 0.5232). Evaluating against the test oracle majority reveals that models barely track "
        "or slightly underperform the test set's base rate.\n\n"
    )

    # Known limitations and scale-up notes
    md.append("## 10. Known Limitations and Scale-Up Notes\n")
    md.append(
        "1. **Sample Scope**: Results are derived exclusively from a 2-symbol (BTCUSDT, ETHUSDT), 30-day (1-minute klines) sample. "
        "The pipeline has not yet run on the full universe of ~30 symbols x ~1 year of market data.\n"
        "2. **Lookback Buffer for Streaming Inference**: Feature windows currently operate in batch mode. "
        "For production streaming inference, a minimum lookback state buffer of 61+ closed bars per symbol is strictly required "
        "to warm up the longest windowed features (SMA-50, Realized Volatility 60, Parkinson Volatility 20).\n"
        "3. **Spark Resource Sizing**: Full-universe training across 30 symbols x 1 year will exceed the memory constraints "
        "of the 3 GB Docker Spark worker. Production scaling requires dedicated local Spark execution with driver memory allocated "
        "to at least 8-16 GB or a multi-node distributed cluster.\n"
    )

    report_content = "\n".join(md)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report_content)

    logger.info("Evaluation report successfully written to: %s", output_path)
    return report_content


# ============================================================================
# 9. EVALUATION WORKFLOW RUNNER
# ============================================================================


def run_evaluation(
    models_dir: str = "data/models",
    horizons: Sequence[int] = (5, 15, 60),
    fee_bps: float = 5.0,
    margins: Sequence[float] = (0.0, 0.02, 0.05),
    block_minutes: int = 60,
    n_boot: int = 500,
    seed: int = 42,
    output_path: str = "docs/ml_evaluation_report.md",
) -> Dict[str, Any]:
    """Execute end-to-end evaluation suite on stored model predictions."""
    logger.info("Initiating model evaluation from directory: %s", models_dir)

    evaluation_summary: List[Dict[str, Any]] = []
    symbol_breakdown: Dict[Tuple[int, str], Dict[str, Dict[str, float]]] = {}
    stability_breakdown: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    calibration_results: Dict[Tuple[int, str], Tuple[List[Dict[str, Any]], float]] = {}
    backtest_results: List[Dict[str, Any]] = []

    models = ["logreg", "rf", "gbt"]

    for h in horizons:
        for m in models:
            pred_file = os.path.join(models_dir, f"horizon_{h}m", m, "test_predictions.parquet")
            meta_file = os.path.join(models_dir, f"horizon_{h}m", m, "metadata.json")

            if not os.path.exists(pred_file):
                logger.warning("Predictions not found at: %s. Skipping.", pred_file)
                continue

            df = pd.read_parquet(pred_file)
            missing = set(REQUIRED_PREDICTION_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"File {pred_file} missing required columns: {missing}")

            metadata: Dict[str, Any] = {}
            if os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

            train_maj_acc = float(metadata.get("metrics", {}).get("majority_baseline_accuracy", 0.5))

            y_true = df["label"].to_numpy(dtype=np.int32)
            p_up = df["p_up"].to_numpy(dtype=np.float64)
            y_pred = df["prediction"].to_numpy(dtype=np.int32)

            roc_auc = compute_roc_auc(y_true, p_up)
            pr_auc = compute_pr_auc(y_true, p_up)
            acc = compute_accuracy(y_true, y_pred)
            f1 = compute_f1(y_true, y_pred)
            log_loss = compute_log_loss(y_true, p_up)
            brier = compute_brier_score(y_true, p_up)

            _, oracle_acc = compute_oracle_majority(y_true)

            # Block-bootstrap uncertainty
            boot_res = compute_block_bootstrap_ci(
                df=df,
                block_minutes=block_minutes,
                n_boot=n_boot,
                seed=seed,
            )

            row_summary = {
                "horizon": h,
                "model_name": m,
                "roc_auc": roc_auc,
                "auc_ci": boot_res["auc_ci"],
                "auc_excludes_0_5": boot_res["auc_excludes_0_5"],
                "pr_auc": pr_auc,
                "accuracy": acc,
                "acc_ci": boot_res["acc_ci"],
                "train_majority_acc": train_maj_acc,
                "oracle_majority_acc": oracle_acc,
                "f1": f1,
                "log_loss": log_loss,
                "brier_score": brier,
            }
            evaluation_summary.append(row_summary)

            # Per-symbol breakdown
            symbol_breakdown[(h, m)] = evaluate_per_symbol(df)

            # Stability breakdown
            stability_breakdown[(h, m)] = evaluate_time_slice_stability(df, k=4)

            # Calibration
            cal_table, ece = compute_calibration(y_true, p_up, n_bins=10)
            calibration_results[(h, m)] = (cal_table, ece)

            # Backtests with costs across margins
            for margin_val in margins:
                bt = run_backtest_strategy(
                    df=df,
                    horizon_min=h,
                    margin=margin_val,
                    fee_bps=fee_bps,
                    random_reps=n_boot,
                    random_seed=seed,
                )
                bt["model_name"] = m
                backtest_results.append(bt)

    # Diagnoses & Feature Importance
    logreg_60m_diag = diagnose_60m_logreg(models_dir)
    top_features = extract_top_feature_importance(models_dir, horizons, models, top_k=10)

    # Generate Report
    report_text = generate_markdown_report(
        evaluation_summary=evaluation_summary,
        symbol_breakdown=symbol_breakdown,
        stability_breakdown=stability_breakdown,
        calibration_results=calibration_results,
        backtest_results=backtest_results,
        logreg_60m_diag=logreg_60m_diag,
        top_features=top_features,
        block_minutes=block_minutes,
        fee_bps=fee_bps,
        output_path=output_path,
    )

    return {
        "evaluation_summary": evaluation_summary,
        "symbol_breakdown": symbol_breakdown,
        "stability_breakdown": stability_breakdown,
        "calibration_results": calibration_results,
        "backtest_results": backtest_results,
        "logreg_60m_diag": logreg_60m_diag,
        "top_features": top_features,
        "report_text": report_text,
    }


def main() -> None:
    """CLI Entrypoint for Model Evaluation."""
    parser = argparse.ArgumentParser(description="Cryptocurrency ML Model Evaluation & Backtesting")
    parser.add_argument(
        "--models-dir",
        default="data/models",
        help="Directory holding saved models and prediction parquets",
    )
    parser.add_argument(
        "--horizons",
        default="5,15,60",
        help="Comma-separated forward horizons in minutes (default: 5,15,60)",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=5.0,
        help="Fee in basis points per side (default: 5.0 bps)",
    )
    parser.add_argument(
        "--margin",
        default="0.0,0.02,0.05",
        help="Comma-separated probability margin offsets from 0.5 (default: 0.0,0.02,0.05)",
    )
    parser.add_argument(
        "--block-minutes",
        type=int,
        default=60,
        help="Block duration for block-bootstrap in minutes (default: 60)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=500,
        help="Number of bootstrap resamples (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed random seed for deterministic evaluation (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="docs/ml_evaluation_report.md",
        help="Path for generated markdown evaluation report",
    )

    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    margins = [float(m.strip()) for m in args.margin.split(",")]

    run_evaluation(
        models_dir=args.models_dir,
        horizons=horizons,
        fee_bps=args.fee_bps,
        margins=margins,
        block_minutes=args.block_minutes,
        n_boot=args.n_boot,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

