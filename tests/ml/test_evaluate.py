"""
==============================================================================
tests/ml/test_evaluate.py
Unit tests for model-agnostic evaluation, block-bootstrap uncertainty,
calibration, and cost-aware backtesting.

Validates:
  1. Exact metric computations against hand-computed analytical values.
  2. Block bootstrap determinism, point estimate containment, noise vs planted signal.
  3. Divergence of Oracle-Majority vs Train-Majority under distribution shift.
  4. Cost-aware backtesting: exact fee deductions, perfect foresight vs random baseline.
  5. Calibration decile counts and ECE invariants.
==============================================================================
"""

import os
import sys
from typing import List

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is available
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spark.apps.ml.evaluate import (
    compute_accuracy,
    compute_block_bootstrap_ci,
    compute_brier_score,
    compute_calibration,
    compute_constant_score_metrics,
    compute_f1,
    compute_log_loss,
    compute_oracle_majority,
    compute_pr_auc,
    compute_roc_auc,
    run_backtest_strategy,
    select_non_overlapping_bars,
)


class TestMetricsAnalytical:
    """Test metrics against known hand-computed ground truths."""

    def test_perfect_and_constant_auc(self) -> None:
        """Verify ROC-AUC and PR-AUC on perfect, reversed, and constant predictions."""
        y_true = np.array([0, 0, 1, 1], dtype=np.int32)

        # 1. Perfect predictions
        p_perfect = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_roc_auc(y_true, p_perfect) == 1.0
        assert compute_pr_auc(y_true, p_perfect) == 1.0

        # 2. Reversed predictions
        p_reversed = np.array([0.9, 0.8, 0.2, 0.1])
        assert compute_roc_auc(y_true, p_reversed) == 0.0

        # 3. Constant predictions
        p_constant = np.array([0.5, 0.5, 0.5, 0.5])
        assert compute_roc_auc(y_true, p_constant) == 0.5
        assert compute_pr_auc(y_true, p_constant) == 0.5

    def test_accuracy_and_f1(self) -> None:
        """Verify Accuracy and F1 score against manual calculations."""
        y_true = np.array([1, 1, 0, 0, 1, 0], dtype=np.int32)
        y_pred = np.array([1, 0, 0, 0, 1, 1], dtype=np.int32)

        # TP=2, FP=1, FN=1, TN=2
        # Accuracy = (2 + 2) / 6 = 4/6 = 2/3
        assert np.isclose(compute_accuracy(y_true, y_pred), 4.0 / 6.0)

        # Precision = 2/3, Recall = 2/3 -> F1 = 2/3
        assert np.isclose(compute_f1(y_true, y_pred), 2.0 / 3.0)

    def test_brier_and_log_loss(self) -> None:
        """Verify Brier score and Log Loss against manual calculations."""
        y_true = np.array([1, 0], dtype=np.int32)
        p_up = np.array([0.8, 0.2], dtype=np.float64)

        # Brier = ((0.8 - 1)^2 + (0.2 - 0)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
        assert np.isclose(compute_brier_score(y_true, p_up), 0.04)

        # Constant 0.5 log loss = -ln(0.5) = ln(2)
        p_half = np.array([0.5, 0.5], dtype=np.float64)
        assert np.isclose(compute_log_loss(y_true, p_half), np.log(2.0))


class TestBlockBootstrapUncertainty:
    """Validate block-bootstrap confidence interval behavior."""

    def test_bootstrap_determinism_and_containment(self) -> None:
        """Verify block bootstrap is deterministic with fixed seed and CI contains point estimate."""
        rng = np.random.default_rng(123)
        n = 300
        base_time = 1_700_000_000_000
        open_times = [base_time + (i * 60_000) for i in range(n)]
        labels = rng.choice([0, 1], size=n)
        scores = np.clip(labels * 0.3 + rng.normal(0.5, 0.2, size=n), 0.0, 1.0)
        preds = (scores >= 0.5).astype(np.int32)

        df = pd.DataFrame({
            "symbol": ["BTCUSDT"] * n,
            "open_time": open_times,
            "label": labels,
            "prediction": preds,
            "p_up": scores,
        })

        res1 = compute_block_bootstrap_ci(df, block_minutes=30, n_boot=100, seed=42)
        res2 = compute_block_bootstrap_ci(df, block_minutes=30, n_boot=100, seed=42)

        # Determinism
        assert np.isclose(res1["auc_ci"][0], res2["auc_ci"][0])
        assert np.isclose(res1["auc_ci"][1], res2["auc_ci"][1])
        assert np.isclose(res1["acc_ci"][0], res2["acc_ci"][0])
        assert np.isclose(res1["acc_ci"][1], res2["acc_ci"][1])

        # Point estimate containment in CI
        assert res1["auc_ci"][0] <= res1["point_auc"] <= res1["auc_ci"][1]
        assert res1["acc_ci"][0] <= res1["point_accuracy"] <= res1["acc_ci"][1]

    def test_bootstrap_noise_vs_planted_signal(self) -> None:
        """Pure noise CI must include 0.5; planted signal CI must exclude 0.5."""
        rng = np.random.default_rng(777)
        n = 600
        base_time = 1_700_000_000_000
        open_times = [base_time + (i * 60_000) for i in range(n)]

        # 1. Pure Noise
        noise_labels = rng.choice([0, 1], size=n)
        noise_scores = rng.uniform(0.0, 1.0, size=n)
        noise_df = pd.DataFrame({
            "symbol": ["BTCUSDT"] * n,
            "open_time": open_times,
            "label": noise_labels,
            "prediction": (noise_scores >= 0.5).astype(np.int32),
            "p_up": noise_scores,
        })
        noise_ci = compute_block_bootstrap_ci(noise_df, block_minutes=60, n_boot=200, seed=42)
        assert noise_ci["auc_ci"][0] <= 0.50 <= noise_ci["auc_ci"][1]
        assert not noise_ci["auc_excludes_0_5"]

        # 2. Planted Strong Signal
        signal_labels = rng.choice([0, 1], size=n)
        signal_scores = np.where(signal_labels == 1, 0.85, 0.15)
        signal_df = pd.DataFrame({
            "symbol": ["BTCUSDT"] * n,
            "open_time": open_times,
            "label": signal_labels,
            "prediction": (signal_scores >= 0.5).astype(np.int32),
            "p_up": signal_scores,
        })
        sig_ci = compute_block_bootstrap_ci(signal_df, block_minutes=60, n_boot=200, seed=42)
        assert sig_ci["auc_ci"][0] > 0.50
        assert sig_ci["auc_excludes_0_5"]


class TestBaselinesAndShift:
    """Validate behavior of majority baselines under regime shift."""

    def test_oracle_vs_train_majority_shift(self) -> None:
        """Oracle majority accuracy differs from train majority when class mix shifts."""
        # Train split had 70% ones, so train majority was 1.0
        train_majority_class = 1

        # Test split has 25% ones, 75% zeros
        y_test = np.array([1] * 25 + [0] * 75, dtype=np.int32)

        oracle_class, oracle_acc = compute_oracle_majority(y_test)
        assert oracle_class == 0
        assert np.isclose(oracle_acc, 0.75)

        # Train majority predicting class 1 on test set yields only 25% accuracy!
        train_majority_acc_on_test = float(np.mean(y_test == train_majority_class))
        assert np.isclose(train_majority_acc_on_test, 0.25)
        assert oracle_acc != train_majority_acc_on_test


class TestBacktestEngine:
    """Validate cost-aware execution and no-lookahead properties."""

    def test_fees_reduce_net_return_by_exact_round_trip(self) -> None:
        """Fees reduce net return by exactly 2 * fee per round trip."""
        base_time = 1_700_000_000_000
        df = pd.DataFrame({
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "open_time": [base_time, base_time + 15 * 60_000],
            "p_up": [0.8, 0.2],  # trade 1 long (+1), trade 2 short (-1)
            "future_return": [0.0100, -0.0100],  # gross returns +1% each
            "current_close": [50000.0, 50500.0],
        })

        fee_bps = 5.0  # 5 bps per side = 0.0005. Round trip = 0.0010 (10 bps)
        res = run_backtest_strategy(df, horizon_min=15, margin=0.0, fee_bps=fee_bps)

        assert res["n_trades"] == 2
        assert np.isclose(res["mean_gross_return"], 0.0100)
        # Net return = 0.0100 - 0.0010 = 0.0090
        assert np.isclose(res["mean_net_return"], 0.0090)
        assert np.isclose(res["round_trip_fee"], 0.0010)

    def test_perfect_foresight_profitable_vs_random_unprofitable(self) -> None:
        """Perfect foresight fixture is profitable; random signal baseline is net unprofitable."""
        rng = np.random.default_rng(42)
        n = 100
        base_time = 1_700_000_000_000
        open_times = [base_time + (i * 15 * 60_000) for i in range(n)]

        # Zero-drift volatile future returns
        fut_returns = rng.normal(0.0, 0.005, size=n)

        # 1. Perfect Foresight (signals match return sign)
        p_up_perfect = np.where(fut_returns > 0, 0.80, 0.20)
        df_perfect = pd.DataFrame({
            "symbol": ["BTCUSDT"] * n,
            "open_time": open_times,
            "p_up": p_up_perfect,
            "future_return": fut_returns,
            "current_close": [100.0] * n,
        })
        res_perf = run_backtest_strategy(df_perfect, horizon_min=15, margin=0.0, fee_bps=5.0)
        assert res_perf["net_total_return"] > 0.0
        assert res_perf["hit_rate_gross"] > 0.90

        # 2. Random Signal (p_up is uninformative noise around 0.5)
        res_rand = run_backtest_strategy(
            df_perfect,
            horizon_min=15,
            margin=0.0,
            fee_bps=5.0,
            random_reps=200,
            random_seed=42,
        )
        assert res_rand["random_baseline_mean_net"] < 0.0

    def test_select_non_overlapping_bars(self) -> None:
        """Ensure no consecutive evaluated bars have open_time difference < horizon_ms."""
        base_time = 1_700_000_000_000
        # 1-minute spacing
        open_times = [base_time + (i * 60_000) for i in range(60)]
        df = pd.DataFrame({
            "symbol": ["BTCUSDT"] * 60,
            "open_time": open_times,
        })

        # Horizon = 15m
        selected = select_non_overlapping_bars(df, horizon_min=15)
        # Should select minute 0, 15, 30, 45 (4 rows)
        assert len(selected) == 4
        diffs = np.diff(selected["open_time"].to_numpy())
        for d in diffs:
            assert d >= 15 * 60_000


class TestCalibrationProperties:
    """Validate calibration table deciles and ECE calculation."""

    def test_calibration_row_counts_sum_to_total(self) -> None:
        """Decile bins must sum to total rows and ECE must be non-negative."""
        rng = np.random.default_rng(999)
        n = 500
        y_true = rng.choice([0, 1], size=n)
        p_up = rng.uniform(0.0, 1.0, size=n)

        cal_table, ece = compute_calibration(y_true, p_up, n_bins=10)
        assert len(cal_table) == 10
        total_counted = sum(b["count"] for b in cal_table)
        assert total_counted == n
        assert ece >= 0.0

