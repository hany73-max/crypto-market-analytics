# ML Layer Overview — Real-Time Cryptocurrency Market Analytics

*Last updated: 2026-10-31 | Branch: `feat/ml-evaluation` → merged into `docs/ml-overview`*

---

## 1. Purpose and Scope

The ML layer provides **binary price-direction forecasts** at three forward horizons (5, 15, and 60 minutes) for individual cryptocurrency symbols. It is self-contained: the only upstream dependency is the canonical kline Parquet dataset produced by the pipeline; all ML outputs (fitted models, test predictions, metadata) are written to `data/models/`. Downstream sinks (PostgreSQL `fact_predictions` table, Airflow retraining DAG) are **not yet implemented** — they depend on pipeline changes agreed with the teammate (see §10).

```
HDFS / local Parquet (klines)
        │
        ▼
 scripts/fetch_sample_data.py      ← one-time data ingestion (sample or full)
        │
        ▼
 spark/apps/ml/features.py         ← gap-safe scale-free feature engineering
 spark/apps/ml/labels.py           ← gap-safe multi-horizon binary labels
        │
        ▼
 spark/apps/ml/splits.py           ← chronological 70/15/15 split + purge
 spark/apps/ml/pipeline.py         ← VectorAssembler → StandardScaler (unfitted)
 spark/apps/ml/train.py            ← fit on train, tune on val, score test once
        │
        ▼
 data/models/horizon_{h}m/{model}/
   ├── pipeline_model/             ← fitted PipelineModel (Spark format)
   ├── metadata.json               ← schema version, hyperparams, metrics, git hash
   ├── feature_importance.json     ← coefficients / Gini importances
   └── test_predictions.parquet    ← common prediction schema
        │
        ▼
 spark/apps/ml/evaluate.py         ← model-agnostic evaluation (reads parquet only)
 docs/ml_evaluation_report.md      ← generated report (never retrain from it)
```

---

## 2. Target Definition

**Why direction and not price?** Raw closing prices are non-stationary, dollar-denominated, and asset-specific. A model trained on price levels cannot generalise across assets or time. Fractional forward returns are stationary and scale-free; their sign (direction) is the binary target the market actually rewards or penalises.

**Label formula** (from [`labels.py`](../../spark/apps/ml/labels.py)):

For horizon $h$ minutes and current close $c_t$:

$$
\text{future\_return}_{h} = \frac{c_{t+h} - c_t}{c_t}
$$

$$
\text{target\_direction}_{h} =
\begin{cases}
1 \;(\text{UP}) & \text{if } \text{future\_return}_{h} > 0 \\
0 \;(\text{DOWN/FLAT}) & \text{otherwise}
\end{cases}
$$

**Gap-safety rule:** The label is set to `NULL` if the lead candle's `open_time` is not exactly $h \times 60{,}000$ ms ahead of the current row, or if `close` is null or non-positive. This ensures labels never cross exchange downtime, maintenance gaps, or symbol partition boundaries. The `Window` is always `partitionBy("symbol").orderBy("open_time")` — no cross-asset leakage.

Defaults: `horizons = (5, 15, 60)`, binary mode (`epsilon = None`). A 3-class mode (UP / FLAT / DOWN) is available via the `epsilon` parameter but is not used in the current training runs.

---

## 3. Features

All 25 features are **scale-free** (dollar prices cancel out), **gap-safe** (evaluate to `NULL` if the required lookback window crosses a missing-minute gap), and derived **without wall-clock columns** (`ingested_at`, `kline_date`, etc.). The minimum required input columns are: `symbol`, `open_time`, `open`, `high`, `low`, `close`, `volume`, `number_of_trades`, `taker_buy_base_volume`.

The longest required lookback to warm up all features is **61 closed bars** per symbol (`realized_vol_60` needs 60 prior 1-minute returns, each needing a prior close — i.e., 61 klines total).

| # | Feature | Formula / Notes |
|---|---------|-----------------|
| 1 | `price_return_1m` | $(c_t - c_{t-1}) / c_{t-1}$ ; gap-checked over 1 min |
| 2 | `price_return_5m` | $(c_t - c_{t-5}) / c_{t-5}$ ; gap-checked over 5 min |
| 3 | `price_return_15m` | $(c_t - c_{t-15}) / c_{t-15}$ ; gap-checked over 15 min |
| 4 | `price_return_60m` | $(c_t - c_{t-60}) / c_{t-60}$ ; gap-checked over 60 min |
| 5 | `close_to_sma_7` | $(c_t / \text{SMA}_7) - 1$ ; SMA over 7 rows; gap-checked |
| 6 | `close_to_sma_20` | $(c_t / \text{SMA}_{20}) - 1$ ; gap-checked over 20 rows |
| 7 | `close_to_sma_50` | $(c_t / \text{SMA}_{50}) - 1$ ; gap-checked over 50 rows |
| 8 | `bb_percent_b` | $(c_t - \text{BB\_lower}) / (\text{BB\_upper} - \text{BB\_lower})$ ; guarded to 0.5 when stddev = 0; gap-checked over 20 rows |
| 9 | `bb_bandwidth` | $(\text{BB\_upper} - \text{BB\_lower}) / \text{SMA}_{20}$ ; guarded to `NULL` when SMA ≤ 0 |
| 10 | `realized_vol_20` | Rolling sample stddev of `price_return_1m` over 20 rows; gap-checked over 21 klines |
| 11 | `realized_vol_60` | Rolling sample stddev of `price_return_1m` over 60 rows; gap-checked over 61 klines |
| 12 | `parkinson_vol_20` | $\sqrt{\overline{\ln(h/l)^2} / (4\ln 2)}$ over 20 rows; guarded against $l \le 0$ or $h < l$ |
| 13 | `candle_body_ratio` | $(c_t - o_t) / o_t$ ; guarded when open ≤ 0 |
| 14 | `candle_range_ratio` | $(h_t - l_t) / c_t$ ; guarded when close ≤ 0 |
| 15 | `volume_ratio_20` | $v_t / \overline{v}_{20}$ ; guarded to 0.0 when avg = 0; gap-checked over 20 rows |
| 16 | `trade_intensity_20` | $n_t / \overline{n}_{20}$ ; same guards as `volume_ratio_20` |
| 17 | `taker_buy_ratio` | `taker_buy_base_volume / volume` ; guarded to 0.0 when volume ≤ 0 |
| 18 | `rsi_14` | **Cutler's RSI** (simple-average, not Wilder's EMA): $100 \times \overline{\text{gain}}_{14} / (\overline{\text{gain}}_{14} + \overline{\text{loss}}_{14})$ ; guarded to 50.0 when both are zero; gap-checked over 15 klines |
| 19 | `macd_line` | $({\rm FastEMA}_{12} - {\rm SlowEMA}_{26}) / c_t$ ; **approximation**: both EMAs are fixed-lookback weighted sums (normalized exponential weights, $\alpha = 2/(n+1)$), not recursive EMAs. Gap-checked over 26 rows. |
| 20 | `macd_signal` | 9-period fixed-lookback weighted average of `macd_line`; gap-checked over 34 rows (26 + 8 additional) |
| 21 | `macd_hist` | `macd_line - macd_signal` ; same validity window as `macd_signal` |
| 22 | `sin_minute_of_day` | $\sin\!\left(2\pi \cdot \lfloor t_{\rm ms} / 60000 \rfloor \bmod 1440 \;/\; 1440\right)$ ; from epoch ms only |
| 23 | `cos_minute_of_day` | Cosine of the same angle |
| 24 | `sin_day_of_week` | $\sin\!\left(2\pi \cdot \bigl(\lfloor t_{\rm ms} / 86400000 \rfloor + 3\bigr) \bmod 7 \;/\; 7\right)$ ; Monday = 0 |
| 25 | `cos_day_of_week` | Cosine of the same angle |

**RSI / MACD approximation notes:**
- RSI uses Cutler's method (simple mean of gains/losses over the window), not Wilder's recursive EMA. Results differ from most charting platforms.
- MACD EMAs are approximated by fixed-lookback weighted sums of lag values rather than recursive state. This is fully deterministic and gap-safe in Spark, but differs from a true online EMA.

**Deliberately excluded:** raw prices (`open`, `high`, `low`, `close` as levels), dollar-denominated volumes (`quote_asset_volume`, `taker_buy_quote_volume`), wall-clock columns (`ingested_at`, `kline_date`, `close_time_dt`), asset metadata (`symbol` as a direct feature), and `data_source`. The optional `btc_return_5m` / `btc_return_15m` cross-asset features are available but **off by default** (`include_btc_market_return=False`).

---

## 4. Validation Design

### 4.1 Chronological 70 / 15 / 15 split

All split boundaries are computed as **global quantiles** of `open_time` across all symbols, so every symbol uses identical cutoffs. The split is strictly chronological: Train < Val < Test.

```
|<───── 70% train ─────>|<── 15% val ──>|<── 15% test ──>|
                        ↑               ↑
                     val_start       test_start
```

**Purge / embargo:** For a model targeting horizon $h$, any training row whose `open_time + h \times 60{,}000 \ge \text{val\_start}` is removed. The same embargo is applied at the val/test boundary. This prevents any forward-return label in train from overlapping with the validation window — a form of label leakage that standard `CrossValidator` does not handle.

### 4.2 Why not `CrossValidator` or random splits?

- Random splits destroy temporal ordering; a label at time $t$ can land in train while its future return at $t+h$ lands in test — a direct look-ahead leak.
- `CrossValidator` with time-series data re-shuffles folds randomly in Spark MLlib, introducing the same problem.
- The implemented approach is a single explicit chronological split with a per-horizon purge, matching quantitative finance walk-forward standards.

### 4.3 Scaler and class weights

`StandardScaler` (zero-mean, unit-variance) is **fit on the training split only** and applied to val and test as a frozen transform. Class weights $w_c = N_{\rm train} / (2 N_c^{\rm train})$ are computed on the training split only and attached as a `weight` column. Neither the scaler nor weights see any val or test data before the model is selected.

### 4.4 Hyperparameter tuning

Each candidate hyperparameter configuration is fit on `train_df`; the resulting model is scored on `val_df` by ROC-AUC. The configuration with the best validation AUC is retained. The test split is scored **exactly once** with the selected model — no further tuning, re-fitting, or ensembling occurs after that point.

---

## 5. Models and Tuning Grids

Three Spark MLlib classifiers, each consuming the same 25-feature `StandardScaler` output:

| Model | Key Parameters | Tuning Grid (as implemented) |
|-------|---------------|------------------------------|
| `LogisticRegression` (`logreg`) | `maxIter=50`, `weightCol` | `{regParam: 0.01, elasticNetParam: 0.1}`, `{regParam: 0.1, elasticNetParam: 0.5}` |
| `RandomForestClassifier` (`rf`) | `seed=42`, `weightCol` | `{numTrees: 20, maxDepth: 5}` (single point) |
| `GBTClassifier` (`gbt`) | `seed=42`, `weightCol` | `{maxIter: 20, maxDepth: 4, stepSize: 0.1}` (single point) |

The grids are deliberately small for laptop-feasible run times. RF and GBT each have a single grid point, so validation selects the one fitted model. Logistic regression compares two regularisation settings; the best by validation AUC is kept.

---

## 6. Evaluation

### 6.1 Metrics

Reported for each (horizon, model) pair:

- **ROC-AUC** with 95% block-bootstrap CI (contiguous 60-minute blocks, 500 resamples per symbol, empirical percentile interval) — accounts for serial autocorrelation in overlapping forward returns.
- **Accuracy** with 95% block-bootstrap CI.
- **PR-AUC** (precision-recall area).
- **F1** (class 1).
- **Log Loss** and **Brier Score** (calibration).
- **Oracle-majority baseline** (always predict the test set's actual majority class) — upper bound for zero-information prediction.
- **Train-majority baseline** — what a model would score if it always predicted the training majority class on the test set. Differs from oracle when the class balance shifts between train and test.

The evaluation is model-agnostic: `evaluate.py` reads only `test_predictions.parquet` and `metadata.json`; it does not load or re-score the PipelineModel.

### 6.2 Results summary (from `docs/ml_evaluation_report.md`)

> **Context:** All results derive from a **~30-day, 2-symbol (BTCUSDT + ETHUSDT)** sample. The test window spans approximately **4.5 days** (15% of ~30 days). Both symbols are highly correlated. The pipeline has **not yet run on the full dataset**. All conclusions below are provisional and subject to revision on the full data run.

**9 models** were evaluated (3 horizons × 3 algorithms):

| Horizon | Model | Test AUC | 95% CI | CI excludes 0.5? | Test Acc | Oracle Maj Acc |
|---------|-------|----------|--------|-----------------|----------|----------------|
| 5m | logreg | 0.5225 | [0.5057, 0.5420] | **Yes** | 0.5145 | 0.5136 |
| 5m | rf | 0.5335 | [0.5173, 0.5498] | **Yes** | 0.5197 | 0.5136 |
| 5m | gbt | 0.5247 | [0.5068, 0.5412] | **Yes** | 0.5170 | 0.5136 |
| 15m | logreg | 0.5208 | [0.4958, 0.5458] | No | 0.5064 | 0.5232 |
| 15m | rf | 0.5258 | [0.5013, 0.5490] | **Yes** | 0.5078 | 0.5232 |
| 15m | gbt | 0.5327 | [0.5095, 0.5581] | **Yes** | 0.5201 | 0.5232 |
| 60m | logreg | 0.5000 | [0.5000, 0.5000] | No | 0.4862 | 0.5138 |
| 60m | rf | 0.4849 | [0.4432, 0.5288] | No | 0.4919 | 0.5138 |
| 60m | gbt | 0.5053 | [0.4627, 0.5478] | No | 0.5013 | 0.5138 |

**Key observations** (stated without claiming statistical significance beyond the CI evidence):
- At 5m, all three models and at 15m two of three (rf, gbt) have 95% CIs that exclude 0.5 on this test window. The absolute AUC margins are small (0.52–0.53). Whether this persists on the full dataset is unknown.
- At 60m, no model's CI excludes 0.5. Signal appears to decay to chance at this horizon on the sample data.
- **Time-slice stability**: 4-quartile AUC standard deviations range from 0.008 (5m gbt) to 0.073 (60m rf). The 15m logistic regression shows high quartile-to-quartile variance (std = 0.048), indicating unstable signal across the test window.
- Buy-and-hold over the ~4.5-day test window lost approximately **−3.56% (5m/15m test window)** and **−3.94% (60m test window)**.
- **No backtest is net-positive after fees** (5.0 bps per side / 10 bps round-trip). Gross returns for several configurations are modestly positive (e.g. 5m GBT: +6.04% gross), but fee drag drives all net returns deeply negative across all 27 tested combinations (margin ∈ {0.00, 0.02, 0.05}).

### 6.3 Degenerate 60m logistic regression

The 60m logreg model selected hyperparameters `{regParam: 0.1, elasticNetParam: 0.5}`. This combination (L1/L2 elastic net with regParam=0.1 on StandardScaler-normalised inputs) shrinks all 25 coefficients to exactly 0.0 (fitted intercept ≈ 3.93 × 10⁻¹³). The model outputs a constant probability for every row, producing AUC = 0.5000 and an F1 of 0.654 (from always predicting the majority class by default threshold). This is a known consequence of the tuning grid; the less-regularised candidate (`regParam: 0.01, elasticNetParam: 0.1`) also achieved AUC = 0.5000 on the validation split, so both candidates tied and the more regularised one was retained. An expanded tuning grid or reduced regularisation should be evaluated on the full dataset.

### 6.4 Calibration

ECE (Expected Calibration Error) is lowest for `logreg` (5m: 0.016, 15m: 0.027, 60m: 0.014 — though the 60m value is trivially low due to degeneracy) and highest for `gbt` (60m: 0.097). RF and GBT compress predicted probabilities into narrower ranges than logistic regression. No model shows probabilities far from 0.5 on this sample, which is consistent with weak signal.

---

## 7. Data Contract

### 7.1 Input: canonical kline schema

The minimum required columns (from [`features.py`](../../spark/apps/ml/features.py)):

| Column | Type | Notes |
|--------|------|-------|
| `symbol` | `string` | Trading pair, e.g. `BTCUSDT` |
| `open_time` | `long` | Candle open timestamp, epoch milliseconds (UTC) |
| `open` | `double` | Open price |
| `high` | `double` | High price |
| `low` | `double` | Low price |
| `close` | `double` | Close price |
| `volume` | `double` | Base asset volume |
| `number_of_trades` | `long` | Trade count per candle |
| `taker_buy_base_volume` | `double` | Taker buy volume |

Additional pipeline columns (`close_time`, `quote_asset_volume`, `taker_buy_quote_volume`, `data_source`, `ingested_at`, `open_time_dt`, `close_time_dt`, `kline_date`) are accepted but not used by the ML layer.

### 7.2 Output: prediction Parquet schema (`prediction_schema_version: "1.0.0"`)

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | `string` | Trading pair |
| `open_time` | `long` | Candle open epoch ms (prediction point) |
| `open_time_dt` | `timestamp` | Human-readable timestamp (UTC) |
| `horizon_min` | `int` | Forward horizon in minutes (5, 15, or 60) |
| `model_name` | `string` | Model identifier (`logreg`, `rf`, `gbt`) |
| `label` | `int` | Realized direction (0 = DOWN/FLAT, 1 = UP) |
| `prediction` | `int` | Model prediction (0 or 1) |
| `p_up` | `double` | Predicted probability of class 1 (UP) |
| `current_close` | `double` | Close price at prediction point |
| `future_return` | `double` | Realized fractional return over horizon |

Only rows where `future_return` is not null are written to the prediction Parquet.

### 7.3 Model artifact layout

```
data/models/
└── horizon_{h}m/                  # h ∈ {5, 15, 60}
    └── {model_name}/              # logreg | rf | gbt
        ├── pipeline_model/        # Spark PipelineModel (VectorAssembler → StandardScaler → Classifier)
        ├── metadata.json          # see keys below
        ├── feature_importance.json
        └── test_predictions.parquet
```

**`metadata.json` keys:** `prediction_schema_version`, `feature_list`, `horizon`, `model_name`, `split_boundaries_and_row_counts` (val_start, test_start, train_end, val_end, purge settings, row counts), `purge_setting`, `hyperparameters`, `metrics` (val_auc, test_auc, test_accuracy, test_f1, majority_baseline_accuracy, training_time_seconds), `seed`, `git_commit_hash`, `pyspark_version`, `created_at_utc`.

---

## 8. How to Run

### 8.1 Environment setup

```powershell
# Python 3.10 required (matches PySpark 3.3.0 runtime)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt
```

### 8.2 Fetch sample data (requires network; Binance public API)

```powershell
python scripts/fetch_sample_data.py --symbols BTCUSDT,ETHUSDT --days 30 --output data/sample_klines.parquet
```

### 8.3 Run tests

```powershell
python -m pytest tests/ml/ -v
```

Tests cover label correctness (gap safety, symbol boundaries, epsilon mode), feature null propagation, chronological split integrity, pipeline assembly, training smoke tests, and evaluation metric implementations. They use a `local[2]` Spark session and do not require Docker.

### 8.4 Training

```powershell
python -m spark.apps.ml.train --input data/sample_klines.parquet --horizons 5,15,60 --models logreg,rf,gbt --output-dir data/models
```

Key CLI flags:
- `--symbols` — restrict to a comma-separated subset (default: all symbols in input)
- `--seed` — fixed random seed (default: 42)
- `--quick` — single hyperparameter point per model (faster iteration)
- `--conservative-purge` — use `max(horizons)` embargo for all splits
- `--exclude-features` — comma-separated feature names to ablate

**Resource note:** Training on the sample dataset (2 symbols × 30 days) runs in `local[*]` with default driver memory. Training on the full dataset (≈30 symbols × 1 year) will exceed the **3 GB Docker Spark worker** constraint — run locally with at least 8–16 GB driver memory allocated, not inside the Docker compose stack.

### 8.5 Evaluation

```powershell
python -m spark.apps.ml.evaluate --models-dir data/models --horizons 5,15,60 --models logreg,rf,gbt --output docs/ml_evaluation_report.md
```

This reads only Parquet prediction files and JSON metadata; it does not reload or retrain models.

---

## 9. Known Limitations

1. **Sample scope only.** All results in `docs/ml_evaluation_report.md` derive from 2 symbols (BTCUSDT, ETHUSDT) and approximately 30 days of 1-minute klines. The full universe (~30 symbols × ~1 year) has not been run. Any signal observed in the CIs should be treated as preliminary.

2. **Highly correlated symbols.** BTCUSDT and ETHUSDT move together; the block-bootstrap CI estimates on the 2-symbol sample do not represent the variance one would see across a diverse symbol universe.

3. **Short test window.** The 15% test split covers approximately 4.5 days. Even with block bootstrapping, CIs computed on ~4.5 days of two correlated symbols carry substantial sampling uncertainty. The full-dataset run is required before making claims about short-horizon signal.

4. **Streaming inference requires a lookback buffer.** The feature engineering is batch-only as implemented. For production streaming inference, each symbol needs a circular buffer of at least **61 closed bars** before any prediction can be made (the warm-up length of `realized_vol_60`). Streaming consumers must also ingest only **closed candles** — using a partially-formed candle as the current bar introduces look-ahead bias. A closed-candle filter in the WebSocket producer is required (agreed with teammate; not yet implemented).

5. **No serving layer.** The PostgreSQL `fact_predictions` table schema exists in `postgres-dw/init.sql` but horizon-specific columns and the inference job that writes to it are not yet implemented. This depends on pipeline changes from the teammate.

6. **No Airflow retraining DAG.** Periodic model retraining is not automated. This is a roadmap item dependent on the teammate's DAG infrastructure.

7. **Degenerate 60m logistic regression.** With the current 2-point tuning grid, both logistic regression candidates achieve AUC = 0.5000 at the 60m horizon on the validation split, resulting in selection of the maximally regularised (degenerate) configuration. An expanded grid with lower regularisation values should be tested on the full dataset.

8. **Batch-only Spark execution.** The current `train.py` and `evaluate.py` use PySpark batch mode. They are not designed to run inside the 3 GB Docker Spark worker for large datasets.

---

## 10. Roadmap

The following items are planned or agreed with the teammate:

| Item | Owner / Dependency |
|------|--------------------|
| Full-dataset training run (30 symbols × 1 year) | ML layer — blocked on HDFS ingestion of full history |
| Closed-candle filter in WebSocket producer | Teammate (agreed) |
| `fact_predictions` horizon handling in PostgreSQL schema | Teammate (agreed) |
| `.gitattributes` LF enforcement | Agreed — not yet committed |
| Inference sink: batch job writing predictions to `fact_predictions` | ML layer, after schema agreement |
| Airflow retraining DAG | ML layer + teammate pipeline |
| Expanded hyperparameter grids (especially 60m logreg) | ML layer |
| Deep-learning baseline (e.g. LSTM or Transformer) | Optional — pluggable through the same `prediction_schema_version: "1.0.0"` contract |
| Order-book / trade-flow features from live WebSocket streams | Optional — producer streams required |
