# Cryptocurrency Direction Prediction: Milestone 4 ML Evaluation Report

*Generated on: 2026-10-31 18:01:40 UTC*

## 1. Evaluation Methodology

This evaluation strictly adheres to production-grade quantitative finance standards:

- **Model-Agnostic Common Schema**: All models (LogisticRegression, RandomForest, GBT) are evaluated strictly through an uncoupled prediction contract (`test_predictions.parquet`) containing `open_time`, `horizon_min`, `model_name`, `label`, `prediction`, scalar `p_up`, `current_close`, and realized `future_return`.
- **Honest Scoring**: The test split is scored exactly once per validated model without re-tuning or data leakage.
- **Contiguous Block-Bootstrap Uncertainty**: Standard i.i.d. bootstrap artificially compresses confidence intervals due to serial overlap in forward returns. We apply a contiguous time block bootstrap (60-minute blocks per symbol, 500 resamples) to compute empirical 95% confidence intervals for ROC-AUC and Accuracy.
- **Baselines**: Models are compared against (1) Train-Majority Accuracy (from training class mix), (2) Oracle-Majority Accuracy (the theoretical ceiling of zero-information majority prediction on the test set), and (3) Constant-Score baselines.
- **Cost-Aware Backtesting**: Realistic execution with 5.0 bps fee per side (10 bps round trip). Trading occurs exclusively on non-overlapping intervals (every $h$-th minute per symbol), strictly precluding trade stacking or forward look-ahead bias.

## 2. Core Metrics and Uncertainty Analysis (95% Block-Bootstrap CI)

| Horizon | Model | Test AUC (95% CI) | AUC Excludes 0.5 | PR-AUC | Test Acc (95% CI) | Train Maj Acc | Oracle Maj Acc | F1 | Log Loss | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5m | logreg | 0.5225 [0.5057, 0.5420] | **YES** | 0.5092 | 0.5145 [0.5000, 0.5291] | 0.5136 | 0.5136 | 0.5339 | 0.6923 | 0.2496 |
| 5m | rf | 0.5335 [0.5173, 0.5498] | **YES** | 0.5159 | 0.5197 [0.5070, 0.5330] | 0.5136 | 0.5136 | 0.5214 | 0.6914 | 0.2491 |
| 5m | gbt | 0.5247 [0.5068, 0.5412] | **YES** | 0.5055 | 0.5170 [0.5043, 0.5300] | 0.5136 | 0.5136 | 0.5150 | 0.6956 | 0.2511 |
| 15m | logreg | 0.5208 [0.4958, 0.5458] | No | 0.5015 | 0.5064 [0.4852, 0.5237] | 0.4768 | 0.5232 | 0.5207 | 0.6933 | 0.2501 |
| 15m | rf | 0.5258 [0.5013, 0.5490] | **YES** | 0.4979 | 0.5078 [0.4884, 0.5273] | 0.4768 | 0.5232 | 0.4867 | 0.6927 | 0.2498 |
| 15m | gbt | 0.5327 [0.5095, 0.5581] | **YES** | 0.5113 | 0.5201 [0.4996, 0.5396] | 0.4768 | 0.5232 | 0.5345 | 0.7016 | 0.2538 |
| 60m | logreg | 0.5000 [0.5000, 0.5000] | No | 0.4862 | 0.4862 [0.4406, 0.5374] | 0.4862 | 0.5138 | 0.6543 | 0.6931 | 0.2500 |
| 60m | rf | 0.4849 [0.4432, 0.5288] | No | 0.4757 | 0.4919 [0.4553, 0.5269] | 0.4862 | 0.5138 | 0.4733 | 0.7037 | 0.2552 |
| 60m | gbt | 0.5053 [0.4627, 0.5478] | No | 0.4986 | 0.5013 [0.4691, 0.5361] | 0.4862 | 0.5138 | 0.4844 | 0.7312 | 0.2660 |


## 3. Per-Symbol Breakdown

| Horizon | Model | Symbol | Count | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5m | gbt | BTCUSDT | 6470 | 0.5227 | 0.5083 | 0.5153 | 0.5100 | 0.2508 |
| 5m | gbt | ETHUSDT | 6471 | 0.5268 | 0.5037 | 0.5186 | 0.5200 | 0.2513 |
| 5m | logreg | BTCUSDT | 6470 | 0.5176 | 0.5106 | 0.5080 | 0.5334 | 0.2497 |
| 5m | logreg | ETHUSDT | 6471 | 0.5268 | 0.5098 | 0.5209 | 0.5345 | 0.2495 |
| 5m | rf | BTCUSDT | 6470 | 0.5311 | 0.5188 | 0.5127 | 0.5177 | 0.2491 |
| 5m | rf | ETHUSDT | 6471 | 0.5360 | 0.5142 | 0.5268 | 0.5253 | 0.2491 |
| 15m | gbt | BTCUSDT | 6469 | 0.5264 | 0.5135 | 0.5179 | 0.5284 | 0.2540 |
| 15m | gbt | ETHUSDT | 6470 | 0.5388 | 0.5101 | 0.5224 | 0.5406 | 0.2537 |
| 15m | logreg | BTCUSDT | 6469 | 0.5115 | 0.4958 | 0.4964 | 0.5238 | 0.2507 |
| 15m | logreg | ETHUSDT | 6470 | 0.5294 | 0.5069 | 0.5164 | 0.5175 | 0.2495 |
| 15m | rf | BTCUSDT | 6469 | 0.5204 | 0.4958 | 0.4971 | 0.4943 | 0.2504 |
| 15m | rf | ETHUSDT | 6470 | 0.5301 | 0.5008 | 0.5185 | 0.4785 | 0.2492 |
| 60m | gbt | BTCUSDT | 6462 | 0.5067 | 0.5066 | 0.5138 | 0.4869 | 0.2661 |
| 60m | gbt | ETHUSDT | 6463 | 0.5011 | 0.4917 | 0.4888 | 0.4820 | 0.2659 |
| 60m | logreg | BTCUSDT | 6462 | 0.5000 | 0.4858 | 0.4858 | 0.6539 | 0.2500 |
| 60m | logreg | ETHUSDT | 6463 | 0.5000 | 0.4866 | 0.4866 | 0.6547 | 0.2500 |
| 60m | rf | BTCUSDT | 6462 | 0.4781 | 0.4706 | 0.4844 | 0.4802 | 0.2567 |
| 60m | rf | ETHUSDT | 6463 | 0.4920 | 0.4820 | 0.4995 | 0.4661 | 0.2536 |


## 4. Time-Slice Stability Analysis (4 Chronological Quartiles)

| Horizon | Model | Q1 AUC | Q2 AUC | Q3 AUC | Q4 AUC | Mean AUC | Std Dev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5m | gbt | 0.5302 | 0.5257 | 0.5357 | 0.5138 | 0.5264 | 0.0081 |
| 5m | logreg | 0.5256 | 0.5338 | 0.5474 | 0.4960 | 0.5257 | 0.0188 |
| 5m | rf | 0.5320 | 0.5431 | 0.5454 | 0.5209 | 0.5353 | 0.0098 |
| 15m | gbt | 0.5204 | 0.5645 | 0.5276 | 0.5279 | 0.5351 | 0.0173 |
| 15m | logreg | 0.4642 | 0.5881 | 0.5418 | 0.4864 | 0.5201 | 0.0483 |
| 15m | rf | 0.4926 | 0.5564 | 0.5219 | 0.5332 | 0.5260 | 0.0230 |
| 60m | gbt | 0.4726 | 0.5655 | 0.4966 | 0.4815 | 0.5041 | 0.0365 |
| 60m | logreg | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| 60m | rf | 0.4083 | 0.5934 | 0.4909 | 0.4248 | 0.4794 | 0.0727 |


## 5. Probability Calibration and Expected Calibration Error (ECE)

| Horizon | Model | ECE | Min p_up | Max p_up | Mean p_up |
| --- | --- | --- | --- | --- | --- |
| 5m | gbt | 0.0254 | 0.1582 | 0.8490 | 0.5042 |
| 5m | logreg | 0.0159 | 0.3797 | 0.6243 | 0.5019 |
| 5m | rf | 0.0141 | 0.3986 | 0.6151 | 0.5006 |
| 15m | gbt | 0.0551 | 0.1551 | 0.8283 | 0.5088 |
| 15m | logreg | 0.0267 | 0.3873 | 0.6166 | 0.5029 |
| 15m | rf | 0.0248 | 0.4699 | 0.6029 | 0.5013 |
| 60m | gbt | 0.0973 | 0.1788 | 0.9153 | 0.5155 |
| 60m | logreg | 0.0138 | 0.5000 | 0.5000 | 0.5000 |
| 60m | rf | 0.0545 | 0.3826 | 0.7141 | 0.5057 |


## 6. Cost-Aware Backtest Performance (Fee: 5.0 bps per side / 10 bps round trip)

| Horizon | Model | Margin | Trades | Gross Hit % | Net Hit % | Mean Net/Trade | Gross Total Ret | Net Total Ret | B&H Net Ret | Random Net Ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5m | logreg | 0.00 | 2589 | 51.5% | 13.8% | -9.9 bps | 2.33% | **-92.33%** | -3.56% | -92.50% |
| 5m | logreg | 0.02 | 933 | 54.8% | 17.7% | -9.8 bps | 1.99% | **-59.90%** | -3.56% | -60.72% |
| 5m | logreg | 0.05 | 99 | 58.6% | 33.3% | -7.0 bps | 2.95% | **-6.76%** | -3.56% | -9.50% |
| 5m | rf | 0.00 | 2589 | 51.8% | 13.9% | -10.0 bps | 0.22% | **-92.48%** | -3.56% | -92.50% |
| 5m | rf | 0.02 | 1095 | 54.2% | 16.6% | -9.8 bps | 1.93% | **-65.92%** | -3.56% | -66.54% |
| 5m | rf | 0.05 | 171 | 60.2% | 32.7% | -8.3 bps | 2.98% | **-13.21%** | -3.56% | -15.76% |
| 5m | gbt | 0.00 | 2589 | 51.9% | 13.8% | -9.8 bps | 6.04% | **-92.05%** | -3.56% | -92.50% |
| 5m | gbt | 0.02 | 1837 | 52.8% | 14.5% | -9.8 bps | 4.37% | **-83.39%** | -3.56% | -84.11% |
| 5m | gbt | 0.05 | 834 | 54.6% | 17.0% | -9.6 bps | 3.21% | **-55.19%** | -3.56% | -56.81% |
| 15m | logreg | 0.00 | 864 | 50.0% | 24.8% | -9.5 bps | 3.81% | **-56.26%** | -3.40% | -57.95% |
| 15m | logreg | 0.02 | 453 | 53.0% | 27.8% | -9.0 bps | 4.30% | **-33.71%** | -3.40% | -36.21% |
| 15m | logreg | 0.05 | 105 | 52.4% | 29.5% | -10.6 bps | -0.67% | **-10.57%** | -3.40% | -10.10% |
| 15m | rf | 0.00 | 864 | 51.4% | 26.5% | -9.4 bps | 5.04% | **-55.75%** | -3.40% | -57.95% |
| 15m | rf | 0.02 | 600 | 51.8% | 25.8% | -9.4 bps | 3.76% | **-43.07%** | -3.40% | -45.14% |
| 15m | rf | 0.05 | 155 | 55.5% | 29.0% | -8.7 bps | 2.04% | **-12.62%** | -3.40% | -14.30% |
| 15m | gbt | 0.00 | 864 | 51.4% | 25.9% | -9.5 bps | 4.12% | **-56.13%** | -3.40% | -57.95% |
| 15m | gbt | 0.02 | 693 | 51.7% | 27.0% | -9.4 bps | 4.27% | **-47.87%** | -3.40% | -49.91% |
| 15m | gbt | 0.05 | 480 | 52.3% | 26.0% | -9.2 bps | 4.03% | **-35.64%** | -3.40% | -38.19% |
| 60m | logreg | 0.00 | 216 | 50.5% | 32.9% | -13.4 bps | -7.26% | **-25.29%** | -3.94% | -19.68% |
| 60m | logreg | 0.02 | 0 | 0.0% | 0.0% | 0.0 bps | 0.00% | **0.00%** | -3.94% | 0.00% |
| 60m | logreg | 0.05 | 0 | 0.0% | 0.0% | 0.0 bps | 0.00% | **0.00%** | -3.94% | 0.00% |
| 60m | rf | 0.00 | 216 | 51.4% | 33.8% | -11.7 bps | -3.83% | **-22.52%** | -3.94% | -19.68% |
| 60m | rf | 0.02 | 155 | 52.3% | 33.5% | -14.2 bps | -6.39% | **-19.84%** | -3.94% | -14.27% |
| 60m | rf | 0.05 | 83 | 50.6% | 33.7% | -11.9 bps | -1.59% | **-9.43%** | -3.94% | -7.99% |
| 60m | gbt | 0.00 | 216 | 56.0% | 38.4% | -7.3 bps | 5.85% | **-14.72%** | -3.94% | -19.68% |
| 60m | gbt | 0.02 | 186 | 55.9% | 36.6% | -8.1 bps | 3.39% | **-14.16%** | -3.94% | -16.98% |
| 60m | gbt | 0.05 | 136 | 53.7% | 31.6% | -13.1 bps | -4.23% | **-16.41%** | -3.94% | -12.67% |


## 7. Diagnosis: Horizon 60m Degenerate Logistic Regression

- **Selected Hyperparameters**: `{"regParam": 0.1, "elasticNetParam": 0.5}`

- **All Coefficients Zero?**: `YES (All 25 features zero)`

- **Fitted Intercept**: `3.927969e-13`

- **Root Cause Analysis**: Degeneracy Root Cause: Hyperparameters {'regParam': 0.1, 'elasticNetParam': 0.5} (regParam=0.1, elasticNetParam=0.5) imposed excessive L1/L2 regularization on standardized features, shrinking ALL 25 coefficients to exactly 0.0 (intercept=3.93e-13). Consequently, the model outputs an identical constant probability for every test row, yielding an AUC of exactly 0.5000.



## 8. Feature Importance Summary (Top 10 by Absolute Magnitude)

> **Note on Signal Strength**: When test ROC-AUC ranges between 0.48 and 0.53, ranking of feature importances is subject to high estimation variance. Feature importances reflect linear/split heuristic fits on the training set and must be interpreted with caution.


### Horizon 5m - GBT

| Rank | Feature | Importance / Coefficient |
| --- | --- | --- |
| 1 | `cos_minute_of_day` | 0.146609 |
| 2 | `sin_minute_of_day` | 0.105905 |
| 3 | `price_return_60m` | 0.089452 |
| 4 | `sin_day_of_week` | 0.077034 |
| 5 | `realized_vol_60` | 0.073960 |
| 6 | `macd_signal` | 0.061234 |
| 7 | `close_to_sma_50` | 0.050428 |
| 8 | `realized_vol_20` | 0.046555 |
| 9 | `macd_hist` | 0.042203 |
| 10 | `cos_day_of_week` | 0.040181 |


### Horizon 5m - LOGREG

| Rank | Feature | Importance / Coefficient |
| --- | --- | --- |
| 1 | `bb_percent_b` | -0.040573 |
| 2 | `realized_vol_60` | -0.032362 |
| 3 | `bb_bandwidth` | 0.032159 |
| 4 | `close_to_sma_7` | -0.022664 |
| 5 | `sin_day_of_week` | -0.021725 |
| 6 | `price_return_5m` | -0.018660 |
| 7 | `candle_range_ratio` | -0.018016 |
| 8 | `realized_vol_20` | 0.016855 |
| 9 | `cos_minute_of_day` | 0.014765 |
| 10 | `trade_intensity_20` | -0.013346 |


### Horizon 5m - RF

| Rank | Feature | Importance / Coefficient |
| --- | --- | --- |
| 1 | `cos_minute_of_day` | 0.083937 |
| 2 | `sin_minute_of_day` | 0.077537 |
| 3 | `price_return_60m` | 0.075193 |
| 4 | `close_to_sma_7` | 0.074361 |
| 5 | `macd_signal` | 0.070779 |
| 6 | `close_to_sma_50` | 0.055898 |
| 7 | `rsi_14` | 0.048574 |
| 8 | `sin_day_of_week` | 0.043081 |
| 9 | `macd_hist` | 0.040922 |
| 10 | `close_to_sma_20` | 0.039537 |


### Horizon 15m - GBT

| Rank | Feature | Importance / Coefficient |
| --- | --- | --- |
| 1 | `sin_minute_of_day` | 0.189080 |
| 2 | `cos_minute_of_day` | 0.169483 |
| 3 | `price_return_60m` | 0.093147 |
| 4 | `cos_day_of_week` | 0.090168 |
| 5 | `sin_day_of_week` | 0.085205 |
| 6 | `realized_vol_60` | 0.071497 |
| 7 | `realized_vol_20` | 0.042105 |
| 8 | `parkinson_vol_20` | 0.039055 |
| 9 | `rsi_14` | 0.037485 |
| 10 | `close_to_sma_50` | 0.033687 |


### Horizon 15m - LOGREG

| Rank | Feature | Importance / Coefficient |
| --- | --- | --- |
| 1 | `realized_vol_60` | -0.063186 |
| 2 | `bb_bandwidth` | 0.057058 |
| 3 | `bb_percent_b` | -0.056237 |
| 4 | `sin_day_of_week` | -0.042806 |
| 5 | `cos_minute_of_day` | 0.042261 |
| 6 | `close_to_sma_50` | 0.035641 |
| 7 | `macd_hist` | -0.034198 |
| 8 | `price_return_60m` | -0.026375 |
| 9 | `sin_minute_of_day` | 0.024163 |
| 10 | `close_to_sma_7` | -0.013255 |


### Horizon 15m - RF

| Rank | Feature | Importance / Coefficient |
| --- | --- | --- |
| 1 | `cos_minute_of_day` | 0.133364 |
| 2 | `sin_minute_of_day` | 0.116456 |
| 3 | `close_to_sma_50` | 0.114466 |
| 4 | `price_return_60m` | 0.088223 |
| 5 | `sin_day_of_week` | 0.068007 |
| 6 | `cos_day_of_week` | 0.065412 |
| 7 | `realized_vol_20` | 0.052751 |
| 8 | `realized_vol_60` | 0.046625 |
| 9 | `rsi_14` | 0.036395 |
| 10 | `close_to_sma_20` | 0.035840 |


### Horizon 60m - GBT

| Rank | Feature | Importance / Coefficient |
| --- | --- | --- |
| 1 | `cos_minute_of_day` | 0.257068 |
| 2 | `sin_minute_of_day` | 0.230631 |
| 3 | `realized_vol_60` | 0.159953 |
| 4 | `sin_day_of_week` | 0.091990 |
| 5 | `cos_day_of_week` | 0.083934 |
| 6 | `price_return_60m` | 0.078929 |
| 7 | `close_to_sma_50` | 0.032361 |
| 8 | `parkinson_vol_20` | 0.029649 |
| 9 | `bb_bandwidth` | 0.014149 |
| 10 | `macd_signal` | 0.007804 |


### Horizon 60m - LOGREG

| Rank | Feature | Importance / Coefficient |
| --- | --- | --- |
| 1 | `price_return_1m` | 0.000000 |
| 2 | `price_return_5m` | 0.000000 |
| 3 | `price_return_15m` | 0.000000 |
| 4 | `price_return_60m` | 0.000000 |
| 5 | `close_to_sma_7` | 0.000000 |
| 6 | `close_to_sma_20` | 0.000000 |
| 7 | `close_to_sma_50` | 0.000000 |
| 8 | `bb_percent_b` | 0.000000 |
| 9 | `bb_bandwidth` | 0.000000 |
| 10 | `realized_vol_20` | 0.000000 |


### Horizon 60m - RF

| Rank | Feature | Importance / Coefficient |
| --- | --- | --- |
| 1 | `cos_minute_of_day` | 0.192022 |
| 2 | `sin_minute_of_day` | 0.180829 |
| 3 | `price_return_60m` | 0.131692 |
| 4 | `realized_vol_60` | 0.094483 |
| 5 | `cos_day_of_week` | 0.074203 |
| 6 | `sin_day_of_week` | 0.072530 |
| 7 | `realized_vol_20` | 0.065583 |
| 8 | `close_to_sma_50` | 0.064553 |
| 9 | `parkinson_vol_20` | 0.026911 |
| 10 | `bb_bandwidth` | 0.019994 |


## 9. Plain-English Conclusions Derived from Empirical Data

1. **Statistical Significance**: At shorter horizons, select models (5m logreg, 5m rf, 5m gbt, 15m rf, 15m gbt) demonstrate weak but statistically significant predictive signal where the 95% block-bootstrap AUC CI strictly excludes 0.5000 (e.g., 5m RF: 0.5335 [0.5173, 0.5498]). However, at the 60m horizon, all models (15m logreg, 60m logreg, 60m rf, 60m gbt) fail to exclude 0.5000, indicating that predictive power is fragile and decays toward chance as the lookahead horizon expands.

2. **Cost Survival**: NO trading strategy survives exchange fees (5 bps per side / 10 bps round trip). While several strategies generate positive gross returns (e.g. 5m GBT +6.04% gross, 15m RF +5.04% gross), frequent trading causes fee drag to overwhelm gross alpha, driving net returns deeply negative across all models and margins (e.g. 5m GBT: -92.05% net, 15m RF: -55.75% net). Increasing the entry margin threshold (margin=0.05) reduces churn but remains net-negative (-6.76% to -55.19% net).

3. **Train-Majority vs. Oracle Baseline**: The train-majority baseline provides a deceptive benchmark whenever regime shift occurs between train and test distributions (e.g. at 15m where train majority accuracy was 0.4768 while oracle test majority was 0.5232). Evaluating against the test oracle majority reveals that models barely track or slightly underperform the test set's base rate.


## 10. Known Limitations and Scale-Up Notes

1. **Sample Scope**: Results are derived exclusively from a 2-symbol (BTCUSDT, ETHUSDT), 30-day (1-minute klines) sample. The pipeline has not yet run on the full universe of ~30 symbols x ~1 year of market data.
2. **Lookback Buffer for Streaming Inference**: Feature windows currently operate in batch mode. For production streaming inference, a minimum lookback state buffer of 61+ closed bars per symbol is strictly required to warm up the longest windowed features (SMA-50, Realized Volatility 60, Parkinson Volatility 20).
3. **Spark Resource Sizing**: Full-universe training across 30 symbols x 1 year will exceed the memory constraints of the 3 GB Docker Spark worker. Production scaling requires dedicated local Spark execution with driver memory allocated to at least 8-16 GB or a multi-node distributed cluster.
