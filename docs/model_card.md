# Model Card

This document describes the bike sharing demand forecasting model following the model card standard introduced by Mitchell et al. (2019). It covers what the model predicts, how it was trained, its performance, intended use cases, and known limitations.

---

## Table of Contents

1. [Model Description](#1-model-description)
2. [Training Data](#2-training-data)
3. [Model Performance](#3-model-performance)
4. [Intended Use](#4-intended-use)
5. [How to Interpret Predictions](#5-how-to-interpret-predictions)
6. [Known Limitations](#6-known-limitations)
7. [Model Versioning and Lifecycle](#7-model-versioning-and-lifecycle)

---

## 1. Model Description

**Model type:** LightGBM Gradient Boosted Trees (regression)  
**Architecture:** Two separate models — one for `registered` riders, one for `casual` riders  
**Prediction horizon:** Trained and validated on H+1 only. Served in production as a recursive h+1..h+12 trajectory — see [`docs/forecasting.md`](forecasting.md) for how the model is rolled out across multiple lead times without retraining.  
**Output:** Predicted total bike demand (bikes per hour)  
**Target during training:** `log(registered + 1)` and `log(casual + 1)` separately; predictions are back-transformed with `expm1` and summed  
**Hyperparameter tuning:** Optuna with 50 trials, optimising RMSE on the validation set  

**Why two models?**  
Registered and casual riders follow fundamentally different demand patterns. Registered users (commuters and subscribers) show a double peak on weekdays (8 AM and 5–6 PM). Casual users (tourists and occasional riders) show a single recreational peak on weekends (midday). A single model trained on total `cnt` must learn both patterns simultaneously, which increases the difficulty of the task. Separate models allow each to specialise. At inference time, predictions are summed: `cnt_pred = expm1(pred_registered) + expm1(pred_casual)`.

---

## 2. Training Data

**Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) — Capital Bikeshare, Washington D.C.  
**Original period:** January 2011 – December 2012  
**Granularity:** Hourly  
**Total records:** 17,379 hours  

**Train/validation split:**  
An 80/20 chronological split is used — the first 80% of records by date form the training set, the remaining 20% form the validation set. Random splits are never used, as they would allow the model to train on future data and produce artificially inflated metrics.

| Split | Period | Records |
|---|---|---|
| Train | Jan 2011 – ~Oct 2012 | ~12,500 rows (after lag warmup) |
| Validation | ~Oct 2012 – Dec 2012 | ~3,100 rows |

**Note on the simulation:**  
In the live simulation, dates are shifted so that the dataset spans a period ending in the near future. The underlying demand patterns remain those of the original 2011–2012 Washington D.C. data. The model is periodically retrained as new "hours" are revealed by the simulation.

**Features used (20 total):**  
See [`docs/feature_engineering.md`](feature_engineering.md) for the full feature set and rationale.

---

## 3. Model Performance

Metrics are reported on the validation set (chronological hold-out, never seen during training or hyperparameter tuning).

| Metric | Value | What it means in plain language |
|---|---|---|
| **RMSE** | ~51 bikes/hr | On a typical hour the model is off by about 51 bikes |
| **RMSLE** | ~0.23 | The model's predictions are off by roughly 23% on a relative scale |
| **R²** | ~0.95 | The model explains about 95% of the variability in hourly demand |

**Context for RMSE:**  
The validation set has a mean demand of ~176 bikes/hr and a standard deviation of ~167 bikes/hr. An RMSE of 51 represents approximately a 29% average error relative to the mean — reasonable for a system with high variability and no access to real-time weather forecasts.

**RMSLE as the primary metric:**  
RMSLE penalises relative errors rather than absolute ones, treating a 50-bike error at low demand (e.g. 100 bikes) as more costly than the same 50-bike error at high demand (e.g. 600 bikes). This is the appropriate metric for right-skewed demand data and is the official evaluation metric for this dataset on Kaggle.

**Comparison to baseline:**

| Model | RMSE | RMSLE | R² |
|---|---|---|---|
| Random Forest (no tuning, single model) | 52.78 | 0.265 | 0.943 |
| LightGBM (tuned, split model) | ~51 | ~0.23 | ~0.95 |
| Improvement | ~3% | ~13% | +0.7pp |

---

## 4. Intended Use

### Recommended use cases

- **Hourly fleet rebalancing decisions** — given current conditions and recent history, estimate how many bikes will be needed in the next hour across the network.
- **Staffing planning** — anticipate high-demand periods to schedule rebalancing crews.
- **Capacity monitoring** — identify hours when supply is likely to fall below predicted demand.

### Not recommended for

- **Forecasting far beyond the primary horizon** — production does serve a multi-hour trajectory (h+1 through h+12) via recursive rollout rather than imputing missing lags (see [`docs/forecasting.md`](forecasting.md)), but forecast skill decays with lead time: `notebooks/04_experimento_horizontes.ipynb` and `05_multi_horizon_forecasting.ipynb` show the model's advantage over a seasonal-naive baseline narrowing past ~h+12 and flattening around ~h+18. This is exactly why `primary_horizon` is pinned to h+1 for all retrain/drift gating — only that lead time is treated as equivalent to the validation RMSE reported in this card.
- **Other cities or systems** — the model was trained exclusively on Washington D.C. Capital Bikeshare data. Demand patterns, weather sensitivities, and seasonal cycles reflect that specific city and infrastructure. Do not expect accurate predictions for other cities without retraining on local data.
- **Event-driven demand spikes** — the model has no knowledge of special events (concerts, marathons, protests). Large crowd events can produce demand spikes that fall far outside the training distribution.
- **Real-time operational decisions requiring sub-hour precision** — the model predicts at hourly granularity. It is not designed for minute-level dispatch decisions.

---

## 5. How to Interpret Predictions

For each origin hour, the model outputs a trajectory of predicted total bike demand across the next several hours (h+1 through h+12 — see [`docs/forecasting.md`](forecasting.md)). Each value should be interpreted as an **estimate of expected demand**, not a guaranteed count, and the h+1 prediction is the most reliable point in the trajectory — accuracy degrades gradually at longer lead times.

**Practical interpretation:**  
If the model predicts 200 bikes for the next hour, the actual demand is likely to fall within roughly ±51 bikes of that figure (one RMSE). However, errors are not uniformly distributed — peak hours (8 AM, 5–6 PM on weekdays) tend to have larger absolute errors because demand variability is highest during those periods.

**Registered vs casual breakdown:**  
The dashboard also shows the registered and casual components separately. This is operationally useful:
- High `registered` and low `casual` → commuter peak; bikes need to be available at transit hubs and residential areas.
- High `casual` and low `registered` → recreational peak (weekend midday); bikes should be available near parks, tourist areas, and waterfronts.

**SHAP values (in evaluation artifacts):**  
SHAP plots in `artifacts/evaluation/` show which features most influenced each prediction. Features pushing the prediction higher appear on the right; features pushing it lower appear on the left. `cnt_lag_24` and `cnt_lag_168` are typically the dominant contributors — recent same-hour demand is the strongest signal.

---

## 6. Known Limitations

**Weather features use observed values, not forecasts**  
In production the model would receive weather forecast data for the next hour. In this simulation, weather values are taken from the historical dataset — the model sees the actual weather at H+1, not a forecast. This means reported metrics are slightly optimistic compared to what a true production deployment would achieve.

**No concept drift detection beyond feature distributions**  
The drift detection system monitors feature distributions using the Kolmogorov-Smirnov test. It does not monitor prediction accuracy over time (performance drift), which would require ground truth labels to be available after each prediction. In a real deployment, both types of drift should be monitored.

**Minimum data requirement for drift-triggered retraining**  
Drift detection requires at least 720 hours (~30 days) of new production data before triggering retraining. In the early weeks of the simulation, drift may go undetected even if it exists, because the current data window is too small for statistical tests to be reliable.

**The model does not learn from its own prediction errors**  
There is no online learning or feedback loop. The model is retrained periodically on accumulated historical data, but does not adjust based on whether its individual predictions were accurate or not.

**Training data is from 2011–2012**  
Urban mobility patterns, bike sharing usage, and infrastructure have changed significantly since 2011. The model should be considered a simulation tool, not a system trained on current real-world data.

**No uncertainty quantification**  
The model outputs a point estimate with no confidence interval. Users cannot distinguish between a high-confidence prediction (e.g. a typical weekday morning) and a low-confidence one (e.g. an unusual weather event). Adding prediction intervals would require a different modelling approach (e.g. quantile regression or conformal prediction).

---

## 7. Model Versioning and Lifecycle

Models are versioned and managed in the MLflow Model Registry hosted on DagsHub.

**Lifecycle stages:**

| Stage | Description |
|---|---|
| Registered (no alias) | Newly trained model, not yet evaluated against production |
| `production` alias | Current model serving predictions in the hourly workflow |
| `archived` alias | Previous production model, superseded by a better version |

**Promotion logic:**  
Rather than promoting `registered` and `casual` independently, retraining evaluates all four combinations of (registered, casual) × (new, production) using their actual summed prediction RMSE on the validation set, and promotes whichever combination has the lowest combined RMSE — which can mean a mixed pair (a new model for one target, the existing production model for the other). Because the current production pair is always one of the four candidates compared, this can never regress combined accuracy. See [`docs/architecture.md`](architecture.md) § 5 for the full rationale.

**Retraining triggers:**  
- **Drift-triggered:** Weekly drift detection exceeds the configured threshold (default: 50% of features drifted) and at least 720 hours of new data have accumulated.
- **Scheduled:** Can be forced at any time with `make retrain-force`, bypassing drift detection.
