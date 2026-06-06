# Feature Engineering

This document describes the exploratory data analysis insights that drove feature engineering decisions, the features created and discarded, and the validation methodology used to confirm that each feature adds signal.

---

## Table of Contents

1. [EDA Key Insights](#1-eda-key-insights)
2. [Target Variable](#2-target-variable)
3. [Lag Features](#3-lag-features)
4. [Calendar Features](#4-calendar-features)
5. [Feature Validation — Mutual Information](#5-feature-validation--mutual-information)
6. [Features Discarded and Why](#6-features-discarded-and-why)
7. [Final Feature Set](#7-final-feature-set)
8. [Limitations](#8-limitations)

---

## 1. EDA Key Insights

The EDA (see `notebooks/01_eda.ipynb`) revealed several patterns that directly shaped the feature set:

**Two distinct demand regimes**  
Weekday demand follows a commuter double-peak (8 AM and 5–6 PM). Weekend demand follows a single recreational midday peak (10 AM–3 PM). Any feature encoding hour-of-day without accounting for this split will learn a blended, less accurate pattern.

**Strong seasonal cycle**  
Demand peaks in fall and dips in spring. The pattern is consistent across years — only the volume level changes, not the shape. This suggests an interaction between hour and season is more informative than season alone.

**Growth trend across years**  
2012 shows consistently higher demand than 2011 in every month. The binary `yr` feature (0/1) captures this coarsely. A continuous day counter captures it more precisely.

**Temperature is the dominant weather predictor**  
Temperature has a clear positive non-linear relationship with demand up to approximately 0.6 (normalised), after which it plateaus. Humidity has a moderate negative effect. Windspeed has a weak effect.

**`atemp` is redundant**  
`atemp` (apparent temperature) correlates with `temp` at r ≈ 0.99. Keeping both adds noise without information and causes multicollinearity in linear models.

**Holiday effect mirrors weekends**  
Public holidays follow a recreational demand profile similar to weekends, not the commuter profile of regular working days. The `workingday` flag already encodes this — `holiday` adds marginal independent signal.

---

## 2. Target Variable

The raw target `cnt` is right-skewed (mean ~176, std ~167, max ~957). Training directly on `cnt` causes models to over-weight peak hours, where absolute errors are large, at the expense of typical hours.

`log(cnt + 1)` produces a more symmetric distribution and aligns the training objective with RMSLE — the official Kaggle evaluation metric. The `+1` prevents `log(0)` for any zero-demand hours.

The model is trained on separate log-transformed targets for `registered` and `casual` riders. Predictions are back-transformed with `expm1` and summed at inference time.

**Why not train on `cnt` directly?**  
Both targets were compared using an untuned Random Forest. `log(cnt+1)` won on RMSLE (0.2612 vs 0.2651) — the more honest metric for a skewed distribution — and is the more principled choice.

---

## 3. Lag Features

Lag features give the model access to past demand as a predictor, capturing the temporal autocorrelation of bike demand. All lags use `shift(n)` to ensure the current hour's demand never leaks into its own predictors.

| Feature | Offset | Signal captured |
|---|---|---|
| `cnt_lag_1` | 1 hour | Short-term momentum — what happened immediately before |
| `cnt_lag_2` | 2 hours | Short-term momentum |
| `cnt_lag_3` | 3 hours | Short-term momentum |
| `cnt_lag_8` | 8 hours | Morning-to-afternoon peak correlation |
| `cnt_lag_24` | 24 hours | Same hour yesterday — strongest individual predictor |
| `cnt_lag_48` | 48 hours | Same hour two days ago |
| `cnt_lag_72` | 72 hours | Same hour three days ago |
| `cnt_lag_168` | 168 hours (7 days) | Same hour last week — second strongest predictor |
| `cnt_rolling_mean_24` | 24-hour rolling avg | Smoothed short-term trend |
| `cnt_rolling_mean_168` | 168-hour rolling avg | Smoothed weekly trend |

**Why lag_8?**  
If the morning commuter peak (8 AM) is high, the afternoon peak (5–6 PM) tends to be high as well — both reflect the same underlying daily activity level. `cnt_lag_8` captures the morning peak as a signal for the afternoon, approximately 8–10 hours later. Rather than hardcoding the exact peak-to-peak distance, lags 8, 9, and 10 were all evaluated; lag_8 was retained and 9 and 10 were discarded as redundant.

**Rolling averages use `shift(1)`**  
`cnt_rolling_mean_24 = df["cnt"].shift(1).rolling(24).mean()`. The `shift(1)` ensures the rolling window never includes the current hour's demand — without it, the feature would leak the target into itself.

**Lag features and the production use case**  
The model is designed for **next-hour forecasting**: given conditions and history up to hour H, predict demand at hour H+1. In this setting all lag features are available — `cnt_lag_1` at H+1 is simply the observed demand at H, which is known at prediction time.

For multi-day horizon forecasting, short-term lags would not be available and would need to be imputed. Validation experiments (see `notebooks/03_batch_scoring_validation.ipynb`) showed that imputing `cnt_lag_1` through `cnt_lag_8` causes RMSE to nearly double (~45 → ~87), confirming that next-hour forecasting is the appropriate use case for this model.

---

## 4. Calendar Features

### Cyclic encoding — `hr_sin`, `hr_cos`, `mnth_sin`, `mnth_cos`

Hour (0–23) and month (1–12) are cyclical variables. A model treating them as plain integers sees hour 23 and hour 0 as maximally distant when they are actually adjacent. Projecting them onto a unit circle with sine and cosine preserves this adjacency.

Two columns are always needed: sine alone is ambiguous (hours 0 and 12 both have sin = 0), but the (sin, cos) pair uniquely identifies every position on the circle — analogous to latitude and longitude.

```
hr_sin = sin(2π × hr / 24)
hr_cos = cos(2π × hr / 24)
```

### Regime separation — `hr_workday`, `hr_weekend`

The EDA showed two fundamentally different hourly demand profiles depending on day type. Rather than a single `hr` feature that blends both, the demand patterns are separated into two independent features:

```
hr_workday = hr × workingday     # active on weekdays, 0 on weekends
hr_weekend = hr × (1 - workingday)  # active on weekends, 0 on weekdays
```

The model can now learn the commuter profile from `hr_workday` and the recreational profile from `hr_weekend` independently. Both features ranked in the top 4 by mutual information, validating this design.

### Season interaction — `hr_x_season`

Demand volume shifts by season (fall > summer > winter > spring) but the hourly shape remains consistent. The interaction `hr × season` encodes both the time-of-day pattern and the seasonal volume level in a single feature. `season` alone has very low mutual information (MI ≈ 0.05) while `hr_x_season` has high mutual information (MI ≈ 0.57), confirming that the signal lives in the interaction, not in season independently.

### Rush hour flag — `is_rush_hour`

An explicit binary flag for commuter peak hours on working days:

```
is_rush_hour = 1 if (7 ≤ hr ≤ 9 or 17 ≤ hr ≤ 19) and workingday == 1
```

Demand in rush hours is approximately 2.7× higher than the average non-rush hour. While `hr_workday` implicitly captures this, the explicit flag gives linear models a direct handle on the highest-demand regime without relying on interactions.

### Growth trend — `days_since_start`

The binary `yr` feature (0 = 2011, 1 = 2012) captures the year-over-year growth trend coarsely. `days_since_start` — the number of days elapsed since the first record in the dataset — encodes the same trend continuously, allowing the model to learn fine-grained growth within each year.

Mutual information comparison:
- `yr`: MI = 0.035
- `days_since_start`: MI = 0.152

`yr` was dropped in favor of `days_since_start`.

---

## 5. Feature Validation — Mutual Information

All features were validated using Mutual Information (MI) regression before being included in the final set. MI measures how much knowing a feature reduces uncertainty about the target, capturing non-linear relationships and operating independently of any specific model — unlike Random Forest feature importance, which can be distorted by feature dominance.

### MI ranking (selected features)

| Feature | MI score | Tier |
|---|---|---|
| `cnt_lag_24` | 0.749 | High |
| `cnt_lag_168` | 0.668 | High |
| `hr_workday` | 0.596 | High |
| `hr_x_season` | 0.572 | High |
| `cnt_lag_2` | 0.482 | High |
| `cnt_lag_48` | 0.442 | Medium |
| `cnt_lag_72` | 0.385 | Medium |
| `hr_cos` | 0.338 | Medium |
| `hr_sin` | 0.332 | Medium |
| `cnt_lag_3` | 0.326 | Medium |
| `hr_weekend` | 0.225 | Medium |
| `cnt_lag_10` | 0.157 | Low |
| `temp` | 0.148 | Low |
| `days_since_start` | 0.152 | Low |
| `is_rush_hour` | 0.126 | Low |
| `cnt_rolling_mean_24` | 0.133 | Low |
| `cnt_rolling_mean_168` | 0.130 | Low |
| `hum` | 0.109 | Low |
| `weathersit` | 0.018 | Very low |
| `yr` | 0.035 | Very low → dropped |

**Note on Random Forest feature dominance**  
When a Random Forest was trained with all lag features included, `cnt_lag_1` captured ~78% of total feature importance. This is a known failure mode: the RF routes most splits through the strongest predictor, making the importance scores of all other features unreliable. MI scores, being model-agnostic, are not affected by this and provide a more honest ranking.

---

## 6. Features Discarded and Why

| Feature | Reason |
|---|---|
| `atemp` | r ≈ 0.99 with `temp` — identical signal, adds multicollinearity |
| `yr` | Replaced by `days_since_start` (MI 0.035 vs 0.152) |
| `cnt_lag_9`, `cnt_lag_10` | Redundant with `cnt_lag_8` per MI ranking |
| `workingday`, `holiday` | Signal fully captured by `hr_workday` and `hr_weekend` |
| `season`, `mnth_sin`, `mnth_cos` | Signal fully captured by `hr_x_season` |
| `windspeed` | Near-zero MI, weak EDA signal |
| `apparent_temp` | MI identical to `temp` (0.148 vs 0.148) — `hum` already in feature set |

---

## 7. Final Feature Set

20 features used in production:

**Lag features (10)**
`cnt_lag_1`, `cnt_lag_2`, `cnt_lag_3`, `cnt_lag_8`, `cnt_lag_24`, `cnt_lag_48`, `cnt_lag_72`, `cnt_lag_168`, `cnt_rolling_mean_24`, `cnt_rolling_mean_168`

**Calendar features (7)**
`hr_sin`, `hr_cos`, `hr_workday`, `hr_weekend`, `hr_x_season`, `is_rush_hour`, `days_since_start`

**Weather and context features (3)**
`temp`, `hum`, `weathersit`

---

## 8. Limitations

**Lag features assume continuous hourly data**  
If the system is restarted after a gap (e.g. missing hours in `hour_past.csv`), lag features will reference incorrect historical values. The current implementation does not detect or handle gaps in the time series.

**Weather features are from the dataset, not a live API**  
In a real production system, temperature, humidity, and weather condition would come from a weather forecast API for the next hour. In this simulation, they are taken from the historical dataset — the model sees the actual weather values, not forecast values. This means the model's real-world performance would likely be slightly worse than reported metrics suggest.

**No external features**  
Events (concerts, sports, holidays not in the dataset), infrastructure changes, or pricing changes are not captured. These could cause large prediction errors in edge cases.

**The model was trained on Washington D.C. data**  
The UCI dataset is from Washington D.C.'s Capital Bikeshare system (2011–2012). Demand patterns, weather relationships, and seasonal cycles reflect that specific city and time period. The model should not be expected to generalise to other cities or decades without retraining.
