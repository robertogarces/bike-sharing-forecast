# Multi-Horizon Forecasting

This document describes how the bike sharing demand forecasting system serves multi-horizon predictions — forecasts from the immediate next hour (h+1) through twelve hours ahead (h+12) — and the design decisions that make this work.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Configuration](#2-configuration)
3. [Design Decision: Horizon in Serving, Not Training](#3-design-decision-horizon-in-serving-not-training)
4. [The Recursive Trajectory Algorithm](#4-the-recursive-trajectory-algorithm)
5. [Storage: Horizon Column and the Dedup Key](#5-storage-horizon-column-and-the-dedup-key)
6. [Per-Horizon Monitoring](#6-per-horizon-monitoring)
7. [The Primary Horizon Filtering Pattern](#7-the-primary-horizon-filtering-pattern)
8. [Supporting Infrastructure: UTC Time and the Shift Invariant](#8-supporting-infrastructure-utc-time-and-the-shift-invariant)
9. [Dashboard Integration](#9-dashboard-integration)

---

## 1. Overview

The underlying LightGBM model is still a **single h+1 (next-hour) model**. It predicts the next hour given data up to the current hour, and its training and validation use only h+1 data. Multi-horizon forecasting is a **serving-layer capability**, not a model capability — it is built on top of the base model through recursive rollout. At prediction time, the model is applied recursively to generate h+1, h+2, ..., h+K predictions (K=12 by default) from a single origin hour.

---

## 2. Configuration

Multi-horizon behavior is controlled via `configs/forecast/default.yaml`:

```yaml
horizon: 12
primary_horizon: 1
```

**`horizon: 12`** — the number of lead times served per prediction run. The system produces K rows per origin: one row for each horizon h+1 through h+K. The value K=12 was chosen empirically from the skill-vs-naive curve in `notebooks/05_multi_horizon_forecasting.ipynb`: the model beats the seasonal-naive baseline comfortably through ~h+12 hours and flattens to parity with naive forecasting beyond ~h+18.

**`primary_horizon: 1`** — the single lead time that governs all single-value decisions: the retrain degradation gate, the output drift check, and the weekly report headline. This must stay at 1 because h+1 is the only horizon directly comparable to the h+1 single-step validation RMSE used as the promotion baseline. Horizons beyond h+1 accumulate recursive-rollout error and aren't apples-to-apples comparable; treating h+2 or beyond as the "representative" horizon would make the degradation gate blind to real deterioration.

**Key invariant:** changing `horizon` requires no retraining. It is a pure serving parameter; the trained model and its validation RMSE do not change.

---

## 3. Design Decision: Horizon in Serving, Not Training

Why not train separate per-horizon models (e.g., a dedicated h+1 model, h+2 model, ..., h+12 model)?

This was tested empirically in `notebooks/04_experimento_horizontes.ipynb`, which compares three
approaches on the real training pipeline (`build_lag_features`/`build_calendar_features`, same
model params as production): a **direct** model per horizon (trained only on the features
available at that distance — short lags aren't available further out), the **recursive** rollout
(reuses the h+1 model, feeding its own predictions back as synthetic lags), and the
**seasonal-naive** baseline (`cnt_lag_168`).

| Horizon | Direct (RMSE) | Recursive (RMSE) |
|---|---|---|
| h+1  | 55.30 | 56.49 |
| h+2  | 73.68 | 67.10 |
| h+3  | 79.59 | 69.23 |
| h+6  | 85.58 | 70.66 |
| h+12 | 85.84 | 74.91 |
| h+24 | 85.84 | 77.98 |

**The result is not just a cost trade-off — recursive wins on accuracy too**, at every horizon
beyond h+1. The direct model loses its short-lag and rolling-mean features as the horizon grows
(they aren't available that far out), so its error rises fast and plateaus around RMSE ~86 from
h+6 onward. The recursive rollout keeps those features by feeding its own predictions back in,
and although its error accumulates step over step, it stays consistently lower than the direct
model's ceiling across the whole studied range (h+2 through h+24).

This means training K dedicated models would be strictly worse than the current design on both
axes: more expensive (K models to tune, register, and monitor instead of one) **and** less
accurate. The only two costs the recursive approach actually pays are (1) error still grows with
horizon — it just grows slower than the direct model's — and (2) it depends on the h+1 model
alone, so any h+1 degradation propagates to every horizon. This is exactly why `primary_horizon`
exists as a separate concept: to keep the retrain gate, drift detection, and operational
reporting anchored to the only horizon the model was genuinely trained and validated on.

---

## 4. The Recursive Trajectory Algorithm

**Location:** `src/bike_sharing/models/predict.py::predict_trajectory()` (lines 150–231).

The core algorithm walks forward one hour at a time from an origin timestamp, recursively predicting K future hours. The key insight is that **predictions are fed back as synthetic actuals**, so the next step's lag features come from the model's own prior output, not from missing-data imputation.

### Walk-Forward Loop

```
For horizon k = 1 to K:
  1. Read the current last row (prior prediction or actual)
  2. Build a synthetic row for the target hour (k steps ahead)
  3. Compute features on that synthetic row (reusing the real training pipeline)
  4. Predict with the trained model
  5. Append the prediction row to the output
  6. Carry the prediction forward as a synthetic actual for the next iteration
```

### Synthetic Row Construction: `build_synthetic_row()`

Each forward step requires a new row with correct values for the target hour. The challenge is that we cannot look ahead (future is unknown), but we can use deterministic knowledge and approximation:

**Calendar features** (`weekday`, `workingday`, `season`, `holiday`, `mnth`, `year`) — these are deterministic and knowable in advance. They are read from `calendar_lookup`, built from `hour_future.csv` (known future dates) plus the shifted training dates. This lookup is critical for correctness: without it, a prediction run that crosses midnight (e.g., starting Friday 23:00 and predicting through Saturday) would wrongly carry Friday's `weekday`/`workingday` values forward into Saturday, corrupting those features. The calendar lookup fixes this.

**Weather features** (`temp`, `atemp`, `hum`, `windspeed`) — genuinely unknown ahead of time. The system uses a lag-24h approximation: weather at `target_dt` is approximated as the actual weather at `target_dt - 24 hours` (same hour, previous day), read from `weather_lookup` (built from historical `hour_past.csv`). This is an educated guess, not perfect foresight, and it is a documented limitation of the production system (see § 6 Known Limitations in `docs/model_card.md`). Falls back to carrying the prior row's weather forward when the lag isn't available (e.g., for horizon beyond the historical window).

### Feature Pipeline Reuse

After building the synthetic row, it is appended to a working history, and **the exact training feature pipeline** (`build_lag_features` + `build_calendar_features`) is applied. This ensures lag features (`cnt_lag_1`, `cnt_lag_3`, `cnt_lag_24`, etc.) are computed identically to training — no duplicated/diverged feature logic, no hidden train/serve skew.

### Prediction Feedback

The prediction (`pred_registered`, `pred_casual`, summed and clipped to `pred_total`) is stored as a **synthetic actual** in the working history:

```python
work.loc[new_index, "cnt"] = pred_total
work.loc[new_index, "registered"] = pred_registered
work.loc[new_index, "casual"] = pred_casual
```

On the next iteration, when `build_lag_features` computes `cnt_lag_1`, it pulls from this synthetic value, not from imputation. This is the crucial difference from the failed imputation approach documented in `docs/feature_engineering.md`: imputation (guessing missing lags with a statistic) nearly doubles error (~51 → ~87 RMSE in historical validation), but recursive feedback (using the model's own prior output) degrades gracefully because the model was trained to handle its own inaccuracies — it has seen "noisy" demand inputs throughout training.

### Backfill Reframing: Missing Origins

The backfill step (filling gaps when a prediction run fails to happen) is now reframed in terms of **origins**, not hours. A missing origin is a timestamp where the recursive trajectory should have been run but wasn't, leaving a gap in the `predictions.csv` log. Because each origin produces up to K rows, backfilling one missing origin requires re-running `predict_trajectory()` on the past-data slice up to that origin.

This is handled by `get_missing_origins()` in `predict.py` — it detects missing origins by checking whether each origin's horizon=1 row exists in the log. For each missing origin, `predict_trajectory()` is re-run (reproducing the exact trajectory that would have been generated on the original run), and the rows are appended.

---

## 5. Storage: Horizon Column and the Dedup Key

### The `horizon` Column

`predictions.csv` gains a `horizon` column (integer, 1 to K). Each row in the log has a `timestamp_predicted` (when the origin was) and a `horizon` (the lead time). For example, an origin at 10:00 produces:
- Row 1: `timestamp_predicted=10:00`, `horizon=1`, `timestamp=11:00`, `pred_total=X`
- Row 2: `timestamp_predicted=10:00`, `horizon=2`, `timestamp=12:00`, `pred_total=Y`
- ...
- Row 12: `timestamp_predicted=10:00`, `horizon=12`, `timestamp=22:00`, `pred_total=Z`

### The Dedup Key

The dedup key for `predictions.csv` is now **`(timestamp_predicted, horizon)`**, not just `timestamp_predicted`. This allows the same hour to be predicted from different origins at different lead times without overwriting, enabling the full trajectory to be stored separately from any other run's trajectory.

### Legacy Compatibility

Rows in `predictions.csv` predating this change do not have a `horizon` column. They are treated as `horizon=1` everywhere downstream for backward compatibility — this "legacy compat" pattern is applied consistently across all modules (`predict.py`, `performance_monitoring.py`, `drift_detection.py`, `retrain.py`, `weekly_report.py`, `dashboard/app.py`).

### Backfill Capacity Note

A backfilled origin produces up to K rows, so the effective backfilled-row count is `missing_origins × horizon`. This matters for capacity planning if a backfill window is ever configured.

---

## 6. Per-Horizon Monitoring

### Rolling Performance by Horizon

`src/bike_sharing/monitoring/performance_monitoring.py::compute_rolling_performance_by_horizon()` computes rolling-window RMSE, RMSLE, MAE, R² independently per horizon. It groups the `(predictions, actuals)` joined frame by `horizon`, and for each group computes the rolling metrics over the most recent `n_hours` observations *at that horizon* — not one pooled window across all horizons, which would be meaningless since error naturally grows with lead time.

### Seasonal-Naive Baseline (Horizon-Independent)

The seasonal-naive baseline (`build_seasonal_naive`) is **horizon-independent**: it always compares the actual demand at time `t` against the actual demand at `t - 168h` (same hour, previous week). This comparison is applied unchanged to every horizon, because the naive forecast doesn't "get worse" with lead time the way the model does. This is intentional: the seasonal-naive baseline is a fixed external yardstick, not a per-horizon model. The skill-vs-naive metric (1 - model_RMSE / naive_RMSE) therefore degrades gracefully: at h+1 the model has high skill; at h+12 it approaches zero or goes negative if the model is worse than naive at that horizon.

### MLflow Logging

Per-horizon metrics are logged to MLflow with `step=horizon`, so all K horizons' metrics land on a single MLflow metric chart keyed by step. This allows visualizing the error/skill curve directly in MLflow's metric UI without manual aggregation.

---

## 7. The Primary Horizon Filtering Pattern

Three production components now filter to `primary_horizon` before making any single-value decision:

**1. `retrain.py::is_performance_degraded()`** — the retrain gate.
Filters `performance_history.csv` to `df["horizon"] == primary_horizon`, reads the latest row at that lead time, and compares against `combined_rmse_baseline`. Mixing horizons into one comparison would corrupt the gate: horizon-12 error is always larger than horizon-1 error, so an unchanged model would appear "degraded" if you mixed the two. Only h+1 is directly comparable to the h+1 validation RMSE from training.

**2. `output_drift_detection.py`** — the output drift check.
Filters `predictions` to `horizon == primary_horizon` *before* calling `split_rolling_windows()`. Mixing lead times into the same rolling window would corrupt both the window span and the drift signal, since the distribution of prediction *residuals* (predicted - actual) differs across horizons.

**3. `weekly_report.py`** — the weekly report headline.
Calls `_load_primary_performance_row(path, primary_horizon)` to fetch the latest row at the primary horizon for the headline metric (comparable to the retrain baseline), and separately calls `_load_horizon_curve(path)` to fetch the latest record per horizon for an optional "Skill by Horizon" section in the report body. The headline number is the single comparable metric; the curve is additional context.

### Rule of Thumb for Future Contributors

**Any code comparing predicted vs. actual for a single-number decision must filter to `primary_horizon` first.** This includes new alerting rules, new reporting metrics, or new gating logic.

---

## 8. Supporting Infrastructure: UTC Time and the Shift Invariant

### `utc_now()` Helper

`src/bike_sharing/utils/datetime_utils.py::utc_now()` (lines 19–25):

```python
def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
```

This helper explicitly reads the current instant in UTC and strips the timezone info, returning a naive datetime in UTC. It is a drop-in replacement for bare `datetime.now()`.

**Why it was needed:** Every script in the simulation/monitoring pipeline (`predict.py`, `retrain.py`, `drift_detection.py`, `hourly_alert.py`, `performance_monitoring.py`, `weekly_report.py`, `shift_dates.py`, `update_simulation.py`) was calling naive `datetime.now()`. This only produced correct results because GitHub Actions runners default to UTC; the code relied on an implicit environmental assumption. Run the pipeline locally on a machine with a non-UTC clock, and naive `datetime.now()` silently returns wall-clock time, causing:
- Simulation clock skew (the "current hour" in the simulation doesn't match the actual predictions' hour)
- Horizon-to-clock-hour mapping errors (horizon-k prediction ends up paired with the wrong hour of the day)

Commit `43bb4d2` replaced all 12 call sites with `utc_now()`, making UTC explicit and testable.

### The Shift Invariant

Documented in `shift_dates.py::shift_dates()` (lines 54–59):

> Invariant: only `dteday` is shifted here. The calendar columns (`weekday`, `workingday`, `season`, `holiday`, `mnth`) travel untouched from the original dataset — their true day-of-week no longer matches the shifted `dteday`. Nothing downstream may ever recompute them from the shifted `dteday`; they must always be read straight from the row, which still carries the original value the model was trained on.

**What this protects:** The simulation advances time by shifting all `dteday` values by a constant offset, so the historical UCI dataset (2011–2012) appears to sit in the near future. The calendar-derived features must *not* be recalculated from this shifted date — they must stay exactly as they were in the original data, because the model was trained on the original 2011–2012 dates' calendar features. If a future code path decided to "fix" this by recomputing `weekday` from the shifted date (which would seem more "correct" on the surface), it would silently desync those features from what the model learned, corrupting predictions.

**Connection to `build_synthetic_row()`:** This invariant is why `build_synthetic_row()` reads calendar features from `calendar_lookup` (deterministic, real dates, built from the unshifted future dates in `hour_future.csv`) rather than naively recomputing them from `target_dt`. The calendar lookup is the single source of truth for calendar features, ensuring the model always sees the weekday/workingday/season it was trained on.

**Test guard:** `tests/test_predict.py::test_predict_trajectory_horizon_maps_to_clock_hour` explicitly asserts that from an origin at hour H, the horizon=k row's target timestamp equals H + k hours. This guards against a bug where horizon and timestamp could be paired only by row order rather than by time arithmetic.

---

## 9. Dashboard Integration

The dashboard exposes multi-horizon forecasts to operators through two new charts and several supporting helpers.

### New Charts

**Operations page — "Forecast Trajectory"** (function `render_operations()`, lines 262–362 in `src/bike_sharing/dashboard/app.py`)
A Plotly filled-area chart showing predicted demand (`pred_total`) vs. `timestamp_predicted` across the full h+1..h+K trajectory. The chart is built via the `latest_trajectory(predictions)` helper, which filters to the most recent origin. Hover text shows the lead time (`+Nh`) and target time. This gives operators a visual sense of the demand curve over the next 12 hours.

**Monitoring page — "Performance by Horizon — Latest"** (function `render_monitoring()`, lines 366–584)
A line chart with two series (model RMSE and seasonal-naive RMSE) plotted against horizon (x-axis: hours ahead). Built via `latest_per_horizon(performance_history)`. Includes a caption reporting the horizon at which the model reaches parity with seasonal-naive, if any — the "multi-horizon payoff" visualization.

### Supporting Helpers

All in `src/bike_sharing/dashboard/app.py`:

- **`ensure_horizon(df)`** (lines 54–64): Normalizes the `horizon` column, filling missing values with 1 (legacy compatibility).
- **`latest_trajectory(predictions)`** (lines 67–81): Filters to the most recent origin and returns its full trajectory sorted by horizon.
- **`filter_to_horizon(predictions, horizon)`** (lines 84–93): Returns only rows at one specific lead time — used to build the primary-horizon view for "Live Performance Over Time."
- **`latest_per_horizon(history)`** (lines 96–104): Extracts the most recent record per horizon from `performance_history.csv` — the current error/skill curve.

### Dashboard Structure Note

The full Operations/Monitoring page structure, rationale for splitting these concerns, and other dashboard design decisions are documented in [Architecture § Dashboard](architecture.md#dashboard).

---

## See Also

- [`docs/feature_engineering.md`](feature_engineering.md) § 3 and § 8 for why recursive rollout solves the multi-day forecasting limitation of naive imputation.
- [`docs/simulation.md`](simulation.md) § 6 for the full backfill-by-origin mechanics, capacity planning, and cold-start behavior.
- [`docs/architecture.md`](architecture.md) § 4 for the per-horizon monitoring step in the weekly workflow and § 5 for design decisions on promotion and alerting.
- [`docs/model_card.md`](model_card.md) § 1, § 4, and § 5 for how multi-horizon serving interacts with model capabilities and intended use.
- Notebooks `04_experimento_horizontes.ipynb` and `05_multi_horizon_forecasting.ipynb` for the empirical analysis behind the horizon=12 choice.
