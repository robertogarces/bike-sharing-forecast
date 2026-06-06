# The Production Simulation

This document explains what the simulation is, why it exists, how it works technically, and how to configure, initialize, and reset it.

---

## Table of Contents

1. [What is the Simulation?](#1-what-is-the-simulation)
2. [How it Works](#2-how-it-works)
3. [Initialization](#3-initialization)
4. [Configuration](#4-configuration)
5. [Day-to-Day Operation](#5-day-to-day-operation)
6. [Resetting the Simulation](#6-resetting-the-simulation)
7. [What Happens When the Future Data Runs Out?](#7-what-happens-when-the-future-data-runs-out)

---

## 1. What is the Simulation?

The UCI Bike Sharing dataset covers 2011–2012 — historical data with no new observations arriving. A model trained on static historical data and evaluated once is a notebook exercise, not a production system.

To make this project behave like a live deployment, dates in the dataset are shifted forward in time so that a configurable slice of the data sits in the near future. As real time advances, those future records become available one hour at a time — exactly as sensor readings or API data would arrive in a real bike sharing system.

This allows the system to genuinely:
- Predict the next hour's demand with real lag features from recent history
- Detect data drift as the distribution of incoming data evolves
- Retrain the model on accumulated new data
- Operate the hourly GitHub Actions workflow on a continuous basis until the future data is exhausted

**This is a deliberate simulation, not a claim of real-time data.** The underlying demand patterns are those of Washington D.C. in 2011–2012. The simulation makes the project behave like a production system without requiring access to a live data feed.

---

## 2. How it Works

### Date shifting

The original dataset spans a fixed range (January 2011 – December 2012). `shift_dates.py` applies a constant offset to every `dteday` value so that the last `future_pct` fraction of records starts exactly at `reference_date`.

For example, with `reference_date = "2026-06-15"` and `future_pct = 0.10`:
- The 10% of records (≈1,737 hours ≈ 72 days) that were at the end of 2012 now start on June 15, 2026
- All earlier records are shifted by the same offset, preserving the temporal structure
- The dataset now spans approximately mid-2024 to mid-August 2026

### The past/future split

After shifting, the dataset is split at `reference_date`:

```
data/raw/hour_past.csv    ← records before reference_date  (model can use this)
data/raw/hour_future.csv  ← records from reference_date onward  (not yet "observed")
```

`hour_past.csv` is what the model trains on and predicts from. `hour_future.csv` is the pool of records that will be revealed one hour at a time.

### Hourly revelation

Every time `update_simulation.py` runs, it compares each record's datetime in `hour_future.csv` against the current system time. Any record whose datetime has already passed is moved to `hour_past.csv`. If the script hasn't run for several hours (e.g. due to a GitHub Actions delay), all records for the elapsed hours are moved at once.

```
Before (19:00):
  hour_past.csv:    ... records up to 18:00
  hour_future.csv:  19:00, 20:00, 21:00, ...

After (20:30):
  hour_past.csv:    ... records up to 20:00  ← 19:00 and 20:00 moved
  hour_future.csv:  21:00, 22:00, ...
```

### The state file

`data/simulation_state.json` is created when the simulation is initialized and serves two purposes:
1. **Guard** — prevents accidental re-initialization. Running `shift_dates.py` a second time will abort with an error pointing to this file.
2. **Metadata** — records the configuration used (reference date, future percentage, simulation boundaries) so the system always knows the current state.

```json
{
  "reference_date": "2026-06-15",
  "future_pct": 0.10,
  "shift_applied_at": "2026-06-04T21:47:46",
  "future_start_date": "2026-06-15",
  "future_end_date": "2026-08-28",
  "n_future_records": 1737,
  "n_past_records": 15642
}
```

---

## 3. Initialization

The simulation must be initialized once before the pipeline can run. Initialization is a one-time operation — it cannot be re-run without explicitly resetting (see Section 6).

**Prerequisites:**
- Dataset downloaded to `data/raw/hour.csv` (via `make setup` or `python src/bike_sharing/data/make_dataset.py`)
- `reference_date` and `future_pct` configured in `configs/simulation/default.yaml`

**Initialize:**

```bash
make setup
```

This runs `make_dataset.py` (downloads the data) followed by `shift_dates.py` (applies the date shift and creates the past/future split).

After initialization, run the full pipeline:

```bash
make repro
```

This builds features from `hour_past.csv`, trains the model, and evaluates it.

---

## 4. Configuration

Two parameters in `configs/simulation/default.yaml` control the simulation:

```yaml
reference_date: "2026-06-15"   # date from which the future window begins
future_pct: 0.10               # fraction of total records to reserve as future
```

**`reference_date`**  
The date from which the simulation's "live" period begins. Records on or after this date are placed in `hour_future.csv` and revealed gradually as real time advances. Choose a date in the near future relative to when you initialize the simulation — if you set it in the past, all future records will be immediately revealed on the first `update_simulation.py` run.

**`future_pct`**  
The fraction of the total dataset (17,379 records) reserved for the simulation. At 10%, approximately 1,737 hours (~72 days) of future data are available. At 5%, approximately 869 hours (~36 days).

| `future_pct` | Future records | Duration |
|---|---|---|
| 0.05 | ~869 hours | ~36 days |
| 0.10 | ~1,737 hours | ~72 days |
| 0.15 | ~2,607 hours | ~109 days |
| 0.20 | ~3,476 hours | ~145 days |

**Important:** These parameters must be set before initialization. Changing them after the simulation has started requires a full reset.

---

## 5. Day-to-Day Operation

Once initialized, the simulation runs automatically via GitHub Actions:

**Every hour** (`hourly.yml`):
```
update_simulation.py  →  reveal records whose datetime has passed
predict.py            →  predict next-hour demand using the latest past data
```

**Every Monday** (`weekly.yml`):
```
drift_detection.py    →  compare recent data distribution to training distribution
retrain.py            →  retrain if drift detected and enough new data accumulated
```

To run manually at any time:

```bash
make predict    # reveal new records + predict next hour
make drift      # run drift detection
make retrain    # retrain if drift detected
```

**Checking simulation progress:**  
The operations dashboard (sidebar) shows the percentage of future data consumed and the date until which data is available. The logs of `update_simulation.py` also report progress:

```
[INFO] - Simulation progress: 92.3% complete
```

---

## 6. Resetting the Simulation

To restart the simulation from scratch — for example, to change `reference_date` or `future_pct` — delete the state file and the simulation data files, then re-initialize:

```bash
# Delete simulation state
rm data/simulation_state.json

# Delete simulation data files
rm data/raw/hour_past.csv
rm data/raw/hour_future.csv
rm data/raw/hour_shifted.csv

# Delete prediction log
rm -rf data/predictions/

# Re-initialize
make setup
make repro
```

⚠️ **This is destructive.** The prediction log and any retraining history will be lost. The MLflow experiment history on DagsHub is not affected — model versions and runs remain.

After resetting locally, push the updated DVC pointer files so GitHub Actions uses the new simulation state:

```bash
dvc push
git add data/raw/hour_past.csv.dvc data/raw/hour_future.csv.dvc data/simulation_state.json.dvc dvc.lock
git commit -m "chore: reset simulation"
git push
```

---

## 7. What Happens When the Future Data Runs Out?

When `hour_future.csv` becomes empty, `update_simulation.py` exits with a warning:

```
[WARNING] - Simulation exhausted — no future records remaining.
```

At this point:
- `predict.py` will continue to run but will always predict based on the same last record — predictions will be stale.
- The hourly GitHub Actions workflow will complete without errors but without producing new predictions.
- The system does not crash or enter an error state.

**Options when the simulation is exhausted:**
1. **Reset with a new reference date** — follow the reset procedure in Section 6, set a new `reference_date` further in the past, and use a larger `future_pct` to extend the simulation window.
2. **Accept the end of the simulation** — the project has served its purpose as a portfolio demonstration. The prediction log, MLflow history, and model artifacts remain intact.
