# System Architecture

This document describes the design of the bike sharing demand forecasting system, the rationale behind key decisions, and the trade-offs considered.

---

## Table of Contents

1. [High-Level Design](#1-high-level-design)
2. [Data Flow](#2-data-flow)
3. [Static Pipeline (DVC)](#3-static-pipeline-dvc)
4. [Dynamic Production Layer (GitHub Actions)](#4-dynamic-production-layer-github-actions)
5. [Key Design Decisions](#5-key-design-decisions)
6. [Infrastructure](#6-infrastructure)

---

## 1. High-Level Design

The system is split into two clearly separated layers with different responsibilities and trigger conditions:

```
┌─────────────────────────────────────────────────────────────┐
│                     STATIC PIPELINE (DVC)                   │
│  Triggered manually or when code / data definitions change  │
│                                                             │
│  make_dataset → shift_dates → build_features → train → evaluate  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ artifacts: models, metrics, plots
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              DYNAMIC PRODUCTION LAYER (GitHub Actions)      │
│                    Triggered on a schedule                  │
│                                                             │
│  hourly:  update_simulation → predict                       │
│  weekly:  drift_detection  → retrain (if needed)            │
└─────────────────────────────────────────────────────────────┘
```

This separation solves a common mistake in ML projects: mixing pipeline orchestration with production automation. The static pipeline is reproducible and deterministic — given the same inputs it always produces the same outputs. The dynamic layer is event-driven and stateful — it advances the simulation and operates on continuously changing data.

---

## 2. Data Flow

```
Kaggle API
    │
    ▼
data/raw/hour.csv                    ← original, never modified
    │
    ▼
shift_dates.py  ──────────────────── configs/simulation/default.yaml
    │
    ├── data/raw/hour_shifted.csv    ← full dataset with shifted dates
    ├── data/raw/hour_past.csv       ← records up to reference_date (grows over time)
    ├── data/raw/hour_future.csv     ← records after reference_date (shrinks over time)
    └── data/simulation_state.json   ← simulation metadata and guard file
    │
    ▼
build_features.py  (reads hour_past.csv)
    │
    ├── data/processed/train.csv
    └── data/processed/val.csv
    │
    ▼
train.py
    │
    ├── artifacts/models/lgbm_registered.txt
    ├── artifacts/models/lgbm_casual.txt
    └── MLflow registry (DagsHub) ── production alias
    │
    ▼
evaluate.py
    │
    ├── artifacts/evaluation/metrics.json
    ├── artifacts/evaluation/residuals.png
    ├── artifacts/evaluation/actual_vs_predicted.png
    ├── artifacts/evaluation/demand_over_time.png
    ├── artifacts/evaluation/shap_registered.png
    └── artifacts/evaluation/shap_casual.png

─── Production (runs hourly) ───────────────────────────────────

update_simulation.py
    │ moves records from hour_future.csv to hour_past.csv
    ▼
predict.py
    │ loads production models (registered + casual) from MLflow registry
    │ recursively rolls out h+1..h+K predictions from a single origin
    │ (see docs/forecasting.md)
    ▼
data/predictions/predictions.csv     ← append-only prediction log
                                        (horizon column; dedup key = timestamp_predicted + horizon)

─── Monitoring (runs hourly) ────────────────────────────────────

output_drift_detection.py
    │ compares recent output distribution (primary horizon only) vs reference
    └── artifacts/drift/output_drift_detected.json
    │
    ▼
hourly_alert.py
    │ checks output drift / data-quality signals, opens/dedupes a GitHub
    │ issue, sends an email alert (never fails the job — see § 5)

─── Monitoring (runs weekly) ────────────────────────────────────

performance_monitoring.py
    │ computes rolling RMSE/RMSLE/MAE/R² per horizon vs seasonal-naive baseline
    └── data/monitoring/performance_history.csv
    │
    ▼
drift_detection.py
    │ compares train.csv (reference snapshot) vs recent hour_past.csv (current)
    └── artifacts/drift/drift_detected.json
    │
    ▼
retrain.py
    │ if drift detected and enough data: dvc repro → evaluate all 4
    │ (registered, casual) × (new, prod) combinations → promote lowest
    │ combined RMSE → snapshot drift reference at promotion
    └── new model version(s) in MLflow registry
    │
    ▼
weekly_report.py
    │ builds a digest (primary-horizon performance + per-horizon curve,
    │ drift status, retrain outcome), opens a GitHub issue, emails it
```

---

## 3. Static Pipeline (DVC)

The static pipeline is managed by DVC and defined in `dvc.yaml`. Each stage declares its inputs (`deps`), outputs (`outs`), and parameters (`params`), which allows DVC to skip stages whose inputs have not changed.

### Stages

**`build_features`** — frozen by default  
Reads `hour_past.csv` and produces `train.csv` and `val.csv`. This stage is frozen (`dvc freeze build_features`) to prevent DVC from re-running it automatically when `hour_past.csv` changes during the simulation. It is unfrozen only during explicit retraining via `retrain.py`.

**`train`**  
Trains two LightGBM models — one for `registered` demand, one for `casual` demand — using the best hyperparameters found by Optuna or loaded from `configs/model/best_params.yaml`. Both models are registered in the MLflow model registry on DagsHub.

**`evaluate`**  
Loads the trained models, runs predictions on the validation set, computes metrics, generates SHAP plots, and logs everything to MLflow.

### Why `build_features` is frozen

`hour_past.csv` is a live file — it grows every hour as `update_simulation.py` moves new records from future to past. Without freezing, `dvc repro` would re-run the entire pipeline on every hourly update, retraining the model every hour. Freezing decouples data arrival from retraining, which is controlled explicitly by `retrain.py`.

---

## 4. Dynamic Production Layer (GitHub Actions)

Three workflows automate the production layer:

### `ci.yml` — Continuous Integration
Runs on every push to `main`. Executes the full test suite (156 unit tests) on a clean Ubuntu environment. Catches regressions before they reach production.

### `hourly.yml` — Hourly Prediction
Runs every hour at minute 0 (`cron: "0 * * * *"`). Steps:
1. Pull latest code and data from GitHub / DagsHub
2. `update_simulation.py` — move newly-revealed records from future to past
3. `predict.py` — load production models from MLflow registry, recursively roll out the h+1..h+K trajectory (see [`docs/forecasting.md`](forecasting.md)), append to `predictions.csv`
4. `output_drift_detection.py` — compare the primary horizon's recent output distribution against a reference window
5. `hourly_alert.py` — check output drift / data-quality signals; open or dedupe a GitHub issue and send an email alert if needed (see § 5, "Alerting fails open, not closed")
6. Push updated simulation files back to DagsHub via DVC
7. Commit updated `.dvc` pointer files to GitHub

### `weekly.yml` — Drift Detection and Retraining
Runs every Monday at midnight (`cron: "0 0 * * 1"`). Steps:
1. Pull latest code and data
2. `performance_monitoring.py` — compute rolling RMSE/RMSLE/MAE/R² per horizon against a seasonal-naive baseline; write `performance_history.csv`
3. `drift_detection.py` — compare feature distributions between the training set (frozen at promotion time — see § 5, "Combination-based promotion") and the most recent 168 hours. Uses the Kolmogorov-Smirnov test. Writes `drift_detected.json`.
4. `retrain.py` — if drift is detected and enough new data has accumulated (≥720 hours), unfreeze `build_features`, run `dvc repro`, evaluate all four (registered, casual) × (new, production) combinations by combined RMSE, and promote whichever combination wins.
5. `weekly_report.py` — build a digest (primary-horizon performance, per-horizon skill curve, drift status, retrain outcome), open a GitHub issue, and email it. Runs with `if: always()`, so a failure earlier in the job still produces and sends a report.

Alerting configuration (SMTP credentials, GitHub issue labels, alert deduplication window) lives in `configs/alerting/default.yaml`.

### Dashboard

The Streamlit dashboard (`src/bike_sharing/dashboard/app.py`) exposes two pages via `st.navigation()`:

**Operations** — what an operator acts on. A KPI row (Now / Next hour / Next peak / Next quietest, with a live-ring animation on "Now"), an hourly-forecast strip, and a "Forecast Trajectory" chart spanning the full h+1..h+K rollout (see [`docs/forecasting.md`](forecasting.md) § 9). This page was redesigned to answer only "what should I act on right now" — it previously mixed in model-quality views (a gauge, a 24h prediction-history chart, a registered/casual breakdown, a full forecast-detail table) that have moved to Monitoring or been removed as redundant.

**Monitoring** — is the model healthy. Model Metrics (Combined always visible; Registered/Casual collapsed into expanders — three fully-expanded blocks of four metrics each was visually saturated), Model Status (live MLflow registry query), Retrain Gate — Last Run, Live Performance Over Time (primary horizon), Performance by Horizon — Latest, Input Drift, and Output Drift Over Time.

Both pages label their subtitle "· All times UTC," since the dashboard's clock is tied to the UTC `hr` feature (see [`docs/forecasting.md`](forecasting.md) § 8 for why `utc_now()` matters here).

Two operational bugs were fixed along the way, worth noting since they explain non-obvious UI elements: fallback predictions (served when hourly data validation fails — a full lag-168 trajectory instead of a model prediction) are now explicitly marked with a banner, a distinct chart marker, and a table column, rather than being indistinguishable from real model output; and a false-alert bug in `hourly_alert.py` (`bool(float('nan'))` evaluates to `True` in Python) was fixed by checking `.get("drift_detected") is True` explicitly instead of a bare truthy check.

---

## 5. Key Design Decisions

### Two models instead of one

The target variable `cnt` is the sum of `registered` and `casual` riders. These two populations follow fundamentally different patterns: registered users show a commuter double-peak on weekdays; casual users show a recreational midday peak on weekends. Training a single model on `cnt` forces it to learn both patterns simultaneously, which increases the difficulty of the learning task.

Training separate models on `log(registered+1)` and `log(casual+1)` and summing their predictions at inference time produced better RMSE and more interpretable SHAP plots. The trade-off is double the training time and two model versions to manage.

### LightGBM over ARIMA

Classical forecasting models (ARIMA, SARIMA) were considered but rejected for three reasons:
- The dataset has multiple overlapping seasonal cycles (hourly, daily, weekly) which SARIMA handles poorly with a single seasonal period parameter.
- Rich external features (temperature, humidity, weather condition, day type) are difficult to incorporate in ARIMA-family models.
- LightGBM with lag features consistently outperforms classical models on demand forecasting tasks in practice.

The approach — a regression model with engineered temporal features — is called feature-based forecasting and is the dominant approach in production ML systems.

### Log-transformed target

`cnt` is right-skewed (mean ~176, std ~167, max ~957). Training on `log(cnt+1)` produces a more symmetric target distribution and aligns the optimization objective with RMSLE, the Kaggle evaluation metric for this dataset. Predictions are back-transformed with `expm1` before evaluation and logging.

### Temporal train/val split

A random train/test split would allow the model to train on future data and validate on the past, inflating all metrics. An 80/20 chronological split ensures the model is evaluated strictly on data it has never seen and that lies after all training data in time — exactly the production scenario.

### Frozen `build_features` stage

Described in Section 3. The key insight is that data arrival (hourly) and model retraining (weekly, when drift is detected) operate on different cadences. Conflating them would cause continuous retraining that undermines the stability of the production model.

### DagsHub for storage and tracking

DagsHub provides DVC remote storage and a hosted MLflow server in a single free platform. This avoids setting up separate S3, EC2, and MLflow infrastructure, while still demonstrating the concepts (data versioning, experiment tracking, model registry) that would apply in a production AWS/GCP environment.

### Horizon lives in serving, not training

Rather than training a dedicated model per lead time, the system reuses the single h+1 model and rolls it out recursively to produce h+1..h+K predictions (see [`docs/forecasting.md`](forecasting.md)). This was validated empirically, not just assumed: `notebooks/04_experimento_horizontes.ipynb` compares a direct per-horizon model against the recursive rollout on the real feature pipeline, and the recursive approach wins on both cost (one model instead of K) and accuracy (its RMSE stays consistently lower than the direct model's ceiling at every horizon beyond h+1, since the direct model loses its short-lag features as the horizon grows).

### Combination-based promotion instead of independent per-model promotion

`registered` and `casual` are no longer promoted independently based on each model's own RMSE improving. Instead, `retrain.py` evaluates all four combinations of (registered, casual) × (new, production) using actual summed and clipped predictions on the same validation set, and promotes whichever combination has the lowest combined RMSE — defaulting to keeping the current pair on a tie. Because the current production pair is always one of the four candidates evaluated, promotion can never make the combined prediction worse. This can produce a mixed pair (e.g. a new `registered` model paired with the existing `casual` model), which is why Azure pins `MODEL_VERSION_REGISTERED`/`MODEL_VERSION_CASUAL` independently (see [`docs/azure.md`](azure.md)). At promotion time, a snapshot of the drift-relevant features is also attached to the registered model's production run as an MLflow artifact, so the drift reference used by `drift_detection.py` reflects what the production model actually learned from, not the ever-advancing live `train.csv`. See [`docs/known-issues.md`](known-issues.md) for the residual atomicity concern in this promotion loop.

### Dashboard split into Operations vs Monitoring

The dashboard's two pages separate "what to act on" from "is the model healthy" — see the Dashboard subsection above for the full breakdown of each page's contents and the redesign rationale.

### Alerting fails open, not closed

`send_email()` never raises — a missing or invalid SMTP secret should not fail the predict/retrain job it's attached to. Failures are instead surfaced as a GitHub Actions `::warning::` annotation, visible directly in the workflow run UI, so a silent alerting failure doesn't go unnoticed just because the pipeline job itself stayed green.

---

## 6. Infrastructure

| Component | Tool | Hosted on |
|---|---|---|
| Code versioning | Git | GitHub |
| Data & artifact versioning | DVC | DagsHub |
| Experiment tracking | MLflow | DagsHub |
| Model registry | MLflow | DagsHub |
| Pipeline orchestration | DVC | Local / GitHub Actions |
| Automation | GitHub Actions | GitHub |
| Dashboard | Streamlit | Docker (local) |
| Configuration | Hydra | Local |

All credentials (DagsHub token, Kaggle token, MLflow tracking URI) are stored as GitHub Secrets and never committed to the repository. Local development uses a `.env` file that is git-ignored.

**Parallel Azure ML deployment:** the same MLOps concepts were also implemented on Azure ML — a DVC remote on Blob Storage, training as an Azure ML Command Job, experiment tracking and model registry in the workspace, a managed online endpoint, and OIDC-based CI/CD — running alongside the DagsHub setup without touching the live system. See [docs/azure.md](azure.md) for the full walkthrough.
