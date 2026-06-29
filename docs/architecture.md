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
    │ loads model from MLflow registry (production alias)
    ▼
data/predictions/predictions.csv     ← append-only prediction log

─── Monitoring (runs weekly) ────────────────────────────────────

drift_detection.py
    │ compares train.csv (reference) vs recent hour_past.csv (current)
    └── artifacts/drift/drift_detected.json
    │
    ▼
retrain.py
    │ if drift detected and enough data: dvc repro → promote if better
    └── new model version in MLflow registry
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
Runs on every push to `main`. Executes the full test suite (41 unit tests) on a clean Ubuntu environment. Catches regressions before they reach production.

### `hourly.yml` — Hourly Prediction
Runs every hour at minute 0 (`cron: "0 * * * *"`). Steps:
1. Pull latest code and data from GitHub / DagsHub
2. `update_simulation.py` — move newly-revealed records from future to past
3. `predict.py` — load production model from MLflow registry, predict next-hour demand, append to `predictions.csv`
4. Push updated simulation files back to DagsHub via DVC
5. Commit updated `.dvc` pointer files to GitHub

### `weekly.yml` — Drift Detection and Retraining
Runs every Monday at midnight (`cron: "0 0 * * 1"`). Steps:
1. Pull latest code and data
2. `drift_detection.py` — compare feature distributions between the training set and the most recent 168 hours. Uses the Kolmogorov-Smirnov test. Writes `drift_detected.json`.
3. `retrain.py` — if drift is detected and enough new data has accumulated (≥720 hours), unfreeze `build_features`, run `dvc repro`, and promote the new model to production only if it improves RMSE.

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
