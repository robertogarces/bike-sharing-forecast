# Setup & Usage

How to install, configure, initialize, and run the project locally. For *what* the system is and *why* it's built the way it is, see the [Technical Overview](overview.md).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install](#2-install)
3. [Configure Credentials](#3-configure-credentials)
4. [Initialize and Run](#4-initialize-and-run)
5. [Docker](#5-docker)
6. [Make Commands](#6-make-commands)
7. [Configuration](#7-configuration)
8. [Development](#8-development)
9. [Project Structure](#9-project-structure)

---

## 1. Prerequisites

- Python 3.11
- A Kaggle API token (for dataset download)
- A DagsHub account (free) for the DVC remote and MLflow server

---

## 2. Install

```bash
conda create -n bike-sharing-forecast python=3.11 -y
conda activate bike-sharing-forecast
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` is compiled from `requirements.in`; install from the pinned `.txt` for a reproducible environment.

---

## 3. Configure Credentials

**Kaggle** — needed to download the dataset:
1. Go to [kaggle.com](https://www.kaggle.com) → Account → Settings → API → Create New Token. This downloads a `kaggle.json` file.
2. Extract the token value and run:
```bash
mkdir -p ~/.kaggle
echo YOUR_TOKEN_HERE > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

**DagsHub** — needed for DVC remote storage and MLflow tracking:
1. Create a free account at [dagshub.com](https://dagshub.com)
2. Create a new repository and connect it to your GitHub repo
3. Go to your DagsHub repo → User Settings → Access Tokens → Generate new token
4. Create a `.env` file in the project root (see `.env.example`):
```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<your-repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```
5. Configure DVC to use DagsHub as remote:
```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-username>
dvc remote modify origin --local password <your-dagshub-token>
```

---

## 4. Initialize and Run

Before running the pipeline you must initialize the simulation. `shift_dates.py` shifts the dataset dates so a configurable fraction of records sits in the "future" relative to a reference date; as real time advances, `update_simulation.py` moves them to the past one hour at a time — exactly as new observations would arrive in production. See [Simulation](simulation.md) for the full mechanics.

Configure two parameters in `configs/simulation/default.yaml` before initializing:
- `reference_date` — the date from which the future window begins (e.g. `"2026-06-04"`)
- `future_pct` — fraction of the dataset to reserve as future data (e.g. `0.10` for 10%)

Then:

```bash
make setup      # download dataset + initialize the simulation (run once)
make repro      # run the DVC pipeline: build_features -> train -> evaluate
make predict    # reveal new records (with data-quality validation) and forecast the next 12 hours
make dashboard  # launch the operations dashboard
```

> ⚠️ Run `make setup` only once. The simulation state is protected — re-running it aborts with a warning. To reset, delete `data/simulation_state.json` explicitly and re-run.

---

## 5. Docker

The dashboard and MLflow UI are containerized with Docker Compose, so you can run the observability stack without installing anything locally beyond Docker. Volumes mount your local `data/` and `artifacts/` directories into the containers, so they always reflect the current state of the simulation.

```bash
docker compose up --build
```

- Dashboard: `http://localhost:8501`
- MLflow UI: `http://localhost:5001`

---

## 6. Make Commands

`make help` prints this list. Every action is a short `make` target rather than a long Python command.

| Command | Description |
|---|---|
| `make setup` | Download the dataset and initialize the simulation (run once) |
| `make repro` | Run the full DVC pipeline (build_features + train + evaluate) |
| `make update` | Reveal new records from future to past (with data-quality validation) |
| `make predict` | Update the simulation and forecast the h+1…h+12 trajectory |
| `make drift` | Run input-drift detection |
| `make drift-report` | Open the latest Evidently drift report (HTML) |
| `make retrain` | Retrain and promote if a trigger fires and enough new data exists |
| `make retrain-force` | Force a retrain regardless of drift (`training.force_retrain=true`) |
| `make dashboard` | Launch the Streamlit operations dashboard |
| `make test` | Run the test suite (`pytest tests/ -v`) |
| `make mlflow` | Launch the MLflow UI locally |

---

## 7. Configuration

All configuration is Hydra-based under `configs/` — nothing tuneable is hardcoded in the source. `configs/config.yaml` composes the config groups:

| Group | Controls |
|---|---|
| `dataset/` | Kaggle dataset source |
| `features/` | Lags, rolling windows, split ratio, dropped columns |
| `model/` | LightGBM params, Optuna search space, `tune` toggle |
| `simulation/` | `reference_date`, `future_pct` |
| `forecast/` | `horizon` (K served) and `primary_horizon` (gating lead time) |
| `training/` | `force_retrain`, `min_new_hours` retrain gate |
| `validation/` | Required columns and valid ranges for data-quality checks |
| `monitoring/` | Drift threshold, lookback window, backfill cap, degradation threshold |
| `alerting/` | Dedup window and GitHub issue labels |
| `paths/` | Artifact and data locations |

Any Hydra value can be overridden on the command line, e.g. `make retrain-force` expands to `... training.force_retrain=true`.

---

## 8. Development

```bash
ruff check .            # lint (matches CI)
ruff format --check .   # format check (matches CI)
pytest tests/ -v        # 156 unit tests
```

CI (`.github/workflows/ci.yml`) runs all three on every push and pull request.

---

## 9. Project Structure

The system is split into a **static pipeline** (runs when code or data definitions change — download, build features, train, evaluate) and a **dynamic production layer** (runs on a schedule via GitHub Actions — reveal data, predict, monitor, retrain). This separation avoids mixing pipeline orchestration with production automation.

```
bike-sharing/
├── configs/                    # Hydra config groups (dataset, features, model, simulation,
│                               #   forecast, training, validation, monitoring, alerting, paths)
├── data/
│   ├── raw/                    # Original + shifted datasets, past/future split
│   ├── processed/              # train.csv / val.csv (build_features output)
│   └── predictions/            # Hourly prediction log (predictions.csv)
├── notebooks/                  # 01 EDA … 05 multi-horizon forecasting experiments
├── src/bike_sharing/
│   ├── data/                   # make_dataset, shift_dates, update_simulation, validate_data
│   ├── features/               # build_features
│   ├── models/                 # train, evaluate, predict, retrain
│   ├── monitoring/             # drift_detection, output_drift_detection,
│   │                           #   performance_monitoring, hourly_alert, weekly_report, suggest_thresholds
│   ├── dashboard/              # Streamlit app (Operations + Monitoring pages)
│   ├── visualization/          # plotting helpers
│   └── utils/                  # alerting, mlflow_utils, datetime_utils, monitoring_utils,
│                               #   simulation_utils, command_utils
├── tests/                      # 156 unit tests (20 modules)
├── artifacts/
│   ├── models/                 # Trained model files
│   ├── evaluation/             # Metrics, SHAP, residual plots
│   ├── drift/                  # Drift reports and flags
│   ├── monitoring/             # Performance/drift history, retrain outcome
│   └── validation/             # Hourly data-quality flag
├── azure/                      # Azure ML deployment (score.py, sync_deployment_versions.py, YAML specs)
├── .github/workflows/          # ci, hourly, weekly, deploy-azure
├── docs/                       # Documentation
├── dvc.yaml                    # Pipeline definition
├── Dockerfile
├── docker-compose.yaml
├── Makefile
└── pyproject.toml
```
