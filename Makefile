# Bike Sharing Demand Forecasting — common tasks. Run `make help` to list targets.
PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help
help:	## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Environment ---
.PHONY: install
install:	## Install runtime dependencies + this package (editable)
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

# --- Setup (run once) ---
.PHONY: setup repro
setup:	## Download the dataset and initialize the simulation (run once)
	$(PYTHON) src/bike_sharing/data/make_dataset.py
	$(PYTHON) src/bike_sharing/data/shift_dates.py

repro:	## Run the full DVC pipeline (build_features -> train -> evaluate)
	dvc repro

# --- Simulation ---
.PHONY: update predict
update:	## Reveal new records from future to past (with data-quality validation)
	$(PYTHON) src/bike_sharing/data/update_simulation.py

predict:	## Update the simulation and forecast the h+1..h+12 trajectory
	$(PYTHON) src/bike_sharing/data/update_simulation.py
	$(PYTHON) src/bike_sharing/models/predict.py

# --- Monitoring ---
.PHONY: drift drift-report performance output-drift suggest-thresholds retrain retrain-force
drift:	## Run input-drift detection
	$(PYTHON) src/bike_sharing/monitoring/drift_detection.py

drift-report:	## Open the latest Evidently drift report (HTML)
	open artifacts/drift/drift_report.html

performance:	## Compute rolling live-performance metrics vs. seasonal-naive baseline
	$(PYTHON) src/bike_sharing/monitoring/performance_monitoring.py

output-drift:	## Run output (prediction) drift detection
	$(PYTHON) src/bike_sharing/monitoring/output_drift_detection.py

suggest-thresholds:	## Suggest drift/degradation thresholds from accumulated history (12+ weeks)
	$(PYTHON) src/bike_sharing/monitoring/suggest_thresholds.py

retrain:	## Retrain and promote if a trigger fires and enough new data exists
	$(PYTHON) src/bike_sharing/models/retrain.py

retrain-force:	## Force a retrain regardless of drift/performance
	$(PYTHON) src/bike_sharing/models/retrain.py training.force_retrain=true

# --- Dashboard ---
.PHONY: dashboard
dashboard:	## Launch the Streamlit operations dashboard
	streamlit run src/bike_sharing/dashboard/app.py

# --- Development ---
.PHONY: lint format test mlflow
lint:	## Lint with ruff (matches CI)
	ruff check .

format:	## Auto-format with ruff (writes changes — CI only checks)
	ruff format .

test:	## Run the test suite
	pytest tests/ -v

mlflow:	## Launch the MLflow UI locally
	mlflow ui

# --- Reset ---
.PHONY: reset-simulation delete-all
reset-simulation:	## Reset the live simulation (state, predictions, monitoring history); keeps the trained model (destructive; requires FORCE=1)
	@if [ "$(FORCE)" != "1" ]; then \
		echo "This will permanently delete:"; \
		echo "  - simulation state + past/future/shifted data (data/simulation_state.json, data/raw/hour_*.csv)"; \
		echo "  - predictions log + last retrain marker (data/predictions/, data/last_retrain.json)"; \
		echo "  - drift/performance/output-drift history + validation flag (artifacts/drift/, artifacts/monitoring/, artifacts/validation/)"; \
		echo "  - matching .dvc pointer files"; \
		echo ""; \
		echo "Trained models (artifacts/models/, artifacts/evaluation/) are NOT touched."; \
		echo ""; \
		echo "Re-run with FORCE=1 to actually delete: make reset-simulation FORCE=1"; \
		exit 1; \
	fi
	rm -f data/simulation_state.json data/simulation_state.json.dvc
	rm -f data/raw/hour_past.csv data/raw/hour_past.csv.dvc
	rm -f data/raw/hour_future.csv data/raw/hour_future.csv.dvc
	rm -f data/raw/hour_shifted.csv
	rm -rf data/predictions/
	rm -f data/last_retrain.json data/last_retrain.json.dvc
	rm -f artifacts/drift/drift_detected.json artifacts/drift/drift_report.html
	rm -f artifacts/monitoring/drift_history.csv artifacts/monitoring/output_drift_history.csv artifacts/monitoring/performance_history.csv artifacts/monitoring/retrain_outcome.json
	rm -f artifacts/validation/hourly_validation.json
	@echo "Simulation state wiped. Run 'make setup' to start over."

delete-all:	## Wipe everything generated -- simulation, trained model, and raw dataset -- back to a pristine clone (destructive; requires FORCE=1)
	@if [ "$(FORCE)" != "1" ]; then \
		echo "This will permanently delete EVERYTHING reset-simulation deletes, plus:"; \
		echo "  - the raw downloaded dataset (data/raw/hour.csv, day.csv, Readme.txt)"; \
		echo "  - the processed features (data/processed/train.csv, val.csv)"; \
		echo "  - the trained models (artifacts/models/)"; \
		echo "  - the evaluation artifacts (artifacts/evaluation/)"; \
		echo ""; \
		echo "You will need 'make setup' (re-downloads from Kaggle) and 'make repro' to rebuild everything from scratch."; \
		echo ""; \
		echo "Re-run with FORCE=1 to actually delete: make delete-all FORCE=1"; \
		exit 1; \
	fi
	rm -f data/simulation_state.json data/simulation_state.json.dvc
	rm -f data/raw/hour_past.csv data/raw/hour_past.csv.dvc
	rm -f data/raw/hour_future.csv data/raw/hour_future.csv.dvc
	rm -f data/raw/hour_shifted.csv
	rm -rf data/predictions/
	rm -f data/last_retrain.json data/last_retrain.json.dvc
	rm -f artifacts/drift/drift_detected.json artifacts/drift/drift_report.html
	rm -f artifacts/monitoring/drift_history.csv artifacts/monitoring/output_drift_history.csv artifacts/monitoring/performance_history.csv artifacts/monitoring/retrain_outcome.json
	rm -f artifacts/validation/hourly_validation.json
	rm -f data/raw/hour.csv data/raw/day.csv data/raw/Readme.txt
	rm -f data/processed/train.csv data/processed/val.csv
	rm -f artifacts/models/lgbm_registered.txt artifacts/models/lgbm_casual.txt
	rm -f artifacts/evaluation/*.png artifacts/evaluation/metrics.json
	@echo "Everything wiped. Run 'make setup && make repro' to rebuild from scratch."
