.PHONY: help setup repro predict update drift retrain dashboard test

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "🚲 Bike Sharing Demand Forecasting"
	@echo ""
	@echo "Setup"
	@echo "  make setup        Download dataset and initialize simulation"
	@echo "  make repro        Run full DVC pipeline (features + train + evaluate)"
	@echo ""
	@echo "Simulation"
	@echo "  make update       Reveal new records from future to past"
	@echo "  make predict      Update simulation and predict next hour demand"
	@echo ""
	@echo "Monitoring"
	@echo "  make drift        Run drift detection"
	@echo "  make retrain      Retrain model if drift detected"
	@echo ""
	@echo "Dashboard"
	@echo "  make dashboard    Launch Streamlit operations dashboard"
	@echo ""
	@echo "Development"
	@echo "  make test         Run all tests"
	@echo "  make mlflow       Launch MLflow UI"
	@echo "  make drift-report Open drift report in browser"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	python src/bike_sharing/data/make_dataset.py
	python src/bike_sharing/data/shift_dates.py

repro:
	dvc repro

# ── Simulation ────────────────────────────────────────────────────────────────
update:
	python src/bike_sharing/data/update_simulation.py

predict:
	python src/bike_sharing/data/update_simulation.py
	python src/bike_sharing/models/predict.py

# ── Monitoring ────────────────────────────────────────────────────────────────
drift:
	python src/bike_sharing/monitoring/drift_detection.py

retrain:
	python src/bike_sharing/models/retrain.py

retrain-force:
	python src/bike_sharing/models/retrain.py training.force_retrain=true

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard:
	streamlit run src/bike_sharing/dashboard/app.py

# ── Development ───────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

mlflow:
	mlflow ui

drift-report:
	open artifacts/drift/drift_report.html