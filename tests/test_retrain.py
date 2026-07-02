import pytest
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from bike_sharing.models.retrain import (
    should_retrain,
    count_new_hours,
    write_retrain_marker,
    evaluate_production_pair,
    promote_models_if_better,
)
from bike_sharing.models.train import FEATURES


@pytest.fixture
def drift_flag_dir():
    """
    Creates a temporary directory with a drift_detected.json file.
    Returns both the directory and the flag path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        flag_path = Path(tmpdir) / "drift_detected.json"
        yield flag_path


def write_flag(flag_path: Path, content: dict) -> None:
    """Helper to write a drift flag JSON file."""
    with open(flag_path, "w") as f:
        json.dump(content, f)


def test_should_retrain_returns_true_when_drift_detected(drift_flag_dir):
    """
    should_retrain should return True when drift_detected is True in the flag.
    """
    write_flag(drift_flag_dir, {
        "drift_detected": True,
        "drift_share":    0.7,
        "threshold":      0.5,
    })

    assert should_retrain(drift_flag_dir, force=False) is True


def test_should_retrain_returns_false_when_no_drift(drift_flag_dir):
    """
    should_retrain should return False when drift_detected is False.
    """
    write_flag(drift_flag_dir, {
        "drift_detected": False,
        "drift_share":    0.2,
        "threshold":      0.5,
    })

    assert should_retrain(drift_flag_dir, force=False) is False


def test_should_retrain_returns_false_when_insufficient_data(drift_flag_dir):
    """
    should_retrain should return False when drift detection was skipped
    due to insufficient data — identified by the presence of 'reason' key.
    """
    write_flag(drift_flag_dir, {
        "drift_detected": False,
        "drift_share":    0.0,
        "threshold":      0.5,
        "reason":         "Not enough data (168 rows < 720 minimum)",
    })

    assert should_retrain(drift_flag_dir, force=False) is False


def test_should_retrain_force_overrides_no_drift(drift_flag_dir):
    """
    When force=True, should_retrain should return True regardless of
    what the drift flag says — used for scheduled retraining.
    """
    write_flag(drift_flag_dir, {
        "drift_detected": False,
        "drift_share":    0.0,
        "threshold":      0.5,
    })

    assert should_retrain(drift_flag_dir, force=True) is True


def test_should_retrain_returns_false_when_no_flag_exists(drift_flag_dir):
    """
    If drift_detected.json doesn't exist, should_retrain should return False
    and warn the user to run drift_detection.py first.
    """
    # Don't write any flag — file doesn't exist
    assert should_retrain(drift_flag_dir, force=False) is False


# ── count_new_hours / write_retrain_marker ────────────────────────────────────

@pytest.fixture
def hour_past_csv():
    """
    Creates a temporary hour_past.csv with 72 consecutive hourly records
    starting at 2026-01-01 00:00, plus a directory for the marker.
    Yields (hour_past_path, marker_path).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        times = pd.date_range("2026-01-01 00:00", periods=72, freq="h")
        df = pd.DataFrame({
            "dteday": times.normalize().strftime("%Y-%m-%d"),
            "hr":     times.hour,
            "cnt":    range(72),
        })
        hour_past_path = tmp / "hour_past.csv"
        df.to_csv(hour_past_path, index=False)
        yield hour_past_path, tmp / "last_retrain.json"


def write_marker(marker_path: Path, cutoff: str) -> None:
    """Helper to write a retrain marker with a given data cutoff."""
    with open(marker_path, "w") as f:
        json.dump({"data_cutoff": cutoff, "written_at": "2026-01-01T00:00:00"}, f)


def test_count_new_hours_returns_none_when_no_marker(hour_past_csv):
    """
    No marker yet → bootstrap signal (None), so the first retrain can proceed.
    """
    hour_past_path, marker_path = hour_past_csv
    assert count_new_hours(hour_past_path, marker_path) is None


def test_count_new_hours_counts_records_after_cutoff(hour_past_csv):
    """
    Cutoff in the middle of the series → counts only records strictly after it.
    72 records (hours 0..71); cutoff at hour 47 → 24 newer records.
    """
    hour_past_path, marker_path = hour_past_csv
    write_marker(marker_path, "2026-01-02T23:00:00")  # 48th record (idx 47)
    assert count_new_hours(hour_past_path, marker_path) == 24


def test_count_new_hours_zero_when_cutoff_at_or_after_max(hour_past_csv):
    """
    Cutoff at the latest record (or beyond) → no new data.
    """
    hour_past_path, marker_path = hour_past_csv
    write_marker(marker_path, "2026-01-03T23:00:00")  # last record (hour 71)
    assert count_new_hours(hour_past_path, marker_path) == 0


def test_write_then_count_roundtrip_is_zero(hour_past_csv):
    """
    Writing a marker from hour_past and immediately counting should yield 0 —
    the boundary is set to the latest record, so nothing is newer.
    """
    hour_past_path, marker_path = hour_past_csv
    write_retrain_marker(hour_past_path, marker_path)
    assert marker_path.exists()
    assert count_new_hours(hour_past_path, marker_path) == 0


# ── evaluate_production_pair ──────────────────────────────────────────────────

class _FakeModel:
    """Model stub whose predict() returns a fixed, caller-supplied array."""

    def __init__(self, log_preds):
        self.log_preds = log_preds

    def predict(self, X):
        return self.log_preds


def test_evaluate_production_pair_perfect_prediction_gives_zero_rmse(monkeypatch):
    """
    evaluate_production_pair combines registered + casual predictions the same
    way evaluate.py does (expm1, sum, clip). If both fakes predict exactly the
    log-space values that reconstruct cnt, the combined RMSE must be 0.
    """
    project = "bike-sharing-forecast"
    val_df = pd.DataFrame({**{f: 0.0 for f in FEATURES}, "cnt": [100, 200]})

    fakes = {
        f"{project}-registered": _FakeModel(np.log1p([60, 150])),
        f"{project}-casual":     _FakeModel(np.log1p([40, 50])),
    }

    def fake_load_model(model_uri):
        for name, model in fakes.items():
            if model_uri == f"models:/{name}@production":
                return model
        raise AssertionError(f"unexpected model_uri: {model_uri}")

    monkeypatch.setattr(mlflow.lightgbm, "load_model", fake_load_model)

    rmse = evaluate_production_pair(project, val_df)

    assert rmse == pytest.approx(0.0)


def test_evaluate_production_pair_returns_none_when_no_production_model(monkeypatch):
    """
    No production alias yet (bootstrap) — nothing to compare against.
    MLflow raises MlflowException when an alias doesn't exist; the function
    should catch it and return None rather than propagate.
    """
    def raise_not_found(model_uri):
        raise mlflow.exceptions.MlflowException("Registered model alias production not found")

    monkeypatch.setattr(mlflow.lightgbm, "load_model", raise_not_found)

    val_df = pd.DataFrame({**{f: 0.0 for f in FEATURES}, "cnt": [100]})

    assert evaluate_production_pair("bike-sharing-forecast", val_df) is None


# ── promote_models_if_better ──────────────────────────────────────────────────

class _FakeVersion:
    def __init__(self, version):
        self.version = version


class _FakeMlflowClient:
    """
    Records set_registered_model_alias calls so tests can assert on them,
    without touching a real MLflow registry.

    Parameters
    ----------
    versions : dict
        {model_name: latest_version_str} — what search_model_versions returns.
    production : dict
        {model_name: current_production_version_str} — used by
        get_model_version_by_alias.
    """

    def __init__(self, versions: dict, production: dict):
        self.versions    = versions
        self.production  = production
        self.alias_calls = []  # list of (model_name, alias, version)

    def search_model_versions(self, filter_str):
        name = filter_str.split("'")[1]
        return [_FakeVersion(self.versions[name])]

    def get_model_version_by_alias(self, name, alias):
        return _FakeVersion(self.production[name])

    def set_registered_model_alias(self, name, alias, version):
        self.alias_calls.append((name, alias, version))


def test_promote_models_if_better_bootstraps_when_no_production():
    """
    prod_rmse=None (no production alias exists yet) → promote both new
    versions directly, no comparison needed.
    """
    project = "bike-sharing-forecast"
    client = _FakeMlflowClient(
        versions={f"{project}-registered": "3", f"{project}-casual": "3"},
        production={},
    )

    promoted = promote_models_if_better(client, project, new_rmse=50.0, prod_rmse=None)

    assert promoted is True
    assert (f"{project}-registered", "production", "3") in client.alias_calls
    assert (f"{project}-casual",     "production", "3") in client.alias_calls


def test_promote_models_if_better_promotes_when_new_beats_production():
    """
    new_rmse < prod_rmse → both new versions become production, both old
    production versions become archived (atomic, both-or-neither).
    """
    project = "bike-sharing-forecast"
    client = _FakeMlflowClient(
        versions={f"{project}-registered": "5", f"{project}-casual": "5"},
        production={f"{project}-registered": "4", f"{project}-casual": "4"},
    )

    promoted = promote_models_if_better(client, project, new_rmse=45.0, prod_rmse=50.0)

    assert promoted is True
    assert (f"{project}-registered", "production", "5") in client.alias_calls
    assert (f"{project}-casual",     "production", "5") in client.alias_calls
    assert (f"{project}-registered", "archived",   "4") in client.alias_calls
    assert (f"{project}-casual",     "archived",   "4") in client.alias_calls


def test_promote_models_if_better_keeps_production_when_new_is_worse():
    """
    new_rmse >= prod_rmse → new versions are archived, production alias is
    left untouched (previous champion retained).
    """
    project = "bike-sharing-forecast"
    client = _FakeMlflowClient(
        versions={f"{project}-registered": "5", f"{project}-casual": "5"},
        production={f"{project}-registered": "4", f"{project}-casual": "4"},
    )

    promoted = promote_models_if_better(client, project, new_rmse=55.0, prod_rmse=50.0)

    assert promoted is False
    assert (f"{project}-registered", "archived", "5") in client.alias_calls
    assert (f"{project}-casual",     "archived", "5") in client.alias_calls
    assert not any(alias == "production" for (_, alias, _) in client.alias_calls)