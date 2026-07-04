import pytest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bike_sharing.models.retrain import (
    should_retrain,
    count_new_hours,
    write_retrain_marker,
    is_performance_degraded,
    snapshot_drift_reference,
    _combination_rmses,
    choose_best_combination,
    promote_best_combination,
)
from bike_sharing.monitoring.drift_detection import DRIFT_FEATURES


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
    write_flag(
        drift_flag_dir,
        {
            "drift_detected": True,
            "drift_share": 0.7,
            "threshold": 0.5,
        },
    )

    assert should_retrain(drift_flag_dir, force=False) is True


def test_should_retrain_returns_false_when_no_drift(drift_flag_dir):
    """
    should_retrain should return False when drift_detected is False.
    """
    write_flag(
        drift_flag_dir,
        {
            "drift_detected": False,
            "drift_share": 0.2,
            "threshold": 0.5,
        },
    )

    assert should_retrain(drift_flag_dir, force=False) is False


def test_should_retrain_returns_false_when_insufficient_data(drift_flag_dir):
    """
    should_retrain should return False when drift detection was skipped
    due to insufficient data — identified by the presence of 'reason' key.
    """
    write_flag(
        drift_flag_dir,
        {
            "drift_detected": False,
            "drift_share": 0.0,
            "threshold": 0.5,
            "reason": "Not enough data (168 rows < 720 minimum)",
        },
    )

    assert should_retrain(drift_flag_dir, force=False) is False


def test_should_retrain_force_overrides_no_drift(drift_flag_dir):
    """
    When force=True, should_retrain should return True regardless of
    what the drift flag says — used for scheduled retraining.
    """
    write_flag(
        drift_flag_dir,
        {
            "drift_detected": False,
            "drift_share": 0.0,
            "threshold": 0.5,
        },
    )

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
        df = pd.DataFrame(
            {
                "dteday": times.normalize().strftime("%Y-%m-%d"),
                "hr": times.hour,
                "cnt": range(72),
            }
        )
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


# ── _combination_rmses ────────────────────────────────────────────────────────


def test_combination_rmses_captures_error_cancellation_not_individual_accuracy():
    """
    Reproduces the session's Example 1: each new model is individually more
    accurate than its production counterpart (error 5 vs 10), but the
    production pair's errors happen to cancel in the sum (+10, -10) while the
    new pair's compound (+5, +5) — so the combined RMSE is WORSE for the
    "individually better" new pair. This is exactly why combined RMSE, not
    per-model RMSE, must drive the promotion decision.
    """
    y_cnt = np.array([150.0, 280.0, 210.0])  # true registered + casual
    new_reg = np.array([105.0, 205.0, 155.0])  # true registered + 5
    new_cas = np.array([55.0, 85.0, 65.0])  # true casual + 5
    prod_reg = np.array([110.0, 210.0, 160.0])  # true registered + 10
    prod_cas = np.array([40.0, 70.0, 50.0])  # true casual - 10

    combos = _combination_rmses(y_cnt, new_reg, new_cas, prod_reg, prod_cas)

    assert combos[("prod", "prod")] == pytest.approx(0.0)
    assert combos[("new", "new")] == pytest.approx(10.0)
    assert combos[("new", "new")] > combos[("prod", "prod")]


# ── choose_best_combination ───────────────────────────────────────────────────


def test_choose_best_combination_picks_mixed_pair_when_it_wins():
    """Reproduces Example 2: a mixed pair beats both pure pairs and production."""
    combos = {
        ("new", "new"): 30.0,
        ("new", "prod"): 0.0,
        ("prod", "new"): 50.0,
        ("prod", "prod"): 20.0,
    }
    assert choose_best_combination(combos) == ("new", "prod")


def test_choose_best_combination_keeps_production_on_tie():
    """A tie with production must not churn aliases for zero real gain."""
    combos = {
        ("new", "new"): 20.0,
        ("new", "prod"): 20.0,
        ("prod", "new"): 25.0,
        ("prod", "prod"): 20.0,
    }
    assert choose_best_combination(combos) == ("prod", "prod")


def test_choose_best_combination_keeps_production_when_it_wins():
    combos = {
        ("new", "new"): 25.0,
        ("new", "prod"): 22.0,
        ("prod", "new"): 30.0,
        ("prod", "prod"): 20.0,
    }
    assert choose_best_combination(combos) == ("prod", "prod")


# ── is_performance_degraded ──────────────────────────────────────────────────


def test_is_performance_degraded_true_when_over_threshold(tmp_path):
    history_path = tmp_path / "performance_history.csv"
    pd.DataFrame([{"rmse": 60.0}, {"rmse": 61.0}]).to_csv(history_path, index=False)

    degraded, live_rmse = is_performance_degraded(
        history_path, baseline_rmse=50.0, degradation_threshold=0.2
    )

    assert degraded is True
    assert live_rmse == 61.0


def test_is_performance_degraded_false_when_within_threshold(tmp_path):
    history_path = tmp_path / "performance_history.csv"
    pd.DataFrame([{"rmse": 55.0}]).to_csv(history_path, index=False)

    degraded, live_rmse = is_performance_degraded(
        history_path, baseline_rmse=50.0, degradation_threshold=0.2
    )

    assert degraded is False
    assert live_rmse == 55.0


def test_is_performance_degraded_false_when_no_baseline(tmp_path):
    history_path = tmp_path / "performance_history.csv"
    pd.DataFrame([{"rmse": 100.0}]).to_csv(history_path, index=False)

    degraded, live_rmse = is_performance_degraded(
        history_path, baseline_rmse=None, degradation_threshold=0.2
    )

    assert degraded is False
    assert live_rmse is None


def test_is_performance_degraded_false_when_history_missing(tmp_path):
    history_path = tmp_path / "does_not_exist.csv"

    degraded, live_rmse = is_performance_degraded(
        history_path, baseline_rmse=50.0, degradation_threshold=0.2
    )

    assert degraded is False
    assert live_rmse is None


# ── promote_best_combination ──────────────────────────────────────────────────


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
        self.versions = versions
        self.production = production
        self.alias_calls = []  # list of (model_name, alias, version)

    def search_model_versions(self, filter_str):
        name = filter_str.split("'")[1]
        return [_FakeVersion(self.versions[name])]

    def get_model_version_by_alias(self, name, alias):
        return _FakeVersion(self.production[name])

    def set_registered_model_alias(self, name, alias, version):
        self.alias_calls.append((name, alias, version))


def test_promote_best_combination_bootstraps_when_no_production():
    """combos=None (no production alias exists yet) → promote both new."""
    project = "bike-sharing-forecast"
    client = _FakeMlflowClient(
        versions={f"{project}-registered": "3", f"{project}-casual": "3"},
        production={},
    )

    promotions = promote_best_combination(client, project, combos=None)

    assert promotions == {"registered": True, "casual": True}
    assert (f"{project}-registered", "production", "3") in client.alias_calls
    assert (f"{project}-casual", "production", "3") in client.alias_calls


def test_promote_best_combination_promotes_mixed_pair():
    """
    A mixed combination wins (new registered + current-production casual):
    only the registered alias moves; casual's production alias stays
    untouched, but its untested new version still gets archived.
    """
    project = "bike-sharing-forecast"
    client = _FakeMlflowClient(
        versions={f"{project}-registered": "5", f"{project}-casual": "5"},
        production={f"{project}-registered": "4", f"{project}-casual": "4"},
    )
    combos = {
        ("new", "new"): 30.0,
        ("new", "prod"): 0.0,  # winner: new registered + prod casual
        ("prod", "new"): 50.0,
        ("prod", "prod"): 20.0,
    }

    promotions = promote_best_combination(client, project, combos)

    assert promotions == {"registered": True, "casual": False}
    assert (f"{project}-registered", "production", "5") in client.alias_calls
    assert (f"{project}-registered", "archived", "4") in client.alias_calls
    assert (f"{project}-casual", "archived", "5") in client.alias_calls
    assert not any(
        name == f"{project}-casual" and alias == "production"
        for (name, alias, _) in client.alias_calls
    )


def test_promote_best_combination_keeps_production_when_it_wins():
    """
    Production pair beats every new/mixed combination → no production alias
    changes for either model; both untested new versions get archived.
    """
    project = "bike-sharing-forecast"
    client = _FakeMlflowClient(
        versions={f"{project}-registered": "5", f"{project}-casual": "5"},
        production={f"{project}-registered": "4", f"{project}-casual": "4"},
    )
    combos = {
        ("new", "new"): 25.0,
        ("new", "prod"): 22.0,
        ("prod", "new"): 30.0,
        ("prod", "prod"): 20.0,
    }

    promotions = promote_best_combination(client, project, combos)

    assert promotions == {"registered": False, "casual": False}
    assert (f"{project}-registered", "archived", "5") in client.alias_calls
    assert (f"{project}-casual", "archived", "5") in client.alias_calls
    assert not any(alias == "production" for (_, alias, _) in client.alias_calls)


# ── snapshot_drift_reference ──────────────────────────────────────────────────


class _FakeVersionWithRunId:
    def __init__(self, run_id):
        self.run_id = run_id


class _FakeClientForSnapshot:
    """
    Stub for get_model_version_by_alias + log_artifact. log_artifact reads
    the CSV immediately (rather than just recording the path) since the
    caller writes it inside a TemporaryDirectory that's gone by the time
    the test could otherwise inspect it.
    """

    def __init__(self, run_id="run123"):
        self.run_id = run_id
        self.logged_artifacts = []

    def get_model_version_by_alias(self, name, alias):
        return _FakeVersionWithRunId(self.run_id)

    def log_artifact(self, run_id, local_path, artifact_path=None):
        self.logged_artifacts.append((run_id, artifact_path, pd.read_csv(local_path)))


def test_snapshot_drift_reference_logs_drift_features_and_mnth(tmp_path):
    """
    The snapshot must contain exactly DRIFT_FEATURES + mnth — no more (other
    train.csv columns), no less (mnth is required for load_reference's
    month matching) — logged to the promoted model's own run.
    """
    train_csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {**{f: [1.0, 2.0] for f in DRIFT_FEATURES}, "mnth": [1, 2], "atemp": [0.5, 0.6]}
    ).to_csv(train_csv_path, index=False)

    client = _FakeClientForSnapshot(run_id="run123")

    snapshot_drift_reference(client, "bike-sharing-forecast", train_csv_path)

    assert len(client.logged_artifacts) == 1
    run_id, artifact_path, logged_df = client.logged_artifacts[0]
    assert run_id == "run123"
    assert artifact_path == "drift_reference"
    assert list(logged_df.columns) == DRIFT_FEATURES + ["mnth"]
