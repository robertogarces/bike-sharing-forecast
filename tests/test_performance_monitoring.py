import tempfile
from pathlib import Path

import pandas as pd
import pytest

from bike_sharing.monitoring.performance_monitoring import (
    join_predictions_with_actuals,
    compute_rolling_performance,
    append_performance_record,
)


# ── join_predictions_with_actuals ───────────────────────────────────────────

def test_join_drops_predictions_without_a_known_actual():
    """
    Predictions for hours whose actual hasn't been revealed yet must be
    dropped — an inner join keeps only resolved (scoreable) predictions.
    """
    predictions = pd.DataFrame({
        "timestamp_predicted": pd.date_range("2026-01-01", periods=5, freq="h"),
        "pred_total": [100.0, 200.0, 150.0, 300.0, 250.0],
    })
    actuals = pd.DataFrame({
        "timestamp_predicted": pd.date_range("2026-01-01", periods=3, freq="h"),
        "actual_total": [90.0, 210.0, 140.0],
    })

    joined = join_predictions_with_actuals(predictions, actuals)

    assert len(joined) == 3
    assert joined["timestamp_predicted"].tolist() == predictions["timestamp_predicted"].tolist()[:3]


def test_join_keeps_all_predictions_when_all_actuals_known():
    predictions = pd.DataFrame({
        "timestamp_predicted": pd.date_range("2026-01-01", periods=3, freq="h"),
        "pred_total": [1.0, 2.0, 3.0],
    })
    actuals = pd.DataFrame({
        "timestamp_predicted": pd.date_range("2026-01-01", periods=3, freq="h"),
        "actual_total": [1.0, 2.0, 3.0],
    })

    joined = join_predictions_with_actuals(predictions, actuals)

    assert len(joined) == 3


# ── compute_rolling_performance ─────────────────────────────────────────────

def test_compute_rolling_performance_perfect_predictions_give_zero_error():
    """
    If pred_total exactly matches actual_total, rmse/mae/rmsle must be 0
    and r2 must be 1 — the cleanest way to verify the metric wiring without
    computing RMSE by hand.
    """
    joined = pd.DataFrame({
        "timestamp_predicted": pd.date_range("2026-01-01", periods=4, freq="h"),
        "pred_total":   [100.0, 200.0, 150.0, 300.0],
        "actual_total": [100.0, 200.0, 150.0, 300.0],
    })

    summary = compute_rolling_performance(joined, n_hours=168)

    assert summary["rmse"] == pytest.approx(0.0)
    assert summary["mae"] == pytest.approx(0.0)
    assert summary["rmsle"] == pytest.approx(0.0)
    assert summary["r2"] == pytest.approx(1.0)
    assert summary["n_resolved"] == 4
    assert summary["n_hours"] == 168


def test_compute_rolling_performance_windows_to_most_recent_n_hours():
    """
    Only the most recent n_hours resolved records should be used — older
    resolved predictions outside the window must not affect the metric.
    """
    joined = pd.DataFrame({
        "timestamp_predicted": pd.date_range("2026-01-01", periods=5, freq="h"),
        # First record is a huge miss; the rest are perfect.
        "pred_total":   [1000.0, 100.0, 200.0, 150.0, 300.0],
        "actual_total": [1.0,    100.0, 200.0, 150.0, 300.0],
    })

    summary = compute_rolling_performance(joined, n_hours=4)

    # Window excludes the first (huge-error) row → error should be exactly 0.
    assert summary["rmse"] == pytest.approx(0.0)
    assert summary["n_resolved"] == 4


# ── append_performance_record ───────────────────────────────────────────────

def test_append_performance_record_creates_file_with_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "performance_history.csv"
        summary = {
            "timestamp": "2026-01-01T00:00:00", "n_hours": 168, "n_resolved": 10,
            "rmse": 50.0, "rmsle": 0.2, "r2": 0.9, "mae": 40.0,
        }

        append_performance_record(summary, history_path)

        assert history_path.exists()
        df = pd.read_csv(history_path)
        assert len(df) == 1
        assert df.iloc[0]["rmse"] == 50.0


def test_append_performance_record_always_appends_never_dedupes():
    """
    Unlike append_prediction (predictions.csv), repeated calls with the same
    summary must each add a new row — every monitoring run is its own
    observation, even if identical to the last.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "performance_history.csv"
        summary = {
            "timestamp": "2026-01-01T00:00:00", "n_hours": 168, "n_resolved": 10,
            "rmse": 50.0, "rmsle": 0.2, "r2": 0.9, "mae": 40.0,
        }

        append_performance_record(summary, history_path)
        append_performance_record(summary, history_path)
        append_performance_record(summary, history_path)

        df = pd.read_csv(history_path)
        assert len(df) == 3
