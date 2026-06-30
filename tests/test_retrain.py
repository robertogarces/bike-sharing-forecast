import pytest
import json
import tempfile
from pathlib import Path

import pandas as pd

from bike_sharing.models.retrain import (
    should_retrain,
    count_new_hours,
    write_retrain_marker,
)


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