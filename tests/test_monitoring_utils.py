import tempfile
from pathlib import Path

import pandas as pd

from bike_sharing.utils.monitoring_utils import append_monitoring_record


def test_append_monitoring_record_creates_file_with_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.csv"
        summary = {
            "timestamp": "2026-01-01T00:00:00",
            "n_hours": 168,
            "n_resolved": 10,
            "rmse": 50.0,
            "rmsle": 0.2,
            "r2": 0.9,
            "mae": 40.0,
        }

        append_monitoring_record(summary, history_path)

        assert history_path.exists()
        df = pd.read_csv(history_path)
        assert len(df) == 1
        assert df.iloc[0]["rmse"] == 50.0


def test_append_monitoring_record_always_appends_never_dedupes():
    """
    Unlike append_prediction (predictions.csv), repeated calls with the same
    summary must each add a new row — every monitoring run is its own
    observation, even if identical to the last.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.csv"
        summary = {
            "timestamp": "2026-01-01T00:00:00",
            "n_hours": 168,
            "n_resolved": 10,
            "rmse": 50.0,
            "rmsle": 0.2,
            "r2": 0.9,
            "mae": 40.0,
        }

        append_monitoring_record(summary, history_path)
        append_monitoring_record(summary, history_path)
        append_monitoring_record(summary, history_path)

        df = pd.read_csv(history_path)
        assert len(df) == 3
