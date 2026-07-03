import pandas as pd

from bike_sharing.monitoring.output_drift_detection import split_rolling_windows


def test_split_rolling_windows_splits_into_current_and_reference():
    """
    current = the most recent n_hours rows; reference = the n_hours
    immediately before that — two disjoint, equal-sized, adjacent windows.
    """
    predictions = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=20, freq="h"),
            "pred_total": range(20),
        }
    )

    reference, current = split_rolling_windows(predictions, n_hours=8)

    assert len(reference) == 8
    assert len(current) == 8
    assert reference["timestamp_predicted"].max() < current["timestamp_predicted"].min()
    assert current["pred_total"].tolist() == list(range(12, 20))
    assert reference["pred_total"].tolist() == list(range(4, 12))


def test_split_rolling_windows_returns_none_when_not_enough_history():
    """
    Fewer than 2*n_hours total rows — not enough for two full windows yet.
    """
    predictions = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=10, freq="h"),
            "pred_total": range(10),
        }
    )

    reference, current = split_rolling_windows(predictions, n_hours=8)

    assert reference is None
    assert current is None


def test_split_rolling_windows_exactly_at_threshold_returns_windows():
    """
    Exactly 2*n_hours rows is enough — the boundary is inclusive.
    """
    predictions = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=16, freq="h"),
            "pred_total": range(16),
        }
    )

    reference, current = split_rolling_windows(predictions, n_hours=8)

    assert reference is not None
    assert len(reference) == 8
    assert len(current) == 8
