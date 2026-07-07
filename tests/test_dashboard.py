import pandas as pd
import pytest

from bike_sharing.dashboard.app import (
    normalize_retrain_outcome,
    compute_gauge_range,
    ensure_horizon,
    latest_trajectory,
    filter_to_horizon,
    latest_per_horizon,
)


# ── normalize_retrain_outcome ───────────────────────────────────────────────────


def test_normalize_retrain_outcome_maps_legacy_promoted_key():
    """
    Older retrain_outcome.json snapshots (pre backlog #10) use a single
    "promoted" bool instead of promoted_registered/promoted_casual — the
    dashboard must read both as if both models moved together, matching what
    that legacy schema actually meant.
    """
    legacy = {"retrain_attempted": True, "promoted": True}

    normalized = normalize_retrain_outcome(legacy)

    assert normalized["promoted_registered"] is True
    assert normalized["promoted_casual"] is True


def test_normalize_retrain_outcome_leaves_new_schema_untouched():
    """New-schema snapshots already have promoted_registered/casual — no rewrite."""
    new = {"retrain_attempted": True, "promoted_registered": True, "promoted_casual": False}

    normalized = normalize_retrain_outcome(new)

    assert normalized == new


# ── compute_gauge_range ─────────────────────────────────────────────────────────


def test_compute_gauge_range_scales_with_historical_max():
    """
    Axis max must clear the historical max (with headroom), not clip it like
    the old hardcoded [0, 900] did against a real max of 977.
    """
    actual_total = pd.Series([100.0, 500.0, 977.0, 300.0])

    axis_max, threshold = compute_gauge_range(actual_total)

    assert axis_max > 977.0
    assert axis_max == pytest.approx(977.0 * 1.1, abs=50)


def test_compute_gauge_range_threshold_is_90th_percentile():
    actual_total = pd.Series(range(1, 101), dtype=float)  # 1..100

    _, threshold = compute_gauge_range(actual_total)

    assert threshold == pytest.approx(actual_total.quantile(0.9))


def test_compute_gauge_range_falls_back_when_no_history():
    axis_max, threshold = compute_gauge_range(pd.Series([], dtype=float))

    assert axis_max == 900.0
    assert threshold == 700.0


# ── ensure_horizon ──────────────────────────────────────────────────────────────


def test_ensure_horizon_adds_column_for_legacy_predictions():
    """Predictions written before multi-horizon have no horizon column — they
    were all next-hour, so they read as horizon=1."""
    df = pd.DataFrame({"timestamp_predicted": pd.date_range("2026-01-01", periods=2, freq="h")})

    result = ensure_horizon(df)

    assert result["horizon"].tolist() == [1, 1]


def test_ensure_horizon_fills_nulls():
    df = pd.DataFrame({"horizon": [2, None, 3]})

    result = ensure_horizon(df)

    assert result["horizon"].tolist() == [2, 1, 3]


# ── latest_trajectory ───────────────────────────────────────────────────────────


def _multi_origin_predictions():
    """Two overlapping trajectories (K=3): an older origin at 10:00 and the
    latest at 11:00. Their targets overlap, which is exactly why the last row
    by timestamp is not the current forecast."""
    rows = []
    for origin_hr, base in [(10, 100), (11, 200)]:
        origin = pd.Timestamp(f"2026-01-01 {origin_hr}:00")
        for k in (1, 2, 3):
            rows.append(
                {
                    "timestamp_predicted": origin + pd.Timedelta(hours=k),
                    "horizon": k,
                    "pred_total": base + k,
                    "prediction_source": "model",
                }
            )
    return pd.DataFrame(rows).sort_values("timestamp_predicted").reset_index(drop=True)


def test_latest_trajectory_selects_most_recent_origin():
    """
    The trajectory must come from the newest origin (11:00), h+1..h+3 in order —
    not the rows with the newest timestamp_predicted (which belong to the older
    origin's far horizon).
    """
    predictions = _multi_origin_predictions()

    traj = latest_trajectory(predictions)

    assert traj["horizon"].tolist() == [1, 2, 3]
    assert traj["pred_total"].tolist() == [201, 202, 203]
    expected_targets = [
        pd.Timestamp("2026-01-01 12:00"),
        pd.Timestamp("2026-01-01 13:00"),
        pd.Timestamp("2026-01-01 14:00"),
    ]
    assert traj["timestamp_predicted"].tolist() == expected_targets


def test_latest_trajectory_empty_input():
    assert latest_trajectory(pd.DataFrame()).empty


# ── filter_to_horizon ───────────────────────────────────────────────────────────


def test_filter_to_horizon_keeps_only_that_lead_time():
    predictions = _multi_origin_predictions()

    h1 = filter_to_horizon(predictions, 1)

    assert (h1["horizon"] == 1).all()
    assert len(h1) == 2  # one per origin


def test_filter_to_horizon_treats_legacy_rows_as_h1():
    predictions = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=3, freq="h"),
            "pred_total": [10.0, 20.0, 30.0],
        }
    )

    h1 = filter_to_horizon(predictions, 1)

    assert len(h1) == 3


# ── latest_per_horizon ──────────────────────────────────────────────────────────


def test_latest_per_horizon_takes_last_row_of_each_horizon():
    """
    performance_history.csv accumulates one row per (run, horizon); the curve is
    the most recent measurement of each horizon.
    """
    history = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-08", "2026-01-08"]),
            "horizon": [1, 2, 1, 2],
            "rmse": [30.0, 40.0, 32.0, 45.0],
        }
    )

    curve = latest_per_horizon(history)

    assert curve["horizon"].tolist() == [1, 2]
    assert curve["rmse"].tolist() == [32.0, 45.0]  # the 2026-01-08 rows
