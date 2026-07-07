import pandas as pd

from bike_sharing.dashboard.app import (
    normalize_retrain_outcome,
    ensure_horizon,
    latest_trajectory,
    filter_to_horizon,
    latest_per_horizon,
    live_model_metrics,
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


# ── live_model_metrics ──────────────────────────────────────────────────────────


def test_live_model_metrics_returns_combined_and_per_model():
    """
    Combined + per-model metrics computed from h+1 predictions vs actuals.
    Perfect predictions → 0 error everywhere, confirming each model reads its
    own prediction/actual columns.
    """
    ts = pd.date_range("2026-01-01", periods=3, freq="h")
    predictions = pd.DataFrame(
        {
            "timestamp_predicted": ts,
            "horizon": 1,
            "pred_total": [100.0, 200.0, 150.0],
            "pred_registered": [80.0, 170.0, 120.0],
            "pred_casual": [20.0, 30.0, 30.0],
            "prediction_source": "model",
        }
    )
    actuals = pd.DataFrame(
        {
            "timestamp_predicted": ts,
            "actual_total": [100.0, 200.0, 150.0],
            "actual_registered": [80.0, 170.0, 120.0],
            "actual_casual": [20.0, 30.0, 30.0],
        }
    )

    out = live_model_metrics(predictions, actuals, n_hours=168)

    assert set(out) == {"Combined", "Registered", "Casual"}
    assert out["Combined"]["rmse"] == 0.0
    assert out["Registered"]["rmse"] == 0.0
    assert out["Casual"]["rmse"] == 0.0


def test_live_model_metrics_excludes_fallback_and_other_horizons():
    """Only resolved h+1 model rows are scored — fallback rows and h+2 rows
    must be dropped before computing."""
    ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"])
    predictions = pd.DataFrame(
        {
            "timestamp_predicted": [ts[0], ts[0], ts[1]],
            "horizon": [1, 2, 1],
            "pred_total": [100.0, 999.0, 200.0],
            "pred_registered": [80.0, 900.0, 170.0],
            "pred_casual": [20.0, 99.0, 30.0],
            "prediction_source": ["model", "model", "fallback_lag168"],
        }
    )
    actuals = pd.DataFrame(
        {
            "timestamp_predicted": ts,
            "actual_total": [100.0, 200.0],
            "actual_registered": [80.0, 170.0],
            "actual_casual": [20.0, 30.0],
        }
    )

    out = live_model_metrics(predictions, actuals, n_hours=168)

    # Only the single h+1 model row at 00:00 survives (01:00 is fallback) → rmse 0.
    assert out["Combined"]["rmse"] == 0.0


def test_live_model_metrics_none_when_no_resolved_rows():
    assert live_model_metrics(pd.DataFrame(), pd.DataFrame(), n_hours=168) is None
