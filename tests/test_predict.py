import json

import pytest
import numpy as np
import pandas as pd

from bike_sharing.models.predict import (
    build_next_hour_features,
    build_synthetic_row,
    predict_trajectory,
    build_calendar_lookup,
    build_weather_lookup,
    get_missing_origins,
    append_prediction,
    get_fallback_prediction,
)


LAGS = [1, 2, 3, 8, 24, 48, 72, 168]
ROLLING_WINDOWS = [24, 168]
DROP_COLS = ["atemp", "yr"]


def _make_past(n: int = 200):
    """Deterministic past dataset — unique cnt per hour makes every hour's
    lag features unique, so tests can detect exactly which hour fed which
    prediction."""
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    past = pd.DataFrame(
        {
            "instant": range(1, n + 1),
            "dteday": dates.date,
            "season": 1,
            "yr": 0,
            "mnth": dates.month,
            "hr": dates.hour,
            "holiday": 0,
            "weekday": dates.dayofweek,
            "workingday": (dates.dayofweek < 5).astype(int),
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.5,
            "hum": 0.5,
            "windspeed": 0.2,
            "casual": 10,
            "registered": 90,
            "cnt": np.arange(1, n + 1),  # unique per hour → unique cnt_lag_1
        }
    )
    return dates, past


# ── build_next_hour_features ────────────────────────────────────────────────


def test_build_next_hour_features_returns_one_row(sample_df_with_datetime, min_date):
    """
    build_next_hour_features should always return exactly one row —
    the feature vector for the next hour prediction.
    """
    result = build_next_hour_features(
        sample_df_with_datetime, min_date, LAGS, ROLLING_WINDOWS, DROP_COLS
    )

    assert len(result) == 1


def test_build_next_hour_features_correct_hour(sample_df_with_datetime, min_date):
    """
    The predicted hour should be exactly one hour after the last record
    in the past dataset.
    """
    last_dt = sample_df_with_datetime["datetime"].iloc[-1]
    result = build_next_hour_features(
        sample_df_with_datetime, min_date, LAGS, ROLLING_WINDOWS, DROP_COLS
    )

    expected_next_dt = last_dt + pd.Timedelta(hours=1)
    actual_next_dt = pd.Timestamp(result["datetime"].values[0])

    assert actual_next_dt == expected_next_dt


def test_build_next_hour_features_lag1_equals_current_cnt(sample_df_with_datetime, min_date):
    """
    cnt_lag_1 of the next hour should equal cnt of the current (last) hour.
    This is the core lag shift — the most recent demand becomes lag_1.
    """
    current_cnt = sample_df_with_datetime["cnt"].iloc[-1]
    result = build_next_hour_features(
        sample_df_with_datetime, min_date, LAGS, ROLLING_WINDOWS, DROP_COLS
    )

    assert result["cnt_lag_1"].values[0] == current_cnt


# ── build_synthetic_row ──────────────────────────────────────────────────────


@pytest.fixture
def current_row():
    return pd.Series(
        {
            "datetime": pd.Timestamp("2024-06-07 23:00"),  # a Friday
            "dteday": pd.Timestamp("2024-06-07"),
            "hr": 23,
            "season": 2,
            "yr": 0,
            "mnth": 6,
            "holiday": 0,
            "weekday": 5,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.3,
            "atemp": 0.3,
            "hum": 0.4,
            "windspeed": 0.1,
            "cnt": 50,
            "casual": 10,
            "registered": 40,
        }
    )


def test_build_synthetic_row_without_lookups_inherits_from_current(current_row):
    """With no lookups given, calendar and weather carry the origin's values
    forward unchanged — the historical single-step behavior."""
    target_dt = current_row["datetime"] + pd.Timedelta(hours=1)
    row = build_synthetic_row(current_row, target_dt)

    assert row["workingday"] == current_row["workingday"]
    assert row["temp"] == current_row["temp"]
    assert row["datetime"] == target_dt
    assert row["hr"] == target_dt.hour
    assert np.isnan(row["cnt"]) and np.isnan(row["registered"]) and np.isnan(row["casual"])


def test_build_synthetic_row_uses_true_calendar_across_midnight(current_row):
    """
    A rollout from Friday 23:00 into Saturday 00:00 must pick up Saturday's
    real workingday (0), not keep Friday's (1) — the stale-calendar bug this
    lookup fixes.
    """
    target_dt = current_row["datetime"] + pd.Timedelta(hours=1)  # Saturday 00:00
    calendar_lookup = pd.DataFrame(
        {"season": [2], "yr": [0], "mnth": [6], "holiday": [0], "weekday": [6], "workingday": [0]},
        index=pd.DatetimeIndex([target_dt]),
    )

    row = build_synthetic_row(current_row, target_dt, calendar_lookup=calendar_lookup)

    assert row["workingday"] == 0
    assert row["weekday"] == 6


def test_build_synthetic_row_falls_back_when_calendar_not_covered(current_row):
    """A target hour outside the calendar lookup's range keeps the origin's
    calendar — the "future exhausted" fallback."""
    target_dt = current_row["datetime"] + pd.Timedelta(hours=1)
    calendar_lookup = pd.DataFrame(
        {"season": [], "yr": [], "mnth": [], "holiday": [], "weekday": [], "workingday": []},
        index=pd.DatetimeIndex([]),
    )

    row = build_synthetic_row(current_row, target_dt, calendar_lookup=calendar_lookup)

    assert row["workingday"] == current_row["workingday"]


def test_build_synthetic_row_uses_lag24_weather(current_row):
    """Weather is approximated from target-24h (same hour, previous day),
    not carried over from the origin."""
    target_dt = current_row["datetime"] + pd.Timedelta(hours=1)
    lag24_dt = target_dt - pd.Timedelta(hours=24)
    weather_lookup = pd.DataFrame(
        {"temp": [0.9], "atemp": [0.9], "hum": [0.9], "windspeed": [0.9]},
        index=pd.DatetimeIndex([lag24_dt]),
    )

    row = build_synthetic_row(current_row, target_dt, weather_lookup=weather_lookup)

    assert row["temp"] == 0.9
    assert row["hum"] == 0.9


def test_build_synthetic_row_falls_back_to_origin_weather_when_lag24_missing(current_row):
    """Beyond the lag24 lookup's coverage (e.g. horizons > 24h), weather
    falls back to persisting the origin's own weather."""
    target_dt = current_row["datetime"] + pd.Timedelta(hours=1)
    weather_lookup = pd.DataFrame(
        {"temp": [], "atemp": [], "hum": [], "windspeed": []}, index=pd.DatetimeIndex([])
    )

    row = build_synthetic_row(current_row, target_dt, weather_lookup=weather_lookup)

    assert row["temp"] == current_row["temp"]


# ── build_calendar_lookup / build_weather_lookup ────────────────────────────


def test_build_calendar_lookup_covers_past_and_future():
    dates, past = _make_past(10)
    past = past.assign(datetime=dates)
    future = past.copy()
    future["datetime"] = future["datetime"] + pd.Timedelta(hours=10)

    lookup = build_calendar_lookup(past, future)

    assert past["datetime"].iloc[0] in lookup.index
    assert future["datetime"].iloc[0] in lookup.index


def test_build_weather_lookup_only_covers_past():
    dates, past = _make_past(10)
    past = past.assign(datetime=dates)

    lookup = build_weather_lookup(past)

    assert list(lookup.columns) == ["temp", "atemp", "hum", "windspeed"]
    assert len(lookup) == 10


# ── predict_trajectory ───────────────────────────────────────────────────────


class _IncrementingModel:
    """registered = cnt_lag_1 + 1 — deterministic, so a K-step rollout
    produces a strictly increasing sequence only if each step's prediction
    is actually fed forward as the next step's cnt_lag_1."""

    def predict(self, X):
        return np.log1p(X["cnt_lag_1"].values + 1)


class _ZeroModel:
    def predict(self, X):
        return np.zeros(len(X))


def _lookups(past):
    from bike_sharing.utils.datetime_utils import reconstruct_datetime

    p = reconstruct_datetime(past.copy())
    return build_calendar_lookup(p, p.iloc[0:0]), build_weather_lookup(p)


def test_predict_trajectory_returns_one_row_per_horizon():
    _, past = _make_past(200)
    calendar_lookup, weather_lookup = _lookups(past)

    trajectory = predict_trajectory(
        past,
        _IncrementingModel(),
        _ZeroModel(),
        5,
        pd.to_datetime(past["dteday"]).min(),
        LAGS,
        ROLLING_WINDOWS,
        DROP_COLS,
        calendar_lookup,
        weather_lookup,
    )

    assert len(trajectory) == 5
    assert trajectory["horizon"].tolist() == [1, 2, 3, 4, 5]


def test_predict_trajectory_feeds_predictions_forward():
    """
    Regression test for the core recursive mechanism: each step's prediction
    must become the next step's cnt_lag_1. With a model that always predicts
    cnt_lag_1 + 1, a rollout that actually feeds forward produces a strictly
    increasing sequence starting at last_cnt + 1; one that (bug) keeps
    re-reading the same origin lag would instead repeat the same value.
    """
    n = 200
    _, past = _make_past(n)
    calendar_lookup, weather_lookup = _lookups(past)
    last_cnt = past["cnt"].iloc[-1]

    trajectory = predict_trajectory(
        past,
        _IncrementingModel(),
        _ZeroModel(),
        4,
        pd.to_datetime(past["dteday"]).min(),
        LAGS,
        ROLLING_WINDOWS,
        DROP_COLS,
        calendar_lookup,
        weather_lookup,
    )

    assert trajectory["pred_registered"].tolist() == pytest.approx(
        [last_cnt + 1, last_cnt + 2, last_cnt + 3, last_cnt + 4]
    )
    assert trajectory["pred_total"].tolist() == pytest.approx(
        trajectory["pred_registered"].tolist()
    )


def test_predict_trajectory_targets_are_consecutive_hours():
    _, past = _make_past(200)
    calendar_lookup, weather_lookup = _lookups(past)

    trajectory = predict_trajectory(
        past,
        _IncrementingModel(),
        _ZeroModel(),
        3,
        pd.to_datetime(past["dteday"]).min(),
        LAGS,
        ROLLING_WINDOWS,
        DROP_COLS,
        calendar_lookup,
        weather_lookup,
    )

    last_dt = pd.to_datetime(past["dteday"].iloc[-1]) + pd.Timedelta(hours=int(past["hr"].iloc[-1]))
    expected = [last_dt + pd.Timedelta(hours=k) for k in (1, 2, 3)]
    assert trajectory["timestamp_predicted"].tolist() == expected


def test_predict_trajectory_horizon_maps_to_clock_hour():
    """
    Explicit per-row guard on the (horizon -> clock hour) pairing, not just
    row order: from an origin at 10:00, the horizon=k row must carry the
    timestamp origin + k hours (h+1 -> 11:00, h+3 -> 13:00, h+4 -> 14:00, ...).
    """
    # Past ending exactly at 10:00 so the scenario reads literally
    # (2024-01-01 00:00 + 202h = 2024-01-09 10:00).
    n = 203
    dates, past = _make_past(n)
    origin = dates[-1]
    assert origin.hour == 10

    calendar_lookup, weather_lookup = _lookups(past)
    trajectory = predict_trajectory(
        past,
        _IncrementingModel(),
        _ZeroModel(),
        6,
        pd.to_datetime(past["dteday"]).min(),
        LAGS,
        ROLLING_WINDOWS,
        DROP_COLS,
        calendar_lookup,
        weather_lookup,
    )

    for horizon, ts in zip(trajectory["horizon"], trajectory["timestamp_predicted"]):
        assert ts == origin + pd.Timedelta(hours=int(horizon))
        assert ts.hour == (origin.hour + int(horizon)) % 24


def test_predict_trajectory_uses_true_calendar_across_day_boundary():
    """A rollout must reflect the real calendar of each future hour, read from
    the lookup, not the origin's — this is what makes multi-hour trajectories
    safe to serve across day boundaries."""
    n = 200
    _, past = _make_past(n)

    calendar_lookup, weather_lookup = _lookups(past)
    origin_dt = pd.to_datetime(past["dteday"].iloc[-1]) + pd.Timedelta(
        hours=int(past["hr"].iloc[-1])
    )
    # Override workingday for the two synthetic future hours to a sentinel value.
    overrides = pd.DataFrame(
        {
            "season": [1, 1],
            "yr": [0, 0],
            "mnth": [1, 1],
            "holiday": [0, 0],
            "weekday": [9, 9],
            "workingday": [9, 9],
        },
        index=pd.DatetimeIndex(
            [origin_dt + pd.Timedelta(hours=1), origin_dt + pd.Timedelta(hours=2)]
        ),
    )
    calendar_lookup = pd.concat([calendar_lookup, overrides])

    trajectory = predict_trajectory(
        past,
        _IncrementingModel(),
        _ZeroModel(),
        2,
        pd.to_datetime(past["dteday"]).min(),
        LAGS,
        ROLLING_WINDOWS,
        DROP_COLS,
        calendar_lookup,
        weather_lookup,
    )

    assert trajectory["workingday"].tolist() == [9, 9]


# ── get_missing_origins ──────────────────────────────────────────────────────


@pytest.fixture
def past_df():
    """Past dataset covering 10 consecutive hours."""
    base = pd.Timestamp("2024-06-01 00:00")
    rows = []
    for i in range(10):
        dt = base + pd.Timedelta(hours=i)
        rows.append({"dteday": dt.date(), "hr": dt.hour, "cnt": 100 + i})
    return pd.DataFrame(rows)


def test_get_missing_origins_no_predictions_file(past_df, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    result = get_missing_origins(past_df, pred_path)
    assert result == []


def test_get_missing_origins_empty_predictions_file(past_df, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(columns=["timestamp_predicted", "horizon"]).to_csv(pred_path, index=False)
    result = get_missing_origins(past_df, pred_path)
    assert result == []


def test_get_missing_origins_detects_gap(past_df, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    # Only origin hour 0's h+1 (target hour 1) was recorded — origins 1..8 are
    # missing (origin 9 has no next hour in past_df, so it's not a candidate).
    pd.DataFrame([{"timestamp_predicted": "2024-06-01 01:00:00", "horizon": 1}]).to_csv(
        pred_path, index=False
    )
    result = get_missing_origins(past_df, pred_path)
    assert len(result) == 8


def test_get_missing_origins_no_gap(past_df, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    # h+1 recorded for every origin 0..8 (targets 1..9) — nothing missing.
    timestamps = [
        {
            "timestamp_predicted": (pd.Timestamp("2024-06-01") + pd.Timedelta(hours=i)).isoformat(),
            "horizon": 1,
        }
        for i in range(1, 10)
    ]
    pd.DataFrame(timestamps).to_csv(pred_path, index=False)
    result = get_missing_origins(past_df, pred_path)
    assert result == []


def test_get_missing_origins_respects_cap(past_df, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame([{"timestamp_predicted": "2024-06-01 01:00:00", "horizon": 1}]).to_csv(
        pred_path, index=False
    )
    result = get_missing_origins(past_df, pred_path, max_backfill_hours=3)
    assert len(result) == 3


def test_get_missing_origins_ignores_non_primary_horizon_rows(past_df, tmp_path):
    """
    An origin is only "resolved" by its horizon=1 row — h+2/h+3 rows for the
    same target existing (from some other origin's trajectory) must not count
    as evidence that this origin's own rollout ran. With no horizon=1 rows at
    all, the function has no anchor and returns early.
    """
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"timestamp_predicted": "2024-06-01 01:00:00", "horizon": 2},
            {"timestamp_predicted": "2024-06-01 01:00:00", "horizon": 3},
        ]
    ).to_csv(pred_path, index=False)
    result = get_missing_origins(past_df, pred_path)
    assert result == []


def test_get_missing_origins_treats_missing_horizon_column_as_h1(past_df, tmp_path):
    """Legacy predictions.csv (pre-multi-horizon) has no horizon column —
    every row is treated as horizon=1."""
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame([{"timestamp_predicted": "2024-06-01 01:00:00"}]).to_csv(pred_path, index=False)
    result = get_missing_origins(past_df, pred_path)
    assert len(result) == 8


# ── append_prediction ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_pred():
    return {
        "predicted_at": "2024-06-01T10:00:00",
        "timestamp_predicted": "2024-06-01T11:00:00",
        "horizon": 1,
        "hr": 11,
        "temp": 0.5,
        "hum": 0.6,
        "weathersit": 1,
        "workingday": 1,
        "pred_registered": 120.0,
        "pred_casual": 30.0,
        "pred_total": 150.0,
    }


def test_append_prediction_creates_file(sample_pred, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    append_prediction(sample_pred, pred_path)
    assert pred_path.exists()
    df = pd.read_csv(pred_path)
    assert len(df) == 1


def test_append_prediction_appends_new(sample_pred, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    append_prediction(sample_pred, pred_path)

    second = sample_pred.copy()
    second["timestamp_predicted"] = "2024-06-01T12:00:00"
    append_prediction(second, pred_path)

    df = pd.read_csv(pred_path)
    assert len(df) == 2


def test_append_prediction_skips_duplicate(sample_pred, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    append_prediction(sample_pred, pred_path)
    append_prediction(sample_pred, pred_path)  # duplicate

    df = pd.read_csv(pred_path)
    assert len(df) == 1


def test_append_prediction_same_timestamp_different_horizon_both_kept(sample_pred, tmp_path):
    """The dedup key is (timestamp_predicted, horizon) — the same target hour
    predicted at two different lead times is not a duplicate."""
    pred_path = tmp_path / "predictions.csv"
    append_prediction(sample_pred, pred_path)

    second = sample_pred.copy()
    second["horizon"] = 2
    append_prediction(second, pred_path)

    df = pd.read_csv(pred_path)
    assert len(df) == 2
    assert sorted(df["horizon"].tolist()) == [1, 2]


def test_append_prediction_defaults_missing_horizon_to_1(tmp_path):
    pred_path = tmp_path / "predictions.csv"
    pred_row = {"timestamp_predicted": "2024-06-01T11:00:00", "pred_total": 10.0}

    append_prediction(pred_row, pred_path)

    df = pd.read_csv(pred_path)
    assert df.iloc[0]["horizon"] == 1


def test_append_prediction_normalizes_legacy_rows_missing_horizon_column(sample_pred, tmp_path):
    """A pre-multi-horizon predictions.csv has no horizon column at all; a new
    row for the same timestamp must be recognized as a duplicate against those
    legacy (implicitly horizon=1) rows."""
    pred_path = tmp_path / "predictions.csv"
    legacy = {k: v for k, v in sample_pred.items() if k != "horizon"}
    pd.DataFrame([legacy]).to_csv(pred_path, index=False)

    append_prediction(sample_pred, pred_path)  # same timestamp, horizon=1

    df = pd.read_csv(pred_path)
    assert len(df) == 1


def test_append_prediction_handles_new_column_without_corrupting_old_rows(sample_pred, tmp_path):
    """
    Schema evolution: if a later prediction includes a column the existing
    file doesn't have (e.g. model_version_registered added after go-live),
    pd.concat must align by name — old rows get NaN for the new column,
    not a corrupted/misaligned CSV.
    """
    pred_path = tmp_path / "predictions.csv"
    append_prediction(sample_pred, pred_path)  # old schema, no model_version

    second = sample_pred.copy()
    second["timestamp_predicted"] = "2024-06-01T12:00:00"
    second["model_version_registered"] = "7"
    append_prediction(second, pred_path)

    df = pd.read_csv(pred_path)
    assert len(df) == 2
    assert pd.isna(df.iloc[0]["model_version_registered"])
    assert df.iloc[1]["model_version_registered"] == 7


# ── run(): stubs shared by the integration tests ──────────────────────────────


class _FakeModel:
    """
    Model stub whose prediction is a deterministic function of the target hour.

    Returns log1p(cnt_lag_1); after predict.py applies expm1, the prediction
    equals cnt_lag_1 — the cnt of the hour right before the target. With a
    unique cnt per hour this makes every hour's prediction unique, so we can
    detect whether the main prediction used the right hour's features.
    """

    def predict(self, X):
        return np.log1p(X["cnt_lag_1"].values)


class _FakeVersion:
    def __init__(self, version):
        self.version = version


class _FakeMlflowClient:
    """
    Stub for MlflowClient.get_model_version_by_alias — avoids the test
    depending on whatever MLflow registry state happens to be reachable
    (e.g. a local mlruns/ store from a previous manual run).
    """

    def get_model_version_by_alias(self, name, alias):
        return _FakeVersion("1")


def _run_cfg(raw_dir, pred_path, state_path, artifacts_dir, horizon=1):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "project": "bike-sharing-forecast",
            "paths": {
                "raw_dir": str(raw_dir),
                "input_file": "hour_past.csv",
                "simulation_state": str(state_path),
                "predictions_path": str(pred_path),
                "artifacts_dir": str(artifacts_dir),
            },
            "monitoring": {"max_backfill_hours": 48},
            "forecast": {"horizon": horizon, "primary_horizon": 1},
            "features": {
                "lags": [1, 2, 3, 8, 24, 48, 72, 168],
                "rolling_windows": [24, 168],
                "drop_cols": ["atemp", "yr"],
            },
        }
    )


# ── run(): backfill must not corrupt the main (next-hour) prediction ──────────


def test_run_main_prediction_uses_next_hour_not_backfill(monkeypatch, tmp_path):
    """
    Regression test for the backfill feature-reuse bug.

    When backfill runs, the final next-hour prediction must be computed from
    next_dt's own features — not the last backfilled hour's. horizon=1 keeps
    each origin's trajectory to a single row, matching the pre-multi-horizon
    shape of this check.
    """
    import mlflow.lightgbm
    from bike_sharing.models import predict as predict_mod

    n = 200
    dates, past = _make_past(n)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    past.to_csv(raw_dir / "hour_past.csv", index=False)

    # h+1 recorded for every origin except the last three → 3-origin backfill.
    pred_path = tmp_path / "predictions.csv"
    seeded = [dates[i] for i in range(1, n - 3)]  # targets = origin + 1h
    pd.DataFrame(
        {
            "predicted_at": [d.isoformat() for d in seeded],
            "timestamp_predicted": [d.isoformat() for d in seeded],
            "horizon": 1,
            "hr": [d.hour for d in seeded],
            "temp": 0.5,
            "hum": 0.5,
            "weathersit": 1,
            "workingday": 1,
            "pred_registered": 1.0,
            "pred_casual": 1.0,
            "pred_total": 2.0,
        }
    ).to_csv(pred_path, index=False)

    state_path = tmp_path / "simulation_state.json"
    state_path.write_text("{}")

    cfg = _run_cfg(raw_dir, pred_path, state_path, tmp_path / "artifacts", horizon=1)

    monkeypatch.setattr(mlflow.lightgbm, "load_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(predict_mod, "MlflowClient", lambda: _FakeMlflowClient())

    predict_mod.run(cfg)

    out = pd.read_csv(pred_path)
    out["ts"] = pd.to_datetime(out["timestamp_predicted"])

    # 1. No row may have an hr that disagrees with its own timestamp
    assert (out["ts"].dt.hour == out["hr"]).all()

    # 2. Every row is horizon=1 (K=1 here)
    assert (out["horizon"] == 1).all()

    # 3. The run wrote the 3 missing origins (backfill) + next_dt (main)
    next_dt = dates[n - 1] + pd.Timedelta(hours=1)
    new = out[out["ts"] > dates[n - 4]].sort_values("ts")
    assert new["ts"].tolist() == [dates[n - 3], dates[n - 2], dates[n - 1], next_dt]

    # 4. The main prediction used next_dt's own lag_1 = last past cnt = n,
    #    not the last backfilled hour's (n - 1) → it is not a duplicate.
    main_row = out[out["ts"] == next_dt].iloc[0]
    assert main_row["hr"] == next_dt.hour
    assert round(main_row["pred_registered"]) == n


def test_run_serves_full_trajectory_with_horizon_greater_than_one(monkeypatch, tmp_path):
    """End-to-end: run() emits K rows (one per horizon) for a single origin
    when forecast.horizon > 1, with the right target timestamps."""
    import mlflow.lightgbm
    from bike_sharing.models import predict as predict_mod

    n = 200
    dates, past = _make_past(n)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    past.to_csv(raw_dir / "hour_past.csv", index=False)

    state_path = tmp_path / "simulation_state.json"
    state_path.write_text("{}")
    pred_path = tmp_path / "predictions.csv"

    cfg = _run_cfg(raw_dir, pred_path, state_path, tmp_path / "artifacts", horizon=6)

    monkeypatch.setattr(mlflow.lightgbm, "load_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(predict_mod, "MlflowClient", lambda: _FakeMlflowClient())

    predict_mod.run(cfg)

    out = pd.read_csv(pred_path)
    assert len(out) == 6
    assert sorted(out["horizon"].tolist()) == [1, 2, 3, 4, 5, 6]

    next_dt = dates[n - 1] + pd.Timedelta(hours=1)
    expected_targets = {(next_dt + pd.Timedelta(hours=k - 1)).isoformat() for k in range(1, 7)}
    assert set(out["timestamp_predicted"]) == expected_targets


# ── get_fallback_prediction ────────────────────────────────────────────────────


def test_get_fallback_prediction_returns_lag168_row():
    """
    The fallback must reuse exactly the same seasonality the model's own
    cnt_lag_168 feature relies on: the actual values from 168h before the
    target hour.
    """
    dates, past = _make_past(200)
    target_dt = dates[-1] + pd.Timedelta(hours=1)

    fallback = get_fallback_prediction(past, target_dt)

    lookback_dt = target_dt - pd.Timedelta(hours=168)
    expected = past[
        pd.to_datetime(past["dteday"]) + pd.to_timedelta(past["hr"], unit="h") == lookback_dt
    ].iloc[0]

    assert fallback == {
        "pred_registered": float(expected["registered"]),
        "pred_casual": float(expected["casual"]),
        "pred_total": float(expected["cnt"]),
    }


def test_get_fallback_prediction_returns_none_when_insufficient_history():
    """
    Less than a week of simulated history — there is no 168h-ago row to fall
    back to yet, so the caller must skip the hour instead of using stale/wrong
    data.
    """
    dates, past = _make_past(50)
    target_dt = dates[-1] + pd.Timedelta(hours=1)

    assert get_fallback_prediction(past, target_dt) is None


# ── run(): fallback when the hourly validation flag is invalid ────────────────


def _write_invalid_flag(artifacts_dir):
    flag_path = artifacts_dir / "validation" / "hourly_validation.json"
    flag_path.parent.mkdir(parents=True)
    flag_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00",
                "n_rows_checked": 1,
                "valid": False,
                "issues": ["1 row(s) with 'hum' outside [0.0, 1.0]"],
            }
        )
    )


def test_run_uses_fallback_trajectory_when_validation_flag_invalid(monkeypatch, tmp_path):
    """
    When the hourly validation flag marks the newly revealed data invalid,
    run() must not call the real model — it serves a fallback trajectory of
    168h-ago actuals (one row per horizon) and tags each so downstream
    monitoring can exclude them.
    """
    import mlflow.lightgbm
    from bike_sharing.models import predict as predict_mod

    n = 200
    dates, past = _make_past(n)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    past.to_csv(raw_dir / "hour_past.csv", index=False)

    pred_path = tmp_path / "predictions.csv"
    state_path = tmp_path / "simulation_state.json"
    state_path.write_text("{}")
    artifacts_dir = tmp_path / "artifacts"
    _write_invalid_flag(artifacts_dir)

    cfg = _run_cfg(raw_dir, pred_path, state_path, artifacts_dir, horizon=3)

    monkeypatch.setattr(mlflow.lightgbm, "load_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(predict_mod, "MlflowClient", lambda: _FakeMlflowClient())

    predict_mod.run(cfg)

    out = pd.read_csv(pred_path)
    assert len(out) == 3
    assert sorted(out["horizon"].tolist()) == [1, 2, 3]
    assert (out["prediction_source"] == "fallback_lag168").all()
    assert out["model_version_registered"].isna().all()

    # h+1 fallback = the actuals from 168h before the h+1 target.
    current_dt = dates[n - 1]
    row_h1 = out[out["horizon"] == 1].iloc[0]
    lookback = (current_dt + pd.Timedelta(hours=1)) - pd.Timedelta(hours=168)
    expected = past[
        pd.to_datetime(past["dteday"]) + pd.to_timedelta(past["hr"], unit="h") == lookback
    ].iloc[0]
    assert row_h1["pred_total"] == round(float(expected["cnt"]), 2)


def test_run_skips_horizons_without_lag168_when_invalid(monkeypatch, tmp_path):
    """
    If the flag is invalid AND there isn't 168h of history yet, run() must not
    write anything — no horizon has a 168h-ago row to fall back to, so the
    file is never created; backfill fills the gap once trustworthy data
    returns.
    """
    import mlflow.lightgbm
    from bike_sharing.models import predict as predict_mod

    n = 50  # less than 168h of history
    dates, past = _make_past(n)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    past.to_csv(raw_dir / "hour_past.csv", index=False)

    pred_path = tmp_path / "predictions.csv"
    state_path = tmp_path / "simulation_state.json"
    state_path.write_text("{}")
    artifacts_dir = tmp_path / "artifacts"
    _write_invalid_flag(artifacts_dir)

    cfg = _run_cfg(raw_dir, pred_path, state_path, artifacts_dir, horizon=3)

    monkeypatch.setattr(mlflow.lightgbm, "load_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(predict_mod, "MlflowClient", lambda: _FakeMlflowClient())

    predict_mod.run(cfg)

    assert not pred_path.exists()
