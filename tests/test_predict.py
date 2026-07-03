import json

import pytest
import numpy as np
import pandas as pd

from bike_sharing.features.build_features import build_lag_features, build_calendar_features
from bike_sharing.models.predict import (
    build_next_hour_features,
    get_missing_hours,
    append_prediction,
    get_fallback_prediction,
)


def test_build_next_hour_features_returns_one_row(sample_df_with_datetime, min_date):
    """
    build_next_hour_features should always return exactly one row —
    the feature vector for the next hour prediction.
    """
    df = build_lag_features(
        sample_df_with_datetime, lags=[1, 2, 3, 8, 24, 48, 72, 168], rolling_windows=[24, 168]
    )
    df = build_calendar_features(df, drop_cols=["atemp", "yr"], min_date=min_date)

    result = build_next_hour_features(df, min_date)

    assert len(result) == 1


def test_build_next_hour_features_correct_hour(sample_df_with_datetime, min_date):
    """
    The predicted hour should be exactly one hour after the last record
    in the past dataset.
    """
    df = build_lag_features(
        sample_df_with_datetime, lags=[1, 2, 3, 8, 24, 48, 72, 168], rolling_windows=[24, 168]
    )
    df = build_calendar_features(df, drop_cols=["atemp", "yr"], min_date=min_date)

    last_dt = df["datetime"].iloc[-1]
    result = build_next_hour_features(df, min_date)

    expected_next_dt = last_dt + pd.Timedelta(hours=1)
    actual_next_dt = pd.Timestamp(result["datetime"].values[0])

    assert actual_next_dt == expected_next_dt


def test_build_next_hour_features_lag1_equals_current_cnt(sample_df_with_datetime, min_date):
    """
    cnt_lag_1 of the next hour should equal cnt of the current (last) hour.
    This is the core lag shift — the most recent demand becomes lag_1.
    """
    df = build_lag_features(
        sample_df_with_datetime, lags=[1, 2, 3, 8, 24, 48, 72, 168], rolling_windows=[24, 168]
    )
    df = build_calendar_features(df, drop_cols=["atemp", "yr"], min_date=min_date)

    current_cnt = df["cnt"].iloc[-1]
    result = build_next_hour_features(df, min_date)

    assert result["cnt_lag_1"].values[0] == current_cnt


def test_build_next_hour_features_cyclic_in_range(sample_df_with_datetime, min_date):
    """
    Cyclic features of the next hour must be in [-1, 1].
    """
    df = build_lag_features(
        sample_df_with_datetime, lags=[1, 2, 3, 8, 24, 48, 72, 168], rolling_windows=[24, 168]
    )
    df = build_calendar_features(df, drop_cols=["atemp", "yr"], min_date=min_date)

    result = build_next_hour_features(df, min_date)

    for col in ["hr_sin", "hr_cos"]:
        val = result[col].values[0]
        assert -1 <= val <= 1, f"{col} = {val} is outside [-1, 1]"


# ── get_missing_hours ─────────────────────────────────────────────────────────


@pytest.fixture
def past_df():
    """Past dataset covering 10 consecutive hours."""
    base = pd.Timestamp("2024-06-01 00:00")
    rows = []
    for i in range(10):
        dt = base + pd.Timedelta(hours=i)
        rows.append({"dteday": dt.date(), "hr": dt.hour, "cnt": 100 + i})
    return pd.DataFrame(rows)


def test_get_missing_hours_no_predictions_file(past_df, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    result = get_missing_hours(past_df, pred_path)
    assert result == []


def test_get_missing_hours_empty_predictions_file(past_df, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(columns=["timestamp_predicted"]).to_csv(pred_path, index=False)
    result = get_missing_hours(past_df, pred_path)
    assert result == []


def test_get_missing_hours_detects_gap(past_df, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    # Only predict hour 0 — hours 1–9 are all missing
    pd.DataFrame([{"timestamp_predicted": "2024-06-01 00:00:00"}]).to_csv(pred_path, index=False)
    result = get_missing_hours(past_df, pred_path)
    assert len(result) == 9


def test_get_missing_hours_no_gap(past_df, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    # Predict all 10 hours in past — nothing missing
    timestamps = [
        {"timestamp_predicted": (pd.Timestamp("2024-06-01") + pd.Timedelta(hours=i)).isoformat()}
        for i in range(10)
    ]
    pd.DataFrame(timestamps).to_csv(pred_path, index=False)
    result = get_missing_hours(past_df, pred_path)
    assert result == []


def test_get_missing_hours_respects_cap(past_df, tmp_path):
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame([{"timestamp_predicted": "2024-06-01 00:00:00"}]).to_csv(pred_path, index=False)
    result = get_missing_hours(past_df, pred_path, max_backfill_hours=3)
    assert len(result) == 3


# ── append_prediction ─────────────────────────────────────────────────────────


@pytest.fixture
def sample_pred():
    return {
        "predicted_at": "2024-06-01T10:00:00",
        "timestamp_predicted": "2024-06-01T11:00:00",
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


# ── run(): backfill must not corrupt the main (next-hour) prediction ──────────


class _FakeModel:
    """
    Model stub whose prediction is a deterministic function of the target hour.

    Returns log1p(cnt_lag_1); after predict.py applies expm1, the prediction
    equals cnt_lag_1 — the cnt of the hour right before the target. With a
    unique cnt per hour this makes every hour's prediction unique, so we can
    detect whether the main prediction used next_dt's features or stale
    backfill features.
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


def _make_past(n: int = 200):
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


def test_run_main_prediction_uses_next_hour_not_backfill(monkeypatch, tmp_path):
    """
    Regression test for the backfill feature-reuse bug.

    When backfill runs, the final next-hour prediction must be computed from
    next_dt's own features — not the last backfilled hour's. The bug overwrote
    X and next_row inside the backfill loop and reused them for the main
    prediction, producing a duplicate of the current hour with the wrong hr.
    """
    import mlflow.lightgbm
    from omegaconf import OmegaConf
    from bike_sharing.models import predict as predict_mod

    n = 200
    dates, past = _make_past(n)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    past.to_csv(raw_dir / "hour_past.csv", index=False)

    # Predictions exist for every hour except the last three → 3-hour backfill
    pred_path = tmp_path / "predictions.csv"
    seeded = [dates[i] for i in range(n - 3)]
    pd.DataFrame(
        {
            "predicted_at": [d.isoformat() for d in seeded],
            "timestamp_predicted": [d.isoformat() for d in seeded],
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

    cfg = OmegaConf.create(
        {
            "project": "bike-sharing-forecast",
            "paths": {
                "raw_dir": str(raw_dir),
                "input_file": "hour_past.csv",
                "simulation_state": str(state_path),
                "predictions_path": str(pred_path),
                "artifacts_dir": str(tmp_path / "artifacts"),
            },
            "monitoring": {"max_backfill_hours": 48},
        }
    )

    monkeypatch.setattr(mlflow.lightgbm, "load_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(predict_mod, "MlflowClient", lambda: _FakeMlflowClient())

    predict_mod.run(cfg)

    out = pd.read_csv(pred_path)
    out["ts"] = pd.to_datetime(out["timestamp_predicted"])

    # 1. No row may have an hr that disagrees with its own timestamp
    assert (out["ts"].dt.hour == out["hr"]).all()

    # 2. The run wrote the 3 missing past hours (backfill) + next_dt (main)
    next_dt = dates[n - 1] + pd.Timedelta(hours=1)
    new = out[out["ts"] > dates[n - 4]].sort_values("ts")
    assert new["ts"].tolist() == [dates[n - 3], dates[n - 2], dates[n - 1], next_dt]

    # 3. The main prediction used next_dt's own lag_1 = last past cnt = n,
    #    not the last backfilled hour's (n - 1) → it is not a duplicate.
    main_row = out[out["ts"] == next_dt].iloc[0]
    assert main_row["hr"] == next_dt.hour
    assert round(main_row["pred_registered"]) == n


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


def _run_cfg(raw_dir, pred_path, state_path, artifacts_dir):
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
        }
    )


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


def test_run_uses_fallback_when_validation_flag_invalid(monkeypatch, tmp_path):
    """
    When the hourly validation flag marks the newly revealed data invalid,
    run() must not call the real model for the main prediction — it serves
    the 168h-ago actuals instead and tags the row so downstream monitoring
    can exclude it.
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

    cfg = _run_cfg(raw_dir, pred_path, state_path, artifacts_dir)

    monkeypatch.setattr(mlflow.lightgbm, "load_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(predict_mod, "MlflowClient", lambda: _FakeMlflowClient())

    predict_mod.run(cfg)

    out = pd.read_csv(pred_path)
    next_dt = dates[n - 1] + pd.Timedelta(hours=1)
    row = out[pd.to_datetime(out["timestamp_predicted"]) == next_dt].iloc[0]

    lookback_dt = next_dt - pd.Timedelta(hours=168)
    expected = past[
        pd.to_datetime(past["dteday"]) + pd.to_timedelta(past["hr"], unit="h") == lookback_dt
    ].iloc[0]

    assert row["prediction_source"] == "fallback_lag168"
    assert pd.isna(row["model_version_registered"])
    assert pd.isna(row["model_version_casual"])
    assert row["pred_registered"] == round(float(expected["registered"]), 2)
    assert row["pred_casual"] == round(float(expected["casual"]), 2)
    assert row["pred_total"] == round(float(expected["cnt"]), 2)


def test_run_skips_prediction_when_invalid_and_no_lag168_data_available(monkeypatch, tmp_path):
    """
    If the flag is invalid AND there isn't 168h of history yet either, run()
    must not write anything for this hour — the existing backfill mechanism
    fills the gap once trustworthy data returns.
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

    cfg = _run_cfg(raw_dir, pred_path, state_path, artifacts_dir)

    monkeypatch.setattr(mlflow.lightgbm, "load_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(predict_mod, "MlflowClient", lambda: _FakeMlflowClient())

    predict_mod.run(cfg)

    assert not pred_path.exists()
