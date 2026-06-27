import pytest
import numpy as np
import pandas as pd

from bike_sharing.features.build_features import build_lag_features, build_calendar_features
from bike_sharing.models.predict import build_next_hour_features, get_missing_hours, append_prediction


def test_build_next_hour_features_returns_one_row(sample_df_with_datetime, min_date):
    """
    build_next_hour_features should always return exactly one row —
    the feature vector for the next hour prediction.
    """
    df = build_lag_features(sample_df_with_datetime, lags=[1, 2, 3, 8, 24, 48, 72, 168], rolling_windows=[24, 168])
    df = build_calendar_features(df, drop_cols=["atemp", "yr"], min_date=min_date)

    result = build_next_hour_features(df, min_date)

    assert len(result) == 1


def test_build_next_hour_features_correct_hour(sample_df_with_datetime, min_date):
    """
    The predicted hour should be exactly one hour after the last record
    in the past dataset.
    """
    df = build_lag_features(sample_df_with_datetime, lags=[1, 2, 3, 8, 24, 48, 72, 168], rolling_windows=[24, 168])
    df = build_calendar_features(df, drop_cols=["atemp", "yr"], min_date=min_date)

    last_dt = df["datetime"].iloc[-1]
    result  = build_next_hour_features(df, min_date)

    expected_next_dt = last_dt + pd.Timedelta(hours=1)
    actual_next_dt   = pd.Timestamp(result["datetime"].values[0])

    assert actual_next_dt == expected_next_dt


def test_build_next_hour_features_lag1_equals_current_cnt(sample_df_with_datetime, min_date):
    """
    cnt_lag_1 of the next hour should equal cnt of the current (last) hour.
    This is the core lag shift — the most recent demand becomes lag_1.
    """
    df = build_lag_features(sample_df_with_datetime, lags=[1, 2, 3, 8, 24, 48, 72, 168], rolling_windows=[24, 168])
    df = build_calendar_features(df, drop_cols=["atemp", "yr"], min_date=min_date)

    current_cnt = df["cnt"].iloc[-1]
    result      = build_next_hour_features(df, min_date)

    assert result["cnt_lag_1"].values[0] == current_cnt


def test_build_next_hour_features_cyclic_in_range(sample_df_with_datetime, min_date):
    """
    Cyclic features of the next hour must be in [-1, 1].
    """
    df = build_lag_features(sample_df_with_datetime, lags=[1, 2, 3, 8, 24, 48, 72, 168], rolling_windows=[24, 168])
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
        "hr": 11, "temp": 0.5, "hum": 0.6, "weathersit": 1, "workingday": 1,
        "pred_registered": 120.0, "pred_casual": 30.0, "pred_total": 150.0,
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