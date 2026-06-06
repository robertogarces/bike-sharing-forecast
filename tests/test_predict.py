import pytest
import numpy as np
import pandas as pd

from bike_sharing.features.build_features import build_lag_features, build_calendar_features
from bike_sharing.models.predict import build_next_hour_features


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