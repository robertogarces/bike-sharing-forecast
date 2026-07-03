import pandas as pd

from bike_sharing.features.build_features import build_lag_features, build_calendar_features


def test_build_lag_features_creates_correct_columns(sample_df_with_datetime):
    """
    build_lag_features should create exactly the lag and rolling columns
    specified in the lags and rolling_windows parameters.
    """
    lags = [1, 2, 24]
    rolling_windows = [24]

    result = build_lag_features(sample_df_with_datetime, lags=lags, rolling_windows=rolling_windows)

    for lag in lags:
        assert f"cnt_lag_{lag}" in result.columns, f"Missing column: cnt_lag_{lag}"

    assert "cnt_rolling_mean_24" in result.columns


def test_build_lag_features_no_leakage(sample_df_with_datetime):
    """
    cnt_lag_1 at row i should equal cnt at row i-1.
    This verifies that shift(1) is applied correctly and the current
    hour's demand does not leak into its own lag.
    """
    result = build_lag_features(sample_df_with_datetime, lags=[1], rolling_windows=[])

    # Row 5: lag_1 should be cnt of row 4
    assert result["cnt_lag_1"].iloc[5] == result["cnt"].iloc[4]


def test_build_lag_features_first_row_is_nan(sample_df_with_datetime):
    """
    The first row should always have NaN for lag_1 since there is no
    previous record to look back at.
    """
    result = build_lag_features(sample_df_with_datetime, lags=[1], rolling_windows=[])

    assert pd.isna(result["cnt_lag_1"].iloc[0])


def test_build_calendar_features_creates_correct_columns(sample_df_with_datetime, min_date):
    """
    build_calendar_features should create all expected engineered columns.
    """
    expected_cols = [
        "hr_sin",
        "hr_cos",
        "mnth_sin",
        "mnth_cos",
        "hr_workday",
        "hr_weekend",
        "hr_x_season",
        "is_rush_hour",
        "days_since_start",
    ]

    result = build_calendar_features(
        sample_df_with_datetime, drop_cols=["atemp", "yr"], min_date=min_date
    )

    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_cyclic_features_in_valid_range(sample_df_with_datetime, min_date):
    """
    Sin and cos encodings must always be in [-1, 1].
    If they fall outside this range, the cyclic encoding formula is wrong.
    """
    result = build_calendar_features(
        sample_df_with_datetime, drop_cols=["atemp", "yr"], min_date=min_date
    )

    for col in ["hr_sin", "hr_cos", "mnth_sin", "mnth_cos"]:
        assert result[col].between(-1, 1).all(), f"{col} has values outside [-1, 1]"


def test_rush_hour_only_on_working_days(sample_df_with_datetime, min_date):
    """
    is_rush_hour should only be 1 on working days during 7-9 AM or 5-7 PM.
    On weekends or outside those hours it must always be 0.
    """
    result = build_calendar_features(
        sample_df_with_datetime, drop_cols=["atemp", "yr"], min_date=min_date
    )

    # is_rush_hour must be 0 on non-working days
    non_working = result[result["workingday"] == 0]
    assert (non_working["is_rush_hour"] == 0).all()

    # is_rush_hour must be 0 outside rush hours on working days
    working_non_rush = result[
        (result["workingday"] == 1)
        & (~result["hr"].between(7, 9))
        & (~result["hr"].between(17, 19))
    ]
    assert (working_non_rush["is_rush_hour"] == 0).all()


def test_drop_cols_are_removed(sample_df_with_datetime, min_date):
    """
    Columns specified in drop_cols should not appear in the output.
    """
    result = build_calendar_features(
        sample_df_with_datetime, drop_cols=["atemp", "yr"], min_date=min_date
    )

    assert "atemp" not in result.columns
    assert "yr" not in result.columns
