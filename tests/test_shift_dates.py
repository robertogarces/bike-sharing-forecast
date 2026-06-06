import pytest
import pandas as pd
import numpy as np

from bike_sharing.data.shift_dates import shift_dates


@pytest.fixture
def sample_raw_df_for_shift():
    """
    Minimal raw dataset for testing shift_dates.
    Uses 1000 records to have a meaningful future split.
    """
    n = 1000
    dates = pd.date_range("2011-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "dteday": dates.normalize(),
        "hr":     dates.hour,
        "cnt":    np.random.randint(10, 300, n),
    })


def test_shift_dates_future_starts_at_reference(sample_raw_df_for_shift):
    """
    The first future record should start exactly at reference_date.
    """
    reference_date = pd.Timestamp("2026-06-04")
    future_pct     = 0.1

    df_shifted, state = shift_dates(sample_raw_df_for_shift, reference_date, future_pct)

    n_future       = int(len(df_shifted) * future_pct)
    first_future   = df_shifted.iloc[-n_future]["dteday"]

    assert first_future == reference_date


def test_shift_dates_preserves_record_count(sample_raw_df_for_shift):
    """
    Shifting dates should not add or remove any records.
    """
    reference_date = pd.Timestamp("2026-06-04")
    df_shifted, _  = shift_dates(sample_raw_df_for_shift, reference_date, 0.1)

    assert len(df_shifted) == len(sample_raw_df_for_shift)


def test_shift_dates_preserves_temporal_order(sample_raw_df_for_shift):
    """
    After shifting, dates should remain in ascending order.
    The temporal structure of the dataset must be fully preserved.
    """
    reference_date = pd.Timestamp("2026-06-04")
    df_shifted, _  = shift_dates(sample_raw_df_for_shift, reference_date, 0.1)

    assert df_shifted["dteday"].is_monotonic_increasing


def test_shift_dates_correct_future_pct(sample_raw_df_for_shift):
    """
    The number of future records should match the requested percentage.
    """
    reference_date = pd.Timestamp("2026-06-04")
    future_pct     = 0.1
    df_shifted, state = shift_dates(sample_raw_df_for_shift, reference_date, future_pct)

    expected_n_future = int(len(sample_raw_df_for_shift) * future_pct)
    assert state["n_future_records"] == expected_n_future


def test_shift_dates_state_contains_required_keys(sample_raw_df_for_shift):
    """
    The state dictionary should contain all required keys for
    simulation_state.json to be valid.
    """
    reference_date = pd.Timestamp("2026-06-04")
    _, state = shift_dates(sample_raw_df_for_shift, reference_date, 0.1)

    required_keys = [
        "reference_date", "future_pct", "shift_applied_at",
        "future_start_date", "future_end_date",
        "n_future_records", "n_past_records",
    ]
    for key in required_keys:
        assert key in state, f"Missing key in state: {key}"