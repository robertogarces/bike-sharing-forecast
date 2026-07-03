import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_raw_df():
    """Minimal raw dataset mimicking hour.csv structure."""
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "instant": range(1, n + 1),
            "dteday": dates.date,
            "season": np.random.randint(1, 5, n),
            "yr": (dates.year - 2024).astype(int),
            "mnth": dates.month,
            "hr": dates.hour,
            "holiday": np.zeros(n, dtype=int),
            "weekday": dates.dayofweek,
            "workingday": (dates.dayofweek < 5).astype(int),
            "weathersit": np.random.randint(1, 4, n),
            "temp": np.random.uniform(0.1, 0.9, n),
            "atemp": np.random.uniform(0.1, 0.9, n),
            "hum": np.random.uniform(0.2, 0.8, n),
            "windspeed": np.random.uniform(0.0, 0.5, n),
            "casual": np.random.randint(0, 50, n),
            "registered": np.random.randint(0, 200, n),
            "cnt": np.random.randint(10, 250, n),
        }
    )


@pytest.fixture
def sample_df_with_datetime(sample_raw_df):
    """Raw dataset with datetime column added."""
    df = sample_raw_df.copy()
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df


@pytest.fixture
def min_date(sample_df_with_datetime):
    return sample_df_with_datetime["dteday"].min()
