import pytest
import json
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from bike_sharing.models.train import FEATURES
from bike_sharing.monitoring.drift_detection import load_reference, run_drift_report


@pytest.fixture
def sample_reference_csv(sample_raw_df):
    """
    Creates a temporary train.csv with all FEATURES columns present.
    """
    df = sample_raw_df.copy()

    # Add all required feature columns with synthetic values
    df["hr_sin"]              = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"]              = np.cos(2 * np.pi * df["hr"] / 24)
    df["mnth_sin"]            = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"]            = np.cos(2 * np.pi * df["mnth"] / 12)
    df["hr_workday"]          = df["hr"] * df["workingday"]
    df["hr_weekend"]          = df["hr"] * (1 - df["workingday"])
    df["hr_x_season"]         = df["hr"] * df["season"]
    df["is_rush_hour"]        = 0
    df["days_since_start"]    = range(len(df))
    df["cnt_lag_1"]           = df["cnt"].shift(1)
    df["cnt_lag_2"]           = df["cnt"].shift(2)
    df["cnt_lag_3"]           = df["cnt"].shift(3)
    df["cnt_lag_8"]           = df["cnt"].shift(8)
    df["cnt_lag_24"]          = df["cnt"].shift(24)
    df["cnt_lag_48"]          = df["cnt"].shift(48)
    df["cnt_lag_72"]          = df["cnt"].shift(72)
    df["cnt_lag_168"]         = df["cnt"].shift(168)
    df["cnt_rolling_mean_24"] = df["cnt"].shift(1).rolling(24).mean()
    df["cnt_rolling_mean_168"]= df["cnt"].shift(1).rolling(168).mean()

    df = df.dropna(subset=FEATURES)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "train.csv"
        df.to_csv(path, index=False)
        yield Path(tmpdir)


def test_load_reference_returns_only_features(sample_reference_csv):
    """
    load_reference should return a DataFrame with only the FEATURES columns.
    No target, no metadata columns should be present.
    """
    result = load_reference(sample_reference_csv)

    assert list(result.columns) == FEATURES


def test_run_drift_report_saves_flag(sample_reference_csv):
    """
    run_drift_report should always save drift_detected.json regardless
    of whether drift was detected or not.
    """
    reference = load_reference(sample_reference_csv)
    current   = reference.tail(50).copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        run_drift_report(reference, current, output_dir, drift_threshold=0.5)

        assert (output_dir / "drift_detected.json").exists()


def test_run_drift_report_flag_has_required_keys(sample_reference_csv):
    """
    drift_detected.json must contain all keys that retrain.py reads.
    Missing keys would cause a KeyError in should_retrain.
    """
    reference = load_reference(sample_reference_csv)
    current   = reference.tail(50).copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        summary = run_drift_report(reference, current, output_dir, drift_threshold=0.5)

        required_keys = ["timestamp", "n_features", "n_drifted", "drift_share", "drift_detected", "threshold"]
        for key in required_keys:
            assert key in summary, f"Missing key in drift summary: {key}"


def test_run_drift_report_no_drift_on_identical_data(sample_reference_csv):
    """
    When current data is identical to reference, drift_share should be 0
    and drift_detected should be False.
    """
    reference = load_reference(sample_reference_csv)
    current   = reference.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        summary = run_drift_report(reference, current, output_dir, drift_threshold=0.5)

        assert summary["drift_detected"] is False
        assert summary["drift_share"] == 0.0