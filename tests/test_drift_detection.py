import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

from bike_sharing.monitoring.drift_detection import load_reference, run_drift_report, DRIFT_FEATURES


@pytest.fixture
def sample_reference_csv(sample_raw_df):
    """
    Creates a temporary train.csv with all FEATURES columns present.
    sample_raw_df spans ~21 days from 2024-01-01, so mnth is always 1.
    """
    df = sample_raw_df.copy()

    # Add all required feature columns with synthetic values
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    df["hr_workday"] = df["hr"] * df["workingday"]
    df["hr_weekend"] = df["hr"] * (1 - df["workingday"])
    df["hr_x_season"] = df["hr"] * df["season"]
    df["is_rush_hour"] = 0
    df["days_since_start"] = range(len(df))
    df["cnt_lag_1"] = df["cnt"].shift(1)
    df["cnt_lag_2"] = df["cnt"].shift(2)
    df["cnt_lag_3"] = df["cnt"].shift(3)
    df["cnt_lag_8"] = df["cnt"].shift(8)
    df["cnt_lag_24"] = df["cnt"].shift(24)
    df["cnt_lag_48"] = df["cnt"].shift(48)
    df["cnt_lag_72"] = df["cnt"].shift(72)
    df["cnt_lag_168"] = df["cnt"].shift(168)
    df["cnt_rolling_mean_24"] = df["cnt"].shift(1).rolling(24).mean()
    df["cnt_rolling_mean_168"] = df["cnt"].shift(1).rolling(168).mean()

    df = df.dropna(subset=DRIFT_FEATURES)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "train.csv"
        df.to_csv(path, index=False)
        yield Path(tmpdir)


def test_load_reference_returns_only_drift_features(sample_reference_csv):
    """
    load_reference should return a DataFrame with only DRIFT_FEATURES columns
    — days_since_start (in FEATURES but not DRIFT_FEATURES) must be excluded,
    since reference and current windows can never overlap in its range.
    """
    result = load_reference(sample_reference_csv, months=[1], min_reference_rows=1)

    assert list(result.columns) == DRIFT_FEATURES
    assert "days_since_start" not in result.columns


def test_run_drift_report_flag_has_required_keys(sample_reference_csv):
    """
    drift_detected.json must contain all keys that retrain.py reads.
    Missing keys would cause a KeyError in should_retrain.
    """
    reference = load_reference(sample_reference_csv, months=[1], min_reference_rows=1)
    current = reference.tail(50).copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        summary = run_drift_report(
            reference, current, output_dir, drift_threshold=0.5, numerical_features=DRIFT_FEATURES
        )

        required_keys = [
            "timestamp",
            "n_features",
            "n_drifted",
            "drift_share",
            "drift_detected",
            "threshold",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key in drift summary: {key}"


def test_run_drift_report_no_drift_on_identical_data(sample_reference_csv):
    """
    When current data is identical to reference, drift_share should be 0
    and drift_detected should be False.
    """
    reference = load_reference(sample_reference_csv, months=[1], min_reference_rows=1)
    current = reference.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        summary = run_drift_report(
            reference, current, output_dir, drift_threshold=0.5, numerical_features=DRIFT_FEATURES
        )

        assert summary["drift_detected"] is False
        assert summary["drift_share"] == 0.0


def test_run_drift_report_save_html_false_skips_html_file(sample_reference_csv):
    """
    save_html=False must never create drift_report.html — used for output
    drift, which runs hourly and would otherwise flood MLflow/DagsHub
    storage with a ~4MB file every hour.
    """
    reference = load_reference(sample_reference_csv, months=[1], min_reference_rows=1)
    current = reference.tail(50).copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        run_drift_report(
            reference,
            current,
            output_dir,
            drift_threshold=0.5,
            numerical_features=DRIFT_FEATURES,
            save_html=False,
        )

        assert not (output_dir / "drift_report.html").exists()
        assert (output_dir / "drift_detected.json").exists()


# ── load_reference: month filtering / widening / fallback ────────────────────


def _make_train_csv(tmpdir, month_counts: dict) -> Path:
    """
    Build a minimal train.csv with DRIFT_FEATURES + mnth columns, with the
    given number of rows per month. Values are synthetic but NaN-free —
    enough to exercise load_reference's month filter/widen/fallback tiers
    without needing a realistic feature pipeline.
    """
    rows = []
    for month, count in month_counts.items():
        for i in range(count):
            row = {f: float(i % 10) for f in DRIFT_FEATURES}
            row["mnth"] = month
            rows.append(row)
    df = pd.DataFrame(rows)
    path = Path(tmpdir) / "train.csv"
    df.to_csv(path, index=False)
    return Path(tmpdir)


def test_load_reference_filters_by_month():
    """
    With enough rows in the exact requested month, only that month's rows
    are returned — other months must not leak into the reference.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_dir = _make_train_csv(tmpdir, {1: 600, 6: 600})

        result = load_reference(processed_dir, months=[1], min_reference_rows=500)

        assert len(result) == 600


def test_load_reference_widens_to_neighboring_months_when_thin():
    """
    If the exact month match is thinner than min_reference_rows, widen to
    month ± 1 before giving up on seasonal precision entirely.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Month 6 alone (50 rows) is too thin for min_reference_rows=300;
        # widening to 5+6+7 (450 rows) clears it. Month 1 is a decoy that
        # must be excluded.
        processed_dir = _make_train_csv(tmpdir, {5: 200, 6: 50, 7: 200, 1: 900})

        result = load_reference(processed_dir, months=[6], min_reference_rows=300)

        assert len(result) == 450


def test_load_reference_falls_back_to_full_when_still_thin():
    """
    If even the widened (month ± 1) window is too thin, fall back to the
    entire reference rather than returning an unreliably small sample.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_dir = _make_train_csv(tmpdir, {6: 10, 1: 900})

        result = load_reference(processed_dir, months=[6], min_reference_rows=500)

        assert len(result) == 910
