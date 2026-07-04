import pandas as pd
import pytest

from bike_sharing.monitoring.suggest_thresholds import (
    suggest_drift_threshold,
    suggest_performance_degradation_threshold,
)


# ── suggest_drift_threshold ──────────────────────────────────────────────────


def test_suggest_drift_threshold_returns_percentile_with_enough_history(tmp_path):
    history_path = tmp_path / "drift_history.csv"
    values = [0.01 * i for i in range(1, 13)]  # 12 weeks: 0.01 .. 0.12
    pd.DataFrame({"drift_share": values}).to_csv(history_path, index=False)

    suggested, n_weeks = suggest_drift_threshold(history_path)

    assert n_weeks == 12
    assert suggested == pytest.approx(0.109)


def test_suggest_drift_threshold_returns_none_with_insufficient_history(tmp_path):
    history_path = tmp_path / "drift_history.csv"
    pd.DataFrame({"drift_share": [0.1, 0.2, 0.3]}).to_csv(history_path, index=False)

    suggested, n_weeks = suggest_drift_threshold(history_path)

    assert suggested is None
    assert n_weeks == 3


def test_suggest_drift_threshold_returns_none_when_file_missing(tmp_path):
    suggested, n_weeks = suggest_drift_threshold(tmp_path / "does_not_exist.csv")

    assert suggested is None
    assert n_weeks == 0


# ── suggest_performance_degradation_threshold ───────────────────────────────


def test_suggest_performance_degradation_threshold_uses_pct_change_not_raw_rmse(tmp_path):
    """
    Constant RMSE across all weeks means zero real variation — the suggested
    threshold must be 0, not the RMSE value itself. Confirms the function
    operates on week-over-week % change, not the raw rmse column.
    """
    history_path = tmp_path / "performance_history.csv"
    pd.DataFrame({"rmse": [50.0] * 13}).to_csv(history_path, index=False)

    suggested, n_weeks = suggest_performance_degradation_threshold(history_path)

    assert n_weeks == 12  # 13 rows -> 12 week-over-week changes
    assert suggested == 0.0


def test_suggest_performance_degradation_threshold_returns_none_with_insufficient_history(
    tmp_path,
):
    history_path = tmp_path / "performance_history.csv"
    pd.DataFrame({"rmse": [50.0, 51.0, 49.0]}).to_csv(history_path, index=False)

    suggested, n_weeks = suggest_performance_degradation_threshold(history_path)

    assert suggested is None
    assert n_weeks == 2  # 3 rows -> 2 changes


def test_suggest_performance_degradation_threshold_returns_none_when_file_missing(tmp_path):
    suggested, n_weeks = suggest_performance_degradation_threshold(tmp_path / "does_not_exist.csv")

    assert suggested is None
    assert n_weeks == 0
