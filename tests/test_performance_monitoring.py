import pandas as pd
import pytest

from bike_sharing.monitoring.performance_monitoring import (
    load_predictions,
    join_predictions_with_actuals,
    build_seasonal_naive,
    compute_rolling_performance,
    compute_rolling_performance_by_horizon,
)


# ── load_predictions ─────────────────────────────────────────────────────────


def test_load_predictions_excludes_fallback_rows(tmp_path):
    """
    fallback_lag168 rows aren't real model output — if this filter silently
    breaks, they'd contaminate RMSE/MAE without any crash or visible error.
    """
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=2, freq="h"),
            "pred_total": [100.0, 200.0],
            "prediction_source": ["model", "fallback_lag168"],
        }
    ).to_csv(pred_path, index=False)

    df = load_predictions(pred_path)

    assert len(df) == 1
    assert df["prediction_source"].tolist() == ["model"]


def test_load_predictions_defaults_missing_horizon_to_1(tmp_path):
    """Legacy predictions.csv (pre-multi-horizon) has no horizon column —
    every row must read as horizon=1 so grouping downstream doesn't break."""
    pred_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=2, freq="h"),
            "pred_total": [100.0, 200.0],
        }
    ).to_csv(pred_path, index=False)

    df = load_predictions(pred_path)

    assert df["horizon"].tolist() == [1, 1]


# ── join_predictions_with_actuals ───────────────────────────────────────────


def test_join_drops_predictions_without_a_known_actual():
    """
    Predictions for hours whose actual hasn't been revealed yet must be
    dropped — an inner join keeps only resolved (scoreable) predictions.
    """
    predictions = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=5, freq="h"),
            "pred_total": [100.0, 200.0, 150.0, 300.0, 250.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=3, freq="h"),
            "actual_total": [90.0, 210.0, 140.0],
        }
    )

    joined = join_predictions_with_actuals(predictions, actuals)

    assert len(joined) == 3
    assert joined["timestamp_predicted"].tolist() == predictions["timestamp_predicted"].tolist()[:3]


# ── build_seasonal_naive ────────────────────────────────────────────────────


def test_build_seasonal_naive_shifts_actual_forward_168h():
    """
    The naive prediction for hour t must be the actual demand at t-168h, i.e.
    each actual's timestamp is pushed forward exactly one week so it lines up
    (via merge) with the hour it predicts.
    """
    actuals = pd.DataFrame(
        {
            "timestamp_predicted": pd.to_datetime(["2026-01-01 05:00", "2026-01-01 06:00"]),
            "actual_total": [100.0, 200.0],
        }
    )

    naive = build_seasonal_naive(actuals)

    assert naive["naive_pred"].tolist() == [100.0, 200.0]
    assert naive["timestamp_predicted"].tolist() == [
        pd.Timestamp("2026-01-08 05:00"),
        pd.Timestamp("2026-01-08 06:00"),
    ]


# ── compute_rolling_performance ─────────────────────────────────────────────


def test_compute_rolling_performance_reports_skill_vs_seasonal_naive():
    """
    naive_rmse and skill_vs_naive must reflect the seasonal-naive baseline over
    the same window. Model off by 5 (rmse 5), naive off by 10 (rmse 10) →
    skill = 1 - 5/10 = 0.5.
    """
    joined = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=4, freq="h"),
            "actual_total": [100.0, 200.0, 300.0, 400.0],
            "pred_total": [105.0, 205.0, 305.0, 405.0],
            "naive_pred": [110.0, 210.0, 310.0, 410.0],
        }
    )

    summary = compute_rolling_performance(joined, n_hours=168)

    assert summary["rmse"] == pytest.approx(5.0)
    assert summary["naive_rmse"] == pytest.approx(10.0)
    assert summary["skill_vs_naive"] == pytest.approx(0.5)


def test_compute_rolling_performance_naive_none_when_column_absent():
    """
    Without a naive_pred column (e.g. a joined frame that predates the baseline),
    naive_rmse/skill must be None rather than raising — keeps the metric wiring
    backward compatible.
    """
    joined = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=3, freq="h"),
            "actual_total": [100.0, 200.0, 150.0],
            "pred_total": [100.0, 200.0, 150.0],
        }
    )

    summary = compute_rolling_performance(joined, n_hours=168)

    assert summary["naive_rmse"] is None
    assert summary["skill_vs_naive"] is None


def test_compute_rolling_performance_perfect_predictions_give_zero_error():
    """
    If pred_total exactly matches actual_total, rmse/mae/rmsle must be 0
    and r2 must be 1 — the cleanest way to verify the metric wiring without
    computing RMSE by hand.
    """
    joined = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=4, freq="h"),
            "pred_total": [100.0, 200.0, 150.0, 300.0],
            "actual_total": [100.0, 200.0, 150.0, 300.0],
        }
    )

    summary = compute_rolling_performance(joined, n_hours=168)

    assert summary["rmse"] == pytest.approx(0.0)
    assert summary["mae"] == pytest.approx(0.0)
    assert summary["rmsle"] == pytest.approx(0.0)
    assert summary["r2"] == pytest.approx(1.0)
    assert summary["n_resolved"] == 4
    assert summary["n_hours"] == 168


def test_compute_rolling_performance_windows_to_most_recent_n_hours():
    """
    Only the most recent n_hours resolved records should be used — older
    resolved predictions outside the window must not affect the metric.
    """
    joined = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=5, freq="h"),
            # First record is a huge miss; the rest are perfect.
            "pred_total": [1000.0, 100.0, 200.0, 150.0, 300.0],
            "actual_total": [1.0, 100.0, 200.0, 150.0, 300.0],
        }
    )

    summary = compute_rolling_performance(joined, n_hours=4)

    # Window excludes the first (huge-error) row → error should be exactly 0.
    assert summary["rmse"] == pytest.approx(0.0)
    assert summary["n_resolved"] == 4


# ── compute_rolling_performance_by_horizon ──────────────────────────────────


def test_by_horizon_returns_one_summary_per_horizon_tagged():
    """Each lead time present gets its own summary, tagged with its horizon."""
    joined = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=6, freq="h"),
            "horizon": [1, 2, 3, 1, 2, 3],
            "pred_total": [100.0, 200.0, 300.0, 110.0, 220.0, 330.0],
            "actual_total": [100.0, 200.0, 300.0, 110.0, 220.0, 330.0],
        }
    )

    summaries = compute_rolling_performance_by_horizon(joined, n_hours=168)

    assert [s["horizon"] for s in summaries] == [1, 2, 3]
    assert all(s["rmse"] == pytest.approx(0.0) for s in summaries)
    assert all(s["n_resolved"] == 2 for s in summaries)


def test_by_horizon_scores_each_horizon_independently():
    """
    Error grows with lead time — a pooled RMSE would blur that. Each horizon's
    summary must reflect only its own rows: here h+1 is perfect (rmse 0) while
    h+2 is off by 10 (rmse 10).
    """
    joined = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=4, freq="h"),
            "horizon": [1, 2, 1, 2],
            "pred_total": [100.0, 210.0, 150.0, 310.0],
            "actual_total": [100.0, 200.0, 150.0, 300.0],
        }
    )

    summaries = {
        s["horizon"]: s for s in compute_rolling_performance_by_horizon(joined, n_hours=168)
    }

    assert summaries[1]["rmse"] == pytest.approx(0.0)
    assert summaries[2]["rmse"] == pytest.approx(10.0)


def test_by_horizon_windows_within_each_horizon():
    """
    The most-recent-n_hours window must apply per horizon, not across the
    pooled frame — otherwise a horizon with many rows would crowd another out
    of the window. With n_hours=1, each horizon keeps only its own latest row.
    """
    joined = pd.DataFrame(
        {
            "timestamp_predicted": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 01:00",
                ]
            ),
            "horizon": [1, 2, 1, 2],
            # Older rows are big misses; latest row of each horizon is perfect.
            "pred_total": [1000.0, 1000.0, 100.0, 200.0],
            "actual_total": [1.0, 1.0, 100.0, 200.0],
        }
    )

    summaries = {s["horizon"]: s for s in compute_rolling_performance_by_horizon(joined, n_hours=1)}

    assert summaries[1]["n_resolved"] == 1
    assert summaries[1]["rmse"] == pytest.approx(0.0)
    assert summaries[2]["n_resolved"] == 1
    assert summaries[2]["rmse"] == pytest.approx(0.0)


def test_by_horizon_seasonal_naive_scored_against_each_horizon():
    """
    The seasonal-naive baseline is horizon-independent (cnt(t-168h)); it must be
    compared against every horizon. Model off by 5 (rmse 5), naive off by 10
    (rmse 10) at both horizons → skill 0.5 for each.
    """
    joined = pd.DataFrame(
        {
            "timestamp_predicted": pd.date_range("2026-01-01", periods=4, freq="h"),
            "horizon": [1, 2, 1, 2],
            "actual_total": [100.0, 200.0, 300.0, 400.0],
            "pred_total": [105.0, 205.0, 305.0, 405.0],
            "naive_pred": [110.0, 210.0, 310.0, 410.0],
        }
    )

    summaries = {
        s["horizon"]: s for s in compute_rolling_performance_by_horizon(joined, n_hours=168)
    }

    assert summaries[1]["skill_vs_naive"] == pytest.approx(0.5)
    assert summaries[2]["skill_vs_naive"] == pytest.approx(0.5)
