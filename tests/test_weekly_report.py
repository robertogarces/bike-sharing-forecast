from bike_sharing.monitoring.weekly_report import build_weekly_digest


def test_build_weekly_digest_handles_first_run_with_no_sources():
    """
    The very first run has no drift report, no retrain outcome, and no
    resolved predictions yet — every section must degrade gracefully instead
    of raising.
    """
    digest = build_weekly_digest(None, None, None)

    assert "Not available yet" in digest
    assert digest.count("Not available yet") == 3


def test_build_weekly_digest_no_drift_no_retrain_with_performance():
    drift_summary = {
        "n_features": 10,
        "n_drifted": 0,
        "drift_share": 0.0,
        "drift_detected": False,
        "threshold": 0.3,
        "drift_by_column": {
            "temp": {"stattest": "ks", "score": 0.05, "drifted": False},
        },
    }
    retrain_outcome = {
        "retrain_attempted": False,
        "skip_reason": "no drift detected (0% <= 30%)",
        "data_quality_checked": False,
        "data_quality_passed": None,
        "data_quality_issues": [],
        "new_rmse": None,
        "prod_rmse": None,
        "promoted_registered": None,
        "promoted_casual": None,
        "performance_degraded": False,
        "baseline_rmse": 30.0,
        "live_rmse": 32.5,
    }
    performance_summary = {
        "n_hours": 168,
        "n_resolved": 165,
        "rmse": 32.5,
        "mae": 20.1,
        "rmsle": 0.15,
        "r2": 0.89,
        "naive_rmse": 65.0,
        "skill_vs_naive": 0.5,
    }

    digest = build_weekly_digest(drift_summary, retrain_outcome, performance_summary)

    assert "No drift detected" in digest
    assert "Attempted: No" in digest
    assert "no drift detected (0% <= 30%)" in digest
    assert "RMSE: 32.50" in digest
    assert "R²: 0.8900" in digest
    assert "Live vs. baseline RMSE: 32.50 vs 30.00 (stable)" in digest
    assert "vs seasonal-naive: 32.50 vs 65.00 RMSE (+50.0% skill)" in digest


def test_build_weekly_digest_drift_detected_and_retrain_promoted():
    drift_summary = {
        "n_features": 10,
        "n_drifted": 2,
        "drift_share": 0.2,
        "drift_detected": True,
        "threshold": 0.1,
        "drift_by_column": {
            "temp": {"stattest": "ks", "score": 0.05, "drifted": False},
            "hum": {"stattest": "ks", "score": 0.5, "drifted": True},
            "windspeed": {"stattest": "ks", "score": 0.4, "drifted": True},
        },
    }
    retrain_outcome = {
        "retrain_attempted": True,
        "skip_reason": None,
        "data_quality_checked": True,
        "data_quality_passed": True,
        "data_quality_issues": [],
        "new_rmse": 28.3,
        "prod_rmse": 31.0,
        "promoted_registered": True,
        "promoted_casual": False,
        "performance_degraded": True,
        "baseline_rmse": 25.0,
        "live_rmse": 32.0,
    }
    performance_summary = None

    digest = build_weekly_digest(drift_summary, retrain_outcome, performance_summary)

    assert "DRIFT DETECTED" in digest
    assert "Drifted: hum, windspeed" in digest
    assert "Attempted: Yes" in digest
    assert "New model RMSE: 28.3000" in digest
    assert "Production model RMSE: 31.0000" in digest
    assert "Promoted to production: registered=Yes, casual=No" in digest
    assert "Live vs. baseline RMSE: 32.00 vs 25.00 (DEGRADED)" in digest


def test_build_weekly_digest_retrain_aborted_by_data_quality_failure():
    retrain_outcome = {
        "retrain_attempted": True,
        "skip_reason": "data quality validation failed: [\"missing column 'hum'\"]",
        "data_quality_checked": True,
        "data_quality_passed": False,
        "data_quality_issues": ["missing column 'hum'"],
        "new_rmse": None,
        "prod_rmse": None,
        "promoted_registered": None,
        "promoted_casual": None,
    }

    digest = build_weekly_digest(None, retrain_outcome, None)

    assert "Data quality check: failed" in digest
    assert "missing column 'hum'" in digest
