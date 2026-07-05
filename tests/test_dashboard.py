import pandas as pd
import pytest

from bike_sharing.dashboard.app import normalize_retrain_outcome, compute_gauge_range


# ── normalize_retrain_outcome ───────────────────────────────────────────────────


def test_normalize_retrain_outcome_maps_legacy_promoted_key():
    """
    Older retrain_outcome.json snapshots (pre backlog #10) use a single
    "promoted" bool instead of promoted_registered/promoted_casual — the
    dashboard must read both as if both models moved together, matching what
    that legacy schema actually meant.
    """
    legacy = {"retrain_attempted": True, "promoted": True}

    normalized = normalize_retrain_outcome(legacy)

    assert normalized["promoted_registered"] is True
    assert normalized["promoted_casual"] is True


def test_normalize_retrain_outcome_leaves_new_schema_untouched():
    """New-schema snapshots already have promoted_registered/casual — no rewrite."""
    new = {"retrain_attempted": True, "promoted_registered": True, "promoted_casual": False}

    normalized = normalize_retrain_outcome(new)

    assert normalized == new


# ── compute_gauge_range ─────────────────────────────────────────────────────────


def test_compute_gauge_range_scales_with_historical_max():
    """
    Axis max must clear the historical max (with headroom), not clip it like
    the old hardcoded [0, 900] did against a real max of 977.
    """
    actual_total = pd.Series([100.0, 500.0, 977.0, 300.0])

    axis_max, threshold = compute_gauge_range(actual_total)

    assert axis_max > 977.0
    assert axis_max == pytest.approx(977.0 * 1.1, abs=50)


def test_compute_gauge_range_threshold_is_90th_percentile():
    actual_total = pd.Series(range(1, 101), dtype=float)  # 1..100

    _, threshold = compute_gauge_range(actual_total)

    assert threshold == pytest.approx(actual_total.quantile(0.9))


def test_compute_gauge_range_falls_back_when_no_history():
    axis_max, threshold = compute_gauge_range(pd.Series([], dtype=float))

    assert axis_max == 900.0
    assert threshold == 700.0
