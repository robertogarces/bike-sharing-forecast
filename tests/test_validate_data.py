import pandas as pd

from bike_sharing.data.validate_data import (
    validate_schema,
    validate_nulls,
    validate_ranges,
    validate_cnt_invariant,
    validate_data_quality,
)

REQUIRED_COLUMNS = ["hr", "weathersit", "temp", "casual", "registered", "cnt"]
RANGES = {"hr": [0, 23], "weathersit": [1, 4], "temp": [0.0, 1.0]}


def _good_df():
    return pd.DataFrame(
        {
            "hr": [0, 12, 23],
            "weathersit": [1, 2, 4],
            "temp": [0.2, 0.5, 0.9],
            "casual": [5, 10, 3],
            "registered": [50, 100, 30],
            "cnt": [55, 110, 33],
        }
    )


# ── validate_schema ────────────────────────────────────────────────────────


def test_validate_schema_returns_empty_when_all_columns_present():
    assert validate_schema(_good_df(), REQUIRED_COLUMNS) == []


def test_validate_schema_detects_missing_columns():
    df = _good_df().drop(columns=["weathersit"])
    assert validate_schema(df, REQUIRED_COLUMNS) == ["weathersit"]


# ── validate_nulls ────────────────────────────────────────────────────────


def test_validate_nulls_returns_empty_when_no_nulls():
    assert validate_nulls(_good_df(), REQUIRED_COLUMNS) == {}


def test_validate_nulls_detects_nulls():
    df = _good_df()
    df.loc[0, "temp"] = None
    assert validate_nulls(df, REQUIRED_COLUMNS) == {"temp": 1}


# ── validate_ranges ────────────────────────────────────────────────────────


def test_validate_ranges_returns_empty_when_in_bounds():
    assert validate_ranges(_good_df(), RANGES) == {}


def test_validate_ranges_detects_out_of_bounds():
    df = _good_df()
    df.loc[0, "hr"] = 30  # invalid, > 23
    assert validate_ranges(df, RANGES) == {"hr": 1}


# ── validate_cnt_invariant ─────────────────────────────────────────────────


def test_validate_cnt_invariant_returns_zero_when_consistent():
    assert validate_cnt_invariant(_good_df()) == 0


def test_validate_cnt_invariant_detects_mismatch():
    df = _good_df()
    df.loc[0, "cnt"] = 999  # doesn't match casual + registered
    assert validate_cnt_invariant(df) == 1


# ── validate_data_quality (runs all checks together) ──────────────────────


def test_validate_data_quality_returns_empty_when_all_valid():
    assert validate_data_quality(_good_df(), REQUIRED_COLUMNS, RANGES) == []


def test_validate_data_quality_reports_each_problem_found():
    df = _good_df()
    df.loc[0, "hr"] = 30
    df.loc[1, "temp"] = None

    issues = validate_data_quality(df, REQUIRED_COLUMNS, RANGES)

    assert len(issues) == 2
    assert any("hr" in issue for issue in issues)
    assert any("temp" in issue for issue in issues)
