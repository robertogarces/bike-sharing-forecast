import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """
    Check that all required columns are present.

    Returns
    -------
    list[str]
        Missing column names, empty if the schema is intact.
    """
    return [col for col in required_columns if col not in df.columns]


def validate_nulls(df: pd.DataFrame, required_columns: list[str]) -> dict[str, int]:
    """
    Check for null values in the required columns.

    Only checks columns that are actually present — a missing column is
    already reported by validate_schema, not double-counted here.

    Returns
    -------
    dict[str, int]
        {column: null_count} for columns with at least one null value.
    """
    present = [col for col in required_columns if col in df.columns]
    null_counts = df[present].isnull().sum()
    return {col: int(count) for col, count in null_counts.items() if count > 0}


def validate_ranges(df: pd.DataFrame, ranges: dict[str, list[float]]) -> dict[str, int]:
    """
    Check that column values fall within their expected [min, max] range.

    Columns not present in df are skipped — validate_schema already
    reports a missing column, so it isn't double-counted here.

    Returns
    -------
    dict[str, int]
        {column: count of out-of-range rows} for columns with at least one
        violation.
    """
    violations = {}
    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < low) | (df[col] > high)).sum()
        if out_of_range > 0:
            violations[col] = int(out_of_range)
    return violations


def validate_cnt_invariant(df: pd.DataFrame) -> int:
    """
    Check that cnt == casual + registered holds for every row — the raw
    dataset's own internal consistency rule, unrelated to any modeling
    assumption.

    Returns
    -------
    int
        Number of rows where the invariant doesn't hold. 0 if the columns
        involved aren't present (already reported by validate_schema).
    """
    if not {"cnt", "casual", "registered"}.issubset(df.columns):
        return 0
    return int((df["cnt"] != df["casual"] + df["registered"]).sum())


def validate_data_quality(
    df: pd.DataFrame,
    required_columns: list[str],
    ranges: dict[str, list[float]],
) -> list[str]:
    """
    Run all data quality checks and collect human-readable issue
    descriptions.

    Meant to run on hour_past.csv right before a retrain — automatically
    retraining when "drift" is actually a data problem (a broken sensor,
    a schema change) would train the new model on bad data, making things
    worse rather than better.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data to validate.
    required_columns : list[str]
        Columns that must be present.
    ranges : dict[str, list[float]]
        {column: [min, max]} plausible bounds, checked for columns that
        are present.

    Returns
    -------
    list[str]
        One description per problem found. Empty list means the data
        passed every check.
    """
    issues = []

    missing = validate_schema(df, required_columns)
    if missing:
        issues.append(f"Missing columns: {missing}")

    nulls = validate_nulls(df, required_columns)
    for col, count in nulls.items():
        issues.append(f"{count} null value(s) in '{col}'")

    range_violations = validate_ranges(df, ranges)
    for col, count in range_violations.items():
        low, high = ranges[col]
        issues.append(f"{count} row(s) with '{col}' outside [{low}, {high}]")

    cnt_violations = validate_cnt_invariant(df)
    if cnt_violations:
        issues.append(f"{cnt_violations} row(s) where cnt != casual + registered")

    return issues
