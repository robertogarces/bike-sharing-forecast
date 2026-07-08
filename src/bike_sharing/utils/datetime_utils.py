from datetime import datetime, timezone

import pandas as pd


def reconstruct_datetime(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    """
    Reconstruct the real timestamp from dteday + hr and store it in `column`.

    dteday is parsed to datetime first — idempotent if it's already parsed
    (pd.to_datetime on an existing datetime column is a no-op), so this is
    safe to call regardless of whether the caller already did that.
    """
    df["dteday"] = pd.to_datetime(df["dteday"])
    df[column] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df


def utc_now() -> datetime:
    """
    Current instant as a naive datetime, explicitly UTC regardless of the
    machine's local timezone. GitHub Actions runners default to UTC, but
    nothing in the simulation clock should silently depend on that.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
