import logging
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_lag_features(
    df: pd.DataFrame, lags: list[int], rolling_windows: list[int]
) -> pd.DataFrame:
    """
    Compute lag and rolling average features from the target column `cnt`.

    Lags capture past demand at specific time offsets (e.g. same hour
    yesterday, same hour last week). Rolling averages smooth short-term
    noise into a trend signal.

    All features use shift(1) as the minimum offset to prevent the current
    hour's demand from leaking into its own predictors.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset sorted chronologically, with a `datetime` and `cnt` column.
    lags : list[int]
        Hourly offsets for point-in-time lag features.
    rolling_windows : list[int]
        Window sizes (in hours) for rolling mean features.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with lag and rolling features appended.
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    for lag in lags:
        df[f"cnt_lag_{lag}"] = df["cnt"].shift(lag)
        logger.debug(f"Created cnt_lag_{lag}")

    for window in rolling_windows:
        df[f"cnt_rolling_mean_{window}"] = df["cnt"].shift(1).rolling(window).mean()
        logger.debug(f"Created cnt_rolling_mean_{window}")

    return df


def build_calendar_features(
    df: pd.DataFrame, drop_cols: list[str], min_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Engineer calendar and interaction features from raw time columns.

    Transformations applied:
    - Cyclic encoding of hour and month (sin/cos) to preserve circular continuity.
    - Regime separation: hr_workday and hr_weekend encode the commuter vs
      recreational demand patterns identified in EDA.
    - Season interaction: hr_x_season captures volume shifts by season at
      each hour of the day.
    - is_rush_hour: explicit flag for commuter peak hours on working days
      (7-9 AM and 17-19 PM). Demand is ~2.7x higher in these windows.
    - days_since_start: continuous growth trend proxy, replacing the binary
      yr feature (MI 0.15 vs 0.04). Computed relative to min_date of the
      full dataset to ensure consistency across train and val.
    - Drop redundant columns (e.g. atemp, yr, low-MI features).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset split (train or val) after lag features have been computed.
    drop_cols : list[str]
        Columns to remove before returning (redundant or low-MI features).
    min_date : pd.Timestamp
        Earliest date in the full dataset. Used to compute days_since_start
        consistently across train and val splits.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with new features and redundant columns removed.
    """
    df = df.copy()

    # Cyclic encoding
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)

    # Regime separation
    df["hr_workday"] = df["hr"] * df["workingday"]
    df["hr_weekend"] = df["hr"] * (1 - df["workingday"])

    # Season interaction
    df["hr_x_season"] = df["hr"] * df["season"]

    # Rush hour flag: commuter peaks on working days
    df["is_rush_hour"] = (
        (df["hr"].between(7, 9) | df["hr"].between(17, 19)) & (df["workingday"] == 1)
    ).astype(int)

    # Continuous growth trend — relative to full dataset min_date
    df["days_since_start"] = (df["dteday"] - min_date).dt.days

    # Drop redundant features
    existing = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing, inplace=True)
    logger.debug(f"Dropped columns: {existing}")

    return df


def build_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Add the model target column to the DataFrame.

    Supports two modes:
    - 'cnt'     : raw demand, suitable when absolute error matters.
    - 'log_cnt' : log(cnt + 1), reduces right skew and aligns with RMSLE.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a `cnt` column.
    target : str
        One of 'cnt' or 'log_cnt'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a `target` column appended.
    """
    if target == "log_cnt":
        df["target"] = np.log1p(df["cnt"])
    elif target == "cnt":
        df["target"] = df["cnt"]
    else:
        raise ValueError(f"Unknown target '{target}'. Expected 'cnt' or 'log_cnt'.")
    return df


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir = Path(cfg.paths.raw_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading raw data")
    df = pd.read_csv(raw_dir / cfg.paths.input_file)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")

    # ── Lag features (on full df before split) ────────────────────────────────
    logger.info("Building lag features")
    df = build_lag_features(
        df,
        lags=list(cfg.features.lags),
        rolling_windows=list(cfg.features.rolling_windows),
    )

    # ── Train / val split ─────────────────────────────────────────────────────
    cutoff = df["dteday"].quantile(cfg.features.split_ratio)
    train = df[df["dteday"] <= cutoff].copy()
    val = df[df["dteday"] > cutoff].copy()
    logger.info(f"Train: {len(train):,} rows | Val: {len(val):,} rows")

    # ── Calendar features ─────────────────────────────────────────────────────────
    logger.info("Building calendar features")
    drop_cols = list(cfg.features.drop_cols)
    min_date = df["dteday"].min()
    train = build_calendar_features(train, drop_cols, min_date)
    val = build_calendar_features(val, drop_cols, min_date)

    # ── Targets ───────────────────────────────────────────────────────────────────
    train["log_cnt"] = np.log1p(train["cnt"])
    train["log_registered"] = np.log1p(train["registered"])
    train["log_casual"] = np.log1p(train["casual"])

    val["log_cnt"] = np.log1p(val["cnt"])
    val["log_registered"] = np.log1p(val["registered"])
    val["log_casual"] = np.log1p(val["casual"])

    # ── Drop NaN rows from train only ─────────────────────────────────────────
    before = len(train)
    train = train.dropna()
    logger.info(f"Dropped {before - len(train):,} NaN rows from train (lag warmup)")

    # ── Save ──────────────────────────────────────────────────────────────────
    train.to_csv(processed_dir / "train.csv", index=False)
    val.to_csv(processed_dir / "val.csv", index=False)
    logger.info(f"Saved processed data to {processed_dir}")


if __name__ == "__main__":
    main()
