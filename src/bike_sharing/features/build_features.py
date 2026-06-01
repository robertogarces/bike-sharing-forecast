import logging
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_lag_features(df: pd.DataFrame, lags: list[int], rolling_windows: list[int]) -> pd.DataFrame:
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


def build_calendar_features(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    """
    Engineer calendar and interaction features from raw time columns.

    Transformations applied:
    - Cyclic encoding of hour and month (sin/cos) to preserve circular continuity.
    - Regime separation: hr_workday and hr_weekend encode the commuter vs
      recreational demand patterns identified in EDA.
    - Season interaction: hr_x_season captures volume shifts by season at
      each hour of the day.
    - Drop redundant columns (e.g. atemp, which is r≈0.99 correlated with temp).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset split (train or val) after lag features have been computed.
    drop_cols : list[str]
        Columns to remove before returning (redundant or low-MI features).

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with new features and redundant columns removed.
    """
    df = df.copy()

    # Cyclic encoding
    df["hr_sin"]   = np.sin(2 * np.pi * df["hr"]   / 24)
    df["hr_cos"]   = np.cos(2 * np.pi * df["hr"]   / 24)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)

    # Regime separation
    df["hr_workday"] = df["hr"] * df["workingday"]
    df["hr_weekend"] = df["hr"] * (1 - df["workingday"])

    # Season interaction
    df["hr_x_season"] = df["hr"] * df["season"]

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
    raw_dir       = Path(cfg.dataset.raw_dir)
    processed_dir = Path(cfg.dataset.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading raw data")
    df = pd.read_csv(raw_dir / "hour.csv")
    df["dteday"]   = pd.to_datetime(df["dteday"])
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
    train  = df[df["dteday"] <= cutoff].copy()
    val    = df[df["dteday"] >  cutoff].copy()
    logger.info(f"Train: {len(train):,} rows | Val: {len(val):,} rows")

    # ── Calendar features ─────────────────────────────────────────────────────
    logger.info("Building calendar features")
    drop_cols = list(cfg.features.drop_cols)
    train = build_calendar_features(train, drop_cols)
    val   = build_calendar_features(val,   drop_cols)

    # ── Target ────────────────────────────────────────────────────────────────
    train = build_target(train, cfg.features.target)
    val   = build_target(val,   cfg.features.target)

    # ── Drop NaN rows from train only ─────────────────────────────────────────
    before = len(train)
    train  = train.dropna()
    logger.info(f"Dropped {before - len(train):,} NaN rows from train (lag warmup)")

    # ── Save ──────────────────────────────────────────────────────────────────
    train.to_csv(processed_dir / "train.csv", index=False)
    val.to_csv(processed_dir / "val.csv",     index=False)
    logger.info(f"Saved processed data to {processed_dir}")


if __name__ == "__main__":
    main()