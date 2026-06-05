import json
import logging
from pathlib import Path
from datetime import datetime

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bike_sharing.features.build_features import build_lag_features, build_calendar_features
from bike_sharing.models.train import FEATURES

logger = logging.getLogger(__name__)

STATE_PATH   = Path("data/simulation_state.json")
PRED_PATH    = Path("data/predictions.csv")


def load_simulation_state() -> dict:
    """
    Load the simulation state from disk.
    Raises if the simulation has not been initialized.
    """
    if not STATE_PATH.exists():
        raise RuntimeError(
            "No simulation state found. "
            "Run shift_dates.py first to initialize the simulation."
        )
    with open(STATE_PATH) as f:
        return json.load(f)


def build_next_hour_features(
    past: pd.DataFrame,
    min_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build the feature vector for the next hour prediction.

    Takes the full past dataset, applies the production feature pipeline,
    and returns a single-row DataFrame representing the next hour after
    the last available record.

    Parameters
    ----------
    past : pd.DataFrame
        Full past dataset sorted chronologically.
    min_date : pd.Timestamp
        Earliest date in the full dataset for days_since_start computation.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all features for the next hour.
    """
    past = past.copy()
    past["dteday"]   = pd.to_datetime(past["dteday"])
    past["datetime"] = past["dteday"] + pd.to_timedelta(past["hr"], unit="h")

    # Build lag features on full past history
    past = build_lag_features(
        past,
        lags=[1, 2, 3, 8, 24, 48, 72, 168],
        rolling_windows=[24, 168]
    )

    # Build calendar features
    past = build_calendar_features(past, drop_cols=["atemp", "yr"], min_date=min_date)

    # Get last available row — this is "current hour"
    current = past.iloc[-1].copy()

    # Next hour datetime
    next_dt  = current["datetime"] + pd.Timedelta(hours=1)
    next_hr  = next_dt.hour
    next_day = next_dt.normalize()

    # Build next hour row by shifting lag features
    next_row = current.copy()
    next_row["datetime"] = next_dt
    next_row["dteday"]   = next_day
    next_row["hr"]       = next_hr

    # Shift lags — lag_1 of next hour = cnt of current hour, etc.
    next_row["cnt_lag_1"]  = current["cnt"]
    next_row["cnt_lag_2"]  = current["cnt_lag_1"]
    next_row["cnt_lag_3"]  = current["cnt_lag_2"]
    next_row["cnt_lag_8"]  = past.iloc[-8]["cnt"]  if len(past) >= 8  else np.nan
    next_row["cnt_lag_24"] = past.iloc[-24]["cnt"] if len(past) >= 24 else np.nan
    next_row["cnt_lag_48"] = past.iloc[-48]["cnt"] if len(past) >= 48 else np.nan
    next_row["cnt_lag_72"] = past.iloc[-72]["cnt"] if len(past) >= 72 else np.nan
    next_row["cnt_lag_168"]= past.iloc[-168]["cnt"] if len(past) >= 168 else np.nan

    next_row["cnt_rolling_mean_24"]  = past["cnt"].iloc[-24:].mean()
    next_row["cnt_rolling_mean_168"] = past["cnt"].iloc[-168:].mean()

    # Update cyclic and calendar features for next hour
    next_row["hr_sin"]          = np.sin(2 * np.pi * next_hr / 24)
    next_row["hr_cos"]          = np.cos(2 * np.pi * next_hr / 24)
    next_row["hr_workday"]      = next_hr * current["workingday"]
    next_row["hr_weekend"]      = next_hr * (1 - current["workingday"])
    next_row["hr_x_season"]     = next_hr * current["season"]
    next_row["is_rush_hour"]    = int(
        (7 <= next_hr <= 9 or 17 <= next_hr <= 19) and current["workingday"] == 1
    )
    next_row["days_since_start"] = (next_day - min_date).days

    return pd.DataFrame([next_row])


def append_prediction(pred_row: dict) -> None:
    """
    Append a prediction record to data/predictions.csv.
    Creates the file with headers if it doesn't exist.
    """
    df_new = pd.DataFrame([pred_row])
    if PRED_PATH.exists():
        df_new.to_csv(PRED_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(PRED_PATH, index=False)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir       = Path(cfg.dataset.raw_dir)
    artifacts_dir = Path("artifacts")

    # ── Load simulation state ─────────────────────────────────────────────────
    state    = load_simulation_state()
    min_date = pd.Timestamp(state["future_start_date"]) - pd.DateOffset(years=2)

    # ── Load past data ────────────────────────────────────────────────────────
    logger.info("Loading past data")
    past = pd.read_csv(raw_dir / "hour_past.csv")
    past["dteday"] = pd.to_datetime(past["dteday"])

    last_record = past.sort_values("dteday").iloc[-1]
    current_dt  = pd.to_datetime(last_record["dteday"]) + pd.Timedelta(hours=int(last_record["hr"]))
    next_dt     = current_dt + pd.Timedelta(hours=1)

    logger.info(f"Current hour: {current_dt} | Predicting: {next_dt}")

    # ── Load models ───────────────────────────────────────────────────────────
    model_registered = lgb.Booster(model_file=str(artifacts_dir / "lgbm_registered.txt"))
    model_casual     = lgb.Booster(model_file=str(artifacts_dir / "lgbm_casual.txt"))

    # ── Build features ────────────────────────────────────────────────────────
    logger.info("Building features for next hour")
    min_date  = past["dteday"].min()
    next_row  = build_next_hour_features(past, min_date)
    X         = next_row[FEATURES]

    # ── Predict ───────────────────────────────────────────────────────────────
    pred_registered = float(np.expm1(model_registered.predict(X))[0])
    pred_casual     = float(np.expm1(model_casual.predict(X))[0])
    pred_total      = max(0, pred_registered + pred_casual)

    logger.info(
        f"Prediction for {next_dt} — "
        f"registered: {pred_registered:.0f} | "
        f"casual: {pred_casual:.0f} | "
        f"total: {pred_total:.0f}"
    )

    # ── Save prediction ───────────────────────────────────────────────────────
    pred_record = {
        "predicted_at":       datetime.now().isoformat(),
        "timestamp_predicted": next_dt.isoformat(),
        "hr":                 int(next_row["hr"].values[0]),
        "temp":               float(next_row["temp"].values[0]),
        "hum":                float(next_row["hum"].values[0]),
        "weathersit":         int(next_row["weathersit"].values[0]),
        "workingday":         int(last_record["workingday"]),
        "pred_registered":    round(pred_registered, 2),
        "pred_casual":        round(pred_casual, 2),
        "pred_total":         round(pred_total, 2),
    }

    append_prediction(pred_record)
    logger.info(f"Prediction saved to {PRED_PATH}")


if __name__ == "__main__":
    main()