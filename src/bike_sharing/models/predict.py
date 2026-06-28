import json
import logging
from pathlib import Path
from datetime import datetime

import hydra
import lightgbm as lgb
import numpy as np
import mlflow
import pandas as pd
from omegaconf import DictConfig

from bike_sharing.features.build_features import build_lag_features, build_calendar_features
from bike_sharing.models.train import FEATURES

logger = logging.getLogger(__name__)


def load_simulation_state(state_path: Path) -> dict:
    """
    Load the simulation state from disk.
    Raises if the simulation has not been initialized.
    """
    if not state_path.exists():
        raise RuntimeError(
            "No simulation state found. "
            "Run shift_dates.py first to initialize the simulation."
        )
    with open(state_path) as f:
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
    next_row["cnt_lag_1"]   = current["cnt"]
    next_row["cnt_lag_2"]   = current["cnt_lag_1"]
    next_row["cnt_lag_3"]   = current["cnt_lag_2"]
    next_row["cnt_lag_8"]   = past.iloc[-8]["cnt"]   if len(past) >= 8   else np.nan
    next_row["cnt_lag_24"]  = past.iloc[-24]["cnt"]  if len(past) >= 24  else np.nan
    next_row["cnt_lag_48"]  = past.iloc[-48]["cnt"]  if len(past) >= 48  else np.nan
    next_row["cnt_lag_72"]  = past.iloc[-72]["cnt"]  if len(past) >= 72  else np.nan
    next_row["cnt_lag_168"] = past.iloc[-168]["cnt"] if len(past) >= 168 else np.nan

    next_row["cnt_rolling_mean_24"]  = past["cnt"].iloc[-24:].mean()
    next_row["cnt_rolling_mean_168"] = past["cnt"].iloc[-168:].mean()

    # Update cyclic and calendar features for next hour
    next_row["hr_sin"]           = np.sin(2 * np.pi * next_hr / 24)
    next_row["hr_cos"]           = np.cos(2 * np.pi * next_hr / 24)
    next_row["hr_workday"]       = next_hr * current["workingday"]
    next_row["hr_weekend"]       = next_hr * (1 - current["workingday"])
    next_row["hr_x_season"]      = next_hr * current["season"]
    next_row["is_rush_hour"]     = int(
        (7 <= next_hr <= 9 or 17 <= next_hr <= 19) and current["workingday"] == 1
    )
    next_row["days_since_start"] = (next_day - min_date).days

    return pd.DataFrame([next_row])


def append_prediction(pred_row: dict, pred_path: Path) -> None:
    """
    Append a prediction record to predictions.csv.
    Skips if a prediction for that hour already exists.
    Creates the file with headers if it doesn't exist.
    """
    df_new = pd.DataFrame([pred_row])

    if pred_path.exists():
        existing = pd.read_csv(pred_path)
        if pred_row["timestamp_predicted"] in existing["timestamp_predicted"].values:
            logger.warning(
                f"Prediction for {pred_row['timestamp_predicted']} already exists — skipping."
            )
            return
        df_new.to_csv(pred_path, mode="a", header=False, index=False)
    else:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(pred_path, index=False)


def get_missing_hours(
    past: pd.DataFrame,
    pred_path: Path,
    max_backfill_hours: int = 48,
) -> list[pd.Timestamp]:
    """
    Identify hours in hour_past.csv that have no corresponding prediction
    in predictions.csv, starting from the last existing prediction.

    If predictions.csv does not exist or is empty, returns an empty list —
    backfill only covers gaps in an already-running system, not cold starts.

    Backfill is capped at max_backfill_hours to avoid filling years of
    history if the system was down for a long time.

    Parameters
    ----------
    past : pd.DataFrame
        Full past dataset sorted chronologically.
    pred_path : Path
        Path to predictions.csv.
    max_backfill_hours : int
        Maximum number of hours to backfill (default: 48).

    Returns
    -------
    list[pd.Timestamp]
        Sorted list of datetimes missing from predictions.csv.
    """
    if not pred_path.exists():
        logger.info("No existing predictions — skipping backfill (cold start)")
        return []

    existing = pd.read_csv(pred_path)
    if existing.empty:
        logger.info("No existing predictions — skipping backfill (cold start)")
        return []

    past = past.copy()
    past["dteday"]   = pd.to_datetime(past["dteday"])
    past["datetime"] = past["dteday"] + pd.to_timedelta(past["hr"], unit="h")

    # Only look for gaps after the last existing prediction
    last_predicted = pd.to_datetime(existing["timestamp_predicted"]).max()
    predicted_hours = set(pd.to_datetime(existing["timestamp_predicted"]).tolist())

    # Past hours after last prediction, excluding the last record (current hour)
    candidate_hours = past[past["datetime"] > last_predicted]["datetime"].tolist()
    candidate_hours = sorted(candidate_hours)

    missing = [h for h in candidate_hours if h not in predicted_hours]

    if len(missing) > max_backfill_hours:
        logger.warning(
            f"Gap of {len(missing)} hours detected — "
            f"capping backfill at {max_backfill_hours} hours. "
            f"Older gaps will not be filled."
        )
        missing = missing[-max_backfill_hours:]

    return missing


def run(cfg: DictConfig) -> None:
    raw_dir       = Path(cfg.paths.raw_dir)
    state_path    = Path(cfg.paths.simulation_state)
    pred_path     = Path(cfg.paths.predictions_path)

    # ── Load simulation state ─────────────────────────────────────────────────
    load_simulation_state(state_path)

    # ── Load past data ────────────────────────────────────────────────────────
    logger.info("Loading past data")
    past = pd.read_csv(raw_dir / cfg.paths.input_file)
    past["dteday"] = pd.to_datetime(past["dteday"])

    last_record = past.sort_values(["dteday", "hr"]).iloc[-1]
    current_dt  = pd.to_datetime(last_record["dteday"]) + pd.Timedelta(hours=int(last_record["hr"]))
    next_dt     = current_dt + pd.Timedelta(hours=1)

    logger.info(f"Current hour: {current_dt} | Predicting: {next_dt}")

    # ── Validate data freshness ───────────────────────────────────────────────
    now      = datetime.now()
    data_lag = now - current_dt.to_pydatetime()

    if data_lag > pd.Timedelta(hours=2):
        logger.warning(
            f"Data lag detected: last record is {data_lag} behind current time. "
            f"Run update_simulation.py first."
        )
    else:
        logger.info(f"Data freshness OK — lag: {data_lag}")

    # ── Load models from MLflow registry ─────────────────────────────────────
    model_registered = mlflow.lightgbm.load_model(
        f"models:/{cfg.project}-registered@production"
    )
    model_casual = mlflow.lightgbm.load_model(
        f"models:/{cfg.project}-casual@production"
    )

    # ── Build features ────────────────────────────────────────────────────────
    logger.info("Building features for next hour")
    min_date = past["dteday"].min()
    next_row = build_next_hour_features(past, min_date)
    X        = next_row[FEATURES]

    # ── Backfill missing predictions ──────────────────────────────────────────
    missing_hours = get_missing_hours(past, pred_path, cfg.monitoring.max_backfill_hours)

    if missing_hours:
        logger.info(f"Backfilling {len(missing_hours)} missing predictions")

        for target_dt in missing_hours:
            # Build past slice up to the hour before target_dt
            past["dteday"]   = pd.to_datetime(past["dteday"])
            past["datetime"] = past["dteday"] + pd.to_timedelta(past["hr"], unit="h")
            past_slice = past[past["datetime"] < target_dt].copy()

            if len(past_slice) < 168:
                logger.warning(f"Not enough history for backfill at {target_dt} — skipping")
                continue

            next_row_bf = build_next_hour_features(past_slice, min_date)
            X_bf        = next_row_bf[FEATURES]

            pred_registered_bf = float(np.expm1(model_registered.predict(X_bf))[0])
            pred_casual_bf     = float(np.expm1(model_casual.predict(X_bf))[0])
            pred_total_bf      = max(0, pred_registered_bf + pred_casual_bf)

            actual_row = past[past["datetime"] == target_dt].iloc[0]

            pred_record = {
                "predicted_at":        datetime.now().isoformat(),
                "timestamp_predicted": target_dt.isoformat(),
                "hr":                  int(actual_row["hr"]),
                "temp":                float(actual_row["temp"]),
                "hum":                 float(actual_row["hum"]),
                "weathersit":          int(actual_row["weathersit"]),
                "workingday":          int(actual_row["workingday"]),
                "pred_registered":     round(pred_registered_bf, 2),
                "pred_casual":         round(pred_casual_bf, 2),
                "pred_total":          round(pred_total_bf, 2),
            }
            append_prediction(pred_record, pred_path)
            logger.info(f"Backfilled prediction for {target_dt}")
            
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
        "predicted_at":        datetime.now().isoformat(),
        "timestamp_predicted": next_dt.isoformat(),
        "hr":                  int(next_row["hr"].values[0]),
        "temp":                float(next_row["temp"].values[0]),
        "hum":                 float(next_row["hum"].values[0]),
        "weathersit":          int(next_row["weathersit"].values[0]),
        "workingday":          int(last_record["workingday"]),
        "pred_registered":     round(pred_registered, 2),
        "pred_casual":         round(pred_casual, 2),
        "pred_total":          round(pred_total, 2),
    }

    append_prediction(pred_record, pred_path)
    logger.info(f"Prediction saved to {pred_path}")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()