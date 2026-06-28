"""
One-off maintenance script to regenerate the live-era predictions series.

The hourly prediction pipeline accumulated corrupted rows while a bug caused
the main prediction to reuse the last backfilled hour's features (wrong hr and
duplicated values), plus skipped hours from an earlier backfill off-by-one.

The corruption only affects the live era (predictions produced by the hourly
workflow from the simulation reference_date onward). An older one-off backfill
block predates it and is left untouched.

For every past hour in the live era this script builds the feature vector as a
genuine 1-step-ahead forecast (using only data strictly before the target hour)
and predicts with the current @production models, producing a clean, gap-free
series.

Note: predictions are regenerated with today's @production model, not the model
version that was live at each historical hour. This produces a coherent series,
not a literal replay of past inference.
"""

import json
import logging
from pathlib import Path

import hydra
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from bike_sharing.models.predict import build_next_hour_features
from bike_sharing.models.train import FEATURES

logger = logging.getLogger(__name__)

MIN_HISTORY = 168  # hours of history required for the longest lag/rolling window


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir    = Path(cfg.paths.raw_dir)
    pred_path  = Path(cfg.paths.predictions_path)
    state_path = Path(cfg.paths.simulation_state)

    # ── Load past data ────────────────────────────────────────────────────────
    logger.info("Loading past data")
    past = pd.read_csv(raw_dir / cfg.paths.input_file)
    past["dteday"]   = pd.to_datetime(past["dteday"])
    past["datetime"] = past["dteday"] + pd.to_timedelta(past["hr"], unit="h")
    past = past.sort_values("datetime").reset_index(drop=True)
    min_date = past["dteday"].min()

    # ── Live-era cutoff (simulation reference date) ───────────────────────────
    with open(state_path) as f:
        state = json.load(f)
    cutoff = pd.Timestamp(state["reference_date"])
    logger.info(f"Live-era cutoff (reference_date): {cutoff}")

    # ── Keep older predictions untouched ──────────────────────────────────────
    existing = pd.read_csv(pred_path)
    existing["_ts"] = pd.to_datetime(existing["timestamp_predicted"])
    old_rows = existing[existing["_ts"] < cutoff].drop(columns="_ts")
    logger.info(f"Keeping {len(old_rows)} existing rows before cutoff (untouched)")

    # ── Load models from MLflow registry ─────────────────────────────────────
    logger.info("Loading models from MLflow registry")
    model_registered = mlflow.lightgbm.load_model(f"models:/{cfg.project}-registered@production")
    model_casual     = mlflow.lightgbm.load_model(f"models:/{cfg.project}-casual@production")

    # ── Regenerate every past hour in the live era ────────────────────────────
    end = past["datetime"].max()
    target_hours = past[(past["datetime"] >= cutoff) & (past["datetime"] <= end)]["datetime"].tolist()
    logger.info(f"Regenerating {len(target_hours)} live-era hours ({cutoff} → {end})")

    records = []
    for target_dt in target_hours:
        past_slice = past[past["datetime"] < target_dt]
        if len(past_slice) < MIN_HISTORY:
            continue

        next_row = build_next_hour_features(past_slice, min_date)
        X        = next_row[FEATURES]

        pred_registered = float(np.expm1(model_registered.predict(X))[0])
        pred_casual     = float(np.expm1(model_casual.predict(X))[0])
        pred_total      = max(0, pred_registered + pred_casual)

        actual_row = past[past["datetime"] == target_dt].iloc[0]

        records.append({
            "predicted_at":        (target_dt - pd.Timedelta(hours=1)).isoformat(),
            "timestamp_predicted": target_dt.isoformat(),
            "hr":                  int(actual_row["hr"]),
            "temp":                float(actual_row["temp"]),
            "hum":                 float(actual_row["hum"]),
            "weathersit":          int(actual_row["weathersit"]),
            "workingday":          int(actual_row["workingday"]),
            "pred_registered":     round(pred_registered, 2),
            "pred_casual":         round(pred_casual, 2),
            "pred_total":          round(pred_total, 2),
        })

    # ── Combine old block + clean live era ────────────────────────────────────
    regenerated = pd.DataFrame(records)
    out = pd.concat([old_rows, regenerated], ignore_index=True)
    out.to_csv(pred_path, index=False)
    logger.info(f"Wrote {len(out)} predictions ({len(old_rows)} kept + {len(regenerated)} regenerated) to {pred_path}")


if __name__ == "__main__":
    main()
