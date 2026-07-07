import json
import logging
from pathlib import Path
from datetime import datetime

import hydra
import numpy as np
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

from bike_sharing.features.build_features import build_lag_features, build_calendar_features
from bike_sharing.models.train import FEATURES
from bike_sharing.utils.datetime_utils import reconstruct_datetime
from bike_sharing.utils.simulation_utils import load_simulation_state

logger = logging.getLogger(__name__)

# Calendar is deterministically knowable in advance (not leakage); weather is not.
CALENDAR_COLS = ["season", "yr", "mnth", "holiday", "weekday", "workingday"]
WEATHER_COLS = ["temp", "atemp", "hum", "windspeed"]


def build_synthetic_row(
    current: pd.Series,
    target_dt: pd.Timestamp,
    calendar_lookup: pd.DataFrame | None = None,
    weather_lookup: pd.DataFrame | None = None,
) -> dict:
    """
    Build the raw (untransformed) row for target_dt, seeded from `current`
    (the last known or already-predicted hour) with calendar and weather
    corrected wherever a real value is available.

    Calendar (season/yr/mnth/holiday/weekday/workingday) is deterministically
    knowable in advance, so it's read from calendar_lookup — a DataFrame
    indexed by datetime spanning past + future — whenever target_dt is
    covered. This is not leakage; it fixes the stale-calendar bug where a
    trajectory crossing midnight would otherwise keep the origin's
    weekday/workingday (e.g. a rollout from Friday 23:00 into Saturday would
    wrongly keep treating Saturday as a working day).

    Weather (temp/atemp/hum/windspeed) is genuinely unknown ahead of time —
    reading the real future weather would be leakage. It's approximated from
    `target_dt - 24h` (same hour, previous day) in weather_lookup, which
    captures the daily cycle; this recovers most of the accuracy lost to a
    naive approximation (see notebooks/05_multi_horizon_forecasting.ipynb).
    Falls back to `current`'s own weather when that lag isn't available
    (horizons beyond 24h, or weather_lookup not given).

    Both lookups are optional — with neither given, this reduces to
    "carry the last row's calendar and weather forward untouched", the
    historical single-step behavior.

    Returns
    -------
    dict
        Raw row for target_dt with cnt/registered/casual set to NaN (not
        yet known) — the caller either predicts them (live rollout) or
        fills them from a fallback source.
    """
    row = current.to_dict()
    row["datetime"] = target_dt
    row["dteday"] = target_dt.normalize()
    row["hr"] = target_dt.hour
    row["cnt"] = np.nan
    row["casual"] = np.nan
    row["registered"] = np.nan

    if calendar_lookup is not None and target_dt in calendar_lookup.index:
        for c in CALENDAR_COLS:
            row[c] = calendar_lookup.loc[target_dt, c]

    if weather_lookup is not None:
        lag24_dt = target_dt - pd.Timedelta(hours=24)
        if lag24_dt in weather_lookup.index:
            for c in WEATHER_COLS:
                row[c] = weather_lookup.loc[lag24_dt, c]

    return row


def _features_for_synthetic_row(
    past: pd.DataFrame,
    raw_row: dict,
    min_date: pd.Timestamp,
    lags: list[int],
    rolling_windows: list[int],
    drop_cols: list[str],
) -> pd.DataFrame:
    """
    Feature row for a synthetic raw_row, computed with the real training
    pipeline (build_lag_features + build_calendar_features) — not a
    hand-written reimplementation that could silently diverge from training
    (e.g. a new lag added there but forgotten here).
    """
    extended = pd.concat([past, pd.DataFrame([raw_row])], ignore_index=True)
    extended = build_lag_features(extended, lags=lags, rolling_windows=rolling_windows)
    extended = build_calendar_features(extended, drop_cols=drop_cols, min_date=min_date)
    return extended.iloc[[-1]]


def build_next_hour_features(
    past: pd.DataFrame,
    min_date: pd.Timestamp,
    lags: list[int],
    rolling_windows: list[int],
    drop_cols: list[str],
    calendar_lookup: pd.DataFrame | None = None,
    weather_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the feature vector for the next hour prediction (a single h+1 step).

    Appends a synthetic row for the next hour and runs it through the same
    feature pipeline used for training. See build_synthetic_row for how its
    calendar and weather are filled in.

    Parameters
    ----------
    past : pd.DataFrame
        Full past dataset sorted chronologically.
    min_date : pd.Timestamp
        Earliest date in the full dataset for days_since_start computation.
    lags : list[int]
        Same lag config used for training.
    rolling_windows : list[int]
        Same rolling window config used for training.
    drop_cols : list[str]
        Same drop_cols config used for training.
    calendar_lookup, weather_lookup : pd.DataFrame | None
        See build_synthetic_row.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all features for the next hour.
    """
    past = past.copy()
    past = reconstruct_datetime(past)
    past = past.sort_values("datetime").reset_index(drop=True)

    current = past.iloc[-1]
    next_dt = current["datetime"] + pd.Timedelta(hours=1)
    raw_row = build_synthetic_row(current, next_dt, calendar_lookup, weather_lookup)

    return _features_for_synthetic_row(past, raw_row, min_date, lags, rolling_windows, drop_cols)


def predict_trajectory(
    past: pd.DataFrame,
    model_registered,
    model_casual,
    horizon: int,
    min_date: pd.Timestamp,
    lags: list[int],
    rolling_windows: list[int],
    drop_cols: list[str],
    calendar_lookup: pd.DataFrame,
    weather_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recursive rollout: predict h+1..h+horizon from the last row of `past`,
    reusing the same h+1 model at every step and feeding each prediction back
    as a synthetic actual so the next step's lag features use it. The horizon
    lives entirely here, in serving — the model itself is unmodified and
    agnostic to how far the rollout goes.

    Parameters
    ----------
    past : pd.DataFrame
        Full past dataset sorted chronologically — the last row is the
        origin the trajectory starts from.
    model_registered, model_casual
        Trained h+1 models (any object exposing .predict(X)).
    horizon : int
        Number of hours ahead to roll out (K).
    min_date, lags, rolling_windows, drop_cols
        Same feature config used for training.
    calendar_lookup, weather_lookup : pd.DataFrame
        See build_synthetic_row.

    Returns
    -------
    pd.DataFrame
        One row per lead time (horizon 1..K), with columns: origin,
        timestamp_predicted, horizon, hr, temp, hum, weathersit, workingday,
        pred_registered, pred_casual, pred_total.
    """
    work = past.copy()
    work = reconstruct_datetime(work)
    work = work.sort_values("datetime").reset_index(drop=True)
    origin_dt = work.iloc[-1]["datetime"]

    rows = []
    for k in range(1, horizon + 1):
        current = work.iloc[-1]
        target_dt = current["datetime"] + pd.Timedelta(hours=1)
        raw_row = build_synthetic_row(current, target_dt, calendar_lookup, weather_lookup)
        feature_row = _features_for_synthetic_row(
            work, raw_row, min_date, lags, rolling_windows, drop_cols
        )
        X = feature_row[FEATURES]

        pred_registered = float(np.clip(np.expm1(model_registered.predict(X))[0], 0, None))
        pred_casual = float(np.clip(np.expm1(model_casual.predict(X))[0], 0, None))
        pred_total = pred_registered + pred_casual

        rows.append(
            {
                "origin": origin_dt,
                "timestamp_predicted": target_dt,
                "horizon": k,
                "hr": int(raw_row["hr"]),
                "temp": float(raw_row["temp"]),
                "hum": float(raw_row["hum"]),
                "weathersit": int(raw_row["weathersit"]),
                "workingday": int(raw_row["workingday"]),
                "pred_registered": pred_registered,
                "pred_casual": pred_casual,
                "pred_total": pred_total,
            }
        )

        # Feed the prediction back as a synthetic actual for the next step's lags.
        raw_row["cnt"] = pred_total
        raw_row["registered"] = pred_registered
        raw_row["casual"] = pred_casual
        work = pd.concat([work, pd.DataFrame([raw_row])[work.columns]], ignore_index=True)

    return pd.DataFrame(rows)


def build_calendar_lookup(past: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar lookup spanning past + future, indexed by datetime.

    Calendar is deterministically knowable in advance (not leakage), so the
    rollout may read the true value for any future hour covered by
    hour_future.csv. Weather must never be sourced from `future` here — that
    would be real leakage — hence this only carries CALENDAR_COLS.
    """
    both = pd.concat([past, future], ignore_index=True) if len(future) else past
    return both.set_index("datetime")[CALENDAR_COLS]


def build_weather_lookup(past: pd.DataFrame) -> pd.DataFrame:
    """
    Weather lookup over past only, indexed by datetime — used for the lag24
    approximation (target_dt - 24h). Never built from `future`: unlike the
    calendar, weather is genuinely unknown ahead of time.
    """
    return past.set_index("datetime")[WEATHER_COLS]


def _trajectory_to_records(
    trajectory: pd.DataFrame,
    model_registered_version: str | None,
    model_casual_version: str | None,
    prediction_source: str = "model",
) -> list[dict]:
    """Convert a predict_trajectory() DataFrame into predictions.csv row dicts."""
    predicted_at = datetime.now().isoformat()
    records = []
    for _, row in trajectory.iterrows():
        records.append(
            {
                "predicted_at": predicted_at,
                "timestamp_predicted": row["timestamp_predicted"].isoformat(),
                "horizon": int(row["horizon"]),
                "hr": int(row["hr"]),
                "temp": float(row["temp"]),
                "hum": float(row["hum"]),
                "weathersit": int(row["weathersit"]),
                "workingday": int(row["workingday"]),
                "pred_registered": round(float(row["pred_registered"]), 2),
                "pred_casual": round(float(row["pred_casual"]), 2),
                "pred_total": round(float(row["pred_total"]), 2),
                "model_version_registered": model_registered_version,
                "model_version_casual": model_casual_version,
                "prediction_source": prediction_source,
            }
        )
    return records


def append_prediction(pred_row: dict, pred_path: Path) -> None:
    """
    Append a prediction record to predictions.csv.

    Dedup key is (timestamp_predicted, horizon) — a target hour gets one
    prediction per lead time, since each origin's trajectory predicts it at a
    different distance. Rows written before the multi-horizon change have no
    horizon column; they're normalized to horizon=1 (that's what they were —
    every run used to emit exactly one, next-hour, prediction).

    Reads the existing file and rewrites it with the new row concatenated,
    rather than a blind mode="a" append — pd.concat aligns columns by name
    and fills NaN for any column missing on either side, so the schema can
    evolve (e.g. a new column added later) without corrupting older rows.
    """
    pred_row = dict(pred_row)
    pred_row.setdefault("horizon", 1)
    df_new = pd.DataFrame([pred_row])

    if pred_path.exists():
        existing = pd.read_csv(pred_path)
        if "horizon" not in existing.columns:
            existing["horizon"] = 1
        else:
            existing["horizon"] = existing["horizon"].fillna(1).astype(int)

        is_duplicate = (
            (existing["timestamp_predicted"] == pred_row["timestamp_predicted"])
            & (existing["horizon"] == pred_row["horizon"])
        ).any()
        if is_duplicate:
            logger.warning(
                f"Prediction for {pred_row['timestamp_predicted']} "
                f"(horizon {pred_row['horizon']}) already exists — skipping."
            )
            return
        updated = pd.concat([existing, df_new], ignore_index=True)
    else:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        updated = df_new

    updated.to_csv(pred_path, index=False)


def get_missing_origins(
    past: pd.DataFrame,
    pred_path: Path,
    max_backfill_hours: int = 48,
) -> list[pd.Timestamp]:
    """
    Identify origins in hour_past.csv whose trajectory was never recorded in
    predictions.csv, starting from the last origin that was.

    An origin's trajectory is considered recorded if its horizon=1 row exists
    — every rollout emits h+1 first, so its presence is a reliable,
    horizon-config-independent signal that the origin's run happened. Rows
    written before the multi-horizon change have no horizon column — treated
    as horizon=1, since that's what they were.

    If predictions.csv does not exist or is empty, returns an empty list —
    backfill only covers gaps in an already-running system, not cold starts.

    Backfill is capped at max_backfill_hours origins to avoid regenerating
    years of history if the system was down for a long time.

    Parameters
    ----------
    past : pd.DataFrame
        Full past dataset sorted chronologically.
    pred_path : Path
        Path to predictions.csv.
    max_backfill_hours : int
        Maximum number of origins to backfill (default: 48).

    Returns
    -------
    list[pd.Timestamp]
        Sorted list of origins missing a trajectory.
    """
    if not pred_path.exists():
        logger.info("No existing predictions — skipping backfill (cold start)")
        return []

    existing = pd.read_csv(pred_path)
    if existing.empty:
        logger.info("No existing predictions — skipping backfill (cold start)")
        return []

    if "horizon" not in existing.columns:
        existing["horizon"] = 1
    else:
        existing["horizon"] = existing["horizon"].fillna(1).astype(int)

    h1_targets = pd.to_datetime(
        existing.loc[existing["horizon"] == 1, "timestamp_predicted"], format="ISO8601"
    )
    if h1_targets.empty:
        return []

    past = past.copy()
    past = reconstruct_datetime(past)

    last_origin = h1_targets.max() - pd.Timedelta(hours=1)
    resolved_targets = set(h1_targets.tolist())

    # Candidate origins are past hours after the last recorded origin, but
    # excluding the very last past hour: that one is the live prediction's own
    # origin, handled by the main rollout below — not a missed backfill.
    current_dt = past["datetime"].max()
    candidate_origins = sorted(
        past[(past["datetime"] > last_origin) & (past["datetime"] < current_dt)][
            "datetime"
        ].tolist()
    )
    missing = [o for o in candidate_origins if (o + pd.Timedelta(hours=1)) not in resolved_targets]

    if len(missing) > max_backfill_hours:
        logger.warning(
            f"Gap of {len(missing)} origins detected — "
            f"capping backfill at {max_backfill_hours} origins. "
            f"Older gaps will not be filled."
        )
        missing = missing[-max_backfill_hours:]

    return missing


def get_fallback_prediction(past: pd.DataFrame, target_dt: pd.Timestamp) -> dict | None:
    """
    Fall back to the actual demand from exactly 168h (one week) ago, same
    hour — the same seasonality the model's own cnt_lag_168 feature already
    relies on. Used when the input data for `target_dt` can't be trusted
    (see load_hourly_validation_flag), so the real model isn't run on data
    that might be corrupted.

    Returns
    -------
    dict | None
        {"pred_registered", "pred_casual", "pred_total"} from that hour's
        real values, or None if it isn't available yet (e.g. less than a
        week of simulated history) — the caller should skip this hour
        entirely in that case; the existing backfill mechanism fills it in
        once trustworthy data returns.
    """
    past = past.copy()
    past = reconstruct_datetime(past)

    lookback_dt = target_dt - pd.Timedelta(hours=168)
    row = past[past["datetime"] == lookback_dt]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "pred_registered": float(row["registered"]),
        "pred_casual": float(row["casual"]),
        "pred_total": float(row["cnt"]),
    }


def load_hourly_validation_flag(flag_path: Path) -> dict | None:
    """
    Load the hourly data validation result written by update_simulation.py.
    Returns None if the flag doesn't exist — treated as "trust the data",
    for back-compat and cold starts before this check existed.
    """
    if not flag_path.exists():
        return None
    with open(flag_path) as f:
        return json.load(f)


def run(cfg: DictConfig) -> None:
    raw_dir = Path(cfg.paths.raw_dir)
    state_path = Path(cfg.paths.simulation_state)
    pred_path = Path(cfg.paths.predictions_path)
    horizon = cfg.forecast.horizon

    # ── Load simulation state ─────────────────────────────────────────────────
    load_simulation_state(state_path)

    # ── Load past data ────────────────────────────────────────────────────────
    logger.info("Loading past data")
    past = pd.read_csv(raw_dir / cfg.paths.input_file)
    past = reconstruct_datetime(past)

    last_record = past.sort_values(["dteday", "hr"]).iloc[-1]
    current_dt = pd.to_datetime(last_record["dteday"]) + pd.Timedelta(hours=int(last_record["hr"]))
    next_dt = current_dt + pd.Timedelta(hours=1)

    logger.info(f"Current hour: {current_dt} | Predicting: h+1..h+{horizon} from here")

    # ── Validate data freshness ───────────────────────────────────────────────
    now = datetime.now()
    data_lag = now - current_dt.to_pydatetime()

    if data_lag > pd.Timedelta(hours=2):
        logger.warning(
            f"Data lag detected: last record is {data_lag} behind current time. "
            f"Run update_simulation.py first."
        )
    else:
        logger.info(f"Data freshness OK — lag: {data_lag}")

    # ── Load models from MLflow registry ─────────────────────────────────────
    client = MlflowClient()
    model_registered_version = client.get_model_version_by_alias(
        f"{cfg.project}-registered", "production"
    ).version
    model_casual_version = client.get_model_version_by_alias(
        f"{cfg.project}-casual", "production"
    ).version

    model_registered = mlflow.lightgbm.load_model(f"models:/{cfg.project}-registered@production")
    model_casual = mlflow.lightgbm.load_model(f"models:/{cfg.project}-casual@production")

    # ── Calendar (past + future, deterministic) and weather (past only) lookups ──
    min_date = past["dteday"].min()
    future_path = raw_dir / "hour_future.csv"
    future = (
        reconstruct_datetime(pd.read_csv(future_path)) if future_path.exists() else past.iloc[0:0]
    )
    calendar_lookup = build_calendar_lookup(past, future)
    weather_lookup = build_weather_lookup(past)

    lags = list(cfg.features.lags)
    rolling_windows = list(cfg.features.rolling_windows)
    drop_cols = list(cfg.features.drop_cols)

    # ── Backfill missing trajectories ─────────────────────────────────────────
    missing_origins = get_missing_origins(past, pred_path, cfg.monitoring.max_backfill_hours)

    if missing_origins:
        logger.info(f"Backfilling {len(missing_origins)} missing origins")

        for origin_dt in missing_origins:
            # Build past slice up to and including the origin — reproduces
            # exactly what the live rollout would have seen at that time,
            # without using any intermediate actuals from the trajectory.
            past_slice = past[past["datetime"] <= origin_dt].copy()

            if len(past_slice) < 168:
                logger.warning(f"Not enough history for backfill at {origin_dt} — skipping")
                continue

            trajectory_bf = predict_trajectory(
                past_slice,
                model_registered,
                model_casual,
                horizon,
                min_date,
                lags,
                rolling_windows,
                drop_cols,
                calendar_lookup,
                weather_lookup,
            )
            for record in _trajectory_to_records(
                trajectory_bf, model_registered_version, model_casual_version
            ):
                append_prediction(record, pred_path)
            logger.info(f"Backfilled trajectory from origin {origin_dt}")

    # ── Check hourly data validation before the main prediction ────────────────
    # Backfill above covers OLDER gaps using each historical slice's own data,
    # independent of this check — only the current origin's trajectory is gated
    # by whether the rows revealed THIS run passed validation.
    validation_flag_path = Path(cfg.paths.artifacts_dir) / "validation" / "hourly_validation.json"
    validation = load_hourly_validation_flag(validation_flag_path)

    if validation is not None and not validation["valid"]:
        logger.error(f"Hourly data validation failed — not using the model: {validation['issues']}")

        predicted_at = datetime.now().isoformat()
        fallback_records = []
        for k in range(1, horizon + 1):
            target_dt = current_dt + pd.Timedelta(hours=k)
            fallback = get_fallback_prediction(past, target_dt)
            if fallback is None:
                logger.warning(
                    f"No data available from 168h ago for h+{k} ({target_dt}) — skipping. "
                    f"Backfill will fill this gap once trustworthy data returns."
                )
                continue
            raw_row = build_synthetic_row(last_record, target_dt, calendar_lookup, weather_lookup)
            fallback_records.append(
                {
                    "predicted_at": predicted_at,
                    "timestamp_predicted": target_dt.isoformat(),
                    "horizon": k,
                    "hr": int(raw_row["hr"]),
                    "temp": float(raw_row["temp"]),
                    "hum": float(raw_row["hum"]),
                    "weathersit": int(raw_row["weathersit"]),
                    "workingday": int(raw_row["workingday"]),
                    "pred_registered": round(fallback["pred_registered"], 2),
                    "pred_casual": round(fallback["pred_casual"], 2),
                    "pred_total": round(fallback["pred_total"], 2),
                    "model_version_registered": None,
                    "model_version_casual": None,
                    "prediction_source": "fallback_lag168",
                }
            )

        for record in fallback_records:
            append_prediction(record, pred_path)
        if fallback_records:
            logger.warning(
                f"Served fallback trajectory (168h lag) from origin {current_dt} "
                f"({len(fallback_records)} horizon(s))"
            )
        return

    # ── Predict trajectory ─────────────────────────────────────────────────────
    trajectory = predict_trajectory(
        past,
        model_registered,
        model_casual,
        horizon,
        min_date,
        lags,
        rolling_windows,
        drop_cols,
        calendar_lookup,
        weather_lookup,
    )

    h1 = trajectory.iloc[0]
    logger.info(
        f"Trajectory from {current_dt} (h+1..h+{horizon}) — "
        f"h+1 ({next_dt}) — registered: {h1['pred_registered']:.0f} | "
        f"casual: {h1['pred_casual']:.0f} | total: {h1['pred_total']:.0f}"
    )

    # ── Save trajectory ────────────────────────────────────────────────────────
    for record in _trajectory_to_records(
        trajectory, model_registered_version, model_casual_version
    ):
        append_prediction(record, pred_path)
    logger.info(f"Trajectory saved to {pred_path} ({len(trajectory)} rows)")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
