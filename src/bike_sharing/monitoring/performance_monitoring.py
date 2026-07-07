import logging
from pathlib import Path
from datetime import datetime

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig

from bike_sharing.models.train import compute_metrics
from bike_sharing.utils.mlflow_utils import setup_mlflow
from bike_sharing.utils.monitoring_utils import append_monitoring_record
from bike_sharing.utils.datetime_utils import reconstruct_datetime

logger = logging.getLogger(__name__)


def load_predictions(pred_path: Path) -> pd.DataFrame:
    """
    Load predictions.csv and parse its timestamp column.

    Excludes fallback_lag168 rows (served when hourly data validation fails)
    — they're not real model output and would corrupt RMSE/MAE/RMSLE/R².
    Older predictions.csv files without a prediction_source column are
    assumed to be all real model predictions.
    """
    df = pd.read_csv(pred_path)
    df["timestamp_predicted"] = pd.to_datetime(df["timestamp_predicted"], format="ISO8601")
    if "prediction_source" in df.columns:
        df = df[df["prediction_source"] != "fallback_lag168"]
    # Multi-horizon: each run emits K rows (one per lead time). Rows written
    # before this change have no horizon column — they were all next-hour
    # predictions, so they read as horizon=1.
    if "horizon" not in df.columns:
        df["horizon"] = 1
    else:
        df["horizon"] = df["horizon"].fillna(1).astype(int)
    return df


def load_actuals(raw_dir: Path, input_file: str) -> pd.DataFrame:
    """
    Load hour_past.csv and reconstruct the actual datetime + demand.

    Same join key logic as dashboard/app.py's load_actuals() — dteday + hr —
    since both need to match predictions.csv's timestamp_predicted exactly.
    """
    df = pd.read_csv(raw_dir / input_file, parse_dates=["dteday"])
    df = reconstruct_datetime(df, column="timestamp_predicted")
    return df[["timestamp_predicted", "cnt"]].rename(columns={"cnt": "actual_total"})


def build_seasonal_naive(actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Seasonal-naive baseline: the prediction for hour t is the actual demand at
    t - 168h (same hour, previous week) — the canonical trivial baseline for
    hourly demand, and the same signal the model's own fallback uses.

    Returned as [timestamp_predicted, naive_pred], ready to left-merge onto the
    joined frame. Horizon-agnostic: the t-168h actual is always available no
    matter how far ahead the model predicts, so this comparison needs no change
    if the forecast horizon ever changes.
    """
    naive = actuals.rename(columns={"actual_total": "naive_pred"}).copy()
    naive["timestamp_predicted"] = naive["timestamp_predicted"] + pd.Timedelta(hours=168)
    return naive


def join_predictions_with_actuals(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join predictions to their actuals by timestamp.

    Inner join: predictions for hours whose actual hasn't been revealed yet
    are dropped — they can't be scored until the real demand is known.
    """
    return predictions.merge(actuals, on="timestamp_predicted", how="inner")


def compute_rolling_performance(joined: pd.DataFrame, n_hours: int) -> dict:
    """
    Compute performance metrics over the most recent n_hours of resolved
    (actual-known) predictions.

    Parameters
    ----------
    joined : pd.DataFrame
        Output of join_predictions_with_actuals — must contain pred_total
        and actual_total.
    n_hours : int
        Number of most recent resolved hours to include in the window.

    Returns
    -------
    dict
        timestamp, n_hours, n_resolved, rmse/rmsle/r2/mae, plus naive_rmse
        (seasonal-naive baseline over the same window) and skill_vs_naive
        (fraction the model improves on it). The last two are None until the
        window has hours with a t-168h actual available.
    """
    recent = joined.sort_values("timestamp_predicted").tail(n_hours)

    metrics = compute_metrics(recent["actual_total"].values, recent["pred_total"].values)

    # Seasonal-naive baseline over the rows in the window that have a t-168h
    # actual. Horizon-agnostic: independent of how far ahead the model predicted.
    naive_rmse = None
    if "naive_pred" in recent.columns:
        scored = recent.dropna(subset=["naive_pred"])
        if len(scored):
            naive_rmse = compute_metrics(
                scored["actual_total"].values, scored["naive_pred"].values
            )["rmse"]
    skill_vs_naive = 1 - metrics["rmse"] / naive_rmse if naive_rmse else None

    return {
        "timestamp": datetime.now().isoformat(),
        "n_hours": n_hours,
        "n_resolved": len(recent),
        **metrics,
        "naive_rmse": naive_rmse,
        "skill_vs_naive": skill_vs_naive,
    }


def compute_rolling_performance_by_horizon(joined: pd.DataFrame, n_hours: int) -> list[dict]:
    """
    Per-horizon performance: group the joined frame by lead time and compute
    the rolling summary within each horizon separately.

    Multi-horizon serving means each target hour carries K predictions, one
    per lead time — and error grows with the horizon, so a single pooled RMSE
    would be meaningless (it would mix h+1 with h+K). Each horizon is its own
    series: grouping first, then windowing to the most recent n_hours within
    the group, keeps the window per-horizon rather than mixing lead times.

    The seasonal-naive baseline is horizon-independent (cnt(t-168h) for the
    target hour), so it's already correct inside each group — compared against
    every horizon without change.

    Returns
    -------
    list[dict]
        One compute_rolling_performance summary per horizon present, each with
        an added "horizon" key, ordered by horizon.
    """
    summaries = []
    for horizon, group in joined.groupby("horizon"):
        summary = compute_rolling_performance(group, n_hours)
        summary["horizon"] = int(horizon)
        summaries.append(summary)
    return summaries


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir = Path(cfg.paths.raw_dir)
    pred_path = Path(cfg.paths.predictions_path)
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    history_path = artifacts_dir / "monitoring" / "performance_history.csv"
    n_hours = cfg.monitoring.n_hours

    # ── Load & join ───────────────────────────────────────────────────────────
    logger.info("Loading predictions and actuals")
    predictions = load_predictions(pred_path)
    actuals = load_actuals(raw_dir, cfg.paths.input_file)

    joined = join_predictions_with_actuals(predictions, actuals)
    joined = joined.merge(build_seasonal_naive(actuals), on="timestamp_predicted", how="left")
    logger.info(f"{len(joined):,}/{len(predictions):,} predictions have a known actual")

    if joined.empty:
        logger.warning("No resolved predictions yet — skipping performance monitoring")
        return

    # ── Compute rolling performance per horizon ──────────────────────────────
    summaries = compute_rolling_performance_by_horizon(joined, n_hours)
    for summary in summaries:
        skill = (
            f" | skill vs naive: {summary['skill_vs_naive']:+.1%}"
            if summary["naive_rmse"] is not None
            else ""
        )
        logger.info(
            f"h+{summary['horizon']:<2d} (last {summary['n_hours']}h, "
            f"{summary['n_resolved']} resolved) — RMSE: {summary['rmse']:.2f} | "
            f"MAE: {summary['mae']:.2f} | R²: {summary['r2']:.4f}{skill}"
        )

    # ── Persist ───────────────────────────────────────────────────────────────
    # One record per (timestamp, horizon) — the horizon column lets downstream
    # consumers pick a lead time (e.g. the retrain gate filters to h+1).
    for summary in summaries:
        append_monitoring_record(summary, history_path)
    logger.info(f"{len(summaries)} per-horizon performance records appended to {history_path}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    # step=horizon logs each metric as a curve over lead time — the
    # error-vs-horizon and skill-vs-horizon curves render natively in the UI.
    setup_mlflow()
    mlflow.set_experiment(cfg.project)
    with mlflow.start_run(run_name="performance_monitoring"):
        for summary in summaries:
            h = summary["horizon"]
            mlflow.log_metric("rmse", summary["rmse"], step=h)
            mlflow.log_metric("mae", summary["mae"], step=h)
            mlflow.log_metric("rmsle", summary["rmsle"], step=h)
            mlflow.log_metric("r2", summary["r2"], step=h)
            mlflow.log_metric("n_resolved", summary["n_resolved"], step=h)
            if summary["naive_rmse"] is not None:
                mlflow.log_metric("naive_rmse", summary["naive_rmse"], step=h)
                mlflow.log_metric("skill_vs_naive", summary["skill_vs_naive"], step=h)
        mlflow.log_param("n_hours", n_hours)
        logger.info(f"Logged per-horizon performance metrics to MLflow ({len(summaries)} horizons)")


if __name__ == "__main__":
    main()
