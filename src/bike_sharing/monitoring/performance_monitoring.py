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
    return df


def load_actuals(raw_dir: Path, input_file: str) -> pd.DataFrame:
    """
    Load hour_past.csv and reconstruct the actual datetime + demand.

    Same join key logic as dashboard/app.py's load_actuals() — dteday + hr —
    since both need to match predictions.csv's timestamp_predicted exactly.
    """
    df = pd.read_csv(raw_dir / input_file, parse_dates=["dteday"])
    df["timestamp_predicted"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df[["timestamp_predicted", "cnt"]].rename(columns={"cnt": "actual_total"})


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
        timestamp, n_hours, n_resolved, and rmse/rmsle/r2/mae.
    """
    recent = joined.sort_values("timestamp_predicted").tail(n_hours)

    metrics = compute_metrics(recent["actual_total"].values, recent["pred_total"].values)

    return {
        "timestamp": datetime.now().isoformat(),
        "n_hours": n_hours,
        "n_resolved": len(recent),
        **metrics,
    }


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
    logger.info(f"{len(joined):,}/{len(predictions):,} predictions have a known actual")

    if joined.empty:
        logger.warning("No resolved predictions yet — skipping performance monitoring")
        return

    # ── Compute rolling performance ──────────────────────────────────────────
    summary = compute_rolling_performance(joined, n_hours)
    logger.info(
        f"Live performance (last {summary['n_hours']}h, {summary['n_resolved']} resolved) — "
        f"RMSE: {summary['rmse']:.2f} | MAE: {summary['mae']:.2f} | "
        f"RMSLE: {summary['rmsle']:.4f} | R²: {summary['r2']:.4f}"
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    append_monitoring_record(summary, history_path)
    logger.info(f"Performance record appended to {history_path}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    setup_mlflow()
    mlflow.set_experiment(cfg.project)
    with mlflow.start_run(run_name="performance_monitoring"):
        mlflow.log_metrics(
            {
                "rmse": summary["rmse"],
                "mae": summary["mae"],
                "rmsle": summary["rmsle"],
                "r2": summary["r2"],
                "n_resolved": summary["n_resolved"],
            }
        )
        mlflow.log_param("n_hours", summary["n_hours"])
        logger.info("Logged live performance metrics to MLflow")


if __name__ == "__main__":
    main()
