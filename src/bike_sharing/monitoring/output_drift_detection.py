import logging
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig

from bike_sharing.monitoring.drift_detection import run_drift_report
from bike_sharing.monitoring.performance_monitoring import load_predictions
from bike_sharing.utils.mlflow_utils import setup_mlflow
from bike_sharing.utils.monitoring_utils import append_monitoring_record

logger = logging.getLogger(__name__)

# Columns compared for output drift — the model's own predictions, not the
# input features (that's drift_detection.py) and not the real outcome
# (that's performance_monitoring.py).
OUTPUT_COLUMNS = ["pred_total", "pred_registered", "pred_casual"]


def split_rolling_windows(
    predictions: pd.DataFrame,
    n_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """
    Split predictions into a rolling reference/current pair: the most recent
    n_hours rows are `current`, and the n_hours immediately before that are
    `reference`.

    Unlike input-feature drift (compared against the training set), there is
    no natural "training-time" distribution for predictions to compare
    against — predictions only exist once the model is live. Comparing two
    adjacent, same-sized recent windows avoids the seasonal mismatch that a
    fixed long-span reference would introduce (see drift_detection.py's
    month-matching fix), at the cost of only catching abrupt week-over-week
    changes rather than long-run drift from the original training data.

    Parameters
    ----------
    predictions : pd.DataFrame
        All predictions, must contain `timestamp_predicted`.
    n_hours : int
        Size of each window.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]
        (reference, current), or (None, None) if there isn't enough history
        yet for two full windows.
    """
    predictions = predictions.sort_values("timestamp_predicted")

    if len(predictions) < 2 * n_hours:
        return None, None

    current = predictions.tail(n_hours)
    reference = predictions.iloc[-2 * n_hours : -n_hours]

    return reference, current


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pred_path = Path(cfg.paths.predictions_path)
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    output_dir = artifacts_dir / "output_drift"
    history_path = artifacts_dir / "monitoring" / "output_drift_history.csv"
    n_hours = cfg.monitoring.n_hours
    drift_threshold = float(cfg.monitoring.drift_threshold)

    # ── Load & split ──────────────────────────────────────────────────────────
    logger.info("Loading predictions")
    predictions = load_predictions(pred_path)

    reference, current = split_rolling_windows(predictions, n_hours)
    if reference is None:
        logger.warning(
            f"Not enough prediction history for a rolling comparison "
            f"(need {2 * n_hours}h, have {len(predictions)}h) — skipping output drift check"
        )
        return

    # ── Run drift report ──────────────────────────────────────────────────────
    # save_html=False: this runs hourly, and the HTML report (~4MB for the
    # 20-feature input-drift case) would flood MLflow/DagsHub storage over
    # time for no real benefit — with only 3 columns, the JSON summary alone
    # is enough to read at a glance.
    logger.info("Running output drift detection")
    summary = run_drift_report(
        reference[OUTPUT_COLUMNS],
        current[OUTPUT_COLUMNS],
        output_dir,
        drift_threshold=drift_threshold,
        numerical_features=OUTPUT_COLUMNS,
        save_html=False,
    )

    logger.info(
        f"Output drift detected: {summary['drift_detected']} — "
        f"{summary['n_drifted']}/{summary['n_features']} prediction columns drifted "
        f"({summary['drift_share']:.0%})"
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    append_monitoring_record(summary, history_path)
    logger.info(f"Output drift record appended to {history_path}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    setup_mlflow()
    mlflow.set_experiment(cfg.project)
    with mlflow.start_run(run_name="output_drift_detection"):
        mlflow.log_metrics(
            {
                "n_drifted_columns": summary["n_drifted"],
                "drift_share": summary["drift_share"],
            }
        )
        mlflow.log_param("drift_detected", summary["drift_detected"])
        logger.info("Logged output drift results to MLflow")


if __name__ == "__main__":
    main()
