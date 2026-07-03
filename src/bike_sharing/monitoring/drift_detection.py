import json
import logging
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
import mlflow
from omegaconf import DictConfig
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

from bike_sharing.models.train import FEATURES
from bike_sharing.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


def load_reference(processed_dir: Path) -> pd.DataFrame:
    """
    Load the training dataset as the drift reference.
    Only feature columns are kept — target and metadata are excluded.
    """
    df = pd.read_csv(processed_dir / "train.csv")
    return df[FEATURES]


def load_current(
    raw_dir: Path,
    input_file: str,
    n_hours: int,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """
    Load the most recent n_hours records from hour_past.csv as current data.

    Parameters
    ----------
    raw_dir : Path
        Directory containing hour_past.csv.
    input_file : str
        Filename of the past dataset.
    n_hours : int
        Number of most recent hours to use as current window.

    Returns
    -------
    pd.DataFrame
        Current data with feature columns only.
    """
    from bike_sharing.features.build_features import build_lag_features, build_calendar_features

    df = pd.read_csv(raw_dir / input_file)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")

    df = build_lag_features(
        df,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    min_date = df["dteday"].min()
    df = build_calendar_features(df, drop_cols=["atemp", "yr"], min_date=min_date)

    df = df.dropna(subset=FEATURES).tail(n_hours)

    return df[FEATURES]


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    drift_threshold: float,
) -> dict:
    """
    Run Evidently drift report and save HTML + JSON summary.

    Uses the Kolmogorov-Smirnov test for numerical features to detect
    statistically significant distribution shifts between reference
    (train set) and current (recent production) data.

    Parameters
    ----------
    reference : pd.DataFrame
        Training data used as drift reference.
    current : pd.DataFrame
        Recent production data to compare against reference.
    output_dir : Path
        Directory to save the drift report and flag.
    drift_threshold : float
        Fraction of drifted features above which retraining is triggered.

    Returns
    -------
    dict
        Drift summary with per-feature results and overall flag.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    column_mapping = ColumnMapping(numerical_features=FEATURES)

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    # Save HTML report
    html_path = output_dir / "drift_report.html"
    report.save_html(str(html_path))
    logger.info(f"Drift report saved to {html_path}")

    # Extract summary
    result = report.as_dict()
    metrics = result["metrics"][0]["result"]
    n_drifted = metrics["number_of_drifted_columns"]
    n_total = metrics["number_of_columns"]
    drift_share = n_drifted / n_total
    drift_detected = drift_share > drift_threshold

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_features": n_total,
        "n_drifted": n_drifted,
        "drift_share": round(drift_share, 4),
        "drift_detected": drift_detected,
        "threshold": drift_threshold,
    }

    # Save drift flag
    flag_path = output_dir / "drift_detected.json"
    with open(flag_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Drift flag saved to {flag_path}")

    return summary


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir = Path(cfg.paths.raw_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    drift_dir = artifacts_dir / "drift"
    drift_threshold = float(cfg.monitoring.drift_threshold)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading reference data (train set)")
    reference = load_reference(processed_dir)

    logger.info(f"Loading current data (last {cfg.monitoring.n_hours} hours)")
    current = load_current(
        raw_dir,
        cfg.paths.input_file,
        n_hours=cfg.monitoring.n_hours,
        lags=list(cfg.features.lags),
        rolling_windows=list(cfg.features.rolling_windows),
    )

    logger.info(f"Reference: {len(reference):,} rows | Current: {len(current):,} rows")

    # ── Run drift report ──────────────────────────────────────────────────────
    logger.info("Running drift detection")
    summary = run_drift_report(
        reference,
        current,
        drift_dir,
        drift_threshold=drift_threshold,
    )

    logger.info(
        f"Drift detected: {summary['drift_detected']} — "
        f"{summary['n_drifted']}/{summary['n_features']} features drifted "
        f"({summary['drift_share']:.0%})"
    )

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    setup_mlflow()
    mlflow.set_experiment(cfg.project)
    with mlflow.start_run(run_name="drift_detection"):
        mlflow.log_metrics(
            {
                "n_drifted_features": summary["n_drifted"],
                "drift_share": summary["drift_share"],
            }
        )
        mlflow.log_param("drift_detected", summary["drift_detected"])
        mlflow.log_artifact(str(drift_dir / "drift_report.html"))
        logger.info("Logged drift results to MLflow")

    if summary["drift_detected"]:
        logger.warning(
            f"Drift threshold exceeded ({summary['drift_share']:.0%} > "
            f"{drift_threshold:.0%}) — retraining recommended."
        )
    else:
        logger.info("No significant drift detected — model is stable.")


if __name__ == "__main__":
    main()
