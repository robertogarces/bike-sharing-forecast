import json
import logging
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

from bike_sharing.models.train import FEATURES
from bike_sharing.utils.mlflow_utils import setup_mlflow
from bike_sharing.utils.monitoring_utils import append_monitoring_record
from bike_sharing.utils.datetime_utils import reconstruct_datetime

logger = logging.getLogger(__name__)

# Excluded from drift comparison: days_since_start is a monotonic trend
# feature — reference (train, older) and current (recent) windows can never
# overlap in its range, so it always shows as "drifted" regardless of real
# degradation. It stays in FEATURES for training/inference; only the drift
# check excludes it.
DRIFT_FEATURES = [f for f in FEATURES if f != "days_since_start"]


def _load_reference_snapshot(client: MlflowClient, project: str) -> pd.DataFrame | None:
    """
    Load the drift reference snapshotted at the production model's own
    promotion time (retrain.py's snapshot_drift_reference). None if no
    snapshot exists yet (model promoted before this existed, or bootstrap)
    — caller falls back to the live train.csv.
    """
    try:
        version = client.get_model_version_by_alias(f"{project}-registered", "production")
        local_path = client.download_artifacts(
            version.run_id, "drift_reference/drift_reference_snapshot.csv"
        )
        return pd.read_csv(local_path)
    except Exception as e:
        logger.warning(f"No drift reference snapshot found for production model — {e}")
        return None


def load_reference(
    processed_dir: Path,
    months: list[int],
    client: MlflowClient | None = None,
    project: str | None = None,
    min_reference_rows: int = 500,
) -> pd.DataFrame:
    """
    Load the drift reference, restricted to rows from the same calendar
    month(s) as the current window (mnth is preserved in train.csv,
    unaffected by the simulation's date-shifting).

    Prefers the snapshot taken when the current production model was
    promoted (see snapshot_drift_reference in retrain.py) — train.csv keeps
    advancing every time dvc repro runs, even for retrain attempts that
    don't get promoted, which would otherwise desync the reference from
    what the production model actually learned from. Falls back to the
    live train.csv if no snapshot exists yet.

    Comparing against all months mixes every season into one distribution,
    so a single week's weather/demand-level features almost always look
    "drifted" against the full-year average — not because of real model
    degradation, but because a week is never representative of a year.

    If the month-matched subset is too thin (e.g. early in the simulation,
    before a given month has repeated enough times in the revealed
    history), widen to the neighboring months (±1) for more statistical
    power, at the cost of being slightly less season-precise. Falls back
    to the full reference only if even that widened window is too thin.

    Parameters
    ----------
    processed_dir : Path
        Directory containing train.csv (fallback source).
    months : list[int]
        Calendar month(s) (1-12) covered by the current window.
    client : MlflowClient | None
        Used to look up the production model's drift reference snapshot.
        Optional — if omitted, skips straight to the train.csv fallback
        (e.g. callers that don't need/have an MLflow client).
    project : str | None
        Registered-model name prefix. Required together with client.
    min_reference_rows : int
        Minimum reference rows required before widening/falling back.
    """
    df = None
    if client is not None and project is not None:
        df = _load_reference_snapshot(client, project)
    if df is None:
        df = pd.read_csv(processed_dir / "train.csv")

    subset = df[df["mnth"].isin(months)]
    if len(subset) < min_reference_rows:
        widened = set(months)
        widened |= {((m - 1 + 1) % 12) + 1 for m in months}  # month + 1
        widened |= {((m - 1 - 1) % 12) + 1 for m in months}  # month - 1
        subset = df[df["mnth"].isin(widened)]

    if len(subset) < min_reference_rows:
        subset = df

    return subset[DRIFT_FEATURES]


def load_current(
    raw_dir: Path,
    input_file: str,
    n_hours: int,
    lags: list[int],
    rolling_windows: list[int],
    drop_cols: list[str],
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
        Current data with DRIFT_FEATURES columns plus `mnth` — the caller
        uses `mnth` to pick a matching reference; it is not itself part of
        the drift comparison.
    """
    from bike_sharing.features.build_features import build_lag_features, build_calendar_features

    df = pd.read_csv(raw_dir / input_file)
    df = reconstruct_datetime(df)

    df = build_lag_features(
        df,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    min_date = df["dteday"].min()
    df = build_calendar_features(df, drop_cols=drop_cols, min_date=min_date)

    df = df.dropna(subset=DRIFT_FEATURES).tail(n_hours)

    return df[DRIFT_FEATURES + ["mnth"]]


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path,
    drift_threshold: float,
    numerical_features: list[str],
    save_html: bool = True,
) -> dict:
    """
    Run Evidently drift report and save a JSON summary (+ optionally HTML).

    Uses the Kolmogorov-Smirnov test for numerical features to detect
    statistically significant distribution shifts between reference and
    current data. Shared between input-feature drift (weekly, ~20 columns,
    HTML is useful for scanning many features at once) and output/prediction
    drift (hourly, 3 columns, HTML would be regenerated every hour for no
    real benefit — save_html=False skips it entirely).

    Parameters
    ----------
    reference : pd.DataFrame
        Reference data (e.g. training set, or an earlier production window).
    current : pd.DataFrame
        Recent data to compare against reference.
    output_dir : Path
        Directory to save the drift report and flag.
    drift_threshold : float
        Fraction of drifted columns above which drift_detected is True.
    numerical_features : list[str]
        Columns to compare (must be present in both reference and current).
    save_html : bool
        Whether to render and save the full HTML report. Default True.

    Returns
    -------
    dict
        Drift summary with per-feature results and overall flag.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    column_mapping = ColumnMapping(numerical_features=numerical_features)

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    if save_html:
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

    # Per-column detail (which columns drifted, with what test/score) — lets a
    # caller attribute drift to a specific column instead of only the
    # aggregate share (e.g. output drift telling registered vs casual apart).
    drift_by_column = {
        col: {
            "stattest": info["stattest_name"],
            "score": round(info["drift_score"], 4),
            "drifted": info["drift_detected"],
        }
        for col, info in result["metrics"][1]["result"]["drift_by_columns"].items()
    }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_features": n_total,
        "n_drifted": n_drifted,
        "drift_share": round(drift_share, 4),
        "drift_detected": drift_detected,
        "threshold": drift_threshold,
        "drift_by_column": drift_by_column,
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

    setup_mlflow()
    client = MlflowClient()

    # ── Load current first — its calendar month(s) drive which reference to use ──
    logger.info(f"Loading current data (last {cfg.monitoring.n_hours} hours)")
    current_raw = load_current(
        raw_dir,
        cfg.paths.input_file,
        n_hours=cfg.monitoring.n_hours,
        lags=list(cfg.features.lags),
        rolling_windows=list(cfg.features.rolling_windows),
        drop_cols=list(cfg.features.drop_cols),
    )
    months = sorted(current_raw["mnth"].unique().tolist())
    current = current_raw[DRIFT_FEATURES]

    logger.info(f"Loading reference data (train set, month(s) {months})")
    reference = load_reference(
        processed_dir,
        months,
        client=client,
        project=cfg.project,
        min_reference_rows=cfg.monitoring.min_reference_rows,
    )

    logger.info(f"Reference: {len(reference):,} rows | Current: {len(current):,} rows")

    # ── Run drift report ──────────────────────────────────────────────────────
    logger.info("Running drift detection")
    summary = run_drift_report(
        reference,
        current,
        drift_dir,
        drift_threshold=drift_threshold,
        numerical_features=DRIFT_FEATURES,
    )

    drifted_features = [col for col, info in summary["drift_by_column"].items() if info["drifted"]]
    logger.info(
        f"Drift detected: {summary['drift_detected']} — "
        f"{summary['n_drifted']}/{summary['n_features']} features drifted "
        f"({summary['drift_share']:.0%}) — drifted: {drifted_features}"
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    history_path = artifacts_dir / "monitoring" / "drift_history.csv"
    flat_summary = {k: v for k, v in summary.items() if k != "drift_by_column"}
    append_monitoring_record(flat_summary, history_path)
    logger.info(f"Drift record appended to {history_path}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
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
