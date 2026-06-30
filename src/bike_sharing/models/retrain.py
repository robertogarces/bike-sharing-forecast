import logging
import subprocess
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig
import json
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from bike_sharing.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


def run_command(cmd: list[str]) -> None:
    """
    Run a shell command and raise if it fails.
    """
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=False)


def should_retrain(drift_flag_path: Path, force: bool) -> bool:
    import json

    if force:
        logger.info("Force retrain — skipping drift check")
        return True

    if not drift_flag_path.exists():
        logger.warning(
            "No drift flag found — run drift_detection.py first. "
            "Use force=True to retrain anyway."
        )
        return False

    with open(drift_flag_path) as f:
        state = json.load(f)

    if "reason" in state:
        logger.warning(f"Drift detection was skipped — {state['reason']}")
        return False

    if state["drift_detected"]:
        logger.info(
            f"Drift detected ({state['drift_share']:.0%} > {state['threshold']:.0%}) "
            f"— proceeding with retraining"
        )
        return True
    else:
        logger.info(
            f"No drift detected ({state['drift_share']:.0%} ≤ {state['threshold']:.0%}) "
            f"— skipping retraining"
        )
        return False


def count_new_hours(hour_past_path: Path, marker_path: Path) -> int | None:
    """
    Count how many hours of data have accumulated since the last retrain.

    Compares the records in hour_past.csv against the data cutoff recorded in
    the last-retrain marker. This answers "is there enough fresh data to make
    retraining worthwhile?" — distinct from drift detection.

    Parameters
    ----------
    hour_past_path : Path
        Path to hour_past.csv (the full accumulated history).
    marker_path : Path
        Path to the last-retrain marker JSON.

    Returns
    -------
    int | None
        Number of records newer than the marker's data cutoff, or None if no
        marker exists yet (first retrain — bootstrap, retrain should proceed).
    """
    if not marker_path.exists():
        return None

    with open(marker_path) as f:
        marker = json.load(f)
    cutoff = pd.to_datetime(marker["data_cutoff"])

    df = pd.read_csv(hour_past_path)
    df["dteday"]   = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")

    return int((df["datetime"] > cutoff).sum())


def write_retrain_marker(hour_past_path: Path, marker_path: Path) -> None:
    """
    Record the data cutoff of the current retrain so future runs can measure
    how much new data has accumulated since.

    The cutoff is the latest datetime present in hour_past.csv at retrain time.
    Written regardless of whether the new model was promoted — the compute was
    already spent, so the boundary should advance to avoid re-running next week.
    """
    df = pd.read_csv(hour_past_path)
    df["dteday"]   = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    data_cutoff = df["datetime"].max()

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marker_path, "w") as f:
        json.dump({
            "data_cutoff": data_cutoff.isoformat(),
            "written_at":  datetime.now().isoformat(),
        }, f, indent=2)
    logger.info(f"Retrain marker updated — data cutoff: {data_cutoff}")


def promote_if_better(
    client: MlflowClient,
    model_name: str,
    new_metrics: dict,
    metric_key: str = "rmse",
) -> bool:
    """
    Promote the latest model version to Production alias if it improves
    on the current Production model.

    Uses MLflow aliases instead of deprecated stages API.
    Lower metric value is considered better.

    Parameters
    ----------
    client : MlflowClient
        MLflow tracking client.
    model_name : str
        Registered model name in MLflow registry.
    new_metrics : dict
        Metrics from the newly trained model.
    metric_key : str
        Metric to compare (default: 'rmse').

    Returns
    -------
    bool
        True if the new model was promoted to Production.
    """
    # Get all versions and find the latest
    all_versions = client.search_model_versions(f"name='{model_name}'")
    new_version  = max(all_versions, key=lambda v: int(v.version))

    # Check if a production alias exists
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
        prod_run     = client.get_run(prod_version.run_id)
        prod_metric  = prod_run.data.metrics[metric_key]
        new_metric   = new_metrics[metric_key]

        if new_metric < prod_metric:
            logger.info(
                f"{model_name} — new model ({metric_key}: {new_metric:.4f}) "
                f"beats production ({metric_key}: {prod_metric:.4f}) → promoting"
            )
            client.set_registered_model_alias(model_name, "production", new_version.version)
            client.set_registered_model_alias(model_name, "archived",   prod_version.version)
            return True
        else:
            logger.warning(
                f"{model_name} — new model ({metric_key}: {new_metric:.4f}) "
                f"does not beat production ({metric_key}: {prod_metric:.4f}) → keeping current"
            )
            client.set_registered_model_alias(model_name, "archived", new_version.version)
            return False

    except mlflow.exceptions.MlflowException:
        # No production alias yet — promote directly
        logger.info(
            f"{model_name} — no production model found → "
            f"promoting version {new_version.version}"
        )
        client.set_registered_model_alias(model_name, "production", new_version.version)
        return True
        

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir   = Path(cfg.paths.artifacts_dir)
    drift_flag_path = artifacts_dir / "drift" / "drift_detected.json"
    marker_path     = Path(cfg.paths.retrain_marker)
    hour_past_path  = Path(cfg.paths.raw_dir) / cfg.paths.input_file
    force           = cfg.training.force_retrain

    # ── Configure MLflow ──────────────────────────────────────────────────────
    setup_mlflow()

    # ── Check if retraining is needed ─────────────────────────────────────────
    if not should_retrain(drift_flag_path, force):
        return

    # ── Check enough new data has accumulated since the last retrain ──────────
    # Even with drift, skip if too little fresh data arrived — retraining the
    # full pipeline for a handful of new hours is not worth the cost/risk.
    # force=True bypasses this guard, same as the drift check.
    if not force:
        new_hours = count_new_hours(hour_past_path, marker_path)
        if new_hours is None:
            logger.info("No retrain marker found — bootstrapping first retrain")
        elif new_hours < cfg.training.min_new_hours:
            logger.warning(
                f"Only {new_hours}h of new data since last retrain "
                f"(< {cfg.training.min_new_hours} required) — skipping retrain"
            )
            return
        else:
            logger.info(
                f"{new_hours}h of new data since last retrain "
                f"(≥ {cfg.training.min_new_hours}) — proceeding"
            )

    # ── Retrain pipeline ──────────────────────────────────────────────────────
    logger.info("Starting retraining pipeline")

    run_command(["dvc", "unfreeze", "build_features"])
    run_command(["dvc", "repro"])

    # ── Promote if better ─────────────────────────────────────────────────────

    client = MlflowClient()

    # Load metrics from latest evaluation
    metrics_path = Path(cfg.paths.artifacts_dir) / "evaluation" / "metrics.json"
    with open(metrics_path) as f:
        new_metrics = json.load(f)

    promoted_registered = promote_if_better(
        client,
        model_name=f"{cfg.project}-registered",
        new_metrics=new_metrics,
    )
    promoted_casual = promote_if_better(
        client,
        model_name=f"{cfg.project}-casual",
        new_metrics=new_metrics,
    )

    if promoted_registered and promoted_casual:
        logger.info("Both models promoted to Production")
    else:
        logger.warning("One or both models not promoted — previous Production model retained")

    run_command(["dvc", "freeze", "build_features"])

    # Advance the data boundary so the next run measures new data from here on.
    write_retrain_marker(hour_past_path, marker_path)

    logger.info("Retraining complete")


if __name__ == "__main__":
    main()