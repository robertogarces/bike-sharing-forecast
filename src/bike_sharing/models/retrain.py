import logging
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig
import json
import mlflow
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
    force           = cfg.training.force_retrain

    # ── Configure MLflow ──────────────────────────────────────────────────────
    setup_mlflow()

    # ── Check if retraining is needed ─────────────────────────────────────────
    if not should_retrain(drift_flag_path, force):
        return

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

    logger.info("Retraining complete")


if __name__ == "__main__":
    main()