import logging
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def run_command(cmd: list[str]) -> None:
    """
    Run a shell command and raise if it fails.
    """
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def should_retrain(drift_flag_path: Path, force: bool) -> bool:
    """
    Determine whether retraining should proceed.

    Retraining is triggered if:
    - force=True (manual or scheduled retrain)
    - drift_detected.json exists and drift_detected is True

    Parameters
    ----------
    drift_flag_path : Path
        Path to drift_detected.json.
    force : bool
        If True, retrain regardless of drift flag.

    Returns
    -------
    bool
        Whether to proceed with retraining.
    """
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


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir   = Path(cfg.paths.artifacts_dir)
    drift_flag_path = artifacts_dir / "drift" / "drift_detected.json"
    force           = cfg.training.force_retrain

    # ── Check if retraining is needed ─────────────────────────────────────────
    if not should_retrain(drift_flag_path, force):
        return

    # ── Retrain pipeline ──────────────────────────────────────────────────────
    logger.info("Starting retraining pipeline")

    run_command(["dvc", "unfreeze", "build_features"])
    run_command(["dvc", "repro"])
    run_command(["dvc", "freeze", "build_features"])

    logger.info("Retraining complete")


if __name__ == "__main__":
    main()