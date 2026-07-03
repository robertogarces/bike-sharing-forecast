import logging
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig
import json
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from bike_sharing.data.validate_data import validate_data_quality
from bike_sharing.models.train import compute_metrics, FEATURES
from bike_sharing.utils.command_utils import run_command
from bike_sharing.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


def should_retrain(drift_flag_path: Path, force: bool) -> bool:
    import json

    if force:
        logger.info("Force retrain — skipping drift check")
        return True

    if not drift_flag_path.exists():
        logger.warning(
            "No drift flag found — run drift_detection.py first. Use force=True to retrain anyway."
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
    df["dteday"] = pd.to_datetime(df["dteday"])
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
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    data_cutoff = df["datetime"].max()

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marker_path, "w") as f:
        json.dump(
            {
                "data_cutoff": data_cutoff.isoformat(),
                "written_at": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    logger.info(f"Retrain marker updated — data cutoff: {data_cutoff}")


def write_retrain_outcome(outcome: dict, path: Path) -> None:
    """
    Persist the result of this retrain.py run — whether a retrain was
    attempted, why not if it wasn't, and the promotion outcome if it was.

    Written on every run via a try/finally in main(), regardless of which
    gate the run exits at, so the weekly report always has something to
    show (most weeks likely skip retraining entirely — that's the common
    case, and today it's invisible without reading CI logs).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(outcome, f, indent=2)
    logger.info(f"Retrain outcome saved to {path}")


def describe_drift_skip_reason(drift_flag_path: Path) -> str:
    """
    Human-readable reason retraining didn't proceed past the drift gate.

    Mirrors should_retrain's branching but returns a message instead of a
    bool — kept separate so should_retrain's signature/tests are untouched.
    """
    if not drift_flag_path.exists():
        return "no drift flag found"

    with open(drift_flag_path) as f:
        state = json.load(f)

    if "reason" in state:
        return f"drift detection was skipped — {state['reason']}"

    if state["drift_detected"]:
        return "drift detected"
    return f"no drift detected ({state['drift_share']:.0%} <= {state['threshold']:.0%})"


def evaluate_production_pair(project: str, val_df: pd.DataFrame) -> float | None:
    """
    Evaluate the current production model pair (registered + casual) on the
    *current* validation set and return the combined RMSE.

    This is the champion side of the champion/challenger comparison. Both the
    new (challenger) and the production (champion) models are scored on the
    SAME val.csv, making the RMSE comparison apples-to-apples — unlike reading
    the champion's metric stored from its own (older) validation period, which
    could differ purely because the holdout differs.

    Parameters
    ----------
    project : str
        Registered-model name prefix (models are '{project}-registered' and
        '{project}-casual').
    val_df : pd.DataFrame
        Current validation set (must contain FEATURES and the 'cnt' column).

    Returns
    -------
    float | None
        Combined RMSE of the production pair on val_df, or None if no
        production alias exists yet (bootstrap — nothing to compare against).
    """
    try:
        model_registered = mlflow.lightgbm.load_model(f"models:/{project}-registered@production")
        model_casual = mlflow.lightgbm.load_model(f"models:/{project}-casual@production")
    except mlflow.exceptions.MlflowException:
        logger.info("No production model pair found — bootstrap (nothing to compare against)")
        return None

    X_val = val_df[FEATURES]
    y_val_cnt = val_df["cnt"].values

    pred_registered = np.expm1(model_registered.predict(X_val))
    pred_casual = np.expm1(model_casual.predict(X_val))
    pred_combined = np.clip(pred_registered + pred_casual, 0, None)

    return float(compute_metrics(y_val_cnt, pred_combined)["rmse"])


def promote_models_if_better(
    client: MlflowClient,
    project: str,
    new_rmse: float,
    prod_rmse: float | None,
) -> bool:
    """
    Atomically promote the new registered+casual model pair to the production
    alias if it beats the current production pair on the same validation set.

    Both models are promoted together (both-or-neither). RMSE is a combined
    metric over the pair (predictions are summed before scoring), so there is
    no measured RMSE for a mixed pair (e.g. new registered + old casual) —
    only for (new+new) and (prod+prod). Promoting one model without the other
    would mean acting on an untested combination. Lower RMSE is better.

    Parameters
    ----------
    client : MlflowClient
        MLflow tracking client.
    project : str
        Registered-model name prefix.
    new_rmse : float
        Combined RMSE of the newly trained pair on the current val set.
    prod_rmse : float | None
        Combined RMSE of the current production pair on the SAME val set, or
        None if there is no production model yet (bootstrap).

    Returns
    -------
    bool
        True if the new pair was promoted to production.
    """
    model_names = [f"{project}-registered", f"{project}-casual"]
    new_versions = {
        name: max(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
        for name in model_names
    }

    # Bootstrap: no production pair yet → promote directly.
    if prod_rmse is None:
        for name, version in new_versions.items():
            logger.info(f"{name} — no production model found → promoting version {version.version}")
            client.set_registered_model_alias(name, "production", version.version)
        return True

    if new_rmse < prod_rmse:
        logger.info(
            f"New model pair (rmse: {new_rmse:.4f}) beats production "
            f"(rmse: {prod_rmse:.4f}) on the current val set → promoting both"
        )
        for name, version in new_versions.items():
            prod_version = client.get_model_version_by_alias(name, "production")
            client.set_registered_model_alias(name, "production", version.version)
            client.set_registered_model_alias(name, "archived", prod_version.version)
        return True

    logger.warning(
        f"New model pair (rmse: {new_rmse:.4f}) does not beat production "
        f"(rmse: {prod_rmse:.4f}) on the current val set → keeping current"
    )
    for name, version in new_versions.items():
        client.set_registered_model_alias(name, "archived", version.version)
    return False


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    drift_flag_path = artifacts_dir / "drift" / "drift_detected.json"
    marker_path = Path(cfg.paths.retrain_marker)
    hour_past_path = Path(cfg.paths.raw_dir) / cfg.paths.input_file
    outcome_path = artifacts_dir / "monitoring" / "retrain_outcome.json"
    force = cfg.training.force_retrain

    # Written on every exit path (see the finally block below) — most weeks
    # likely skip retraining entirely (no drift), which is otherwise
    # invisible without reading CI logs.
    outcome = {
        "timestamp": datetime.now().isoformat(),
        "retrain_attempted": False,
        "skip_reason": None,
        "data_quality_checked": False,
        "data_quality_passed": None,
        "data_quality_issues": [],
        "new_rmse": None,
        "prod_rmse": None,
        "promoted": None,
    }

    try:
        # ── Configure MLflow ──────────────────────────────────────────────────
        setup_mlflow()

        # ── Check if retraining is needed ─────────────────────────────────────
        if not should_retrain(drift_flag_path, force):
            outcome["skip_reason"] = describe_drift_skip_reason(drift_flag_path)
            return

        # ── Check enough new data has accumulated since the last retrain ──────
        # Even with drift, skip if too little fresh data arrived — retraining
        # the full pipeline for a handful of new hours is not worth the
        # cost/risk. force=True bypasses this guard, same as the drift check.
        if not force:
            new_hours = count_new_hours(hour_past_path, marker_path)
            if new_hours is None:
                logger.info("No retrain marker found — bootstrapping first retrain")
            elif new_hours < cfg.training.min_new_hours:
                logger.warning(
                    f"Only {new_hours}h of new data since last retrain "
                    f"(< {cfg.training.min_new_hours} required) — skipping retrain"
                )
                outcome["skip_reason"] = (
                    f"insufficient new hours: {new_hours}/{cfg.training.min_new_hours} required"
                )
                return
            else:
                logger.info(
                    f"{new_hours}h of new data since last retrain "
                    f"(≥ {cfg.training.min_new_hours}) — proceeding"
                )

        # ── Validate data quality ───────────────────────────────────────────────
        # Runs even with force=True — this isn't a "should we retrain" business
        # decision like the two gates above, it's a safety check: retraining on
        # corrupted data (a broken sensor, a schema change) makes the model
        # worse, regardless of how confident we are that retraining is
        # otherwise warranted.
        hour_past_df = pd.read_csv(hour_past_path)
        issues = validate_data_quality(
            hour_past_df,
            required_columns=list(cfg.validation.required_columns),
            ranges={k: list(v) for k, v in cfg.validation.ranges.items()},
        )
        outcome["data_quality_checked"] = True
        outcome["data_quality_issues"] = issues
        if issues:
            outcome["data_quality_passed"] = False
            outcome["skip_reason"] = f"data quality validation failed: {issues}"
            logger.error(f"Data quality check failed — aborting retrain: {issues}")
            return
        outcome["data_quality_passed"] = True
        logger.info("Data quality check passed")

        # ── Retrain pipeline ────────────────────────────────────────────────────
        logger.info("Starting retraining pipeline")
        outcome["retrain_attempted"] = True

        run_command(["dvc", "unfreeze", "build_features"])
        run_command(["dvc", "repro"])

        # ── Promote if better ───────────────────────────────────────────────────
        # Champion/challenger comparison on the SAME (current) val set: the new
        # pair's RMSE comes from evaluate.py (metrics.json), and the production
        # pair is re-scored on that same val.csv here — apples-to-apples.
        client = MlflowClient()

        # New (challenger) pair's RMSE — already computed by evaluate.py on the
        # current val set during the dvc repro above.
        metrics_path = Path(cfg.paths.artifacts_dir) / "evaluation" / "metrics.json"
        with open(metrics_path) as f:
            new_metrics = json.load(f)
        new_rmse = new_metrics["rmse"]
        outcome["new_rmse"] = new_rmse

        # Production (champion) pair, re-scored on the current val set.
        val_df = pd.read_csv(Path(cfg.paths.processed_dir) / "val.csv")
        prod_rmse = evaluate_production_pair(cfg.project, val_df)
        outcome["prod_rmse"] = prod_rmse

        if prod_rmse is None:
            logger.info(
                f"Bootstrap promotion — new pair rmse: {new_rmse:.4f} (no production baseline)"
            )
        else:
            logger.info(
                f"Champion/challenger on current val — "
                f"new: {new_rmse:.4f} | production: {prod_rmse:.4f}"
            )

        promoted = promote_models_if_better(client, cfg.project, new_rmse, prod_rmse)
        outcome["promoted"] = promoted

        if promoted:
            logger.info("New model pair promoted to Production")
        else:
            logger.warning("New model pair not promoted — previous Production pair retained")

        run_command(["dvc", "freeze", "build_features"])

        # Advance the data boundary so the next run measures new data from here on.
        write_retrain_marker(hour_past_path, marker_path)

        logger.info("Retraining complete")
    finally:
        write_retrain_outcome(outcome, outcome_path)


if __name__ == "__main__":
    main()
