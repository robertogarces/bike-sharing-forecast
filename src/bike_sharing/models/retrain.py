import logging
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig
import json
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from bike_sharing.data.validate_data import validate_data_quality
from bike_sharing.models.train import compute_metrics, FEATURES
from bike_sharing.utils.command_utils import run_command
from bike_sharing.utils.datetime_utils import reconstruct_datetime, utc_now
from bike_sharing.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


def should_retrain(drift_flag_path: Path, force: bool) -> bool:
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
    df = reconstruct_datetime(df)

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
    df = reconstruct_datetime(df)
    data_cutoff = df["datetime"].max()

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marker_path, "w") as f:
        json.dump(
            {
                "data_cutoff": data_cutoff.isoformat(),
                "written_at": utc_now().isoformat(),
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


def get_production_baseline_rmse(client: MlflowClient, project: str) -> float | None:
    """
    Combined RMSE of the current production pair at the time it was
    promoted. Read first from the "combined_rmse_baseline" tag on the
    registered model's "production" version — logged by
    promote_best_combination after each promotion, since with mixed pairs
    the RMSE no longer corresponds to a single training run. Falls back to
    the "rmse" metric train.py logs on the run (versions promoted before
    this tag existed).

    None if no "production" alias exists yet (bootstrap — nothing to compare
    against).
    """
    try:
        version = client.get_model_version_by_alias(f"{project}-registered", "production")
    except mlflow.exceptions.MlflowException:
        logger.info("No production model found — performance gate not evaluated (bootstrap)")
        return None
    if "combined_rmse_baseline" in version.tags:
        return float(version.tags["combined_rmse_baseline"])
    run = client.get_run(version.run_id)
    return run.data.metrics.get("rmse")


def is_performance_degraded(
    performance_history_path: Path,
    baseline_rmse: float | None,
    degradation_threshold: float,
    primary_horizon: int = 1,
) -> tuple[bool, float | None]:
    """
    Compare the most recent rolling RMSE at primary_horizon (the last row for
    that lead time in performance_history.csv, written per-horizon by
    performance_monitoring.py) against the production model's validation RMSE
    at promotion time.

    Only primary_horizon (h+1) is comparable to the baseline: the baseline is a
    single-step validation RMSE, while longer horizons carry recursive-rollout
    error and would read as permanently "degraded" against it. Rows written
    before the multi-horizon change have no horizon column — treated as
    horizon=1, which is what they were.

    Returns (degraded, live_rmse). degraded is always False when there is no
    baseline, no performance history yet, or no row at primary_horizon (cold
    start) — live_rmse is None in those cases.
    """
    if baseline_rmse is None or not performance_history_path.exists():
        return False, None

    df = pd.read_csv(performance_history_path)
    if df.empty:
        return False, None

    if "horizon" not in df.columns:
        df["horizon"] = 1
    else:
        df["horizon"] = df["horizon"].fillna(1).astype(int)
    df = df[df["horizon"] == primary_horizon]
    if df.empty:
        return False, None

    live_rmse = float(df.iloc[-1]["rmse"])
    degraded = live_rmse > baseline_rmse * (1 + degradation_threshold)
    return degraded, live_rmse


def _combination_rmses(
    y_cnt: np.ndarray,
    new_reg: np.ndarray,
    new_cas: np.ndarray,
    prod_reg: np.ndarray,
    prod_cas: np.ndarray,
) -> dict[tuple[str, str], float]:
    """
    Combined RMSE for each of the four deployable (registered, casual) source
    combinations, keyed by ("new"|"prod", "new"|"prod"). Predictions must be on
    the original cnt scale (expm1 already applied); the pair sum is clipped at 0
    before scoring, matching evaluate_models / evaluate.py.

    The combined RMSE is not decomposable into per-model RMSEs — it depends on
    how the two models' errors covary (they can cancel or compound). Scoring the
    actual sums is the only way to pick the best deployable pair.
    """
    preds = {
        ("new", "reg"): new_reg,
        ("prod", "reg"): prod_reg,
        ("new", "cas"): new_cas,
        ("prod", "cas"): prod_cas,
    }
    out = {}
    for r in ("new", "prod"):
        for c in ("new", "prod"):
            combined = np.clip(preds[(r, "reg")] + preds[(c, "cas")], 0, None)
            out[(r, c)] = float(compute_metrics(y_cnt, combined)["rmse"])
    return out


def evaluate_promotion_combinations(
    project: str, val_df: pd.DataFrame, models_dir: Path
) -> dict[tuple[str, str], float] | None:
    """
    Score all four deployable (registered, casual) combinations on the combined
    RMSE over the current val set. The two just-trained models are loaded from
    models_dir (same local .txt files evaluate.py wrote); the two production
    models by alias. All four are scored on the SAME val.csv, making the
    comparison apples-to-apples.

    Returns the combination→RMSE dict, or None if no production pair exists yet
    (bootstrap — only new+new is deployable).
    """
    X_val = val_df[FEATURES]
    y_cnt = val_df["cnt"].values

    new_reg = lgb.Booster(model_file=str(models_dir / "lgbm_registered.txt"))
    new_cas = lgb.Booster(model_file=str(models_dir / "lgbm_casual.txt"))

    try:
        prod_reg = mlflow.lightgbm.load_model(f"models:/{project}-registered@production")
        prod_cas = mlflow.lightgbm.load_model(f"models:/{project}-casual@production")
    except mlflow.exceptions.MlflowException:
        logger.info("No production model pair found — bootstrap (nothing to compare against)")
        return None

    return _combination_rmses(
        y_cnt,
        np.expm1(new_reg.predict(X_val)),
        np.expm1(new_cas.predict(X_val)),
        np.expm1(prod_reg.predict(X_val)),
        np.expm1(prod_cas.predict(X_val)),
    )


def choose_best_combination(combos: dict[tuple[str, str], float]) -> tuple[str, str]:
    """
    Source (registered, casual) of the lowest combined-RMSE combination — but
    keeps the current production pair ("prod", "prod") on a tie, to avoid
    churning aliases for no real gain.
    """
    prod_rmse = combos[("prod", "prod")]
    best = min(combos, key=lambda k: combos[k])
    return best if combos[best] < prod_rmse else ("prod", "prod")


def promote_best_combination(
    client: MlflowClient,
    project: str,
    combos: dict[tuple[str, str], float] | None,
) -> dict[str, bool]:
    """
    Promote the best deployable (registered, casual) pair — which may be mixed
    (e.g. new registered + current-production casual). The current production
    pair is one of the four candidates, so this never makes the combined
    prediction worse: an alias only moves if another combination strictly beats
    it. combos is None on bootstrap (no production pair yet) → promote both new.

    When an alias actually moves, the deployed pair's combined RMSE is frozen
    as a "combined_rmse_baseline" tag on the registered production version —
    the baseline get_production_baseline_rmse reads. It is deliberately NOT
    refreshed when nothing is promoted: re-tagging on every attempt would let
    the baseline follow the data distribution, dulling is_performance_degraded
    exactly when the model is getting worse.

    Returns {"registered": bool, "casual": bool} — whether each slot moved to
    the newly trained version.
    """
    slots = {"registered": f"{project}-registered", "casual": f"{project}-casual"}
    new_versions = {
        slot: max(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version))
        for slot, name in slots.items()
    }

    # Bootstrap: no production pair yet → promote both new.
    if combos is None:
        for slot, name in slots.items():
            version = new_versions[slot].version
            logger.info(f"{name} — no production model found → promoting version {version}")
            client.set_registered_model_alias(name, "production", version)
        return {"registered": True, "casual": True}

    decision = choose_best_combination(combos)
    sources = {"registered": decision[0], "casual": decision[1]}
    logger.info(
        f"Best combination on current val: registered={sources['registered']}, "
        f"casual={sources['casual']} (rmse {combos[decision]:.4f} vs "
        f"production {combos[('prod', 'prod')]:.4f})"
    )

    promotions = {}
    for slot, name in slots.items():
        new_version = new_versions[slot].version
        if sources[slot] == "new":
            prod_version = client.get_model_version_by_alias(name, "production")
            client.set_registered_model_alias(name, "production", new_version)
            client.set_registered_model_alias(name, "archived", prod_version.version)
            promotions[slot] = True
        else:
            # New version was trained but not deployed → archive it, leave the
            # production alias untouched.
            client.set_registered_model_alias(name, "archived", new_version)
            promotions[slot] = False

    if promotions["registered"] or promotions["casual"]:
        registered_prod_version = (
            new_versions["registered"].version
            if promotions["registered"]
            else client.get_model_version_by_alias(slots["registered"], "production").version
        )
        client.set_model_version_tag(
            slots["registered"],
            registered_prod_version,
            "combined_rmse_baseline",
            str(combos[decision]),
        )

    return promotions


def snapshot_drift_reference(
    client: MlflowClient, project: str, train_csv_path: Path, promotions: dict[str, bool]
) -> None:
    """
    Snapshot the columns drift_detection.py needs (DRIFT_FEATURES + mnth)
    from the just-regenerated train.csv, and attach it as an artifact on the
    run of whichever model actually got promoted — registered if it moved,
    otherwise casual. Always using registered's run regardless of which
    model moved would, with a mixed pair, write over a snapshot already
    logged there for an unrelated promotion.

    Without this, drift_detection.py reads the live train.csv as its
    reference — which keeps advancing every time dvc repro runs, even for
    retrain attempts that don't get promoted. That desyncs the reference
    from what the production model actually learned from, understating
    drift relative to the model that's actually deployed.
    """
    from bike_sharing.monitoring.drift_detection import DRIFT_FEATURES

    df = pd.read_csv(train_csv_path)
    snapshot = df[DRIFT_FEATURES + ["mnth"]]
    model_name = "registered" if promotions["registered"] else "casual"
    version = client.get_model_version_by_alias(f"{project}-{model_name}", "production")

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / "drift_reference_snapshot.csv"
        snapshot.to_csv(snapshot_path, index=False)
        client.log_artifact(version.run_id, str(snapshot_path), artifact_path="drift_reference")

    logger.info(f"Drift reference snapshot logged to production run {version.run_id}")


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
        "timestamp": utc_now().isoformat(),
        "retrain_attempted": False,
        "skip_reason": None,
        "data_quality_checked": False,
        "data_quality_passed": None,
        "data_quality_issues": [],
        "new_rmse": None,
        "prod_rmse": None,
        "promoted_registered": None,
        "promoted_casual": None,
        "combination_rmses": None,
        "performance_degraded": None,
        "baseline_rmse": None,
        "live_rmse": None,
    }

    try:
        # ── Configure MLflow ──────────────────────────────────────────────────
        setup_mlflow()
        client = MlflowClient()

        # ── Check if retraining is needed ─────────────────────────────────────
        # Two independent triggers, combined with OR: input drift (a proxy —
        # can miss concept drift where X looks the same but the X→Y
        # relationship changed) and live performance degradation (the direct
        # signal, possible here because real actuals arrive ~1h later).
        # Over-triggering isn't risky — promotion below only takes effect if
        # the new pair is strictly better — so OR is the safer default.
        drift_says_retrain = should_retrain(drift_flag_path, force)

        baseline_rmse = get_production_baseline_rmse(client, cfg.project)
        performance_history_path = artifacts_dir / "monitoring" / "performance_history.csv"
        performance_degraded, live_rmse = is_performance_degraded(
            performance_history_path,
            baseline_rmse,
            cfg.monitoring.performance_degradation_threshold,
            cfg.forecast.primary_horizon,
        )
        outcome["performance_degraded"] = performance_degraded
        outcome["baseline_rmse"] = baseline_rmse
        outcome["live_rmse"] = live_rmse

        if not drift_says_retrain and not performance_degraded:
            reason = describe_drift_skip_reason(drift_flag_path)
            if baseline_rmse is not None and live_rmse is not None:
                pct = (live_rmse / baseline_rmse - 1) * 100
                reason += (
                    f"; performance stable (live rmse {live_rmse:.2f} vs "
                    f"baseline {baseline_rmse:.2f}, {pct:+.1f}%)"
                )
            outcome["skip_reason"] = reason
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

        # ── Promote the best combination ─────────────────────────────────────────
        # Evaluate all four deployable (registered, casual) combinations on the
        # combined RMSE over the current val set, and promote the best pair —
        # which may be mixed. The current production pair is one of the four, so
        # this never makes the combined prediction worse.
        val_df = pd.read_csv(Path(cfg.paths.processed_dir) / "val.csv")
        combos = evaluate_promotion_combinations(cfg.project, val_df, Path(cfg.paths.models_dir))

        if combos is None:
            logger.info("Bootstrap promotion — no production baseline to compare against")
        else:
            outcome["new_rmse"] = combos[("new", "new")]
            outcome["prod_rmse"] = combos[("prod", "prod")]
            outcome["combination_rmses"] = {f"{r}+{c}": v for (r, c), v in combos.items()}

        promotions = promote_best_combination(client, cfg.project, combos)
        outcome["promoted_registered"] = promotions["registered"]
        outcome["promoted_casual"] = promotions["casual"]

        if promotions["registered"] or promotions["casual"]:
            logger.info(
                f"Promoted — registered: {promotions['registered']}, casual: {promotions['casual']}"
            )
            snapshot_drift_reference(
                client, cfg.project, Path(cfg.paths.processed_dir) / "train.csv", promotions
            )
        else:
            logger.warning("No model promoted — previous production pair retained")

        run_command(["dvc", "freeze", "build_features"])

        # Advance the data boundary so the next run measures new data from here on.
        write_retrain_marker(hour_past_path, marker_path)

        logger.info("Retraining complete")
    finally:
        write_retrain_outcome(outcome, outcome_path)


if __name__ == "__main__":
    main()
