import logging
import tempfile
import yaml
from pathlib import Path

import hydra
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import (
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
    mean_absolute_error,
)
from bike_sharing.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

FEATURES = [
    # Calendar
    "hr_sin",
    "hr_cos",
    "hr_workday",
    "hr_weekend",
    "hr_x_season",
    # New
    "is_rush_hour",
    "days_since_start",
    # Context
    "temp",
    "hum",
    "weathersit",
    # Lags
    "cnt_lag_1",
    "cnt_lag_2",
    "cnt_lag_3",
    "cnt_lag_8",
    "cnt_lag_24",
    "cnt_lag_48",
    "cnt_lag_72",
    "cnt_lag_168",
    "cnt_rolling_mean_24",
    "cnt_rolling_mean_168",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, RMSLE, R² and MAE on the original cnt scale.

    Predictions must be on the original scale (not log) before calling.
    y_pred is clipped at 0 to avoid negative values breaking RMSLE.
    """
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "rmsle": np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None))),
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def load_best_params(path: Path, fixed_params: dict) -> dict:
    """
    Load best hyperparameters from a previous Optuna run.
    Falls back to fixed_params (LightGBM defaults) if file doesn't exist.
    """
    if path.exists():
        logger.info(f"Loading best params from {path}")
        with open(path) as f:
            return {**fixed_params, **yaml.safe_load(f)}
    else:
        logger.warning(f"No best_params found at {path} — using default LightGBM params")
        return fixed_params


def save_best_params(params: dict, fixed_params: dict, path: Path) -> None:
    """
    Save tuned hyperparameters to yaml, excluding fixed params.
    Only the Optuna-tuned params are stored.
    """
    tuned_only = {k: v for k, v in params.items() if k not in fixed_params}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(tuned_only, f)
    logger.info(f"Best params saved to {path}")


def make_objective(
    X_train: pd.DataFrame,
    y_train_registered: pd.Series,
    y_train_casual: pd.Series,
    X_val: pd.DataFrame,
    y_val_cnt: pd.Series,
    fixed_params: dict,
    search_space: DictConfig,
):
    """
    Build an Optuna objective function for LightGBM hyperparameter tuning.

    Tuning is performed on the registered target (dominant population, ~80%
    of total demand). The same hyperparameters are reused for the casual model.

    Each trial is logged as a nested MLflow run. The objective metric is
    RMSLE on the validation set (original cnt scale).
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            **fixed_params,
            "num_leaves": trial.suggest_int(
                "num_leaves", search_space.num_leaves.low, search_space.num_leaves.high
            ),
            "max_depth": trial.suggest_int(
                "max_depth", search_space.max_depth.low, search_space.max_depth.high
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                search_space.learning_rate.low,
                search_space.learning_rate.high,
                log=search_space.learning_rate.log,
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", search_space.n_estimators.low, search_space.n_estimators.high
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                search_space.min_child_samples.low,
                search_space.min_child_samples.high,
            ),
            "subsample": trial.suggest_float(
                "subsample", search_space.subsample.low, search_space.subsample.high
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                search_space.colsample_bytree.low,
                search_space.colsample_bytree.high,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                search_space.reg_alpha.low,
                search_space.reg_alpha.high,
                log=search_space.reg_alpha.log,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                search_space.reg_lambda.low,
                search_space.reg_lambda.high,
                log=search_space.reg_lambda.log,
            ),
        }

        model_reg = lgb.LGBMRegressor(**params)
        model_reg.fit(X_train, y_train_registered)

        model_cas = lgb.LGBMRegressor(**params)
        model_cas.fit(X_train, y_train_casual)

        pred_combined = np.expm1(model_reg.predict(X_val)) + np.expm1(model_cas.predict(X_val))
        metrics = compute_metrics(y_val_cnt.values, pred_combined)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

        return metrics["rmse"]

    return objective


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    processed_dir = Path(cfg.paths.processed_dir)
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    best_params_path = Path(cfg.model.best_params_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.models_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading processed data")
    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "val.csv")

    X_train = train[FEATURES]
    X_val = val[FEATURES]
    y_val_cnt = val["cnt"]

    fixed_params = dict(cfg.model.lgbm)

    # ── MLflow parent run ─────────────────────────────────────────────────────
    setup_mlflow()
    mlflow.set_experiment(cfg.project)

    with mlflow.start_run(run_name="lgbm_optuna_split"):
        mlflow.log_params(
            {
                "tune": cfg.model.tune,
                "target": cfg.features.target,
                "split_ratio": cfg.features.split_ratio,
                "n_features": len(FEATURES),
                "strategy": "split_registered_casual",
            }
        )

        # ── Tune or load ──────────────────────────────────────────────────────
        if cfg.model.tune:
            logger.info(f"Starting Optuna study — {cfg.model.n_trials} trials")
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=cfg.model.random_state),
            )
            study.optimize(
                make_objective(
                    X_train,
                    train["log_registered"],
                    train["log_casual"],
                    X_val,
                    y_val_cnt,
                    fixed_params,
                    cfg.model.search_space,
                ),
                n_trials=cfg.model.n_trials,
                show_progress_bar=True,
            )

            best_params = {**fixed_params, **study.best_params}
            logger.info(f"Best RMSE: {study.best_value:.4f}")
            save_best_params(best_params, fixed_params, best_params_path)

        else:
            logger.info("Skipping Optuna — loading best params")
            best_params = load_best_params(best_params_path, fixed_params)

        # ── Train final models ────────────────────────────────────────────────
        logger.info("Training model — registered users")
        model_registered = lgb.LGBMRegressor(**best_params)
        model_registered.fit(X_train, train["log_registered"])

        logger.info("Training model — casual users")
        model_casual = lgb.LGBMRegressor(**best_params)
        model_casual.fit(X_train, train["log_casual"])

        # ── Evaluate ──────────────────────────────────────────────────────────
        pred_registered = np.expm1(model_registered.predict(X_val))
        pred_casual = np.expm1(model_casual.predict(X_val))
        pred_combined = pred_registered + pred_casual

        metrics = compute_metrics(y_val_cnt.values, pred_combined)
        logger.info(
            f"Final model — "
            f"RMSE: {metrics['rmse']:.2f} | "
            f"RMSLE: {metrics['rmsle']:.4f} | "
            f"R²: {metrics['r2']:.4f}"
        )

        # ── Log to MLflow ─────────────────────────────────────────────────────
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        run_id = mlflow.active_run().info.run_id
        client = mlflow.MlflowClient()

        for model, name in [
            (model_registered, "registered"),
            (model_casual, "casual"),
        ]:
            artifact_path = f"model_{name}"
            registered_name = f"{cfg.project}-{name}"

            with tempfile.TemporaryDirectory() as tmpdir:
                mlflow.lightgbm.save_model(model, f"{tmpdir}/{artifact_path}")
                mlflow.log_artifacts(f"{tmpdir}/{artifact_path}", artifact_path=artifact_path)

            try:
                client.create_registered_model(registered_name)
            except Exception:
                pass

            client.create_model_version(
                name=registered_name,
                source=mlflow.get_artifact_uri(artifact_path),
                run_id=run_id,
            )

        # ── Save models locally ───────────────────────────────────────────────
        model_registered.booster_.save_model(Path(cfg.paths.models_dir) / "lgbm_registered.txt")
        model_casual.booster_.save_model(Path(cfg.paths.models_dir) / "lgbm_casual.txt")
        logger.info(f"Models saved to {cfg.paths.models_dir}")

        # ── Save models locally ───────────────────────────────────────────────
        model_registered.booster_.save_model(Path(cfg.paths.models_dir) / "lgbm_registered.txt")
        model_casual.booster_.save_model(Path(cfg.paths.models_dir) / "lgbm_casual.txt")
        logger.info(f"Models saved to {cfg.paths.models_dir}")


if __name__ == "__main__":
    main()
