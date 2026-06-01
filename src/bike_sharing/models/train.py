import logging
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
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

logger = logging.getLogger(__name__)

FEATURES = [
    "hr_sin", "hr_cos",
    "hr_workday", "hr_weekend", "hr_x_season",
    "temp", "hum", "weathersit", "yr",
    "cnt_lag_1", "cnt_lag_2", "cnt_lag_3",
    "cnt_lag_8", "cnt_lag_24", "cnt_lag_168",
    "cnt_rolling_mean_24", "cnt_rolling_mean_168",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, RMSLE and R² on the original cnt scale.

    If the model was trained on log(cnt+1), y_pred must be
    back-transformed with expm1 before calling this function.
    """
    return {
        "rmse":  np.sqrt(mean_squared_error(y_true, y_pred)),
        "rmsle": np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None))),
        "r2":    r2_score(y_true, y_pred),
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
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val_cnt: pd.Series,
    fixed_params: dict,
    search_space: DictConfig,
    use_log_target: bool,
):
    """
    Build an Optuna objective function for LightGBM hyperparameter tuning.

    Each trial is logged as a nested MLflow run under the parent experiment.
    The objective metric is RMSLE on the validation set (original cnt scale).
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            **fixed_params,
            "num_leaves":        trial.suggest_int("num_leaves",         search_space.num_leaves.low,        search_space.num_leaves.high),
            "max_depth":         trial.suggest_int("max_depth",          search_space.max_depth.low,         search_space.max_depth.high),
            "learning_rate":     trial.suggest_float("learning_rate",    search_space.learning_rate.low,     search_space.learning_rate.high,     log=search_space.learning_rate.log),
            "n_estimators":      trial.suggest_int("n_estimators",       search_space.n_estimators.low,      search_space.n_estimators.high),
            "min_child_samples": trial.suggest_int("min_child_samples",  search_space.min_child_samples.low, search_space.min_child_samples.high),
            "subsample":         trial.suggest_float("subsample",        search_space.subsample.low,         search_space.subsample.high),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", search_space.colsample_bytree.low,  search_space.colsample_bytree.high),
            "reg_alpha":         trial.suggest_float("reg_alpha",        search_space.reg_alpha.low,         search_space.reg_alpha.high,         log=search_space.reg_alpha.log),
            "reg_lambda":        trial.suggest_float("reg_lambda",       search_space.reg_lambda.low,        search_space.reg_lambda.high,        log=search_space.reg_lambda.log),
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        if use_log_target:
            pred = np.expm1(pred)

        metrics = compute_metrics(y_val_cnt.values, pred)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

        return metrics["rmsle"]

    return objective


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    processed_dir    = Path(cfg.dataset.processed_dir)
    artifacts_dir    = Path("artifacts")
    best_params_path = Path(cfg.model.best_params_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading processed data")
    train = pd.read_csv(processed_dir / "train.csv")
    val   = pd.read_csv(processed_dir / "val.csv")

    use_log_target = cfg.features.target == "log_cnt"

    X_train   = train[FEATURES]
    y_train   = train["target"]
    X_val     = val[FEATURES]
    y_val_cnt = val["cnt"]

    fixed_params = dict(cfg.model.lgbm)

    # ── MLflow parent run ─────────────────────────────────────────────────────
    mlflow.set_experiment(cfg.project)

    with mlflow.start_run(run_name="lgbm_optuna") as parent_run:
        mlflow.log_params({
            "tune":        cfg.model.tune,
            "target":      cfg.features.target,
            "split_ratio": cfg.features.split_ratio,
            "n_features":  len(FEATURES),
        })

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
                    X_train, y_train,
                    X_val, y_val_cnt,
                    fixed_params,
                    cfg.model.search_space,
                    use_log_target,
                ),
                n_trials=cfg.model.n_trials,
                show_progress_bar=True,
            )

            best_params = {**fixed_params, **study.best_params}
            logger.info(f"Best RMSLE: {study.best_value:.4f}")
            save_best_params(best_params, fixed_params, best_params_path)

        else:
            logger.info("Skipping Optuna — loading best params")
            best_params = load_best_params(best_params_path, fixed_params)

        # ── Train final model ─────────────────────────────────────────────────
        logger.info("Training final model with best hyperparameters")
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────────────────
        pred = final_model.predict(X_val)
        if use_log_target:
            pred = np.expm1(pred)

        metrics = compute_metrics(y_val_cnt.values, pred)
        logger.info(
            f"Final model — "
            f"RMSE: {metrics['rmse']:.2f} | "
            f"RMSLE: {metrics['rmsle']:.4f} | "
            f"R²: {metrics['r2']:.4f}"
        )

        # ── Log final model to MLflow ─────────────────────────────────────────
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(final_model, name="model")

        # ── Save model locally ────────────────────────────────────────────────
        model_path = artifacts_dir / "lgbm_model.txt"
        final_model.booster_.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()