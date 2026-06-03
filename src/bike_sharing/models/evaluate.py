import logging
from pathlib import Path

import hydra
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from omegaconf import DictConfig

from bike_sharing.models.train import compute_metrics, FEATURES

logger = logging.getLogger(__name__)

def plot_shap_summary(model: lgb.Booster, X_val: pd.DataFrame, path: Path) -> None:
    """
    SHAP summary plot showing feature importance and impact direction.

    Each point represents one prediction. Position on the x-axis shows
    whether the feature pushed the prediction higher (positive) or lower
    (negative). Color shows the feature value — red = high, blue = low.

    Unlike RF feature importance, SHAP values are based on game theory
    and account for feature interactions, making them a more honest
    measure of each feature's contribution.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary plot to {path}")

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """
    Residuals vs predicted values plot.
    A well-calibrated model shows residuals randomly scattered around zero
    with no systematic pattern. A funnel shape indicates heteroscedasticity
    — larger errors at higher demand levels.
    """
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted cnt")
    ax.set_ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved residuals plot to {path}")


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """
    Actual vs predicted scatter plot.
    Points aligned along the diagonal indicate accurate predictions.
    Systematic deviation above or below the diagonal indicates bias.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.2, s=10)
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual cnt")
    ax.set_ylabel("Predicted cnt")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved actual vs predicted plot to {path}")


def plot_demand_over_time(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.Series, path: Path) -> None:
    """
    Actual vs predicted demand over the validation period.
    Shows how well the model tracks the temporal pattern of demand,
    including peaks, troughs, and seasonal effects.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, y_true, label="Actual",    alpha=0.7, linewidth=0.8)
    ax.plot(dates, y_pred, label="Predicted", alpha=0.7, linewidth=0.8)
    ax.set_title("Actual vs Predicted Demand — Validation Period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bikes rented per hour")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved demand over time plot to {path}")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    processed_dir = Path(cfg.dataset.processed_dir)
    artifacts_dir = Path("artifacts")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading validation data")
    val = pd.read_csv(processed_dir / "val.csv")
    val["dteday"] = pd.to_datetime(val["dteday"])

    X_val     = val[FEATURES]
    y_val_cnt = val["cnt"].values

    # ── Load models ───────────────────────────────────────────────────────────
    logger.info("Loading models from artifacts")
    model_registered = lgb.Booster(model_file=str(artifacts_dir / "lgbm_registered.txt"))
    model_casual     = lgb.Booster(model_file=str(artifacts_dir / "lgbm_casual.txt"))

    # ── Predict ───────────────────────────────────────────────────────────────
    pred_registered = np.expm1(model_registered.predict(X_val))
    pred_casual     = np.expm1(model_casual.predict(X_val))
    pred_combined   = np.clip(pred_registered + pred_casual, 0, None)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_val_cnt, pred_combined)
    logger.info(
        f"Evaluation — "
        f"RMSE: {metrics['rmse']:.2f} | "
        f"RMSLE: {metrics['rmsle']:.4f} | "
        f"R²: {metrics['r2']:.4f}"
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating evaluation plots")
    plot_residuals(
        y_val_cnt, pred_combined,
        artifacts_dir / "residuals.png"
    )
    plot_actual_vs_predicted(
        y_val_cnt, pred_combined,
        artifacts_dir / "actual_vs_predicted.png"
    )
    plot_demand_over_time(
        y_val_cnt, pred_combined,
        val["dteday"],
        artifacts_dir / "demand_over_time.png"
    )

    logger.info("Generating SHAP plots")
    plot_shap_summary(
        model_registered, X_val,
        artifacts_dir / "shap_registered.png"
    )
    plot_shap_summary(
        model_casual, X_val,
        artifacts_dir / "shap_casual.png"
)

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    mlflow.set_experiment(cfg.project)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(artifacts_dir / "residuals.png"))
        mlflow.log_artifact(str(artifacts_dir / "actual_vs_predicted.png"))
        mlflow.log_artifact(str(artifacts_dir / "demand_over_time.png"))
        mlflow.log_artifact(str(artifacts_dir / "shap_registered.png"))
        mlflow.log_artifact(str(artifacts_dir / "shap_casual.png"))
        logger.info("Logged metrics and artifacts to MLflow")


if __name__ == "__main__":
    main()