import json
import logging
import os

import mlflow.lightgbm
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES = [
    "hr_sin", "hr_cos",
    "hr_workday", "hr_weekend", "hr_x_season",
    "is_rush_hour", "days_since_start",
    "temp", "hum", "weathersit",
    "cnt_lag_1", "cnt_lag_2", "cnt_lag_3",
    "cnt_lag_8", "cnt_lag_24", "cnt_lag_48", "cnt_lag_72", "cnt_lag_168",
    "cnt_rolling_mean_24", "cnt_rolling_mean_168",
]

model_registered = None
model_casual = None


def init():
    global model_registered, model_casual
    project = os.getenv("PROJECT_NAME", "bike-sharing-forecast")
    version = os.getenv("MODEL_VERSION", "6")
    logger.info(f"Loading models v{version} from MLflow registry")
    model_registered = mlflow.lightgbm.load_model(f"models:/{project}-registered/{version}")
    model_casual     = mlflow.lightgbm.load_model(f"models:/{project}-casual/{version}")
    logger.info("Models loaded")


def run(raw_data):
    data = json.loads(raw_data)
    df = pd.DataFrame(data["data"])
    X = df[FEATURES]

    pred_registered = float(np.expm1(model_registered.predict(X))[0])
    pred_casual = float(np.expm1(model_casual.predict(X))[0])
    pred_total = max(0, pred_registered + pred_casual)

    return {
        "registered": round(pred_registered, 1),
        "casual": round(pred_casual, 1),
        "total": round(pred_total, 1),
    }
