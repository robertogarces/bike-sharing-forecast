"""
Scoring script for the Azure ML Managed Online Endpoint.

Azure ML calls ``init()`` once at container startup (after environment variables
are injected) to load both LightGBM models from the Model Registry, then calls
``run(raw_data)`` per request. ``run()`` mirrors the production ``predict.py``
logic: it back-transforms each model's log prediction with ``expm1`` and returns
the registered/casual split plus the total.

Models are loaded by explicit version (``MODEL_VERSION_REGISTERED`` /
``MODEL_VERSION_CASUAL``) rather than by alias, because Azure ML's MLflow API
does not implement model aliases. See docs/azure.md.

Registered and casual are pinned independently — since backlog #10, the
production pair can be mixed (each model promoted on its own merit), so a
single version number can no longer describe both.
"""

import json
import logging
import os

import mlflow.lightgbm
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES = [
    "hr_sin",
    "hr_cos",
    "hr_workday",
    "hr_weekend",
    "hr_x_season",
    "is_rush_hour",
    "days_since_start",
    "temp",
    "hum",
    "weathersit",
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

model_registered = None
model_casual = None


def init():
    global model_registered, model_casual
    project = os.getenv("PROJECT_NAME", "bike-sharing-forecast")
    version_registered = os.getenv("MODEL_VERSION_REGISTERED")
    version_casual = os.getenv("MODEL_VERSION_CASUAL")
    if version_registered is None or version_casual is None:
        raise RuntimeError(
            "MODEL_VERSION_REGISTERED and MODEL_VERSION_CASUAL environment variables must be set."
        )
    logger.info(f"Loading registered v{version_registered}, casual v{version_casual}")
    model_registered = mlflow.lightgbm.load_model(
        f"models:/{project}-registered/{version_registered}"
    )
    model_casual = mlflow.lightgbm.load_model(f"models:/{project}-casual/{version_casual}")
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
