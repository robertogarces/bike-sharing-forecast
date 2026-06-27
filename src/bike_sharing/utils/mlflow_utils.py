import os
from pathlib import Path
from dotenv import load_dotenv


def setup_mlflow() -> None:
    """
    Configure MLflow tracking URI from environment variables.

    Loads credentials from .env file if present (local development),
    or uses environment variables directly (GitHub Actions, Docker).

    Required environment variables:
    - MLFLOW_TRACKING_URI
    - MLFLOW_TRACKING_USERNAME
    - MLFLOW_TRACKING_PASSWORD
    """
    # Load .env file if it exists (local development)
    env_path = Path(__file__).parents[3] / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. "
            "Add it to your .env file or environment variables."
        )

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)

    import logging
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azureml").setLevel(logging.WARNING)
