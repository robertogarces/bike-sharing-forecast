"""
Sync deployment.yaml's MODEL_VERSION_REGISTERED/MODEL_VERSION_CASUAL to the
latest model versions in the Azure ML Model Registry.

Azure ML's MLflow-compatible registry doesn't support aliases (see
docs/azure.md), so there's no "production" pointer to read — this takes the
highest version number instead. That's unambiguous here because the Azure
training job is a manual, occasional demo step, not the continuously
running retrain pipeline, so there is only ever one model to promote to.

Run manually after a new Azure training job (azure/job.yaml), before
triggering the deploy-azure.yml workflow. Requires MLFLOW_TRACKING_URI set
to the Azure ML workspace's tracking URI and an active `az login` session.
"""

import os
import re
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

PROJECT = "bike-sharing-forecast"
DEPLOYMENT_PATH = Path(__file__).parent / "deployment.yaml"


def latest_version(client: MlflowClient, name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    return max(versions, key=lambda v: int(v.version)).version


def update_deployment_versions(content: str, version_registered: str, version_casual: str) -> str:
    """Rewrite just the two MODEL_VERSION_* lines, leaving the rest of the YAML untouched."""
    content = re.sub(
        r'MODEL_VERSION_REGISTERED: ".*"',
        f'MODEL_VERSION_REGISTERED: "{version_registered}"',
        content,
    )
    content = re.sub(
        r'MODEL_VERSION_CASUAL: ".*"',
        f'MODEL_VERSION_CASUAL: "{version_casual}"',
        content,
    )
    return content


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri.startswith("azureml://"):
        sys.exit("MLFLOW_TRACKING_URI must point at the Azure ML workspace (azureml://...)")

    client = MlflowClient()
    version_registered = latest_version(client, f"{PROJECT}-registered")
    version_casual = latest_version(client, f"{PROJECT}-casual")

    content = update_deployment_versions(
        DEPLOYMENT_PATH.read_text(), version_registered, version_casual
    )
    DEPLOYMENT_PATH.write_text(content)

    print(f"deployment.yaml updated — registered v{version_registered}, casual v{version_casual}")


if __name__ == "__main__":
    main()
