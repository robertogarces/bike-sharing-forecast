import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "azure" / "sync_deployment_versions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sync_deployment_versions", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_update_deployment_versions_replaces_only_the_two_version_lines():
    """
    Must update MODEL_VERSION_REGISTERED/CASUAL without touching anything
    else in the file — comments, $schema, MLFLOW_TRACKING_URI must survive
    byte-for-byte, since this rewrites a real, hand-maintained deployment.yaml.
    """
    module = _load_module()
    content = (
        "# Deployment attached to the endpoint\n"
        "$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json\n"
        "environment_variables:\n"
        "  MLFLOW_TRACKING_URI: azureml://example\n"
        '  MODEL_VERSION_REGISTERED: "6"\n'
        '  MODEL_VERSION_CASUAL: "6"\n'
    )

    updated = module.update_deployment_versions(content, "9", "7")

    assert '  MODEL_VERSION_REGISTERED: "9"\n' in updated
    assert '  MODEL_VERSION_CASUAL: "7"\n' in updated
    assert "# Deployment attached to the endpoint\n" in updated
    assert "  MLFLOW_TRACKING_URI: azureml://example\n" in updated
