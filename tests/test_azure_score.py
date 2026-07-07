import importlib.util
from pathlib import Path

from bike_sharing.models.train import FEATURES

SCORE_PATH = Path(__file__).parents[1] / "azure" / "score.py"


def _load_score_module():
    spec = importlib.util.spec_from_file_location("azure_score", SCORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_azure_score_features_matches_train_features():
    """
    azure/score.py keeps its own copy of FEATURES because the Azure ML
    endpoint's isolated environment (conda_endpoint.yaml) doesn't install
    the bike_sharing package — it can't import from train.py at runtime.
    This test converts a silent desync (features change in training, Azure
    endpoint quietly falls behind) into a loud, caught-before-deploy failure.
    """
    score_module = _load_score_module()
    assert score_module.FEATURES == FEATURES
