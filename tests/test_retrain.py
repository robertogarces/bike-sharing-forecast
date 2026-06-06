import pytest
import json
import tempfile
from pathlib import Path

from bike_sharing.models.retrain import should_retrain


@pytest.fixture
def drift_flag_dir():
    """
    Creates a temporary directory with a drift_detected.json file.
    Returns both the directory and the flag path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        flag_path = Path(tmpdir) / "drift_detected.json"
        yield flag_path


def write_flag(flag_path: Path, content: dict) -> None:
    """Helper to write a drift flag JSON file."""
    with open(flag_path, "w") as f:
        json.dump(content, f)


def test_should_retrain_returns_true_when_drift_detected(drift_flag_dir):
    """
    should_retrain should return True when drift_detected is True in the flag.
    """
    write_flag(drift_flag_dir, {
        "drift_detected": True,
        "drift_share":    0.7,
        "threshold":      0.5,
    })

    assert should_retrain(drift_flag_dir, force=False) is True


def test_should_retrain_returns_false_when_no_drift(drift_flag_dir):
    """
    should_retrain should return False when drift_detected is False.
    """
    write_flag(drift_flag_dir, {
        "drift_detected": False,
        "drift_share":    0.2,
        "threshold":      0.5,
    })

    assert should_retrain(drift_flag_dir, force=False) is False


def test_should_retrain_returns_false_when_insufficient_data(drift_flag_dir):
    """
    should_retrain should return False when drift detection was skipped
    due to insufficient data — identified by the presence of 'reason' key.
    """
    write_flag(drift_flag_dir, {
        "drift_detected": False,
        "drift_share":    0.0,
        "threshold":      0.5,
        "reason":         "Not enough data (168 rows < 720 minimum)",
    })

    assert should_retrain(drift_flag_dir, force=False) is False


def test_should_retrain_force_overrides_no_drift(drift_flag_dir):
    """
    When force=True, should_retrain should return True regardless of
    what the drift flag says — used for scheduled retraining.
    """
    write_flag(drift_flag_dir, {
        "drift_detected": False,
        "drift_share":    0.0,
        "threshold":      0.5,
    })

    assert should_retrain(drift_flag_dir, force=True) is True


def test_should_retrain_returns_false_when_no_flag_exists(drift_flag_dir):
    """
    If drift_detected.json doesn't exist, should_retrain should return False
    and warn the user to run drift_detection.py first.
    """
    # Don't write any flag — file doesn't exist
    assert should_retrain(drift_flag_dir, force=False) is False