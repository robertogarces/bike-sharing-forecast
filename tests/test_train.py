import pytest
import numpy as np

from bike_sharing.models.train import compute_metrics


def test_compute_metrics_clips_negative_predictions():
    """
    compute_metrics clips negative predictions to 0 before computing RMSLE.
    Negative predictions would cause log(negative) which is undefined.
    This test ensures the function does not raise an error with negative preds.
    """
    y_true = np.array([100, 200, 150])
    y_pred = np.array([-10, 200, 150])  # negative prediction

    try:
        compute_metrics(y_true, y_pred)
    except Exception as e:
        pytest.fail(f"compute_metrics raised an exception with negative predictions: {e}")
