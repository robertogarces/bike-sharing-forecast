import pytest
import numpy as np

from bike_sharing.models.train import compute_metrics


def test_compute_metrics_returns_required_keys():
    """
    compute_metrics should always return a dict with rmse, rmsle and r2.
    If any key is missing, downstream code that reads metrics.json will break.
    """
    y_true = np.array([100, 200, 150, 300])
    y_pred = np.array([110, 190, 160, 280])

    result = compute_metrics(y_true, y_pred)

    assert "rmse" in result
    assert "rmsle" in result
    assert "r2" in result


def test_compute_metrics_perfect_prediction():
    """
    When predictions are identical to true values, RMSE and RMSLE should
    be 0 and R² should be 1.
    """
    y_true = np.array([100, 200, 150, 300])
    y_pred = np.array([100, 200, 150, 300])

    result = compute_metrics(y_true, y_pred)

    assert result["rmse"] == pytest.approx(0.0)
    assert result["rmsle"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


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


def test_compute_metrics_rmse_is_positive():
    """
    RMSE should always be a positive number regardless of prediction direction.
    """
    y_true = np.array([100, 200, 150, 300])
    y_pred = np.array([50, 250, 100, 350])

    result = compute_metrics(y_true, y_pred)

    assert result["rmse"] > 0
