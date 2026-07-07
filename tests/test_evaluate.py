import numpy as np
import pandas as pd
import pytest

from bike_sharing.models.evaluate import evaluate_models
from bike_sharing.models.train import FEATURES


class _FakeModel:
    """Model stub whose predict() returns a fixed, caller-supplied array (log-space)."""

    def __init__(self, log_preds):
        self.log_preds = log_preds

    def predict(self, X):
        return self.log_preds


def _make_val(cnt):
    return pd.DataFrame({**{f: 0.0 for f in FEATURES}, "cnt": cnt})


def test_evaluate_models_perfect_prediction_gives_zero_rmse():
    """
    evaluate_models combines registered + casual predictions the same way
    _combination_rmses does (expm1, sum, clip). If both fakes predict
    exactly the log-space values that reconstruct cnt, RMSE must be 0.
    """
    val = _make_val([100, 200, 150])

    model_registered = _FakeModel(np.log1p([60, 150, 100]))
    model_casual = _FakeModel(np.log1p([40, 50, 50]))

    metrics, pred_combined = evaluate_models(model_registered, model_casual, val)

    assert metrics["rmse"] == pytest.approx(0.0)
    assert pred_combined == pytest.approx([100, 200, 150])


def test_evaluate_models_uses_both_registered_and_casual_predictions():
    """
    A bug that swaps registered/casual or only wires up one model would
    silently corrupt metrics.json (a direct input to retrain.py's promotion
    decision) without this — the combined prediction must reflect the SUM
    of both, not either one alone.
    """
    val = _make_val([100, 100])

    model_registered = _FakeModel(np.log1p([80, 80]))
    model_casual = _FakeModel(np.log1p([20, 20]))

    _, pred_combined = evaluate_models(model_registered, model_casual, val)

    assert pred_combined == pytest.approx([100, 100])
    assert not np.allclose(pred_combined, 80)
    assert not np.allclose(pred_combined, 20)
