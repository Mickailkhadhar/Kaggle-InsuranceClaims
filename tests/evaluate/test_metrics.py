from __future__ import annotations
import numpy as np
import pytest
from src.evaluate.metrics import rmse


def test_rmse_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_rmse_known_value() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert pytest.approx(rmse(y_true, y_pred), rel=1e-5) == np.sqrt(12.5)


def test_rmse_returns_float() -> None:
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 3.0])
    assert isinstance(rmse(y_true, y_pred), float)