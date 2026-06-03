from __future__ import annotations
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
