from __future__ import annotations
import pandas as pd
import pytest
from xgboost import XGBRegressor
from src.training.trainer import Trainer


@pytest.fixture
def model_config() -> dict:
    """Minimal model config for fast test runs."""
    return {
        "xgboost": {"random_state": 42, "n_jobs": 1, "n_estimators": 10},
    }


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small dataset for fast test runs."""
    X = pd.DataFrame({"f1": range(20), "f2": range(20, 40)})
    y = pd.Series([float(i * 5 + 1) for i in range(20)])
    return X, y


def test_fit_returns_xgb_regressor(model_config: dict, sample_data: tuple) -> None:
    X, y = sample_data
    model = Trainer(model_config).fit(X, y)
    assert isinstance(model, XGBRegressor)


def test_fit_produces_predictions(model_config: dict, sample_data: tuple) -> None:
    X, y = sample_data
    model = Trainer(model_config).fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
