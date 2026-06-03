from __future__ import annotations
import pandas as pd
import pytest
from src.training.tuner import Tuner


@pytest.fixture
def model_config() -> dict:
    """Minimal model config for fast test runs."""
    return {
        "xgboost": {
            "random_state": 42,
            "n_jobs": 1,
            "enable_categorical": False,
            "objective": "reg:squarederror",
        },
        "cv": {"n_folds": 3, "n_trials": 2},
        "search_space": {
            "n_estimators": [10, 20],
            "learning_rate": [0.01, 0.3],
            "max_depth": [3, 6],
            "subsample": [0.6, 1.0],
            "colsample_bytree": [0.6, 1.0],
            "min_child_weight": [1, 5],
            "reg_alpha": [0.0, 1.0],
            "reg_lambda": [0.0, 1.0],
            "tweedie_variance_power": [1.0, 2.0],
            "gamma": [0.0, 1.0],
            "colsample_bylevel": [0.6, 1.0],
            "max_delta_step": [0.0, 1.0],
        },
    }


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small dataset for fast test runs."""
    X = pd.DataFrame({"f1": range(30), "f2": range(30, 60)})
    y = pd.Series([float(i * 10 + 1) for i in range(30)])
    return X, y


def test_tune_returns_dict(model_config: dict, sample_data: tuple) -> None:
    X, y = sample_data
    best_params = Tuner(model_config).tune(X, y)
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params


def test_trials_report_shape(model_config: dict, sample_data: tuple) -> None:
    X, y = sample_data
    tuner = Tuner(model_config)
    tuner.tune(X, y)
    report = tuner.trials_report()
    assert isinstance(report, pd.DataFrame)
    assert len(report) == model_config["cv"]["n_trials"]
    assert "rmse" in report.columns


def test_trials_report_sorted_by_rmse(model_config: dict, sample_data: tuple) -> None:
    X, y = sample_data
    tuner = Tuner(model_config)
    tuner.tune(X, y)
    report = tuner.trials_report()
    assert report["rmse"].is_monotonic_increasing


def test_trials_report_raises_before_tune(model_config: dict) -> None:
    with pytest.raises(RuntimeError):
        Tuner(model_config).trials_report()
