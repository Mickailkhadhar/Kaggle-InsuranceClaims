from __future__ import annotations
import pandas as pd
from loguru import logger
from xgboost import XGBRegressor

class Trainer:
    """Fits an XGBoost regressor using config-driven parameters.

    Args:
        config: Model config dict loaded from model.yaml.
    """
    def __init__(self, config: dict) -> None:
        self._config = config

    def fit(self, X: pd.DataFrame, y: pd.Series, extra_params: dict | None = None) -> XGBRegressor:
        """Fit an XGBRegressor using config params, optionally overridden by extra_params.

        Args:
            X: Feature matrix.
            y: Target series.
            extra_params: Optional dict of params to override/extend the config ones (e.g. from Optuna).

        Returns:
            Fitted XGBRegressor instance.
        """
        params = {**self._config["xgboost"], **(extra_params or {})}
        model = XGBRegressor(**params)
        model.fit(X, y)
        logger.info(f"Model fitted — {len(X)} rows, {X.shape[1]} features")
        return model