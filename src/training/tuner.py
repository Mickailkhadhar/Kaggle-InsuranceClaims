from __future__ import annotations
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from src.evaluate.metrics import rmse


class Tuner:
    """Hyperparameter tuning for XGBoost using Optuna with stratified K-Fold CV.

    Args:
        config: Model config dict loaded from model.yaml.
    """
    def __init__(self, config: dict) -> None:
        self._config = config
        self._study: optuna.Study | None = None

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optuna objective: mean CV RMSE for a given set of hyperparameters.

        Args:
            trial: Optuna trial object.
            X: Feature matrix.
            y: Target series.

        Returns:
            Mean CV RMSE across all folds.
        """
        space = self._config["search_space"]
        params = {
            "random_state": self._config["xgboost"]["random_state"],
            "n_jobs": self._config["xgboost"]["n_jobs"],
            "enable_categorical": self._config["xgboost"]["enable_categorical"],
            "objective": self._config["xgboost"]["objective"],
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "subsample": trial.suggest_float("subsample", *space["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *space["min_child_weight"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"]),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", *space["tweedie_variance_power"]),
            "gamma": trial.suggest_float("gamma", *space["gamma"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", *space["colsample_bylevel"]),
            "max_delta_step": trial.suggest_float("max_delta_step", *space["max_delta_step"]),
        }
        bins = pd.qcut(y, q=self._config["cv"]["n_folds"], labels=False, duplicates="drop")
        kf = StratifiedKFold(
            n_splits=self._config["cv"]["n_folds"],
            shuffle=True,
            random_state=self._config["xgboost"]["random_state"],
        )
        scores = []
        for train_idx, val_idx in kf.split(X, bins):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(rmse(y_val.to_numpy(), preds))
        return float(np.mean(scores))

    def tune(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Run Optuna hyperparameter search.

        Args:
            X: Feature matrix.
            y: Target series.

        Returns:
            Best params dict found by Optuna.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = self._config["cv"]["n_trials"]
        n_folds = self._config["cv"]["n_folds"]
        logger.info(f"Starting tuning — {n_trials} trials, {n_folds}-fold CV")

        self._study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        self._study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=n_trials,
        )
        logger.info(f"Best RMSE: {self._study.best_value:.4f} | Params: {self._study.best_params}")
        return self._study.best_params

    def trials_report(self) -> pd.DataFrame:
        """Return all Optuna trials with their params and CV RMSE.

        Returns:
            DataFrame with one row per trial, sorted by RMSE ascending.

        Raises:
            RuntimeError: If tune() has not been called yet.
        """
        if self._study is None:
            raise RuntimeError("Call tune() before trials_report().")
        rows = []
        for trial in self._study.trials:
            row = {"trial": trial.number, "rmse": trial.value}
            row.update(trial.params)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)