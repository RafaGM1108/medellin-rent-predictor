"""Cross-validation and metrics shared by every model.

Models are scikit-learn estimators trained on ``log_rent`` (decision 1 in
``docs/decisions.md``); predictions are turned back into COP before scoring, so MAE, RMSE
and MAPE are in COP, the unit a renter understands. Every model is scored on the same folds.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from medellin_rent.features.build import FEATURES
from medellin_rent.features.model_input import LOG_TARGET, TARGET

ModelFactory = Callable[[], BaseEstimator]


def metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """MAE, RMSE and MAPE (%) of rent predictions in COP."""
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    error = pred - true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error) / true) * 100),
    }


def folds(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The k shuffled (train, validation) index pairs used for every model."""
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)))


def predict_rent(model: Any, data: pd.DataFrame) -> np.ndarray:
    """Predict monthly rent in COP with a model trained on ``log_rent``."""
    return np.asarray(np.exp(model.predict(data[FEATURES])), dtype=float)


def cross_validate(
    factory: ModelFactory, train: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """Fit a fresh model on each fold and score it on the held-out part.

    Args:
        factory: Returns a new, unfitted estimator.
        train: Training set from ``05_model_input`` (features + targets).
        splits: Output of :func:`folds`.

    Returns:
        One row per fold with ``fold``, ``n`` and the metrics.
    """
    rows = []
    for i, (fit_idx, val_idx) in enumerate(splits):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        model = factory().fit(fit[FEATURES], fit[LOG_TARGET])
        rows.append({"fold": i, "n": len(val), **metrics(val[TARGET], predict_rent(model, val))})
    return pd.DataFrame(rows)


def summarize(cv: pd.DataFrame) -> dict[str, float]:
    """Mean and standard deviation of each metric across folds."""
    return {
        f"{name}_{stat}": float(getattr(cv[name], stat)())
        for name in ("mae", "rmse", "mape")
        for stat in ("mean", "std")
    }
