"""Baseline: median rent of the comuna, per m² when the area is known (decision 9).

Every other model must beat it. Medians are learned in :meth:`fit`, so in cross-validation
they come from the training folds only.
"""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class ComunaMedianBaseline(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """Predicts ``log(rent)`` from the comuna's median rent (per m² times area when known)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:  # noqa: N803 - scikit-learn API
        """Learn median rent and median rent per m² per comuna (plus global fallbacks).

        Args:
            X: Features with ``comuna_code`` and ``area_m2``.
            y: ``log_rent``.
        """
        rent = np.exp(np.asarray(y, dtype=float))
        df = pd.DataFrame(
            {"comuna": X["comuna_code"].astype("str").to_numpy(), "rent": rent}
        ).assign(per_m2=rent / X["area_m2"].to_numpy())
        self.median_rent_ = df.groupby("comuna")["rent"].median().to_dict()
        self.median_per_m2_ = (
            df.dropna(subset=["per_m2"]).groupby("comuna")["per_m2"].median().to_dict()
        )
        self.global_rent_ = float(df["rent"].median())
        self.global_per_m2_ = float(df["per_m2"].median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803 - scikit-learn API
        """Return predicted ``log(rent)``."""
        comuna = X["comuna_code"].astype("str")
        per_m2 = comuna.map(self.median_per_m2_).fillna(self.global_per_m2_).to_numpy()
        rent = comuna.map(self.median_rent_).fillna(self.global_rent_).to_numpy()
        area = X["area_m2"].to_numpy(dtype=float)
        prediction = np.where(np.isnan(area), rent, per_m2 * area)
        return np.log(prediction)
