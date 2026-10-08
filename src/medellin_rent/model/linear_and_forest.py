"""Ridge and Random Forest as scikit-learn pipelines.

Both share the missing-value handling of decision 3 (``docs/decisions.md``): the area is
imputed with the median area of listings with the same number of bedrooms, learned in
``fit`` (training folds only); ``area_missing`` already flags the imputed rows. Barrio is
not used here (decision 5 keeps it for the LightGBM experiment).
"""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from medellin_rent.features.build import AMENITIES, CATEGORICAL

NUMERIC = ["area_m2", "bedrooms", "bathrooms", "lat", "lon"]
FLAGS = ["area_missing", *AMENITIES]


class AreaByBedroomsImputer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Fill a missing ``area_m2`` with the median area of listings with as many bedrooms."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:  # noqa: N803 - scikit-learn API
        """Learn the median area per number of bedrooms and overall."""
        known = X.dropna(subset=["area_m2"])
        self.by_bedrooms_ = known.groupby("bedrooms")["area_m2"].median().to_dict()
        self.overall_ = float(known["area_m2"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803 - scikit-learn API
        """Return a copy with ``area_m2`` filled."""
        out = X.copy()
        fill = out["bedrooms"].map(self.by_bedrooms_).fillna(self.overall_)
        out["area_m2"] = out["area_m2"].fillna(fill)
        return out


def _to_float(x: pd.DataFrame) -> pd.DataFrame:
    """Booleans as 0/1 (a named function, so fitted pipelines can be pickled)."""
    return x.astype(float)


def _columns(categorical: object, numeric: list[object]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", make_pipeline(*numeric), NUMERIC),
            ("flags", FunctionTransformer(_to_float), FLAGS),
            ("categorical", categorical, CATEGORICAL),
        ]
    )


def make_ridge() -> Pipeline:
    """Ridge on log area, scaled numeric features and one-hot categories (alpha by inner CV)."""
    log_area = ColumnTransformer(
        [("log_area", FunctionTransformer(np.log), ["area_m2"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline(
        [
            ("area", AreaByBedroomsImputer()),
            ("log_area", log_area),
            (
                "columns",
                _columns(
                    OneHotEncoder(handle_unknown="ignore"),
                    [SimpleImputer(strategy="median"), StandardScaler()],
                ),
            ),
            ("model", RidgeCV(alphas=np.logspace(-2, 3, 11))),
        ]
    )


def make_random_forest(seed: int) -> Pipeline:
    """Random Forest on imputed numeric features and ordinal-encoded categories."""
    return Pipeline(
        [
            ("area", AreaByBedroomsImputer()),
            (
                "columns",
                _columns(
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    [SimpleImputer(strategy="median")],
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=3,
                    max_features=0.5,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
