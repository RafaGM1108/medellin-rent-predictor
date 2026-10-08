"""Predict the monthly rent of new listings with the saved model.

New listings are described in a small, user-facing format (``NewListingSchema``): the comuna
by its official name, plus whatever is known about the apartment. They are turned into the
model's features with the same code as training. The model's predictions are in
**2020-2021 prices** (the period of the training data).

The 80% interval multiplies the prediction by the 10th and 90th percentiles of
``actual / predicted`` rent on the test set, stored in the model metadata at training time.

With a rent index record (``conf/rent_index.json``, see :mod:`medellin_rent.data.prices`),
every amount is also given at current prices (columns ending in ``_current``).
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa

from medellin_rent.features.build import (
    AMENITIES,
    COMUNA_NAMES,
    PARKING,
    UNKNOWN,
    build_features,
)
from medellin_rent.model.evaluate import predict_rent

PRICE_REFERENCE_DATE = pd.Timestamp("2021-08-19")  # last listing date in the training data
_CODES_BY_NAME = {name: code for code, name in COMUNA_NAMES.items()}
OPTIONAL = ["estrato", "area_m2", "bedrooms", "bathrooms", "lat", "lon"]

NewListingSchema = pa.DataFrameSchema(
    {
        "comuna": pa.Column(str, pa.Check.isin(list(_CODES_BY_NAME)), nullable=False),
        "estrato": pa.Column("Int64", pa.Check.isin(range(1, 7)), nullable=True),
        "area_m2": pa.Column(float, pa.Check.in_range(10, 1_000), nullable=True),
        "bedrooms": pa.Column("Int64", pa.Check.in_range(0, 10), nullable=True),
        "bathrooms": pa.Column("Int64", pa.Check.in_range(0, 10), nullable=True),
        "parking": pa.Column(str, pa.Check.isin(PARKING), nullable=False),
        "lat": pa.Column(float, pa.Check.in_range(6.1, 6.4), nullable=True),
        "lon": pa.Column(float, pa.Check.in_range(-75.75, -75.45), nullable=True),
        **{amenity: pa.Column(bool) for amenity in AMENITIES},
    },
    coerce=True,
    strict="filter",
    name="NewListings",
)


def to_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Validate new listings and build the model's features for them.

    Args:
        listings: One row per listing with the columns of ``NewListingSchema``. Only
            ``comuna`` is required: missing optional columns mean "unknown", a missing
            ``parking`` is ``unknown`` and missing amenities are ``False``.

    Returns:
        Feature table in the training format.
    """
    df = listings.copy()
    defaults: dict[str, bool | str | None] = dict.fromkeys(OPTIONAL) | {"parking": UNKNOWN}
    defaults |= dict.fromkeys(AMENITIES, False)
    for column, value in defaults.items():
        if column not in df:
            df[column] = value
    df = NewListingSchema.validate(df, lazy=True)
    primary_like = pd.DataFrame(
        {
            "listing_id": [f"new-{i}" for i in range(len(df))],
            "comuna_code": df["comuna"].map(_CODES_BY_NAME).astype("str").to_numpy(),
            "estrato": df["estrato"].to_numpy(),
            "area_m2": df["area_m2"].to_numpy(),
            "bedrooms": df["bedrooms"].to_numpy(),
            "bathrooms": df["bathrooms"].to_numpy(),
            "has_parking": df["parking"].map({"yes": True, "no": False}).astype("boolean").array,
            "lat": df["lat"].to_numpy(),
            "lon": df["lon"].to_numpy(),
            "barrio_name": np.full(len(df), None),
            "start_date": PRICE_REFERENCE_DATE,
            **{amenity: df[amenity].to_numpy() for amenity in AMENITIES},
        }
    )
    return build_features(primary_like)


def load_model(models_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load the saved model and its metadata from ``06_models``."""
    model = joblib.load(models_dir / "model.joblib")
    metadata = json.loads((models_dir / "model_metadata.json").read_text())
    return model, metadata


def predict(
    model: Any,
    metadata: dict[str, Any],
    listings: pd.DataFrame,
    rent_index: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Predicted monthly rent (COP) with an 80% interval, optionally at current prices.

    Args:
        model: Fitted model from ``06_models``.
        metadata: Its ``model_metadata.json``.
        listings: New listings in the ``NewListingSchema`` format.
        rent_index: Price adjustment record (``conf/rent_index.json``), or ``None``.

    Returns:
        ``predicted_rent_cop`` (data-period prices) and, if the metadata has an interval,
        ``interval_low_cop`` and ``interval_high_cop``; with ``rent_index`` the same
        columns again at current prices with a ``_current`` suffix. Rows follow ``listings``.
    """
    rent = predict_rent(model, to_features(listings))
    out = pd.DataFrame({"predicted_rent_cop": np.round(rent, -3)}, index=listings.index)
    interval = metadata.get("interval")
    if interval:
        out["interval_low_cop"] = np.round(rent * interval["low_factor"], -3)
        out["interval_high_cop"] = np.round(rent * interval["high_factor"], -3)
    if rent_index:
        current = rent * rent_index["factor"]
        out["predicted_rent_cop_current"] = np.round(current, -3)
        if interval:
            out["interval_low_cop_current"] = np.round(current * interval["low_factor"], -3)
            out["interval_high_cop_current"] = np.round(current * interval["high_factor"], -3)
    return out
