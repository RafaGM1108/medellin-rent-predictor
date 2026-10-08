"""Logic behind the "Is this rent fair?" app: the verdict and the per-listing explanation.

Kept out of the Streamlit script so it can be tested.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from medellin_rent.analysis.interpret import as_percent, shap_values
from medellin_rent.features.build import COMUNA_NAMES
from medellin_rent.inference.predict import to_features

LABELS = {
    "comuna_code": "Comuna",
    "estrato": "Estrato",
    "area_m2": "Area",
    "area_missing": "Area not given",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "lat": "Location (latitude)",
    "lon": "Location (longitude)",
    "parking": "Parking",
    "listing_quarter": "Listing date",
    "has_elevator": "Elevator",
    "has_pool": "Pool",
    "has_gym": "Gym",
    "has_balcony": "Balcony",
    "has_doorman": "Doorman / security",
    "is_furnished": "Furnished",
}


@dataclass(frozen=True)
class Verdict:
    """How a listed rent compares with the model's range."""

    label: str  # "good deal", "fair" or "expensive"
    difference_pct: float  # listed rent vs the prediction


def verdict(listed: float, predicted: float, low: float, high: float) -> Verdict:
    """Compare a listed rent with the predicted rent and its 80% interval.

    Below the interval is a good deal, inside it is fair and above it is expensive.
    """
    if listed < low:
        label = "good deal"
    elif listed > high:
        label = "expensive"
    else:
        label = "fair"
    return Verdict(label=label, difference_pct=100 * (listed / predicted - 1))


def _value(feature: str, value: Any) -> str:
    """Readable value of a model input for the explanation table."""
    if pd.isna(value):
        return "not given"
    if feature == "comuna_code":
        return COMUNA_NAMES.get(str(value), str(value)).title()
    if isinstance(value, bool | np.bool_):
        return "yes" if value else "no"
    if feature == "area_m2":
        return f"{value:.0f} m²"
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def explain(model: Any, listing: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """The features that moved this listing's prediction the most.

    Args:
        model: Saved LightGBM pipeline.
        listing: One listing in the ``NewListingSchema`` format.
        top: Number of features to return.

    Returns:
        ``feature``, ``value`` and ``effect_pct`` (percentage change of the predicted rent
        caused by the feature, relative to the model's average listing), largest first.
    """
    values, inputs = shap_values(model, to_features(listing))
    effects = values.drop(columns="base_value").iloc[0]
    order = effects.abs().sort_values(ascending=False).index[:top]
    return pd.DataFrame(
        {
            "feature": [LABELS.get(f, f) for f in order],
            "value": [_value(f, inputs.iloc[0][f]) for f in order],
            "effect_pct": [float(as_percent(effects[f])) for f in order],
        }
    )
