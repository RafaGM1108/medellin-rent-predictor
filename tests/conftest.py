import numpy as np
import pandas as pd
import pytest

from medellin_rent.features import build
from medellin_rent.features.model_input import LOG_TARGET, TARGET


def synthetic_model_input(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Tiny model input in the real schema: rent grows with area and comuna 14 is pricier."""
    rng = np.random.default_rng(seed)
    comuna = rng.choice(["11", "14"], n)
    area = rng.uniform(40, 150, n)
    area_known = rng.random(n) < 0.6
    per_m2 = np.where(comuna == "14", 30_000, 20_000)
    rent = area * per_m2 * rng.uniform(0.9, 1.1, n)
    df = pd.DataFrame(
        {
            "listing_id": [f"s{i}" for i in range(n)],
            "area_m2": np.where(area_known, area, np.nan),
            "bedrooms": np.clip(np.round(area / 35), 1, 5),
            "bathrooms": np.clip(np.round(area / 50), 1, 4),
            "lat": np.where(comuna == "14", 6.21, 6.25) + rng.normal(0, 0.005, n),
            "lon": np.where(comuna == "14", -75.57, -75.59) + rng.normal(0, 0.005, n),
            "area_missing": ~area_known,
            **{a: rng.random(n) < 0.3 for a in build.AMENITIES},
            "comuna_code": pd.Categorical(comuna, build.COMUNAS),
            "estrato": pd.Categorical(np.where(comuna == "14", "6", "4"), build.ESTRATOS),
            "parking": pd.Categorical(rng.choice(["yes", "unknown"], n), build.PARKING),
            "listing_quarter": pd.Categorical(rng.choice(["2020Q4", "2021Q1"], n), build.QUARTERS),
            "barrio_name": pd.Series(rng.choice(["A", "B", "C", None], n), dtype="str"),
            TARGET: rent,
        }
    )
    df[LOG_TARGET] = np.log(df[TARGET])
    return df


@pytest.fixture
def model_input() -> pd.DataFrame:
    return synthetic_model_input()
