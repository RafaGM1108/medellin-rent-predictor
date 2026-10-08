import numpy as np
import pandas as pd
import pytest

from medellin_rent.model.baseline import ComunaMedianBaseline


def _xy() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "comuna_code": pd.Categorical(["14", "14", "11", "11"]),
            "area_m2": [100.0, float("nan"), 50.0, 100.0],
        }
    )
    rent = pd.Series([3_000_000.0, 2_000_000.0, 1_000_000.0, 2_000_000.0])
    return features, pd.Series(np.log(rent))


def test_uses_rent_per_m2_when_area_is_known_and_median_rent_otherwise() -> None:
    features, y = _xy()
    model = ComunaMedianBaseline().fit(features, y)
    new = pd.DataFrame(
        {"comuna_code": pd.Categorical(["14", "14", "11"]), "area_m2": [50.0, float("nan"), 80.0]}
    )
    prediction = np.exp(model.predict(new))
    assert prediction[0] == pytest.approx(30_000 * 50)  # comuna 14: one listing with area
    assert prediction[1] == pytest.approx(2_500_000)  # comuna 14 median rent
    assert prediction[2] == pytest.approx(20_000 * 80)  # comuna 11: 20k/m2 both listings


def test_unseen_comuna_falls_back_to_global_medians() -> None:
    features, y = _xy()
    model = ComunaMedianBaseline().fit(features, y)
    new = pd.DataFrame({"comuna_code": ["90", "90"], "area_m2": [10.0, float("nan")]})
    prediction = np.exp(model.predict(new))
    assert prediction[0] == pytest.approx(model.global_per_m2_ * 10)
    assert prediction[1] == pytest.approx(model.global_rent_)
