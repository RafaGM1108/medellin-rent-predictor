from pathlib import Path

import pandas as pd
import pandera.errors as pe
import pytest

from medellin_rent.features.build import FEATURES
from medellin_rent.inference.predict import load_model, predict, to_features

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "01_raw" / "new_listings_sample.csv"


def _listing(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "comuna": "EL POBLADO",
        "estrato": 5,
        "area_m2": 80.0,
        "bedrooms": 2,
        "bathrooms": 2,
        "parking": "yes",
        "lat": None,
        "lon": None,
    }
    return pd.DataFrame([row | overrides])


def test_to_features_matches_training_format() -> None:
    features = to_features(_listing())
    assert features.columns.tolist() == ["listing_id", *FEATURES]
    row = features.iloc[0]
    assert row["comuna_code"] == "14"
    assert row["estrato"] == "5"
    assert row["parking"] == "yes"
    assert row["listing_quarter"] == "2021Q3"  # predictions at the latest data prices
    assert not row["has_pool"]  # missing amenity columns default to False
    assert not row["area_missing"]


def test_to_features_handles_unknown_values() -> None:
    features = to_features(_listing(estrato=None, area_m2=None, parking="unknown"))
    row = features.iloc[0]
    assert row["estrato"] == "unknown"
    assert row["area_missing"]
    assert row["parking"] == "unknown"


@pytest.mark.parametrize(
    ("column", "value"),
    [("comuna", "NARNIA"), ("estrato", 7), ("area_m2", 5.0), ("parking", "maybe"), ("lat", 4.6)],
)
def test_to_features_rejects_invalid_listings(column: str, value: object) -> None:
    with pytest.raises(pe.SchemaErrors, match=column):
        to_features(_listing(**{column: value}))


def test_the_sample_file_is_valid() -> None:
    assert len(to_features(pd.read_csv(SAMPLE))) == 5


def test_predict_with_interval(saved_model: Path) -> None:
    model, metadata = load_model(saved_model)
    listings = pd.concat([_listing(), _listing(comuna="LAURELES", estrato=4)], ignore_index=True)
    listings.index = [10, 20]  # keeps the caller's index
    out = predict(model, metadata, listings)
    assert out.index.tolist() == [10, 20]
    assert (out["interval_low_cop"] < out["predicted_rent_cop"]).all()
    assert (out["predicted_rent_cop"] < out["interval_high_cop"]).all()
    assert (out["predicted_rent_cop"] % 1000 == 0).all()  # rounded to thousands of COP


def test_predict_without_interval(saved_model: Path) -> None:
    model, _ = load_model(saved_model)
    out = predict(model, {"model": "x"}, _listing())
    assert out.columns.tolist() == ["predicted_rent_cop"]


def test_only_the_comuna_is_required() -> None:
    row = to_features(pd.DataFrame([{"comuna": "BELEN"}])).iloc[0]
    assert row["comuna_code"] == "16"
    assert row["estrato"] == "unknown"
    assert row["parking"] == "unknown"
    assert row["area_missing"]
