import pandas as pd
import pandera.errors as pe
import pytest

from medellin_rent.data.listings import COLUMNS
from medellin_rent.data.parse import parse_listings
from medellin_rent.data.schemas import IntermediateSchema, PrimarySchema


def _intermediate(n: int = 2) -> pd.DataFrame:
    rows = [
        dict.fromkeys(COLUMNS, "")
        | {
            "id": f"id{i}",
            "price": "1600000",
            "lat": "6.2",
            "lon": "-75.57",
            "description": "70 m2, 3 habitaciones, 2 baños, estrato 4",
        }
        for i in range(n)
    ]
    return parse_listings(pd.DataFrame(rows))


def _primary(n: int = 2) -> pd.DataFrame:
    df = _intermediate(n)
    df["barrio_name"] = pd.Series(["Laureles"] * n, dtype="str")
    df["comuna_code"] = pd.Series(["11"] * n, dtype="str")
    df["comuna_name"] = pd.Series(["LAURELES ESTADIO"] * n, dtype="str")
    return df


def test_parsed_listings_pass_intermediate_schema() -> None:
    IntermediateSchema.validate(_intermediate(), lazy=True)


def test_intermediate_keeps_odd_but_possible_values() -> None:
    df = _intermediate()
    df.loc[0, "area_m2"] = 0.0
    df.loc[0, "bedrooms"] = 65
    IntermediateSchema.validate(df, lazy=True)


@pytest.mark.parametrize(
    ("column", "value"),
    [("estrato", 7), ("rent_cop", -1.0), ("lat", 95.0), ("bathrooms", -1)],
)
def test_intermediate_rejects_impossible_values(column: str, value: float) -> None:
    df = _intermediate()
    df.loc[0, column] = value
    with pytest.raises(pe.SchemaErrors, match=column):
        IntermediateSchema.validate(df, lazy=True)


def test_intermediate_rejects_unexpected_columns() -> None:
    df = _intermediate().assign(description="text")
    with pytest.raises(pe.SchemaErrors, match="description"):
        IntermediateSchema.validate(df, lazy=True)


def test_primary_accepts_clean_listings() -> None:
    PrimarySchema.validate(_primary(), lazy=True)


@pytest.mark.parametrize(
    ("column", "value"),
    [("rent_cop", float("nan")), ("rent_cop", 0.0), ("area_m2", 0.0), ("bedrooms", 65)],
)
def test_primary_rejects_what_cleaning_must_remove(column: str, value: float) -> None:
    df = _primary()
    df.loc[0, column] = value
    with pytest.raises(pe.SchemaErrors, match=column):
        PrimarySchema.validate(df, lazy=True)


def test_primary_rejects_duplicate_listings() -> None:
    df = _primary()
    df.loc[1, "listing_id"] = df.loc[0, "listing_id"]
    with pytest.raises(pe.SchemaErrors, match="listing_id"):
        PrimarySchema.validate(df, lazy=True)


def test_primary_requires_comuna_columns() -> None:
    with pytest.raises(pe.SchemaErrors, match="comuna_code"):
        PrimarySchema.validate(_intermediate(), lazy=True)
