import pandas as pd
import pytest

from medellin_rent.data.listings import COLUMNS
from medellin_rent.data.parse import extract_number, normalize_text, parse_listings


def _raw(**overrides: str) -> pd.DataFrame:
    row = dict.fromkeys(COLUMNS, "") | {"id": "x", "price": "1600000"}
    return pd.DataFrame([row | overrides])


def _parse_text(description: str) -> pd.Series:
    return parse_listings(_raw(description=description)).iloc[0]


def test_normalize_text_unescapes_strips_tags_and_lowercases() -> None:
    text = pd.Series(["Ba&ntilde;o<br />\n  AMPLIO", None])
    assert normalize_text(text).tolist() == ["baño amplio", ""]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("área 80 m2", 80.0), ("80,5m²", 80.5), ("120 mts", 120.0), ("75 metros cuadrados", 75.0)],
)
def test_area_from_text(text: str, expected: float) -> None:
    assert _parse_text(text)["area_m2"] == expected


def test_structured_values_win_over_text() -> None:
    raw = _raw(
        surface_covered="",
        surface_total="95",
        bedrooms="",
        rooms="4",
        bathrooms="3",
        description="80 m2, 2 habitaciones, 1 baño",
    )
    row = parse_listings(raw).iloc[0]
    assert (row["area_m2"], row["bedrooms"], row["bathrooms"]) == (95.0, 4, 3)


def test_fields_from_text() -> None:
    row = _parse_text(
        "3 alcobas, 2 baños, 1 parqueadero, piso 8, estrato 4, 10 años de construido, "
        "ascensor, piscina, gimnasio, balcón, portería, amoblado"
    )
    assert row["bedrooms"] == 3
    assert row["bathrooms"] == 2
    assert row["parking_spots"] == 1
    assert row["has_parking"]
    assert row["floor"] == 8
    assert row["estrato"] == 4
    assert row["building_age_years"] == 10
    assert all(
        row[c]
        for c in [
            "has_elevator",
            "has_pool",
            "has_gym",
            "has_balcony",
            "has_doorman",
            "is_furnished",
        ]
    )


def test_missing_fields_are_na_and_amenities_false() -> None:
    row = _parse_text("Apartamento iluminado")
    assert pd.isna(row["area_m2"])
    assert pd.isna(row["bedrooms"])
    assert pd.isna(row["estrato"])
    assert pd.isna(row["has_parking"])  # not mentioned -> unknown, not False
    assert not row["has_pool"]


def test_no_parking_is_false() -> None:
    assert not _parse_text("Sin parqueadero")["has_parking"]


def test_output_is_typed_and_has_no_text() -> None:
    out = parse_listings(_raw(lat="6.2", lon="-75.5", start_date="2021-01-10", l4="Laureles"))
    assert "title" not in out.columns
    assert "description" not in out.columns
    assert out["rent_cop"].dtype == "float64"
    assert out["bedrooms"].dtype == "Int64"
    assert str(out["start_date"].dtype).startswith("datetime64")
    assert out["barrio_raw"].iloc[0] == "Laureles"
    assert pd.isna(parse_listings(_raw())["barrio_raw"].iloc[0])


def test_extract_number_returns_nan_without_match() -> None:
    assert extract_number(pd.Series(["nada"]), r"(\d+) m2").isna().all()
