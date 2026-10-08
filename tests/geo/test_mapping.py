import logging

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from medellin_rent.geo.mapping import assign_location, normalize_name


def _barrios() -> gpd.GeoDataFrame:
    # Two synthetic barrios side by side, in two comunas (lon/lat degrees).
    return gpd.GeoDataFrame(
        {
            "barrio_code": ["1101", "1401"],
            "barrio_name": ["Laureles", "Patio Bonito"],
            "comuna_code": ["11", "14"],
            "comuna_name": ["LAURELES", "EL POBLADO"],
            "area_type": ["U", "U"],
        },
        geometry=[box(-75.60, 6.24, -75.58, 6.26), box(-75.58, 6.20, -75.56, 6.22)],
        crs="EPSG:4326",
    )


def _listings(rows: list[tuple[float | None, float | None, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": [f"l{i}" for i in range(len(rows))],
            "lat": pd.Series([r[0] for r in rows], dtype="float64"),
            "lon": pd.Series([r[1] for r in rows], dtype="float64"),
            "barrio_raw": pd.Series([r[2] for r in rows], dtype="str"),
        }
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("El Poblado", "POBLADO"),
        ("EL POBLADO", "POBLADO"),
        ("Candelaria", "CANDELARIA"),
        ("LA CANDELARIA", "CANDELARIA"),
        ("San Cristóbal", "SAN CRISTOBAL"),
        ("  Belén ", "BELEN"),
        ("Villa-Hermosa", "VILLA HERMOSA"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(pd.Series([raw])).iloc[0] == expected


def test_normalize_name_keeps_missing_values() -> None:
    assert normalize_name(pd.Series([None], dtype="str")).isna().all()


def test_assign_location_by_name_and_coordinates() -> None:
    listings = _listings(
        [
            (6.25, -75.59, "Laureles"),  # name and point agree -> barrio + comuna
            (None, None, "El Poblado"),  # name only -> comuna, no barrio
            (6.21, -75.57, None),  # point only -> barrio + comuna
            (6.25, -75.59, "El Poblado"),  # conflict -> declared comuna, no barrio
            (None, None, None),  # nothing
            (7.00, -75.00, None),  # point outside every barrio
        ]
    )

    out = assign_location(listings, _barrios())

    assert out["comuna_code"].tolist()[:4] == ["11", "14", "14", "14"]
    assert out["comuna_name"].tolist()[:4] == ["LAURELES", "EL POBLADO", "EL POBLADO", "EL POBLADO"]
    assert out["barrio_name"].tolist()[0] == "Laureles"
    assert out["barrio_name"].tolist()[2] == "Patio Bonito"
    assert out.loc[[1, 3, 4, 5], "barrio_name"].isna().all()
    assert out.loc[[4, 5], "comuna_code"].isna().all()
    assert listings.columns.tolist() == [
        "listing_id",
        "lat",
        "lon",
        "barrio_raw",
    ]  # input untouched


def test_assign_location_logs_unmatched_names(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("medellin_rent.geo.mapping")
    logger.addHandler(caplog.handler)
    try:
        out = assign_location(_listings([(None, None, "Narnia")]), _barrios())
    finally:
        logger.removeHandler(caplog.handler)

    assert out["comuna_code"].isna().all()
    assert "Narnia" in caplog.text
