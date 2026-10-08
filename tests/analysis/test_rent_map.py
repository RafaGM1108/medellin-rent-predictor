from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from medellin_rent.analysis import rent_map


def _comunas() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"comuna_code": ["14", "11", "50"], "comuna_name": ["EL POBLADO", "LAURELES", "PALMITAS"]},
        geometry=[
            box(-75.58, 6.18, -75.55, 6.22),
            box(-75.60, 6.24, -75.58, 6.26),
            box(-75.70, 6.30, -75.65, 6.35),
        ],
        crs="EPSG:4326",
    )


def _primary() -> pd.DataFrame:
    rows = [("14", "EL POBLADO", 3_000_000.0, 100.0)] * 30 + [("11", "LAURELES", 2e6, 100.0)] * 5
    return pd.DataFrame(rows, columns=["comuna_code", "comuna_name", "rent_cop", "area_m2"])


def test_comuna_layer_masks_comunas_with_too_little_data() -> None:
    layer = rent_map.comuna_layer(_primary(), _comunas()).set_index("comuna_code")
    assert layer.loc["14", "median_rent_per_m2"] == 30_000
    assert layer.loc["14", "n"] == 30
    assert pd.isna(layer.loc["11", "median_rent_per_m2"])  # n = 5 < 30
    assert layer.loc["50", "n"] == 0  # no listings at all
    assert len(layer) == 3


def test_run_writes_png_and_html_with_attribution(tmp_path: Path) -> None:
    png, html = rent_map.run(_primary(), _comunas(), tmp_path)
    assert png.name == "rent_map_comunas.png"
    assert png.stat().st_size > 0
    page = html.read_text()
    assert page.count('"type": "Feature"') == 3
    assert "GeoMedellín" in page
    assert '"rent_per_m2": "30,000 COP"' in page
