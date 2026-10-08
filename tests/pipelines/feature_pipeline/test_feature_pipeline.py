import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from medellin_rent.data.listings import RAW_FILE
from medellin_rent.pipelines.feature_pipeline.pipeline import run
from medellin_rent.utils.config import Config, PathsConfig, load_config

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "data" / "fixtures" / "co_properties.csv"


def _config(tmp_path: Path) -> Config:
    config = load_config(ROOT / "conf" / "base.yaml")
    paths = PathsConfig(**{name: tmp_path / name for name in PathsConfig.model_fields})
    paths.raw_listings.mkdir()
    shutil.copy(FIXTURE, paths.raw_listings / RAW_FILE)
    return config.model_copy(update={"paths": paths})


def _write_geo(directory: Path) -> None:
    # One synthetic barrio and estrato polygon around the fixture's El Poblado listing.
    directory.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "codigo": ["1401"],
            "nombre_barrio": ["Barrio Test"],
            "comuna": ["14"],
            "nombre_comuna": ["EL POBLADO"],
            "indicador_ur": ["U"],
        },
        geometry=[box(-75.57, 6.20, -75.56, 6.21)],
        crs="EPSG:4326",
    ).to_crs("EPSG:9377").to_file(directory / "barrios.geojson", driver="GeoJSON")
    gpd.GeoDataFrame(
        {"OBJECTID": [1], "codigo_barrio": ["1401"], "estrato": [6]},
        geometry=[box(-75.57, 6.20, -75.56, 6.21)],
        crs="EPSG:4326",
    ).to_crs("EPSG:9377").to_file(directory / "estrato.geojson", driver="GeoJSON")


def test_run_writes_intermediate_and_primary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_geo(config.paths.raw_geo)

    out = run(config)

    intermediate = pd.read_parquet(config.paths.intermediate / "listings.parquet")
    primary = pd.read_parquet(out)
    assert out == config.paths.primary / "listings.parquet"
    assert intermediate["listing_id"].tolist() == ["fx1", "fx2"]
    assert primary["listing_id"].tolist() == ["fx1", "fx2"]
    assert primary["comuna_name"].tolist()[0] == "EL POBLADO"
    assert primary["barrio_name"].tolist()[0] == "Barrio Test"
    assert pd.isna(primary["comuna_code"].iloc[1])  # fx2 has no location
    assert primary["estrato"].tolist() == [6, 3]  # fx1 from the layer, fx2 declared in text
    assert primary["estrato_source"].tolist() == ["layer", "listing"]
    assert "description" not in primary.columns


def test_run_asks_for_the_boundaries_first(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make geo"):
        run(_config(tmp_path))
