"""Download and load Medellín's official comuna, barrio and estrato layers.

Source: GeoMedellín open data (Alcaldía de Medellín), license CC BY-SA 4.0 with the extra
condition that the data "no puede ser comercializada o transferida". The files are
downloaded at run time into ``data/01_raw/geo/`` and never committed.

The published layers use EPSG:9377 (MAGNA-SIRGAS, origen nacional). :func:`load_layer`
reprojects them to EPSG:4326 so they can be joined with listing coordinates.
"""

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.request import Request, urlopen

import geopandas as gpd

from medellin_rent.utils.log import get_logger

logger = get_logger(__name__)

LayerName = Literal["comunas", "barrios", "estrato"]

USER_AGENT = "medellin-rent-predictor/0.1 (+https://github.com/RafaGM1108/medellin-rent-predictor)"
TARGET_CRS = "EPSG:4326"
_BASE_URL = (
    "https://www.medellin.gov.co/apigeomedellin/atributos/archivos/openDataExt/Gis/open_data"
)


@dataclass(frozen=True)
class Layer:
    """A GeoMedellín open data layer."""

    url: str
    page: str
    columns: dict[str, str]  # source column -> project column


LAYERS: dict[LayerName, Layer] = {
    "comunas": Layer(
        url=f"{_BASE_URL}/OD1043/geojson_limite_catastral_de_comun.zip",
        page="https://www.medellin.gov.co/geomedellin/datosAbiertos/1043",
        columns={"comuna": "comuna_code", "nombre": "comuna_name"},
    ),
    "barrios": Layer(
        url=f"{_BASE_URL}/OD1044/geojson_limite_barrio_vereda_cata.zip",
        page="https://www.medellin.gov.co/geomedellin/datosAbiertos/1044",
        columns={
            "codigo": "barrio_code",
            "nombre_barrio": "barrio_name",
            "comuna": "comuna_code",
            "nombre_comuna": "comuna_name",
            "indicador_ur": "area_type",
        },
    ),
    "estrato": Layer(
        url=f"{_BASE_URL}/OD396/geojson_estrato_socioeconomico.zip",
        page="https://www.medellin.gov.co/geomedellin/datosAbiertos/396",
        columns={"estrato": "estrato"},
    ),
}

LICENSE = "CC BY-SA 4.0; 'no puede ser comercializada o transferida' (Alcaldía de Medellín)"


def download_layer(name: LayerName, dest_dir: Path, timeout: float = 120) -> Path:
    """Download a layer's GeoJSON into ``dest_dir`` and record where it came from.

    Writes ``<name>.geojson`` and ``<name>.source.json`` (URL, license, retrieval time).

    Args:
        name: Layer to download.
        dest_dir: Target directory, usually ``config.paths.raw_geo``.
        timeout: Request timeout in seconds.

    Returns:
        Path to the downloaded GeoJSON file.
    """
    layer = LAYERS[name]
    logger.info("Downloading %s from %s", name, layer.url)
    request = Request(layer.url, headers={"User-Agent": USER_AGENT})
    # The URL is one of the fixed https constants in LAYERS, never user input.
    with urlopen(request, timeout=timeout) as response:  # nosec B310
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [m for m in archive.namelist() if m.endswith(".geojson")]
        if len(members) != 1:
            raise ValueError(f"Expected one .geojson in {layer.url}, found {members}")
        data = archive.read(members[0])

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{name}.geojson"
    path.write_bytes(data)
    source = {
        "url": layer.url,
        "page": layer.page,
        "license": LICENSE,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (dest_dir / f"{name}.source.json").write_text(json.dumps(source, indent=2) + "\n")
    logger.info("Saved %s (%d bytes)", path, len(data))
    return path


def load_layer(name: LayerName, path: Path) -> gpd.GeoDataFrame:
    """Load a downloaded layer with project column names, in EPSG:4326.

    Args:
        name: Layer the file belongs to.
        path: GeoJSON file written by :func:`download_layer`.

    Returns:
        GeoDataFrame with the columns in ``LAYERS[name].columns`` plus ``geometry``.
    """
    columns = LAYERS[name].columns
    gdf = gpd.read_file(path)
    missing = set(columns) - set(gdf.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    gdf = gdf[[*columns, "geometry"]].rename(columns=columns)
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS")
    return gdf.to_crs(TARGET_CRS)
