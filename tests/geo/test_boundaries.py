import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from medellin_rent.geo import boundaries
from medellin_rent.geo.boundaries import LAYERS, download_layer, load_layer

# Two squares in EPSG:9377 near Medellín (synthetic, not real boundaries).
_X, _Y = 4_717_000.0, 2_250_000.0


def _square(x: float, y: float, size: float = 1000.0) -> list[list[list[float]]]:
    return [[[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]]


def _barrios_geojson() -> dict[str, Any]:
    features = [
        {
            "type": "Feature",
            "properties": {
                "OBJECTID": i,
                "comuna": "14",
                "barrio": f"0{i}",
                "codigo": f"140{i}",
                "nombre_barrio": f"Barrio {i}",
                "indicador_ur": "U",
                "sector": 1,
                "nombre_comuna": "EL POBLADO",
            },
            "geometry": {"type": "Polygon", "coordinates": _square(_X + i * 1000, _Y)},
        }
        for i in (1, 2)
    ]
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:9377"}},
        "features": features,
    }


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_download_layer_extracts_geojson_and_records_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.dumps(_barrios_geojson()).encode()
    seen: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        seen["url"] = request.full_url
        seen["agent"] = request.get_header("User-agent")
        return _FakeResponse(_zip_bytes({"limite_barrio_vereda_cata.geojson": payload}))

    monkeypatch.setattr(boundaries, "urlopen", fake_urlopen)
    path = download_layer("barrios", tmp_path / "geo")

    assert path == tmp_path / "geo" / "barrios.geojson"
    assert path.read_bytes() == payload
    assert seen["url"] == LAYERS["barrios"].url
    assert "medellin-rent-predictor" in seen["agent"]
    source = json.loads((tmp_path / "geo" / "barrios.source.json").read_text())
    assert source["url"] == LAYERS["barrios"].url
    assert "CC BY-SA" in source["license"]
    assert source["retrieved_at"]


def test_download_layer_rejects_unexpected_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        boundaries, "urlopen", lambda request, timeout: _FakeResponse(_zip_bytes({"a.txt": b""}))
    )
    with pytest.raises(ValueError, match=r"Expected one \.geojson"):
        download_layer("comunas", tmp_path)


def test_load_layer_renames_columns_and_reprojects(tmp_path: Path) -> None:
    path = tmp_path / "barrios.geojson"
    path.write_text(json.dumps(_barrios_geojson()))

    gdf = load_layer("barrios", path)

    assert list(gdf.columns) == [*LAYERS["barrios"].columns.values(), "geometry"]
    assert gdf.crs.to_epsg() == 4326
    assert gdf["comuna_name"].tolist() == ["EL POBLADO", "EL POBLADO"]
    minx, miny, maxx, maxy = gdf.total_bounds
    # Reprojected coordinates fall in the Medellín area (lon ≈ -75.6, lat ≈ 6.2).
    assert -76.0 < minx < maxx < -75.0
    assert 5.5 < miny < maxy < 7.0


def test_load_layer_rejects_missing_columns(tmp_path: Path) -> None:
    data = _barrios_geojson()
    for feature in data["features"]:
        del feature["properties"]["nombre_barrio"]
    path = tmp_path / "barrios.geojson"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="nombre_barrio"):
        load_layer("barrios", path)
