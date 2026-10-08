import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from medellin_rent.data.listings import COLUMNS, load_listings, record_source

HEADER = ["ad_type", *COLUMNS[:6], "l1", "l2", *COLUMNS[6:]]  # raw file has extra columns


def _row(**overrides: str) -> dict[str, str]:
    row = dict.fromkeys(HEADER, "")
    row |= {
        "id": "a1",
        "l3": "Medellín",
        "operation_type": "Arriendo",
        "property_type": "Apartamento",
        "currency": "COP",
        "price": "1600000",
        "title": "Apartamento en arriendo",
        "description": "3 habitaciones. Llamar al 300 123 4567 o escribir a ana@example.com",
    }
    return row | overrides


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_listings_keeps_medellin_apartment_rents_in_cop(tmp_path: Path) -> None:
    rows = [
        _row(id="keep"),
        _row(id="other-city", l3="Bogotá"),
        _row(id="sale", operation_type="Venta"),
        _row(id="house", property_type="Casa"),
        _row(id="usd", currency="USD"),
    ]
    path = _write_csv(tmp_path / "co_properties.csv", rows)

    listings = load_listings(path, chunksize=2)  # several chunks

    assert listings["id"].tolist() == ["keep"]
    assert list(listings.columns) == COLUMNS
    assert all(pd.api.types.is_string_dtype(dtype) for dtype in listings.dtypes)


def test_load_listings_redacts_phones_and_emails(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "co_properties.csv", [_row()])

    description = load_listings(path)["description"].iloc[0]

    assert "300 123 4567" not in description
    assert "ana@example.com" not in description
    assert description == "3 habitaciones. Llamar al [redacted] o escribir a [redacted]"


def test_record_source_writes_checksum(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "co_properties.csv", [_row()])

    source = json.loads(record_source(path).read_text())

    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source["bytes"] == path.stat().st_size
    assert source["file"] == "co_properties.csv"
    assert "kaggle.com" in source["url"]
