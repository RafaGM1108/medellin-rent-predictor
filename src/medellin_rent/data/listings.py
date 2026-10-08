"""Load Medellín apartment rentals from the Properati Colombia file.

Source: Properati Data, mirrored on Kaggle as "Colombian Properties"
(``co_properties.csv``). Scraping portals is not allowed (see ``data/README.md``), so the
file is downloaded manually into ``data/01_raw/listings/`` and never committed.

Phone numbers and emails in the listing text are redacted on load, so no personal data
reaches later layers.
"""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from medellin_rent.utils.log import get_logger

logger = get_logger(__name__)

RAW_FILE = "co_properties.csv"
SOURCE_URL = "https://www.kaggle.com/datasets/lauramartinezortiz/colombian-properties"
LICENSE = "Unknown on Kaggle (Properati Data); original license not verified - do not redistribute"

COLUMNS = [
    "id",
    "start_date",
    "end_date",
    "created_on",
    "lat",
    "lon",
    "l3",
    "l4",
    "l5",
    "l6",
    "rooms",
    "bedrooms",
    "bathrooms",
    "surface_total",
    "surface_covered",
    "price",
    "currency",
    "price_period",
    "title",
    "description",
    "property_type",
    "operation_type",
]
TEXT_COLUMNS = ["title", "description"]

# Limit: catches emails and Colombian mobile numbers (3xx xxx xxxx, optional +57); landlines
# are not matched because 7-digit runs collide with prices. Extend if audits find leaks.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_MOBILE = re.compile(r"(?<!\d)(?:\+?57[\s.-]?)?3\d{2}[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
REDACTED = "[redacted]"


def redact_personal_data(text: pd.Series) -> pd.Series:
    """Replace emails and mobile phone numbers in a text column with ``[redacted]``."""
    return text.str.replace(_EMAIL, REDACTED, regex=True).str.replace(_MOBILE, REDACTED, regex=True)


def load_listings(
    path: Path,
    city: str = "Medellín",
    operation: str = "Arriendo",
    property_type: str = "Apartamento",
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Read the raw file and keep the rentals of one property type in one city, priced in COP.

    All columns are read as strings; typing and cleaning happen in the intermediate layer.

    Args:
        path: Path to ``co_properties.csv``.
        city: Value of the ``l3`` (city) column to keep.
        operation: Value of ``operation_type`` to keep (``Arriendo`` is a monthly rent).
        property_type: Value of ``property_type`` to keep.
        chunksize: Rows per chunk, to keep memory bounded on the ~600 MB file.

    Returns:
        The matching listings with ``COLUMNS``, personal data redacted.

    Raises:
        ValueError: If the file is missing any of ``COLUMNS``.
    """
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(COLUMNS) - set(header))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")

    chunks = [
        chunk[
            (chunk["l3"] == city)
            & (chunk["operation_type"] == operation)
            & (chunk["property_type"] == property_type)
            & (chunk["currency"] == "COP")
        ]
        for chunk in pd.read_csv(
            path, usecols=COLUMNS, dtype=str, keep_default_na=False, chunksize=chunksize
        )
    ]
    listings = pd.concat(chunks, ignore_index=True)[COLUMNS]
    for column in TEXT_COLUMNS:
        listings[column] = redact_personal_data(listings[column])
    logger.info("Loaded %d %s rentals in %s from %s", len(listings), property_type, city, path)
    return listings


def record_source(path: Path) -> Path:
    """Write ``listings.source.json`` next to the raw file: origin, license and checksum.

    Args:
        path: Path to the manually downloaded raw file.

    Returns:
        Path to the metadata file.
    """
    sha256 = hashlib.sha256()
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1 << 20), b""):
            sha256.update(block)
    source = {
        "file": path.name,
        "url": SOURCE_URL,
        "license": LICENSE,
        "sha256": sha256.hexdigest(),
        "bytes": path.stat().st_size,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = path.parent / "listings.source.json"
    out.write_text(json.dumps(source, indent=2) + "\n")
    return out
