"""Parse raw listings into the typed intermediate table (``02_intermediate``).

Structured columns are typed as they are. The Properati file leaves most structured fields
empty (see ``data/README.md``), so area, rooms, bathrooms, parking, floor, building age,
estrato and amenities are also extracted from the listing text, preferring the structured
value when there is one. No business rules here: implausible values are kept and handled in
the cleaning step (#15).

``title`` and ``description`` are dropped from the output because they can contain agent
names.
"""

import html
import re

import pandas as pd

_NUMBER = r"(\d{1,4}(?:[.,]\d{1,2})?)"
_PATTERNS = {
    "area_m2": rf"{_NUMBER}\s*(?:m2|m²|mts2?|metros(?:\s+cuadrados)?)\b",
    "bedrooms": r"(\d{1,2})\s*(?:habitaciones|habitación|habitacion|alcobas?|cuartos?)\b",
    "bathrooms": r"(\d{1,2})\s*baños?\b",
    "parking_spots": r"(\d)\s*(?:parqueaderos?|garajes?)\b",
    "floor": r"\bpiso\s*(?:n[°o.]?\s*)?(\d{1,2})\b",
    "building_age_years": r"(\d{1,2})\s*años\s+de\s+(?:construid[oa]|antigüedad)",
    "estrato": r"\bestrato\s*([1-6])\b",
}
_NO_PARKING = r"\b(?:sin|no\s+tiene|no\s+incluye)\s+(?:parqueadero|garaje)"
_PARKING = r"\b(?:parqueadero|garaje)"
AMENITIES = {
    "has_elevator": r"\bascensor",
    "has_pool": r"\bpiscina",
    "has_gym": r"\bgimnasio",
    "has_balcony": r"\bbalc[oó]n",
    "has_doorman": r"\bporter[ií]a|\bvigilancia",
    "is_furnished": r"\b(?:amoblad|amueblad)",
}


def normalize_text(text: pd.Series) -> pd.Series:
    """Unescape HTML entities, strip tags, collapse whitespace and lowercase."""
    return (
        text.fillna("")
        .map(html.unescape)
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )


def _to_number(values: pd.Series) -> pd.Series:
    """Convert strings like ``"80"``, ``"80,5"`` or ``""`` to floats (NaN when empty)."""
    return pd.to_numeric(values.str.replace(",", ".", regex=False), errors="coerce").astype(
        "float64"
    )


def extract_number(text: pd.Series, pattern: str) -> pd.Series:
    """Return the first number captured by ``pattern`` in each text, as float (NaN if none)."""
    return _to_number(text.str.extract(pattern, flags=re.IGNORECASE)[0].fillna(""))


def parse_listings(raw: pd.DataFrame) -> pd.DataFrame:
    """Type the raw listings and extract fields from their text.

    Args:
        raw: Output of :func:`medellin_rent.data.listings.load_listings` (all strings).

    Returns:
        One row per raw listing, typed, without ``title`` or ``description``.
    """
    text = normalize_text(raw["title"] + " " + raw["description"])

    def structured_or_text(column: str | None, field: str) -> pd.Series:
        from_text = extract_number(text, _PATTERNS[field])
        if column is None:
            return from_text
        return _to_number(raw[column]).fillna(from_text)

    area = _to_number(raw["surface_covered"]).fillna(_to_number(raw["surface_total"]))
    parking_mentioned = text.str.contains(_PARKING, regex=True)
    no_parking = text.str.contains(_NO_PARKING, regex=True)

    out = pd.DataFrame(
        {
            "listing_id": raw["id"],
            "start_date": pd.to_datetime(raw["start_date"], errors="coerce"),
            "end_date": pd.to_datetime(raw["end_date"], errors="coerce"),
            "created_on": pd.to_datetime(raw["created_on"], errors="coerce"),
            "lat": _to_number(raw["lat"]),
            "lon": _to_number(raw["lon"]),
            "barrio_raw": raw["l4"].replace("", pd.NA),
            "rent_cop": _to_number(raw["price"]),
            "area_m2": area.fillna(extract_number(text, _PATTERNS["area_m2"])),
            "bedrooms": _to_number(raw["bedrooms"])
            .fillna(_to_number(raw["rooms"]))
            .fillna(extract_number(text, _PATTERNS["bedrooms"])),
            "bathrooms": structured_or_text("bathrooms", "bathrooms"),
            "parking_spots": structured_or_text(None, "parking_spots"),
            # Unknown (NA) when parking is not mentioned: silence is not "no parking".
            "has_parking": (parking_mentioned & ~no_parking)
            .astype("boolean")
            .mask(~parking_mentioned),
            "floor": structured_or_text(None, "floor"),
            "building_age_years": structured_or_text(None, "building_age_years"),
            "estrato": structured_or_text(None, "estrato"),
        }
    )
    for name, pattern in AMENITIES.items():
        out[name] = text.str.contains(pattern, regex=True)

    int_columns = ["bedrooms", "bathrooms", "parking_spots", "floor", "building_age_years"]
    out[[*int_columns, "estrato"]] = out[[*int_columns, "estrato"]].round().astype("Int64")
    return out
