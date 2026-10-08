from pathlib import Path

import pandas as pd
import pytest

from medellin_rent.data.listings import COLUMNS, load_listings, redact_personal_data

FIXTURE = Path(__file__).parent / "fixtures" / "co_properties.csv"


def test_fixture_keeps_only_medellin_apartment_rents_in_cop() -> None:
    listings = load_listings(FIXTURE)

    # fx3 is a sale, fx4 is in Bogotá, fx5 is priced in USD, fx6 is a house.
    assert listings["id"].tolist() == ["fx1", "fx2"]


def test_fixture_multiline_text_and_empty_values_are_preserved() -> None:
    listings = load_listings(FIXTURE).set_index("id")

    description = str(listings.loc["fx1", "description"])
    assert "<br />\nValor $2.500.000." in description  # quoted newlines kept, HTML untouched
    assert listings.loc["fx2", "lat"] == ""  # empty strings, not NaN
    assert listings.loc["fx2", "end_date"] == ""


def test_fixture_redacts_contacts_but_not_prices_or_references() -> None:
    listings = load_listings(FIXTURE).set_index("id")

    fx1 = str(listings.loc["fx1", "description"])
    assert fx1.endswith("Informes: [redacted]")
    assert "$2.500.000" in fx1
    assert "SimiCRM62214300 622-14300" in fx1
    fx2 = str(listings.loc["fx2", "description"])
    assert fx2 == "80 m2, estrato 3. Escribir a [redacted] o llamar al [redacted]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Cel 3001234567", "Cel [redacted]"),
        ("Cel 300 123 4567", "Cel [redacted]"),
        ("Cel 300.123.4567", "Cel [redacted]"),
        ("WhatsApp +57 300 123 4567", "WhatsApp [redacted]"),
        ("Precio 3.200.000", "Precio 3.200.000"),
        ("Area 300 m2", "Area 300 m2"),
        ("Codigo 13001234567890", "Codigo 13001234567890"),  # longer digit run, not a phone
        ("a.b-c+d@mail.example.com", "[redacted]"),
        ("", ""),
    ],
)
def test_redact_personal_data_cases(text: str, expected: str) -> None:
    assert redact_personal_data(pd.Series([text])).iloc[0] == expected


def test_load_listings_rejects_files_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "co_properties.csv"
    path.write_text(",".join(c for c in COLUMNS if c != "price") + "\n")

    with pytest.raises(ValueError, match="price"):
        load_listings(path)


def test_load_listings_returns_empty_frame_when_nothing_matches() -> None:
    listings = load_listings(FIXTURE, city="Cali")

    assert listings.empty
    assert list(listings.columns) == COLUMNS
