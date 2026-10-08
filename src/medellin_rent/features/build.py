"""Feature engineering: primary listings -> one row of features per listing (``04_feature``).

Follows the decisions in ``docs/decisions.md``. Every feature is a function of the listing's
own columns only: nothing is computed from other rows (no medians, no target encoding), so
the table cannot leak the target. Imputation and barrio target encoding are fitted inside
the training folds (Phase 5). Floor, number of parking spots and building age are not used
(too sparse, decision 7).
"""

import pandas as pd

AMENITIES = ["has_elevator", "has_pool", "has_gym", "has_balcony", "has_doorman", "is_furnished"]
UNKNOWN = "unknown"

# Fixed category levels, so a single listing at inference gets the same encoding as training.
COMUNA_NAMES = {  # official code -> name (GeoMedellín), 16 comunas and 5 corregimientos
    "01": "POPULAR",
    "02": "SANTA CRUZ",
    "03": "MANRIQUE",
    "04": "ARANJUEZ",
    "05": "CASTILLA",
    "06": "DOCE DE OCTUBRE",
    "07": "ROBLEDO",
    "08": "VILLA HERMOSA",
    "09": "BUENOS AIRES",
    "10": "LA CANDELARIA",
    "11": "LAURELES",
    "12": "LA AMERICA",
    "13": "SAN JAVIER",
    "14": "EL POBLADO",
    "15": "GUAYABAL",
    "16": "BELEN",
    "50": "PALMITAS",
    "60": "SAN CRISTOBAL",
    "70": "ALTAVISTA",
    "80": "SAN ANTONIO DE PRADO",
    "90": "SANTA ELENA",
}
COMUNAS = list(COMUNA_NAMES)
ESTRATOS = [str(e) for e in range(1, 7)] + [UNKNOWN]
PARKING = ["yes", "no", UNKNOWN]
QUARTERS = ["2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3", UNKNOWN]  # data period

CATEGORICAL = ["comuna_code", "estrato", "parking", "listing_quarter"]
NUMERIC = ["area_m2", "bedrooms", "bathrooms", "lat", "lon"]
BOOLEAN = ["area_missing", *AMENITIES]
FEATURES = [*NUMERIC, *BOOLEAN, *CATEGORICAL, "barrio_name"]


def size_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Area in m² (NaN when unknown) and a flag for the missing area."""
    area = listings["area_m2"].astype("float64")
    return pd.DataFrame({"area_m2": area, "area_missing": area.isna()})


def room_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Bedrooms and bathrooms as floats (NaN when unknown)."""
    return listings[["bedrooms", "bathrooms"]].astype("float64")


def parking_feature(listings: pd.DataFrame) -> pd.Series:
    """Parking as three levels: ``yes``, ``no`` or ``unknown`` (not mentioned is not "no")."""
    parking = listings["has_parking"].map({True: "yes", False: "no"}).fillna(UNKNOWN)
    return pd.Series(pd.Categorical(parking, PARKING), index=listings.index, name="parking")


def amenity_features(listings: pd.DataFrame) -> pd.DataFrame:
    """The six amenity flags, as booleans."""
    return listings[AMENITIES].astype(bool)


def location_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Comuna and estrato as categories (estrato has an ``unknown`` level), lat/lon, barrio."""
    estrato = listings["estrato"].astype("Int64").astype("str").where(listings["estrato"].notna())
    return pd.DataFrame(
        {
            "comuna_code": pd.Categorical(listings["comuna_code"], COMUNAS),
            "estrato": pd.Categorical(estrato.fillna(UNKNOWN), ESTRATOS),
            "lat": listings["lat"].astype("float64").to_numpy(),
            "lon": listings["lon"].astype("float64").to_numpy(),
            "barrio_name": listings["barrio_name"].astype("str").to_numpy(),
        },
        index=listings.index,
    )


def time_features(listings: pd.DataFrame) -> pd.Series:
    """Quarter in which the listing was published (e.g. ``2021Q1``), ``unknown`` if no date."""
    quarter = listings["start_date"].dt.to_period("Q").astype("str")
    quarter = quarter.where(listings["start_date"].notna() & quarter.isin(QUARTERS), UNKNOWN)
    return pd.Series(
        pd.Categorical(quarter, QUARTERS), index=listings.index, name="listing_quarter"
    )


def build_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Build every feature for the primary listings.

    Args:
        listings: Primary listings (``03_primary``).

    Returns:
        ``listing_id`` plus ``FEATURES``, one row per listing, in the input order.
    """
    features = pd.concat(
        [
            size_features(listings),
            room_features(listings),
            parking_feature(listings),
            amenity_features(listings),
            location_features(listings),
            time_features(listings),
        ],
        axis=1,
    )
    return pd.concat([listings["listing_id"], features[FEATURES]], axis=1)
