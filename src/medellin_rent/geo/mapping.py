"""Assign each listing its barrio and comuna.

In the Properati file ``l4`` (``barrio_raw``) holds the **comuna** name (its 21 values are
Medellín's comunas and corregimientos), so it is matched by normalized name against the
comuna names in the barrios layer. The barrio itself can only come from the coordinates,
via a point-in-polygon join. When both are present and disagree, the declared comuna wins
and the barrio is left empty, because the coordinates are then unreliable.
"""

import unicodedata

import geopandas as gpd
import pandas as pd

from medellin_rent.geo.boundaries import TARGET_CRS
from medellin_rent.utils.log import get_logger

logger = get_logger(__name__)

_ARTICLES = ("LA ", "EL ", "LOS ", "LAS ")


def normalize_name(name: pd.Series) -> pd.Series:
    """Uppercase; strip accents, punctuation and a leading article (La Candelaria -> CANDELARIA)."""
    ascii_name = name.map(
        lambda s: (
            unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
            if isinstance(s, str)
            else s
        )
    ).astype("str")  # keeps the .str accessor when every value is missing
    cleaned = (
        ascii_name.str.upper()
        .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    for article in _ARTICLES:
        cleaned = cleaned.str.removeprefix(article)
    return cleaned


def assign_location(listings: pd.DataFrame, barrios: gpd.GeoDataFrame) -> pd.DataFrame:
    """Add ``barrio_name``, ``comuna_code`` and ``comuna_name`` to the listings.

    Args:
        listings: Intermediate listings with ``lat``, ``lon`` and ``barrio_raw``.
        barrios: Barrios layer from :func:`medellin_rent.geo.boundaries.load_layer`.

    Returns:
        A copy of ``listings`` with the three columns (NA where unknown).
    """
    comunas = barrios[["comuna_code", "comuna_name"]].drop_duplicates()
    by_name = dict(zip(normalize_name(comunas["comuna_name"]), comunas["comuna_code"], strict=True))
    names = dict(zip(comunas["comuna_code"], comunas["comuna_name"], strict=True))

    declared = normalize_name(listings["barrio_raw"]).map(by_name)

    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(listings["lon"], listings["lat"]),
        index=listings.index,
        crs=TARGET_CRS,
    )
    has_point = listings["lat"].notna() & listings["lon"].notna()
    joined = gpd.sjoin(
        points[has_point], barrios[["barrio_name", "comuna_code", "geometry"]], predicate="within"
    )
    joined = joined[~joined.index.duplicated()]  # a point on a shared border: keep the first
    spatial_comuna = joined["comuna_code"].reindex(listings.index)
    spatial_barrio = joined["barrio_name"].reindex(listings.index)

    conflict = declared.notna() & spatial_comuna.notna() & (declared != spatial_comuna)
    comuna_code = declared.fillna(spatial_comuna)

    out = listings.copy()
    out["barrio_name"] = spatial_barrio.where(~conflict).astype("str")
    out["comuna_code"] = comuna_code.astype("str")
    out["comuna_name"] = comuna_code.map(names).astype("str")

    unknown_names = listings.loc[
        listings["barrio_raw"].notna() & declared.isna(), "barrio_raw"
    ].value_counts()
    logger.info(
        "Location: comuna by name %d, by coordinates only %d, none %d of %d; "
        "barrio %d; name/coordinate conflicts %d",
        declared.notna().sum(),
        (declared.isna() & spatial_comuna.notna()).sum(),
        comuna_code.isna().sum(),
        len(listings),
        out["barrio_name"].notna().sum(),
        conflict.sum(),
    )
    if not unknown_names.empty:
        logger.warning("Unmatched comuna names: %s", unknown_names.to_dict())
    return out
