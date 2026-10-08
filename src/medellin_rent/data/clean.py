"""Cleaning rules: intermediate listings -> primary listings (``03_primary``).

Rules, applied in order (thresholds chosen from the real distribution, see
``data/README.md``):

1. **Duplicates.** Listings with the same text and rent are re-publications of one ad; keep
   the most recent (latest ``start_date``).
2. **Rent.** Drop listings with a monthly rent outside ``RENT_RANGE``. Rent is the target, so
   a row with an implausible rent is not usable.
3. **Area.** Set ``area_m2`` to NA outside ``AREA_RANGE`` or when rent per m² falls outside
   ``RENT_PER_M2_RANGE`` (almost always a parsing error); the listing is kept.
4. **Counts and floor.** Set bedrooms, bathrooms and parking spots above ``MAX_COUNT`` and
   floors above ``MAX_FLOOR`` to NA.
"""

import pandas as pd

from medellin_rent.utils.log import get_logger

logger = get_logger(__name__)

RENT_RANGE = (300_000, 30_000_000)  # COP per month
AREA_RANGE = (20, 1_000)  # m²
RENT_PER_M2_RANGE = (5_000, 150_000)  # COP per m² per month
MAX_COUNT = 10
MAX_FLOOR = 50


def clean_listings(listings: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules and log how many rows or values each one affects.

    Args:
        listings: Intermediate listings (optionally with location columns).

    Returns:
        A cleaned copy with a fresh index.
    """
    n = len(listings)
    df = (
        listings.sort_values(["start_date", "listing_id"], ascending=[False, True])
        .drop_duplicates(["text_hash", "rent_cop"])
        .sort_index()
    )
    logger.info("Rule 1 duplicates: dropped %d of %d", n - len(df), n)

    n = len(df)
    df = df[df["rent_cop"].between(*RENT_RANGE)].copy()
    logger.info("Rule 2 rent outside %s COP: dropped %d of %d", RENT_RANGE, n - len(df), n)

    rent_per_m2 = df["rent_cop"] / df["area_m2"]
    bad_area = df["area_m2"].notna() & (
        ~df["area_m2"].between(*AREA_RANGE) | ~rent_per_m2.between(*RENT_PER_M2_RANGE)
    )
    df.loc[bad_area, "area_m2"] = float("nan")
    logger.info("Rule 3 implausible area: cleared %d values", bad_area.sum())

    for column, limit in [
        ("bedrooms", MAX_COUNT),
        ("bathrooms", MAX_COUNT),
        ("parking_spots", MAX_COUNT),
        ("floor", MAX_FLOOR),
    ]:
        too_high = df[column] > limit
        df.loc[too_high.fillna(False), column] = pd.NA
        logger.info("Rule 4 %s above %d: cleared %d values", column, limit, too_high.sum())

    return df.reset_index(drop=True)
