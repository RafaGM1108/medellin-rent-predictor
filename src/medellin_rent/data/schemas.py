"""Pandera schemas for the intermediate and primary listings tables.

Intermediate is a faithful parse: it checks types and domains (estrato 1-6, valid
coordinates, no negative numbers) but keeps odd values such as 0 m² or 65 bedrooms. Primary
adds what cleaning guarantees: a positive rent on every row, positive area, sensible counts,
unique listings and the comuna columns. Plausibility thresholds beyond that (e.g. a 10 m²
apartment) are cleaning rules (#15), not schema rules.
"""

import pandera.pandas as pa

_NON_NEGATIVE = pa.Check.ge(0)

IntermediateSchema = pa.DataFrameSchema(
    {
        "listing_id": pa.Column(str, nullable=False),
        "start_date": pa.Column("datetime64[us]", nullable=True),
        "end_date": pa.Column("datetime64[us]", nullable=True),
        "created_on": pa.Column("datetime64[us]", nullable=True),
        "lat": pa.Column(float, pa.Check.in_range(-90, 90), nullable=True),
        "lon": pa.Column(float, pa.Check.in_range(-180, 180), nullable=True),
        "barrio_raw": pa.Column(str, nullable=True),
        "rent_cop": pa.Column(float, _NON_NEGATIVE, nullable=True),
        "area_m2": pa.Column(float, _NON_NEGATIVE, nullable=True),
        "bedrooms": pa.Column("Int64", _NON_NEGATIVE, nullable=True),
        "bathrooms": pa.Column("Int64", _NON_NEGATIVE, nullable=True),
        "parking_spots": pa.Column("Int64", _NON_NEGATIVE, nullable=True),
        "has_parking": pa.Column("boolean", nullable=True),
        "floor": pa.Column("Int64", _NON_NEGATIVE, nullable=True),
        "building_age_years": pa.Column("Int64", _NON_NEGATIVE, nullable=True),
        "estrato": pa.Column("Int64", pa.Check.isin(range(1, 7)), nullable=True),
        "has_elevator": pa.Column(bool),
        "has_pool": pa.Column(bool),
        "has_gym": pa.Column(bool),
        "has_balcony": pa.Column(bool),
        "has_doorman": pa.Column(bool),
        "is_furnished": pa.Column(bool),
    },
    strict=True,
    coerce=False,
    name="IntermediateListings",
)

PrimarySchema = (
    IntermediateSchema.add_columns(
        {
            "barrio_name": pa.Column(str, nullable=True),
            "comuna_code": pa.Column(str, nullable=True),
            "comuna_name": pa.Column(str, nullable=True),
        }
    )
    .update_columns(
        {
            "listing_id": {"unique": True},
            "rent_cop": {"nullable": False, "checks": [pa.Check.gt(0)]},
            "area_m2": {"checks": [pa.Check.gt(0)]},
            "bedrooms": {"checks": [_NON_NEGATIVE, pa.Check.le(20)]},
            "bathrooms": {"checks": [_NON_NEGATIVE, pa.Check.le(20)]},
            "parking_spots": {"checks": [_NON_NEGATIVE, pa.Check.le(20)]},
        }
    )
    .set_name("PrimaryListings")
)
