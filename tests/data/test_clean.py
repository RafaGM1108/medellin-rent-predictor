from typing import Any

import pandas as pd

from medellin_rent.data.clean import clean_listings


def _listings(**columns: Any) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    base: dict[str, object] = {
        "listing_id": [f"l{i}" for i in range(n)],
        "text_hash": pd.Series(range(n), dtype="uint64"),
        "start_date": pd.to_datetime(["2021-01-01"] * n),
        "rent_cop": [1_600_000.0] * n,
        "area_m2": [70.0] * n,
        "bedrooms": pd.array([3] * n, dtype="Int64"),
        "bathrooms": pd.array([2] * n, dtype="Int64"),
        "parking_spots": pd.array([1] * n, dtype="Int64"),
        "floor": pd.array([5] * n, dtype="Int64"),
    }
    return pd.DataFrame(base | columns)


def test_rule1_keeps_the_most_recent_duplicate() -> None:
    df = _listings(
        text_hash=pd.Series([7, 7, 8], dtype="uint64"),
        start_date=pd.to_datetime(["2021-01-01", "2021-03-01", "2021-02-01"]),
    )
    assert clean_listings(df)["listing_id"].tolist() == ["l1", "l2"]


def test_rule1_same_text_different_rent_is_not_a_duplicate() -> None:
    df = _listings(text_hash=pd.Series([7, 7], dtype="uint64"), rent_cop=[1e6, 2e6])
    assert len(clean_listings(df)) == 2


def test_rule2_drops_implausible_rents() -> None:
    df = _listings(rent_cop=[299_999.0, 300_000.0, 30_000_000.0, 450_000_000.0])
    assert clean_listings(df)["rent_cop"].tolist() == [300_000.0, 30_000_000.0]


def test_rule3_clears_implausible_area_but_keeps_the_listing() -> None:
    out = clean_listings(_listings(area_m2=[8.0, 70.0, 1_200.0, float("nan")]))
    assert len(out) == 4
    assert out["area_m2"].isna().tolist() == [True, False, True, True]


def test_rule3_clears_area_with_implausible_rent_per_m2() -> None:
    # 1.6M COP for 400 m2 = 4,000 COP/m2 (below range); for 25 m2 = 64,000 COP/m2 (fine).
    out = clean_listings(_listings(area_m2=[400.0, 25.0]))
    assert out["area_m2"].isna().tolist() == [True, False]


def test_rule4_clears_counts_and_floors_above_limits() -> None:
    df = _listings(
        bedrooms=pd.array([65, 3], dtype="Int64"),
        bathrooms=pd.array([2, 13], dtype="Int64"),
        floor=pd.array([99, pd.NA], dtype="Int64"),
    )
    out = clean_listings(df)
    assert out["bedrooms"].isna().tolist() == [True, False]
    assert out["bathrooms"].isna().tolist() == [False, True]
    assert out["floor"].isna().tolist() == [True, True]
    assert out["parking_spots"].tolist() == [1, 1]
