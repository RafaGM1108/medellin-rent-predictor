import pandas as pd

from medellin_rent.features import build


def _primary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": ["a", "b", "c"],
            "area_m2": [70.0, float("nan"), 120.0],
            "bedrooms": pd.array([3, pd.NA, 4], dtype="Int64"),
            "bathrooms": pd.array([2, 1, pd.NA], dtype="Int64"),
            "has_parking": pd.array([True, False, pd.NA], dtype="boolean"),
            "estrato": pd.array([4, pd.NA, 6], dtype="Int64"),
            "comuna_code": pd.Series(["11", "14", None], dtype="str"),
            "lat": [6.25, float("nan"), 6.20],
            "lon": [-75.59, float("nan"), -75.57],
            "barrio_name": pd.Series(["Laureles", None, "Patio Bonito"], dtype="str"),
            "start_date": pd.to_datetime(pd.Series(["2020-08-01", "2021-03-15", None])).to_numpy(),
            **{a: [True, False, False] for a in build.AMENITIES},
        },
        index=[10, 20, 30],  # not a range index: features must keep the listing's index
    )


def test_size_features() -> None:
    out = build.size_features(_primary())
    assert out["area_missing"].tolist() == [False, True, False]
    assert out["area_m2"].iloc[0] == 70.0


def test_room_features_are_float_with_nan() -> None:
    out = build.room_features(_primary())
    assert out.dtypes.eq("float64").all()
    assert pd.isna(out.loc[20, "bedrooms"])


def test_parking_three_levels() -> None:
    out = build.parking_feature(_primary())
    assert out.tolist() == ["yes", "no", "unknown"]
    assert out.cat.categories.tolist() == build.PARKING


def test_location_features_fixed_categories() -> None:
    out = build.location_features(_primary())
    assert out["estrato"].tolist() == ["4", "unknown", "6"]
    assert out["comuna_code"].cat.categories.tolist() == build.COMUNAS
    assert pd.isna(out.loc[30, "comuna_code"])
    assert out.index.tolist() == [10, 20, 30]
    assert out.loc[10, "lat"] == 6.25


def test_time_features() -> None:
    out = build.time_features(_primary())
    assert out.tolist() == ["2020Q3", "2021Q1", "unknown"]
    assert out.cat.categories.tolist() == build.QUARTERS


def test_quarter_outside_the_data_period_is_unknown() -> None:
    df = _primary().assign(start_date=pd.to_datetime(["2026-10-01"] * 3))
    assert build.time_features(df).eq("unknown").all()


def test_build_features_columns_and_single_row_encoding() -> None:
    full = build.build_features(_primary())
    assert full.columns.tolist() == ["listing_id", *build.FEATURES]
    assert "rent_cop" not in full.columns  # no target in the feature table
    one = build.build_features(_primary().iloc[[1]])
    for column in build.CATEGORICAL:
        assert one[column].cat.categories.equals(full[column].cat.categories)
