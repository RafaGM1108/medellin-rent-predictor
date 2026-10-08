from pathlib import Path

import pandas as pd

from medellin_rent.analysis import eda


def _primary() -> pd.DataFrame:
    n = 4
    data: dict[str, object] = dict.fromkeys(eda.QUALITY_FIELDS, [None] * n)
    data |= {
        "rent_cop": [1_000_000.0, 1_600_000.0, 2_000_000.0, 4_000_000.0],
        "area_m2": [50.0, 80.0, float("nan"), float("nan")],
        "bedrooms": pd.array([1, 3, 3, pd.NA], dtype="Int64"),
        "bathrooms": pd.array([1, 2, 2, 3], dtype="Int64"),
        "comuna_code": ["11", "14", None, "14"],
    }
    return pd.DataFrame(data)


def test_summary_includes_rent_per_m2() -> None:
    table = eda.summary(_primary()).set_index("field")
    assert table.loc["rent_per_m2", "count"] == 2
    assert table.loc["rent_per_m2", "50%"] == (20_000 + 20_000) / 2
    assert table.loc["rent_cop", "50%"] == 1_800_000


def test_missing_values_sorted_by_completeness() -> None:
    table = eda.missing_values(_primary())
    assert table.iloc[0].tolist() == ["rent_cop", 1.0]
    shares = dict(zip(table["field"], table["share_present"], strict=True))
    assert shares["comuna_code"] == 0.75
    assert shares["area_m2"] == 0.5
    assert shares["floor"] == 0.0
    assert table["share_present"].is_monotonic_decreasing


def test_run_writes_tables_and_figures(tmp_path: Path) -> None:
    paths = eda.run(_primary(), tmp_path / "reporting")
    assert [p.name for p in paths] == [
        "eda_summary.csv",
        "eda_missing_values.csv",
        "eda_rent.png",
        "eda_area.png",
        "eda_bedrooms.png",
        "eda_missing_values.png",
    ]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    assert pd.read_csv(paths[0])["field"].tolist() == eda.NUMERIC
