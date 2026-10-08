from pathlib import Path

import pandas as pd

from medellin_rent.analysis import price_m2


def _primary() -> pd.DataFrame:
    # Comuna 14: 30 listings at 30k/m2 (estrato 6); comuna 11: 30 at 20k/m2 (estrato 4)
    # plus 2 at 10k/m2 (estrato 3, too few for the figures); one without area.
    rows = (
        [("14", "EL POBLADO", 6, 3_000_000.0, 100.0)] * 30
        + [("11", "LAURELES", 4, 2_000_000.0, 100.0)] * 30
        + [("11", "LAURELES", 3, 1_000_000.0, 100.0)] * 2
        + [("11", "LAURELES", 3, 1_000_000.0, float("nan"))]
    )
    df = pd.DataFrame(
        rows, columns=["comuna_code", "comuna_name", "estrato", "rent_cop", "area_m2"]
    )
    df["estrato"] = df["estrato"].astype("Int64")
    return df


def test_by_comuna_sorted_by_median_and_counts_only_listings_with_area() -> None:
    table = price_m2.by_comuna(_primary())
    assert table["comuna_name"].tolist() == ["EL POBLADO", "LAURELES"]
    assert table["n"].tolist() == [30, 32]
    assert table.loc[0, "median_rent_per_m2"] == 30_000
    assert table.loc[1, "median_rent_per_m2"] == 20_000


def test_by_estrato_and_cross_table() -> None:
    estrato = price_m2.by_estrato(_primary())
    assert estrato["estrato"].tolist() == [3, 4, 6]
    assert estrato["median_rent_per_m2"].tolist() == [10_000, 20_000, 30_000]
    cross = price_m2.comuna_estrato(_primary()).set_index(["comuna_name", "estrato"])
    assert cross.loc[("LAURELES", 3), "n"] == 2
    assert cross.loc[("LAURELES", 4), "q25_rent_per_m2"] == 20_000


def test_run_writes_tables_and_figures(tmp_path: Path) -> None:
    paths = price_m2.run(_primary(), tmp_path)
    assert {p.suffix for p in paths} == {".csv", ".png"}
    assert len(paths) == 6
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    # Small groups stay in the CSV with their n.
    assert 2 in pd.read_csv(tmp_path / "price_m2_comuna_estrato.csv")["n"].tolist()
