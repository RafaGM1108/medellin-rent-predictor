import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medellin_rent.analysis import hypothesis


def test_holm_adjustment() -> None:
    # Sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone 0.06.
    assert hypothesis.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert hypothesis.holm([0.9, 0.8]) == [1.0, 1.0]


def test_compare_two_effect_sizes() -> None:
    rng = np.random.default_rng(0)
    a = pd.Series([2.0, 3.0, 4.0] * 10)
    b = pd.Series([1.0, 1.5, 2.0] * 10)
    result = hypothesis.compare_two(a, b, rng)
    assert result["median_ratio"] == 2.0
    assert result["median_ratio_ci95"][0] <= 2.0 <= result["median_ratio_ci95"][1]
    assert 0.8 < result["rank_biserial"] <= 1.0
    assert result["p_value"] < 0.001


def _listings() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for estrato, base in [(3, 1.0e6), (4, 1.5e6), (6, 3.0e6)]:
        for has_parking, factor, n in [(True, 1.3, 40), (False, 1.0, 25), (None, 1.0, 10)]:
            rent = base * factor * rng.uniform(0.9, 1.1, n)
            rows += [(estrato, has_parking, r, 60.0) for r in rent]
    rows += [(1, True, 5e5, 50.0)] * 5  # too few for the estrato test
    df = pd.DataFrame(rows, columns=["estrato", "has_parking", "rent_cop", "area_m2"])
    df["estrato"] = df["estrato"].astype("Int64")
    df["has_parking"] = df["has_parking"].astype("boolean")
    return df


def test_parking_effect_excludes_unknown_and_stratifies() -> None:
    result = hypothesis.parking_effect(_listings(), seed=42)
    overall = result["overall"]
    assert (overall["n_a"], overall["n_b"]) == (125, 75)  # NA excluded
    assert [r["estrato"] for r in result["by_estrato"]] == [3, 4, 6]  # estrato 1: no "without"
    assert all(1.2 < r["median_ratio"] < 1.4 for r in result["by_estrato"])
    assert all(r["p_value_holm"] >= r["p_value"] for r in result["by_estrato"])


def test_estrato_effect_skips_small_estratos() -> None:
    result = hypothesis.estrato_effect(_listings(), seed=42)
    assert result["estratos"] == [3, 4, 6]
    assert result["spearman"]["rho"] > 0.5
    assert [p["estratos"] for p in result["adjacent_pairs"]] == [[3, 4], [4, 6]]
    assert 0 < result["kruskal_wallis"]["epsilon_squared"] <= 1


def test_run_is_reproducible(tmp_path: Path) -> None:
    first = hypothesis.run(_listings(), tmp_path / "a", seed=7)[0].read_text()
    second = hypothesis.run(_listings(), tmp_path / "b", seed=7)[0].read_text()
    assert first == second
    assert set(json.loads(first)) == {"parking", "estrato"}
