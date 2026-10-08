"""Hypothesis tests: does parking, and does estrato, change the rent?

Rents are right-skewed and far from normal, so the tests are rank-based (Mann-Whitney U,
Kruskal-Wallis, Spearman). Every test reports an effect size next to its p-value:

- **Parking.** H0: the rent distribution is the same for listings that state they have
  parking and listings that state they have none. Listings that do not mention parking are
  excluded (unknown, not "no"). Effect sizes: ratio of medians with a bootstrap 95% CI, and
  the rank-biserial correlation. Parking is more common in higher estratos, so the test is
  repeated within each estrato (Holm-adjusted p-values).
- **Estrato.** H0: rent per m² has the same distribution in every estrato. Effect sizes:
  epsilon² for Kruskal-Wallis and Spearman's rho for the ordered trend; adjacent estratos
  are compared pairwise (Holm-adjusted).

Assumptions: listings are independent (duplicates were removed in cleaning, but one agency
can still publish similar units) and groups are compared on medians (Mann-Whitney is a test
of stochastic dominance; it reads as a median shift only if the shapes are similar).

Writes ``hypothesis_tests.json`` to ``data/08_reporting``.
"""

import json
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from medellin_rent.analysis.eda import with_rent_per_m2

MIN_N = 20  # per group, for a within-estrato comparison
N_BOOTSTRAP = 2_000


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


def compare_two(a: pd.Series, b: pd.Series, rng: np.random.Generator) -> dict[str, Any]:
    """Mann-Whitney U of ``a`` vs ``b`` with median ratio (bootstrap CI) and rank-biserial r."""
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    result = stats.mannwhitneyu(x, y, alternative="two-sided")
    boot = [
        np.median(rng.choice(x, x.size)) / np.median(rng.choice(y, y.size))
        for _ in range(N_BOOTSTRAP)
    ]
    return {
        "n_a": int(x.size),
        "n_b": int(y.size),
        "median_a": float(np.median(x)),
        "median_b": float(np.median(y)),
        "median_ratio": float(np.median(x) / np.median(y)),
        "median_ratio_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "rank_biserial": float(2 * result.statistic / (x.size * y.size) - 1),
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def parking_effect(listings: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Rent of listings with vs without parking, overall and within each estrato."""
    rng = np.random.default_rng(seed)
    known = listings[listings["has_parking"].notna()]
    with_parking = known["has_parking"].astype(bool)
    overall = compare_two(
        known.loc[with_parking, "rent_cop"], known.loc[~with_parking, "rent_cop"], rng
    )

    by_estrato = []
    for estrato, group in known.dropna(subset=["estrato"]).groupby("estrato"):
        has = group["has_parking"].astype(bool)
        if has.sum() >= MIN_N and (~has).sum() >= MIN_N:
            result = compare_two(group.loc[has, "rent_cop"], group.loc[~has, "rent_cop"], rng)
            by_estrato.append({"estrato": int(cast(int, estrato)), **result})
    for row, adjusted in zip(by_estrato, holm([r["p_value"] for r in by_estrato]), strict=True):
        row["p_value_holm"] = adjusted

    return {
        "hypothesis": "Rent is the same with and without parking (a = with, b = without)",
        "test": "Mann-Whitney U, two-sided",
        "measure": "rent_cop",
        "excluded": "listings that do not mention parking",
        "overall": overall,
        "by_estrato": by_estrato,
        "min_n_per_group_by_estrato": MIN_N,
    }


def estrato_effect(listings: pd.DataFrame, seed: int, min_n: int = 30) -> dict[str, Any]:
    """Rent per m² across estratos: Kruskal-Wallis, Spearman trend and adjacent pairs."""
    rng = np.random.default_rng(seed)
    df = with_rent_per_m2(listings).dropna(subset=["rent_per_m2", "estrato"])
    counts = df["estrato"].value_counts()
    df = df[df["estrato"].isin(counts[counts >= min_n].index)]
    groups = {int(cast(int, e)): g["rent_per_m2"] for e, g in df.groupby("estrato")}

    kruskal = stats.kruskal(*groups.values())
    spearman = stats.spearmanr(df["estrato"].astype(int), df["rent_per_m2"])
    levels = sorted(groups)
    pairs = [
        {"estratos": [lo, hi], **compare_two(groups[hi], groups[lo], rng)}
        for lo, hi in pairwise(levels)
    ]
    for row, adjusted in zip(pairs, holm([p["p_value"] for p in pairs]), strict=True):
        row["p_value_holm"] = adjusted

    n = len(df)
    return {
        "hypothesis": "Rent per m² has the same distribution in every estrato",
        "measure": "rent_per_m2",
        "estratos": levels,
        "n": n,
        "kruskal_wallis": {
            "h_statistic": float(kruskal.statistic),
            "p_value": float(kruskal.pvalue),
            "epsilon_squared": float(kruskal.statistic / (n - 1)),
        },
        "spearman": {"rho": float(spearman.statistic), "p_value": float(spearman.pvalue)},
        "adjacent_pairs": pairs,
        "pairs_note": "a = higher estrato, b = lower estrato",
    }


def run(listings: pd.DataFrame, out_dir: Path, seed: int) -> list[Path]:
    """Run both tests and write ``hypothesis_tests.json`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "parking": parking_effect(listings, seed),
        "estrato": estrato_effect(listings, seed),
    }
    path = out_dir / "hypothesis_tests.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    return [path]
