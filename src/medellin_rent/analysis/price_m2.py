"""Rent per m² by comuna and by estrato.

Only listings that state their area can have a rent per m². Groups with fewer than
``MIN_N`` such listings are kept in the CSVs (with their ``n``) but left out of the figures,
because their medians are not reliable.

Writes to ``data/08_reporting``:

- ``price_m2_by_comuna.csv``, ``price_m2_by_estrato.csv``, ``price_m2_comuna_estrato.csv``
- ``price_m2_by_comuna.png``, ``price_m2_by_estrato.png``, ``price_m2_comuna_estrato.png``
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from medellin_rent.analysis.eda import with_rent_per_m2
from medellin_rent.analysis.plotting import (
    BLUES,
    GRID,
    NO_DATA,
    SERIES,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    label_axes,
    new_figure,
    save,
)

MIN_N = 30


def _thousands(value: float, _: object = None) -> str:
    return f"{value / 1e3:,.0f}k"


def by_group(listings: pd.DataFrame, group: str | list[str]) -> pd.DataFrame:
    """Median and quartiles of rent per m² per group, with the number of listings behind it.

    Args:
        listings: Primary listings.
        group: Column(s) to group by.

    Returns:
        One row per group: ``n``, ``median``, ``q25``, ``q75`` of rent per m² (COP) and the
        median monthly rent of the same listings.
    """
    df = with_rent_per_m2(listings).dropna(subset=["rent_per_m2"])
    df = df.dropna(subset=[group] if isinstance(group, str) else group)
    grouped = df.groupby(group, observed=True)
    table = pd.DataFrame(
        {
            "n": grouped.size(),
            "median_rent_per_m2": grouped["rent_per_m2"].median(),
            "q25_rent_per_m2": grouped["rent_per_m2"].quantile(0.25),
            "q75_rent_per_m2": grouped["rent_per_m2"].quantile(0.75),
            "median_rent": grouped["rent_cop"].median(),
        }
    )
    return table.reset_index()


def by_comuna(listings: pd.DataFrame) -> pd.DataFrame:
    """Rent per m² by comuna, most expensive first."""
    table = by_group(listings, ["comuna_code", "comuna_name"])
    return table.sort_values("median_rent_per_m2", ascending=False, ignore_index=True)


def by_estrato(listings: pd.DataFrame) -> pd.DataFrame:
    """Rent per m² by estrato (1-6)."""
    table = by_group(listings, "estrato")
    table["estrato"] = table["estrato"].astype(int)
    return table.sort_values("estrato", ignore_index=True)


def comuna_estrato(listings: pd.DataFrame) -> pd.DataFrame:
    """Rent per m² for every comuna and estrato combination present."""
    table = by_group(listings, ["comuna_name", "estrato"])
    table["estrato"] = table["estrato"].astype(int)
    return table


def plot_by_comuna(table: pd.DataFrame, path: Path) -> Path:
    """Horizontal bars of median rent per m² by comuna, with the interquartile range."""
    shown = table[table["n"] >= MIN_N].iloc[::-1]
    names = shown["comuna_name"].str.title()
    fig, ax = new_figure(f"Median rent per m² by comuna (n ≥ {MIN_N})", height=5.5)
    ax.barh(names, shown["median_rent_per_m2"], color=SERIES, height=0.65)
    ax.errorbar(
        shown["median_rent_per_m2"],
        names,
        xerr=[
            shown["median_rent_per_m2"] - shown["q25_rent_per_m2"],
            shown["q75_rent_per_m2"] - shown["median_rent_per_m2"],
        ],
        fmt="none",
        ecolor=TEXT_SECONDARY,
        elinewidth=1,
        capsize=0,
    )
    for y, (value, q75, n) in enumerate(
        zip(shown["median_rent_per_m2"], shown["q75_rent_per_m2"], shown["n"], strict=True)
    ):
        ax.text(
            q75,
            y,
            f"  {_thousands(value)}  (n={n:,})",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, shown["q75_rent_per_m2"].max() * 1.35)
    label_axes(ax, "COP per m² per month (bar: median; line: 25th-75th percentile)", "")
    return save(fig, path)


def plot_by_estrato(table: pd.DataFrame, path: Path) -> Path:
    """Bars of median rent per m² by estrato, with the interquartile range."""
    shown = table[table["n"] >= MIN_N]
    labels = shown["estrato"].astype(str)
    fig, ax = new_figure(f"Median rent per m² by estrato (n ≥ {MIN_N})")
    ax.bar(labels, shown["median_rent_per_m2"], color=SERIES, width=0.6)
    ax.errorbar(
        labels,
        shown["median_rent_per_m2"],
        yerr=[
            shown["median_rent_per_m2"] - shown["q25_rent_per_m2"],
            shown["q75_rent_per_m2"] - shown["median_rent_per_m2"],
        ],
        fmt="none",
        ecolor=TEXT_SECONDARY,
        elinewidth=1,
        capsize=0,
    )
    for x, (value, n) in enumerate(zip(shown["median_rent_per_m2"], shown["n"], strict=True)):
        ax.text(
            x,
            0,
            f"{_thousands(value)}\nn={n:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=SURFACE,
        )
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    label_axes(ax, "Estrato", "COP per m² per month")
    return save(fig, path)


def plot_comuna_estrato(table: pd.DataFrame, comunas: list[str], path: Path) -> Path:
    """Heatmap of median rent per m², comunas (rows) by estrato (columns)."""
    grid = table.pivot(index="comuna_name", columns="estrato", values="median_rent_per_m2")
    counts = table.pivot(index="comuna_name", columns="estrato", values="n")
    grid = grid.reindex(index=comunas, columns=range(1, 7))
    counts = counts.reindex(index=comunas, columns=range(1, 7))
    values = grid.where(counts >= MIN_N)

    fig, ax = new_figure("Median rent per m² by comuna and estrato", width=7.5, height=6.0)
    ax.set_facecolor(NO_DATA)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    finite = values.to_numpy(dtype=float)
    vmin, vmax = np.nanmin(finite), np.nanmax(finite)
    ax.imshow(np.ma.masked_invalid(finite), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for (i, j), value in np.ndenumerate(finite):
        if not np.isnan(value):
            dark = (value - vmin) / (vmax - vmin) > 0.5
            ax.text(
                j,
                i,
                _thousands(value),
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if dark else TEXT_PRIMARY,
            )
    ax.set_xticks(range(6), [str(e) for e in range(1, 7)])
    ax.set_yticks(range(len(comunas)), [c.title() for c in comunas])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    label_axes(ax, f"Estrato (grey: fewer than {MIN_N} listings)", "")
    return save(fig, path)


def run(listings: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write the rent-per-m² tables and figures to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    comuna, estrato, cross = by_comuna(listings), by_estrato(listings), comuna_estrato(listings)
    for name, table in [("by_comuna", comuna), ("by_estrato", estrato), ("comuna_estrato", cross)]:
        table.to_csv(out_dir / f"price_m2_{name}.csv", index=False, float_format="%.0f")
    comunas = comuna.loc[comuna["n"] >= MIN_N, "comuna_name"].tolist()
    return [
        out_dir / "price_m2_by_comuna.csv",
        out_dir / "price_m2_by_estrato.csv",
        out_dir / "price_m2_comuna_estrato.csv",
        plot_by_comuna(comuna, out_dir / "price_m2_by_comuna.png"),
        plot_by_estrato(estrato, out_dir / "price_m2_by_estrato.png"),
        plot_comuna_estrato(cross, comunas, out_dir / "price_m2_comuna_estrato.png"),
    ]
