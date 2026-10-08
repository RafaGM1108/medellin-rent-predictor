"""Exploratory analysis of the primary listings: distributions and data quality.

Writes to ``data/08_reporting``:

- ``eda_summary.csv``: count, mean and quantiles of rent, area, rent per m², bedrooms and
  bathrooms
- ``eda_missing_values.csv``: share of listings with a value, per field
- ``eda_rent.png``, ``eda_area.png``, ``eda_bedrooms.png``, ``eda_missing_values.png``
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from medellin_rent.analysis.plotting import (
    GRID,
    SERIES,
    SURFACE,
    TEXT_SECONDARY,
    label_axes,
    new_figure,
    save,
)

NUMERIC = ["rent_cop", "area_m2", "rent_per_m2", "bedrooms", "bathrooms"]
QUALITY_FIELDS = [
    "rent_cop",
    "comuna_code",
    "estrato",
    "bathrooms",
    "bedrooms",
    "has_parking",
    "lat",
    "barrio_name",
    "area_m2",
    "parking_spots",
    "floor",
    "building_age_years",
]


def with_rent_per_m2(listings: pd.DataFrame) -> pd.DataFrame:
    """Add ``rent_per_m2`` (NaN where the area is unknown)."""
    return listings.assign(rent_per_m2=listings["rent_cop"] / listings["area_m2"])


def summary(listings: pd.DataFrame) -> pd.DataFrame:
    """Count, mean and quantiles of the main numeric fields (one row per field)."""
    numeric = with_rent_per_m2(listings)[NUMERIC].astype("float64")
    table = numeric.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    return table.rename_axis("field").reset_index()


def missing_values(listings: pd.DataFrame) -> pd.DataFrame:
    """Share of listings with a value for each field, sorted from most to least complete."""
    present = listings[QUALITY_FIELDS].notna().mean()
    table = pd.DataFrame({"field": present.index, "share_present": present.to_numpy()})
    return table.sort_values("share_present", ascending=False, ignore_index=True)


def _millions(value: float, _: object) -> str:
    return f"{value / 1e6:g}M"


def plot_rent(listings: pd.DataFrame, path: Path) -> Path:
    """Histogram of monthly rent on a log scale."""
    rent = listings["rent_cop"].dropna()
    fig, ax = new_figure(f"Monthly rent (n = {len(rent):,})")
    bins = np.logspace(np.log10(rent.min()), np.log10(rent.max()), 40).tolist()
    ax.hist(rent, bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_millions))
    ax.axvline(rent.median(), color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.annotate(
        f"median {rent.median() / 1e6:.2f}M",
        (rent.median(), ax.get_ylim()[1] * 0.92),
        xytext=(4, 0),
        textcoords="offset points",
        color=TEXT_SECONDARY,
        fontsize=9,
    )
    label_axes(ax, "COP per month (log scale)", "Listings")
    return save(fig, path)


def plot_area(listings: pd.DataFrame, path: Path) -> Path:
    """Histogram of area for listings that state it."""
    area = listings["area_m2"].dropna()
    fig, ax = new_figure(f"Area, listings that state it (n = {len(area):,})")
    ax.hist(area.clip(upper=400), bins=40, color=SERIES, edgecolor=SURFACE, linewidth=2)
    label_axes(ax, "m² (values above 400 shown at 400)", "Listings")
    return save(fig, path)


def plot_bedrooms(listings: pd.DataFrame, path: Path) -> Path:
    """Bar chart of listings by number of bedrooms."""
    counts = listings["bedrooms"].dropna().astype(int).value_counts().sort_index()
    fig, ax = new_figure(f"Bedrooms (n = {counts.sum():,})")
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=SERIES, width=0.7)
    label_axes(ax, "Bedrooms", "Listings")
    return save(fig, path)


def plot_missing_values(table: pd.DataFrame, path: Path) -> Path:
    """Horizontal bars: share of listings with each field."""
    fig, ax = new_figure("Share of listings with each field", height=4.5)
    ordered = table.iloc[::-1]
    ax.barh(ordered["field"], ordered["share_present"], color=SERIES, height=0.7)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for y, share in enumerate(ordered["share_present"]):
        ax.text(share + 0.01, y, f"{share:.0%}", va="center", fontsize=8, color=TEXT_SECONDARY)
    label_axes(ax, "Listings with a value", "")
    return save(fig, path)


def run(listings: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write every EDA table and figure to ``out_dir``.

    Args:
        listings: Primary listings table.
        out_dir: Reporting directory, usually ``config.paths.reporting``.

    Returns:
        Paths of the written files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = missing_values(listings)
    summary(listings).to_csv(out_dir / "eda_summary.csv", index=False, float_format="%.1f")
    quality.to_csv(out_dir / "eda_missing_values.csv", index=False, float_format="%.4f")
    return [
        out_dir / "eda_summary.csv",
        out_dir / "eda_missing_values.csv",
        plot_rent(listings, out_dir / "eda_rent.png"),
        plot_area(listings, out_dir / "eda_area.png"),
        plot_bedrooms(listings, out_dir / "eda_bedrooms.png"),
        plot_missing_values(quality, out_dir / "eda_missing_values.png"),
    ]
