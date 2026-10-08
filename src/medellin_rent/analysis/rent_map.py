"""Choropleth of median rent per m² by comuna.

Boundaries: GeoMedellín (Alcaldía de Medellín), CC BY-SA 4.0. The layer may not be
transferred, so the interactive HTML (which embeds the polygons) is written to
``data/08_reporting`` but git-ignored; only the static PNG is committed.

Writes ``rent_map_comunas.png`` and ``rent_map_comunas.html`` to ``data/08_reporting``.
"""

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex

from medellin_rent.analysis.plotting import (
    BLUES,
    NO_DATA,
    SURFACE,
    TEXT_SECONDARY,
    new_figure,
    save,
)
from medellin_rent.analysis.price_m2 import MIN_N, by_comuna

ATTRIBUTION = "Boundaries: Alcaldía de Medellín (GeoMedellín), CC BY-SA 4.0"


def comuna_layer(listings: pd.DataFrame, comunas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Comuna polygons with ``n`` and ``median_rent_per_m2`` (NaN where n < ``MIN_N``)."""
    table = by_comuna(listings)[["comuna_code", "n", "median_rent_per_m2"]]
    layer = comunas.merge(table, on="comuna_code", how="left")
    layer["n"] = layer["n"].fillna(0).astype(int)
    layer.loc[layer["n"] < MIN_N, "median_rent_per_m2"] = float("nan")
    return layer


def plot_static(layer: gpd.GeoDataFrame, path: Path) -> Path:
    """PNG choropleth on the sequential blue ramp; grey where there is too little data."""
    fig, ax = new_figure("Median rent per m² by comuna (COP per month)", width=6.5, height=6.0)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    layer.plot(
        ax=ax,
        column="median_rent_per_m2",
        cmap=cmap,
        edgecolor=SURFACE,
        linewidth=1,
        legend=True,
        legend_kwds={"shrink": 0.6, "format": lambda v, _: f"{v / 1e3:,.0f}k"},
        missing_kwds={"color": NO_DATA, "edgecolor": SURFACE, "linewidth": 1},
    )
    ax.set_axis_off()
    fig.axes[-1].tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)  # colorbar
    fig.axes[-1].set_frame_on(False)
    ax.text(
        0,
        -0.02,
        f"Grey: fewer than {MIN_N} listings with area. {ATTRIBUTION}",
        transform=ax.transAxes,
        fontsize=7,
        color=TEXT_SECONDARY,
        va="top",
    )
    return save(fig, path)


def plot_interactive(layer: gpd.GeoDataFrame, path: Path) -> Path:
    """Folium choropleth with a tooltip per comuna (git-ignored output)."""
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    values = layer["median_rent_per_m2"]
    vmin = values.min()
    span = (values.max() - vmin) or 1.0  # one comuna with data: avoid dividing by zero

    def style(feature: dict[str, dict[str, float]]) -> dict[str, object]:
        value = feature["properties"]["median_rent_per_m2"]
        fill = NO_DATA if pd.isna(value) else to_hex(cmap((value - vmin) / span))
        return {"fillColor": fill, "color": SURFACE, "weight": 1, "fillOpacity": 0.85}

    shown = layer.assign(
        comuna=layer["comuna_name"].str.title(),
        rent_per_m2=values.map(lambda v: "n < 30" if pd.isna(v) else f"{v:,.0f} COP"),
        geometry=layer.geometry.simplify(0.0002),
    )
    center = layer.geometry.union_all().centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="OpenStreetMap")
    folium.GeoJson(
        shown[["comuna", "rent_per_m2", "n", "median_rent_per_m2", "geometry"]],
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=["comuna", "rent_per_m2", "n"],
            aliases=["Comuna", "Median rent per m²", "Listings with area"],
        ),
        attr=ATTRIBUTION,
    ).add_to(fmap)
    fmap.get_root().html.add_child(  # type: ignore[attr-defined]
        folium.Element(f'<p style="font:12px sans-serif;margin:4px">{ATTRIBUTION}</p>')
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(path)
    return path


def run(listings: pd.DataFrame, comunas: gpd.GeoDataFrame, out_dir: Path) -> list[Path]:
    """Write the static and interactive comuna maps to ``out_dir``."""
    layer = comuna_layer(listings, comunas)
    return [
        plot_static(layer, out_dir / "rent_map_comunas.png"),
        plot_interactive(layer, out_dir / "rent_map_comunas.html"),
    ]
