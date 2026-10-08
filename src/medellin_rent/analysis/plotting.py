"""Shared figure style for static PNGs in ``data/08_reporting``.

Colors are the reference palette of the data-viz guide: one blue series on a light surface,
text in neutral ink, recessive grid. Figures are light-mode PNGs (they are embedded in the
README and docs).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: scripts and CI have no display

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
# Sequential blue ramp, steps 100 -> 700; grey for cells/areas without enough data.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NO_DATA = "#f0efec"


def new_figure(title: str, width: float = 7.0, height: float = 4.0) -> tuple[Figure, Axes]:
    """Create a figure with the project style and a left-aligned title."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12, pad=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def label_axes(ax: Axes, x: str, y: str) -> None:
    """Set axis labels in secondary ink."""
    ax.set_xlabel(x, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(y, color=TEXT_SECONDARY, fontsize=9)


def save(fig: Figure, path: Path) -> Path:
    """Write the figure as PNG and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
