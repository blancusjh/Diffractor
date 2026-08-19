"""Shared plotting style for the example figures.

Same palette as `benchmarks/fig_fields.py`, so the README gallery and the
validation figures look like one document.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"

BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#008300", "#e34948"
YELLOW, VIOLET, AQUA = "#eda100", "#4a3aa7", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
EDGE = "#7fd4ff"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF})


def ttl(a, t, s=None):
    """Left-aligned bold title with an optional grey subtitle."""
    a.set_title(t, fontsize=10.5, fontweight="bold", pad=9 if s is None else 17,
                loc="left")
    if s:
        a.text(0, 1.025, s, transform=a.transAxes, fontsize=8.5, color=MUTED)


def save(fig, name, dpi=115):
    IMAGES.mkdir(parents=True, exist_ok=True)
    out = IMAGES / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(f"  -> {out.relative_to(IMAGES.parents[1])}")
