"""Shared plotting style and data access for the notebooks.

Keeps the notebooks about physics rather than about matplotlib.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "benchmarks" / "golden"

BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#008300", "#e34948"
YELLOW, VIOLET, AQUA = "#eda100", "#4a3aa7", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURF, EDGE = "#e1e0d9", "#fcfcfb", "#7fd4ff"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5, "figure.dpi": 110,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF})


def ttl(ax, title, sub=None):
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left",
                 pad=9 if sub is None else 17)
    if sub:
        ax.text(0, 1.025, sub, transform=ax.transAxes, fontsize=8.5, color=MUTED)


def load(name):
    """Load a frozen result from benchmarks/golden."""
    return np.load(GOLDEN / f"{name}.npz")


def mirror(rg, F):
    """Half-plane field (n_r, n_z) -> full meridional section."""
    return np.concatenate([-rg[::-1], rg[1:]]), np.vstack([F[::-1], F[1:]])


def meridional(ax, rg, zg, psi, ref, *, decades=2.5, cmap="magma"):
    """log10 |psi|/ref over the meridional plane; returns the image."""
    rr, PP = mirror(rg, psi)
    A = np.abs(PP) / ref
    im = ax.imshow(np.log10(np.maximum(A, 10 ** -decades)),
                   extent=[zg[0], zg[-1], rr[0], rr[-1]], origin="lower",
                   aspect="equal", cmap=cmap, vmin=-decades, vmax=0.2,
                   interpolation="bilinear")
    ax.set_xlabel("z  [λ]")
    ax.set_ylabel("r  [λ]")
    ax.grid(False)
    return im


def focal_peak(D, psi=None):
    """Peak |psi| within 1.5 lambda of the design focus, on the axis."""
    rg, zg = D["rg"], D["zg"]
    w = (np.abs(zg - float(D["zi"])) < 1.5)[None, :] & (rg < 0.4)[:, None]
    return np.abs(D["psi"] if psi is None else psi)[w].max()
