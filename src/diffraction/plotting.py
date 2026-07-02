"""Plotting helpers shared by the example scripts."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .grids import Array, Grid

__all__ = ["intensity", "plot_intensity"]


def intensity(U: Array, *, normalize: bool = True) -> Array:
    """Return ``|U|²``, optionally normalized to a unit maximum."""
    I = np.abs(U) ** 2
    if normalize and I.max() > 0:
        I = I / I.max()
    return I


def plot_intensity(
    ax,
    U: Array,
    grid: Grid,
    *,
    title: Optional[str] = None,
    log: bool = True,
    vmin: float = -6.0,
    vmax: float = 0.0,
    cmap: str = "hot",
):
    """Draw the (log-)intensity of a field on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib axis
        Target axis.
    U : 2D complex array
        Field to display.
    grid : (x, y)
        Coordinates of the field samples, used for the plot extent.
    log : bool
        If True (default), show ``log10`` of the normalized intensity
        clipped to ``[vmin, vmax]`` decades. The default 6-decade range
        matches the useful dynamic range of FFT propagation with
        grid-sampled apertures; deeper floors mostly display the
        edge-quantization noise of the sampled masks.
    """
    x, y = grid
    I = intensity(U)
    data = np.log10(I + 10.0 ** (vmin - 4)) if log else I

    # Fringe patterns are usually finer than the screen resolution;
    # antialiased RGBA-stage interpolation avoids moiré when imshow
    # downsamples them for display.
    im = ax.imshow(
        data,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap=cmap,
        vmin=vmin if log else None,
        vmax=vmax if log else None,
        interpolation="antialiased",
        interpolation_stage="rgba",
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return im
