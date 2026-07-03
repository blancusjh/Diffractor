"""Plotting helpers shared by the example scripts."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .field import Field
from .grids import Array, Grid
from .longitudinal import LongitudinalSection

__all__ = ["intensity", "plot_intensity", "plot_longitudinal", "plot_rgb"]


def intensity(U: Union[Field, Array], *, normalize: bool = True) -> Array:
    """Return ``|U|²`` of a field or array, optionally unit-normalized."""
    values = U.values if isinstance(U, Field) else U
    I = np.abs(values) ** 2
    if normalize and I.max() > 0:
        I = I / I.max()
    return I


def plot_intensity(
    ax,
    field: Field,
    grid: Optional[Grid] = None,
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
    field : Field
        Field to display (its grid supplies the plot extent).
    grid : Grid, optional
        Deprecated / optional override of the extent grid; defaults to
        ``field.grid``.
    log : bool
        If True (default), show ``log10`` of the normalized intensity
        clipped to ``[vmin, vmax]`` decades. The default 6-decade range
        matches the useful dynamic range of FFT propagation with
        grid-sampled apertures; deeper floors mostly display the
        edge-quantization noise of the sampled masks.
    """
    if grid is None:
        grid = field.grid
    x, y = grid
    I = intensity(field)
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


def plot_longitudinal(
    ax,
    section: LongitudinalSection,
    *,
    title: Optional[str] = None,
    log: bool = True,
    vmin: float = -4.0,
    vmax: float = 0.0,
    cmap: str = "inferno",
):
    """Draw an axial (``x–z``/``y–z``) intensity cross-section.

    Propagation distance ``z`` runs along the horizontal axis and the sliced
    transverse coordinate along the vertical, so a focusing cone, a beam waist,
    or a grating's Talbot carpet reads directly off the image. Takes the
    :class:`~diffraction.longitudinal.LongitudinalSection` returned by
    :func:`~diffraction.longitudinal.longitudinal_field`.
    """
    data = section.intensity.T  # (n_t, n_z): transverse vertical, z horizontal
    peak = data.max()
    norm = data / peak if peak > 0 else data
    shown = np.log10(norm + 10.0 ** (vmin - 4)) if log else norm
    im = ax.imshow(
        shown,
        extent=[section.z.min(), section.z.max(), section.t.min(), section.t.max()],
        origin="lower",
        aspect="auto",  # z and transverse scales differ by orders of magnitude
        cmap=cmap,
        vmin=vmin if log else None,
        vmax=vmax if log else None,
        interpolation="antialiased",
        interpolation_stage="rgba",
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("z [m]")
    ax.set_ylabel(f"{section.axis} [m]")
    return im


def plot_rgb(ax, rgb: Array, grid: Grid, *, title: Optional[str] = None):
    """Draw an ``(H, W, 3)`` sRGB image on a matplotlib axis.

    Companion to :func:`plot_intensity` for polychromatic results (see
    :func:`diffraction.polychromatic.propagate_polychromatic`).
    """
    x, y = grid
    im = ax.imshow(
        np.clip(rgb, 0.0, 1.0),
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        interpolation="antialiased",
        interpolation_stage="rgba",
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return im
