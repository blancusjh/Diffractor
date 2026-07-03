"""Diffraction gratings — periodic amplitude and phase elements.

Amplitude gratings are ``(x, y, ...) -> Array`` transmission kernels, used just
like the aperture masks — wrap them with
:func:`~diffractor.physics.apertures.antialiased` to get a ``Field`` with grey edges,
or multiply directly onto a source field. Phase gratings are built with
:func:`phase_grating`, which imprints a periodic optical-path profile through
:func:`~diffractor.physics.surfaces.thin_element_phase`, so their phase is correctly
chromatic (``k0 = 2π/λ``).

A grating of period ``d`` sends order ``m`` to angle ``sin θ_m = m λ / d``; in
a far-field / matrix-DFT propagation to distance ``z`` the orders land at
``x_m = m λ z / d``.

All amplitude kernels return *float* transmissions (compare the boolean
aperture masks); combine them arithmetically. The 1D amplitude gratings take
``orientation`` in ``{"x", "y"}`` with 2D crossings provided by
:func:`cross_grating` / :func:`polar_grating`, while :func:`phase_grating`
additionally accepts ``orientation="both"`` (it crosses its own relief).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .field import Field
from ..mathutils.grids import Array, Grid
from .surfaces import thin_element_phase

__all__ = [
    "ronchi_grating",
    "sinusoidal_amplitude_grating",
    "cross_grating",
    "polar_grating",
    "phase_grating",
]

Center = Tuple[float, float]


def _coord(x: Array, y: Array, orientation: str, center: Center) -> Array:
    """The coordinate along the grating vector, shifted by the center."""
    x0, y0 = center
    if orientation == "x":
        return x - x0
    if orientation == "y":
        return y - y0
    raise ValueError("orientation must be 'x' or 'y'.")


def ronchi_grating(
    x: Array,
    y: Array,
    period: float,
    *,
    duty: float = 0.5,
    orientation: str = "x",
    center: Center = (0.0, 0.0),
) -> Array:
    """Binary (Ronchi) amplitude grating: transmitting bars of duty cycle ``duty``.

    Transmission is 1 over a fraction ``duty`` of each period and 0 elsewhere.
    """
    if period <= 0:
        raise ValueError("period must be positive.")
    if not 0.0 < duty < 1.0:
        raise ValueError("duty must be in (0, 1).")
    u = _coord(x, y, orientation, center)
    frac = (u / period) % 1.0
    return (frac < duty).astype(float)


def sinusoidal_amplitude_grating(
    x: Array,
    y: Array,
    period: float,
    *,
    contrast: float = 1.0,
    orientation: str = "x",
    center: Center = (0.0, 0.0),
) -> Array:
    """Sinusoidal amplitude grating ``t = ½(1 + m cos(2π u / d))``.

    A pure cosine transmittance diffracts only into orders 0 and ±1. ``contrast``
    is the modulation depth ``m`` in ``[0, 1]``.
    """
    if period <= 0:
        raise ValueError("period must be positive.")
    if not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must be in [0, 1].")
    u = _coord(x, y, orientation, center)
    return 0.5 * (1.0 + contrast * np.cos(2.0 * np.pi * u / period))


def cross_grating(
    x: Array,
    y: Array,
    period: float,
    *,
    kind: str = "ronchi",
    duty: float = 0.5,
    contrast: float = 1.0,
    center: Center = (0.0, 0.0),
) -> Array:
    """2D crossed grating: the product of two 1D gratings along x and y.

    Produces a 2D lattice of diffraction orders. ``kind`` selects the 1D
    grating (``"ronchi"`` or ``"sinusoidal"``).
    """
    if kind == "ronchi":
        gx = ronchi_grating(x, y, period, duty=duty, orientation="x", center=center)
        gy = ronchi_grating(x, y, period, duty=duty, orientation="y", center=center)
    elif kind == "sinusoidal":
        gx = sinusoidal_amplitude_grating(x, y, period, contrast=contrast, orientation="x", center=center)
        gy = sinusoidal_amplitude_grating(x, y, period, contrast=contrast, orientation="y", center=center)
    else:
        raise ValueError("kind must be 'ronchi' or 'sinusoidal'.")
    return gx * gy


def polar_grating(
    x: Array,
    y: Array,
    *,
    radial_period: float | None = None,
    n_spokes: int | None = None,
    duty: float = 0.5,
    center: Center = (0.0, 0.0),
) -> Array:
    """Binary grating periodic in polar coordinates — the polar `cross_grating`.

    Combines two independent Ronchi-like binary profiles in polar coordinates:

    * **concentric rings** — a square wave of radial period ``radial_period``
      (a circular grating), which diffracts into ring-shaped orders at radii
      ``m λ f / radial_period`` at a lens focus;
    * **angular spokes** — ``n_spokes`` transmitting sectors over the full turn
      (a Siemens-star pattern), which spreads energy into azimuthal orders.

    Give either one alone, or both — with both, the result is their product, a
    polar lattice of apertures whose focal pattern is a polar arrangement of
    orders (the rotational analogue of the Cartesian lattice from
    :func:`cross_grating`). ``duty`` is the transmitting fraction of each
    period. At least one of ``radial_period`` / ``n_spokes`` must be given.
    """
    if radial_period is None and n_spokes is None:
        raise ValueError("give at least one of radial_period or n_spokes.")
    if not 0.0 < duty < 1.0:
        raise ValueError("duty must be in (0, 1).")
    x0, y0 = center
    xr = x - x0
    yr = y - y0
    t = np.ones(np.broadcast(xr, yr).shape, dtype=float)
    if radial_period is not None:
        if radial_period <= 0:
            raise ValueError("radial_period must be positive.")
        r = np.hypot(xr, yr)
        t = t * ((r / radial_period) % 1.0 < duty)
    if n_spokes is not None:
        if int(n_spokes) != n_spokes or n_spokes < 1:
            raise ValueError("n_spokes must be a positive integer.")
        theta = np.arctan2(yr, xr)
        t = t * ((n_spokes * theta / (2.0 * np.pi)) % 1.0 < duty)
    return t.astype(float)


def _grating_sag(u: Array, period: float, height: float, profile: str) -> Array:
    """Periodic surface profile of peak-to-valley ``height`` [m]."""
    frac = (u / period) % 1.0
    if profile == "sinusoidal":
        return 0.5 * height * (1.0 + np.cos(2.0 * np.pi * frac))
    if profile == "binary":
        return height * (frac < 0.5).astype(float)
    if profile == "blazed":
        return height * frac  # sawtooth: ramp 0 -> height over each period
    raise ValueError("profile must be 'sinusoidal', 'binary' or 'blazed'.")


def phase_grating(
    grid: Grid,
    wavelength: float,
    *,
    period: float,
    height: float,
    n1: float = 1.0,
    n2: float = 1.5,
    profile: str = "sinusoidal",
    orientation: str = "x",
    center: Center = (0.0, 0.0),
) -> Field:
    """Periodic phase grating as a thin element, returned as a ``Field``.

    A periodic surface relief of peak-to-valley ``height`` between media ``n1``
    and ``n2`` imprints ``exp(i k0 (n1 - n2) sag)`` (via
    :func:`~diffractor.physics.surfaces.thin_element_phase`), so the phase depth is
    ``2π (n2 - n1) height / λ`` and correctly wavelength-dependent.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    wavelength : float
        Vacuum wavelength [m].
    period : float
        Grating period ``d`` [m].
    height : float
        Peak-to-valley relief height [m].
    n1, n2 : float
        Indices before / after the element.
    profile : {"sinusoidal", "binary", "blazed"}
        Relief shape. ``"blazed"`` is a sawtooth that steers energy into one
        order (the spectrometer case).
    orientation : {"x", "y", "both"}
        Grating vector; ``"both"`` crosses two identical 1D reliefs (2D).
    center : (x0, y0)
        Phase origin.

    Returns
    -------
    Field
        Complex phase transmission on ``grid``.
    """
    if period <= 0:
        raise ValueError("period must be positive.")
    x, y = grid
    if orientation == "both":
        sag = _grating_sag(_coord(x, y, "x", center), period, height, profile) + _grating_sag(
            _coord(x, y, "y", center), period, height, profile
        )
    else:
        sag = _grating_sag(_coord(x, y, orientation, center), period, height, profile)
    return Field(grid, thin_element_phase(sag, wavelength, n1, n2))
