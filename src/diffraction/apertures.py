"""Aperture (pupil) masks.

Each function evaluates a boolean transmission mask on ``(x, y)`` coordinate
arrays. Cast to complex (or multiply directly into a complex field) to use
them as thin amplitude screens.

Hard-edged masks sampled on a Cartesian grid have jagged, pixelated edges
whose spurious high-frequency content shows up as anisotropic streaks in
the propagated far field. Wrap any mask function with :func:`antialiased`
to evaluate it with grey (area-coverage) edge pixels instead.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .field import Field
from .grids import Array, Grid, grid_spacing

__all__ = [
    "annular_aperture",
    "antialiased",
    "circular_aperture",
    "elliptical_aperture",
    "rectangular_aperture",
    "slit_aperture",
    "square_aperture",
]

Center = Tuple[float, float]


def antialiased(
    f: Callable[..., Array], grid: Grid, *params, factor: int = 4, **kwargs
) -> Field:
    """Evaluate a boolean aperture with grey (area-coverage) edge pixels.

    The mask is sampled on a ``factor × factor`` subgrid inside every pixel
    and averaged, so edge pixels take the fractional value of the aperture
    area they cover. This suppresses the pixelation artifacts of hard-edged
    masks by several orders of magnitude in the propagated far field.

    Parameters
    ----------
    f : callable
        Any mask function from this module, ``f(x, y, *params, **kwargs)``.
    grid : Grid
        Spatial coordinate grid.
    factor : int
        Subsamples per pixel and axis (default 4, i.e. 16 per pixel).

    Returns
    -------
    Field
        Complex transmission field in ``[0, 1]``, ready to use as a source or
        to multiply onto another field on the same grid.
    """
    if factor < 1:
        raise ValueError("factor must be at least 1.")
    x, y = grid
    dx, dy = grid_spacing(grid)
    offsets = (np.arange(factor) + 0.5) / factor - 0.5

    acc = np.zeros(x.shape, dtype=float)
    for ox in offsets:
        for oy in offsets:
            acc += f(x + ox * dx, y + oy * dy, *params, **kwargs)
    return Field(grid, (acc / factor**2).astype(np.complex128))


def circular_aperture(x: Array, y: Array, R: float, *, center: Center = (0.0, 0.0)) -> Array:
    """Circular pupil of radius ``R``: True inside, False outside."""
    x0, y0 = center
    return (x - x0) ** 2 + (y - y0) ** 2 <= R * R


def rectangular_aperture(
    x: Array, y: Array, Wx: float, Wy: float, *, center: Center = (0.0, 0.0)
) -> Array:
    """Axis-aligned rectangle of full widths ``Wx`` and ``Wy``."""
    x0, y0 = center
    return (np.abs(x - x0) <= 0.5 * Wx) & (np.abs(y - y0) <= 0.5 * Wy)


def square_aperture(x: Array, y: Array, W: float, *, center: Center = (0.0, 0.0)) -> Array:
    """Square of side ``W``."""
    return rectangular_aperture(x, y, W, W, center=center)


def annular_aperture(
    x: Array, y: Array, R_inner: float, R_outer: float, *, center: Center = (0.0, 0.0)
) -> Array:
    """Annulus: ``R_inner <= r <= R_outer``."""
    x0, y0 = center
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return (R_inner * R_inner <= r2) & (r2 <= R_outer * R_outer)


def elliptical_aperture(
    x: Array, y: Array, a: float, b: float, *, center: Center = (0.0, 0.0)
) -> Array:
    """Ellipse with semi-axes ``a`` (x) and ``b`` (y)."""
    x0, y0 = center
    X = (x - x0) / a
    Y = (y - y0) / b
    return X * X + Y * Y <= 1.0


def slit_aperture(
    x: Array, y: Array, W: float, *, orientation: str = "x", center: Center = (0.0, 0.0)
) -> Array:
    """Infinite slit of width ``W``.

    ``orientation='x'`` means the slit is long along x and limited in y
    (``|y - y0| <= W/2``); ``orientation='y'`` is the transposed case.
    """
    x0, y0 = center
    h = 0.5 * W
    if orientation == "x":
        return np.abs(y - y0) <= h
    if orientation == "y":
        return np.abs(x - x0) <= h
    raise ValueError("orientation must be 'x' or 'y'.")
