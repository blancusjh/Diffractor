"""Source fields.

Helpers that build common input fields on a sampling grid, always returned
as complex arrays ready to feed the propagators.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .grids import Array, Grid

__all__ = ["gaussian_beam", "plane_wave"]


def gaussian_beam(grid: Grid, waist: float, *, center: Tuple[float, float] = (0.0, 0.0)) -> Array:
    """Fundamental Gaussian beam at its waist plane.

    Amplitude profile ``exp(-r² / w0²)`` with ``1/e`` field radius ``waist``.
    """
    if waist <= 0:
        raise ValueError("waist must be positive.")
    x, y = grid
    x0, y0 = center
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return np.exp(-r2 / waist**2).astype(np.complex128)


def plane_wave(
    grid: Grid,
    wavelength: float,
    *,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
    n: float = 1.0,
) -> Array:
    """Unit-amplitude plane wave, optionally tilted.

    Parameters
    ----------
    grid : (x, y)
        Spatial coordinate grids.
    wavelength : float
        Vacuum wavelength [m].
    theta_x, theta_y : float
        Small tilt angles [rad] about the y and x axes respectively.
    n : float
        Refractive index of the medium.
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    x, y = grid
    k = 2 * np.pi * n / wavelength
    return np.exp(1.0j * k * (np.sin(theta_x) * x + np.sin(theta_y) * y))
