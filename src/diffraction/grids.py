"""Sampling-grid helpers.

Every routine in the package works on a ``Grid``: a tuple ``(x, y)`` of 2D
coordinate arrays produced by :func:`numpy.meshgrid`. All lengths are in
meters (SI units throughout the package).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

Array = np.ndarray
Grid = Tuple[Array, Array]

__all__ = ["Array", "Grid", "make_grid", "grid_spacing"]


def make_grid(n: int, length: float) -> Grid:
    """Build a square, symmetric sampling grid.

    Parameters
    ----------
    n : int
        Number of samples per side.
    length : float
        Physical side length of the grid [m].

    Returns
    -------
    (x, y) : tuple of 2D arrays
        Coordinates spanning ``[-length/2, length/2)`` with spacing
        ``length / n``. The half-open interval keeps the grid compatible
        with the FFT sampling convention.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if length <= 0:
        raise ValueError("length must be positive.")

    coords = np.linspace(-length / 2, length / 2, n, endpoint=False)
    return tuple(np.meshgrid(coords, coords))


def grid_spacing(grid: Grid) -> Tuple[float, float]:
    """Return the sample spacings ``(dx, dy)`` of a meshgrid pair."""
    x, y = grid
    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    if dx <= 0 or dy <= 0:
        raise ValueError("Grid must be sampled in increasing x and y order.")
    return dx, dy
