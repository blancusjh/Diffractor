"""Sampling grids.

A :class:`Grid` bundles the two 2D coordinate arrays ``(x, y)`` that every
routine in the package samples fields on, and owns the coordinate-level
queries — spacing, shape, coordinate access. All lengths are in meters (SI
units throughout the package).

``Grid`` is iterable and indexable as the pair ``(x, y)``, so the historical
``x, y = grid`` / ``grid[0]`` idiom keeps working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np

Array = np.ndarray

__all__ = ["Array", "Grid", "make_grid", "grid_spacing"]


@dataclass(frozen=True, eq=False)
class Grid:
    """A pair of 2D coordinate arrays ``(x, y)`` with coordinate queries.

    Parameters
    ----------
    x, y : 2D arrays
        Coordinate grids, as produced by :func:`numpy.meshgrid` (equal shape,
        uniformly sampled). They may be NumPy or CuPy arrays.

    Notes
    -----
    ``eq=False`` keeps the dataclass hashable-by-identity and avoids invoking
    the ambiguous element-wise ``==`` of the underlying arrays.
    """

    x: Array
    y: Array

    def __iter__(self) -> Iterator[Array]:
        """Yield ``x`` then ``y``, so ``x, y = grid`` unpacks as before."""
        return iter((self.x, self.y))

    def __getitem__(self, index: int) -> Array:
        """Index as the pair ``(x, y)``: ``grid[0] is x``, ``grid[1] is y``."""
        return (self.x, self.y)[index]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the coordinate arrays."""
        return self.x.shape

    @property
    def spacing(self) -> Tuple[float, float]:
        """Sample spacings ``(dx, dy)``.

        Raises
        ------
        ValueError
            If the grid is not sampled in increasing x and y order.
        """
        dx = float(self.x[0, 1] - self.x[0, 0])
        dy = float(self.y[1, 0] - self.y[0, 0])
        if dx <= 0 or dy <= 0:
            raise ValueError("Grid must be sampled in increasing x and y order.")
        return dx, dy


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
    Grid
        Coordinates spanning ``[-length/2, length/2)`` with spacing
        ``length / n``. The half-open interval keeps the grid compatible
        with the FFT sampling convention.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if length <= 0:
        raise ValueError("length must be positive.")

    coords = np.linspace(-length / 2, length / 2, n, endpoint=False)
    x, y = np.meshgrid(coords, coords)
    return Grid(x, y)


def grid_spacing(grid: Grid) -> Tuple[float, float]:
    """Return the sample spacings ``(dx, dy)`` of a grid.

    Thin wrapper over :attr:`Grid.spacing`, kept for call-site compatibility.
    """
    return grid.spacing
