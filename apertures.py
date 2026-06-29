from __future__ import annotations

from typing import Callable, Tuple, Any
import numpy as np

Array = np.ndarray
Grid = Tuple[Array, Array]


class Aperture:
    """
    Represents an aperture mask defined on a 2D grid (x, y).

    Parameters
    ----------
    f : callable
        A function f(x, y, *params) -> 2D array phase mask.
    grid : (x, y)
        Tuple of numpy arrays with the same shape.
    *params :
        Extra parameters forwarded to f.
    """

    def __init__(self, f: Callable[..., Array], grid: Grid, *params: Any):
        self.f = f
        self.grid = grid
        self.params = params

        x, y = grid

        A = f(x, y, *params)
        self.A = A.astype(np.complex128)



# Aperture Functions 

def circular_aperture(x: Array, y: Array, R: float) -> Array:
    """Boolean circular pupil: 1 inside radius R, 0 outside."""
    return (x * x + y * y) <= (R * R)


def rectangular_aperture(x: Array, y: Array, Wx: float, Wy: float, *, center=(0.0, 0.0)) -> Array:
    """
    Axis-aligned rectangle of width Wx and height Wy (both full-width), centered at `center`.
    Returns a boolean mask.
    """
    x0, y0 = center
    hx, hy = 0.5 * Wx, 0.5 * Wy
    return (np.abs(x - x0) <= hx) & (np.abs(y - y0) <= hy)


def square_aperture(x: Array, y: Array, W: float, *, center=(0.0, 0.0)) -> Array:
    """Convenience: square of side W."""
    return rectangular_aperture(x, y, W, W, center=center)


def annular_aperture(x: Array, y: Array, R_inner: float, R_outer: float, *, center=(0.0, 0.0)) -> Array:
    """Annulus: R_inner <= r <= R_outer."""
    x0, y0 = center
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return (R_inner * R_inner <= r2) & (r2 <= R_outer * R_outer)


def elliptical_aperture(x: Array, y: Array, a: float, b: float, *, center=(0.0, 0.0)) -> Array:
    """Ellipse: (x/a)^2 + (y/b)^2 <= 1."""
    x0, y0 = center
    X = (x - x0) / a
    Y = (y - y0) / b
    return (X * X + Y * Y) <= 1.0


def slit_aperture(x: Array, y: Array, W: float, *, orientation: str = "x", center=(0.0, 0.0)) -> Array:
    """
    Infinite slit of width W.
    orientation='x' means the slit is long along x and limited in y: |y-y0|<=W/2.
    orientation='y' means long along y and limited in x: |x-x0|<=W/2.
    """
    x0, y0 = center
    h = 0.5 * W
    if orientation == "x":
        return np.abs(y - y0) <= h
    if orientation == "y":
        return np.abs(x - x0) <= h
    raise ValueError("orientation must be 'x' or 'y'.")
