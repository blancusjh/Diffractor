"""Centered 2D Fourier-transform utilities.

These wrappers keep the zero-frequency component at the center of the
array, so fields sampled on symmetric grids (from :func:`diffraction.grids.make_grid`)
transform without extra bookkeeping.
"""

from __future__ import annotations

import numpy as np

from .grids import Grid, grid_spacing

__all__ = ["FFT2", "IFFT2", "frequency_grid"]


def FFT2(g: np.ndarray) -> np.ndarray:
    """Centered 2D FFT: shift, transform, shift back."""
    g_ = np.fft.ifftshift(g)
    G_ = np.fft.fft2(g_)
    return np.fft.fftshift(G_)


def IFFT2(G: np.ndarray) -> np.ndarray:
    """Centered inverse 2D FFT: shift, inverse transform, shift back."""
    G_ = np.fft.ifftshift(G)
    g_ = np.fft.ifft2(G_)
    return np.fft.fftshift(g_)


def frequency_grid(grid: Grid) -> Grid:
    """Return centered spatial-frequency grids ``(fx, fy)`` for a spatial grid.

    Parameters
    ----------
    grid : (x, y)
        Spatial coordinate grids from :func:`numpy.meshgrid` (2D arrays of
        equal shape, uniformly sampled).

    Returns
    -------
    (fx, fy) : tuple of 2D arrays
        Spatial-frequency coordinates in cycles per unit length, centered
        so that ``fx = fy = 0`` sits at the middle of the array.
    """
    x, y = grid
    if x.shape != y.shape:
        raise ValueError("x and y grids must have the same shape.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")

    dx, dy = grid_spacing(grid)
    ny, nx = x.shape

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    return tuple(np.meshgrid(fx, fy))
