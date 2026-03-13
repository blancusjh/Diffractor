"""
Fourier Transform Utilities
"""

import numpy as np


def FFT2(g: np.ndarray) -> np.ndarray:
    """Centered 2D FFT."""
    g_ = np.fft.ifftshift(g)
    G_ = np.fft.fft2(g_)
    return np.fft.fftshift(G_)


def IFFT2(G: np.ndarray) -> np.ndarray:
    """Centered inverse 2D FFT."""
    G_ = np.fft.ifftshift(G)
    g_ = np.fft.ifft2(G_)
    return np.fft.fftshift(g_)


def frequency_grid(grid):
    """Return centered spatial-frequency grids (fx, fy) from (x, y) sampling grids."""
    x, y = grid
    if x.shape != y.shape:
        raise ValueError("x and y grids must have the same shape.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")

    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    ny, nx = x.shape

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    return np.meshgrid(fx, fy)
