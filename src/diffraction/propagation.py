import numpy as np

from .fourier import FFT2, IFFT2, frequency_grid

exp = np.exp
π = np.pi


def fresnel_output_grid(grid, z, λ, n=1):
    """Return output-plane spatial grids for Fresnel propagation."""
    λm = λ / n
    fx, fy = frequency_grid(grid)
    return λm * z * fx, λm * z * fy


def fresnel_propagator(U, grid, z, λ, n=1):
    """
    Fresnel propagation of a scalar field U over distance z.

    Parameters
    ----------
    U : 2D complex array
        Input field on the spatial grid.
    grid : tuple (x, y)
        Spatial coordinate grids from np.meshgrid.
    z : float
        Propagation distance.
    λ : float
        Wavelength in vacuum.
    n : float
        Refractive index of medium.
    """
    x, y = grid

    dx = float(x[0, 1] - x[0, 0])
    dy = float(y[1, 0] - y[0, 0])
    λm = λ / n
    k = 2 * π / λm
    x_, y_ = fresnel_output_grid(grid, z, λ, n=n)

    T0 = exp(1.0j * k * z) / (1.0j * λm * z)
    T1 = exp(1.0j * k * (x_**2 + y_**2) / (2.0 * z))
    T2 = exp(1.0j * k * (x**2 + y**2) / (2.0 * z))

    # dx*dy scales FFT2 into a Riemann-sum approximation of the continuous FT integral.
    return T0 * T1 * (FFT2(T2 * U) * dx * dy)


def asm_propagator(U, grid, z, λ, n=1, include_evanescent=False):
    """
    Angular spectrum propagation of a scalar field U over distance z.

    Parameters
    ----------
    U : 2D complex array
        Input field on the spatial grid.
    grid : tuple (x, y)
        Spatial coordinate grids from np.meshgrid.
    z : float
        Propagation distance.
    λ : float
        Wavelength in vacuum.
    n : float
        Refractive index of medium.
    include_evanescent : bool
        If True, includes evanescent components with exponential decay.
    """
    λm = λ / n
    fx, fy = frequency_grid(grid)
    f2 = fx**2 + fy**2
    cutoff = (1.0 / λm) ** 2

    if include_evanescent:
        kz = 2.0 * π * np.sqrt(cutoff - f2 + 0.0j)
    else:
        kz = 2.0 * π * np.sqrt(np.maximum(cutoff - f2, 0.0))

    H = exp(1.0j * kz * z)
    return IFFT2(FFT2(U) * H)
