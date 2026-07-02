"""Scalar diffraction propagators.

Implemented methods
-------------------
- :func:`fresnel_propagator` — single-FFT Fresnel (near-field, paraxial)
  diffraction integral. The output plane has its own coordinates, given by
  :func:`fresnel_output_grid`.
- :func:`fraunhofer_propagator` — far-field limit of the Fresnel integral
  (drops the quadratic input phase). Same output grid as the Fresnel method.
- :func:`fresnel_zoom_propagator` — same Fresnel integral evaluated by a
  direct (matrix) Fourier transform on a user-chosen output window, for
  zooming into focal spots and other compact features.
- :func:`asm_propagator` — angular-spectrum method (exact scalar transfer
  function). Input and output share the same grid; supports negative ``z``
  for back-propagation.

Conventions
-----------
Fields are 2D complex arrays sampled on a ``(x, y)`` meshgrid pair, all
distances in meters, and ``wavelength`` is the vacuum wavelength. The medium
is characterized by its refractive index ``n``.
"""

from __future__ import annotations

import numpy as np

from .fourier import FFT2, IFFT2, frequency_grid
from .grids import Array, Grid, grid_spacing

__all__ = [
    "asm_propagator",
    "fraunhofer_propagator",
    "fresnel_output_grid",
    "fresnel_propagator",
    "fresnel_zoom_propagator",
]


def _validate_medium(wavelength: float, n: float) -> None:
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    if n <= 0:
        raise ValueError("n must be positive.")


def fresnel_output_grid(grid: Grid, z: float, wavelength: float, n: float = 1.0) -> Grid:
    """Output-plane coordinates for the single-FFT Fresnel/Fraunhofer methods.

    The single-FFT method evaluates the diffraction integral on a scaled
    grid ``x' = λ z fx``, ``y' = λ z fy``, so the physical window of the
    output plane grows linearly with the propagation distance.
    """
    _validate_medium(wavelength, n)
    if z == 0:
        raise ValueError("z must be non-zero for Fresnel propagation.")

    wavelength_medium = wavelength / n
    fx, fy = frequency_grid(grid)
    return wavelength_medium * z * fx, wavelength_medium * z * fy


def fresnel_propagator(U: Array, grid: Grid, z: float, wavelength: float, n: float = 1.0) -> Array:
    """Propagate a scalar field with the single-FFT Fresnel method.

    Parameters
    ----------
    U : 2D complex array
        Input field sampled on ``grid``.
    grid : (x, y)
        Spatial coordinate grids from :func:`numpy.meshgrid`.
    z : float
        Propagation distance [m], non-zero.
    wavelength : float
        Vacuum wavelength [m].
    n : float
        Refractive index of the propagation medium.

    Returns
    -------
    2D complex array
        Field on the output plane whose coordinates are given by
        :func:`fresnel_output_grid`.
    """
    x, y = grid
    dx, dy = grid_spacing(grid)
    wavelength_medium = wavelength / n
    k = 2 * np.pi / wavelength_medium
    x_, y_ = fresnel_output_grid(grid, z, wavelength, n=n)

    T0 = np.exp(1.0j * k * z) / (1.0j * wavelength_medium * z)
    T1 = np.exp(1.0j * k * (x_**2 + y_**2) / (2.0 * z))
    T2 = np.exp(1.0j * k * (x**2 + y**2) / (2.0 * z))

    # dx*dy scales FFT2 into a Riemann-sum approximation of the continuous FT integral.
    return T0 * T1 * (FFT2(T2 * U) * dx * dy)


def fraunhofer_propagator(U: Array, grid: Grid, z: float, wavelength: float, n: float = 1.0) -> Array:
    """Propagate a scalar field to the far field (Fraunhofer regime).

    Identical to :func:`fresnel_propagator` but without the quadratic input
    phase, valid when ``z >> k * max(x² + y²) / 2``. The output plane
    coordinates are given by :func:`fresnel_output_grid`.
    """
    dx, dy = grid_spacing(grid)
    wavelength_medium = wavelength / n
    k = 2 * np.pi / wavelength_medium
    x_, y_ = fresnel_output_grid(grid, z, wavelength, n=n)

    T0 = np.exp(1.0j * k * z) / (1.0j * wavelength_medium * z)
    T1 = np.exp(1.0j * k * (x_**2 + y_**2) / (2.0 * z))

    return T0 * T1 * (FFT2(U) * dx * dy)


def fresnel_zoom_propagator(
    U: Array,
    grid: Grid,
    z: float,
    wavelength: float,
    n: float = 1.0,
    *,
    output_half_width: float,
    output_samples: int = 512,
) -> tuple[Array, Grid]:
    """Fresnel propagation onto a freely chosen output window.

    The single-FFT method locks the output sampling to ``λ z / L``, which
    often under-resolves compact features such as focal spots. This variant
    evaluates the same Fresnel integral with a direct (matrix) Fourier
    transform on user-chosen output coordinates, decoupling the output
    window and sampling from the input grid.

    Parameters
    ----------
    U : 2D complex array
        Input field sampled on ``grid``.
    grid : (x, y)
        Spatial coordinate grids from :func:`numpy.meshgrid`.
    z : float
        Propagation distance [m], non-zero.
    wavelength : float
        Vacuum wavelength [m].
    n : float
        Refractive index of the propagation medium.
    output_half_width : float
        Half-width of the (square, centered) output window [m].
    output_samples : int
        Samples per side of the output window.

    Returns
    -------
    (U_out, (x_out, y_out))
        Field and coordinate grids on the output window.

    Notes
    -----
    Cost scales as ``output_samples × N²`` (two matrix products) instead
    of ``N² log N``, so keep the output window modest (≲ 1024 samples).
    """
    _validate_medium(wavelength, n)
    if z == 0:
        raise ValueError("z must be non-zero for Fresnel propagation.")
    if output_half_width <= 0:
        raise ValueError("output_half_width must be positive.")
    if output_samples < 2:
        raise ValueError("output_samples must be at least 2.")

    x, y = grid
    dx, dy = grid_spacing(grid)
    wavelength_medium = wavelength / n
    k = 2 * np.pi / wavelength_medium

    xo = np.linspace(-output_half_width, output_half_width, output_samples)
    x_, y_ = np.meshgrid(xo, xo)
    fx = xo / (wavelength_medium * z)  # output positions as spatial frequencies

    # Direct Fourier transform of T2*U evaluated at (fx, fy), separably:
    # rows first (over y), then columns (over x).
    T2 = np.exp(1.0j * k * (x**2 + y**2) / (2.0 * z))
    Ey = np.exp(-2.0j * np.pi * np.outer(fx, y[:, 0]))
    Ex = np.exp(-2.0j * np.pi * np.outer(x[0, :], fx))
    G = (Ey @ (T2 * U) @ Ex) * (dx * dy)

    T0 = np.exp(1.0j * k * z) / (1.0j * wavelength_medium * z)
    T1 = np.exp(1.0j * k * (x_**2 + y_**2) / (2.0 * z))
    return T0 * T1 * G, (x_, y_)


def asm_propagator(
    U: Array,
    grid: Grid,
    z: float,
    wavelength: float,
    n: float = 1.0,
    include_evanescent: bool = False,
    bandlimit: bool = True,
    pad_factor: int = 1,
) -> Array:
    """Propagate a scalar field with the angular-spectrum method.

    Parameters
    ----------
    U : 2D complex array
        Input field sampled on ``grid``.
    grid : (x, y)
        Spatial coordinate grids from :func:`numpy.meshgrid`.
    z : float
        Propagation distance [m]. Negative values back-propagate.
    wavelength : float
        Vacuum wavelength [m].
    n : float
        Refractive index of the propagation medium.
    include_evanescent : bool
        If True, evanescent components are kept and decay exponentially.
        If False (default), they are filtered out.
    bandlimit : bool
        If True (default), apply the Matsushima–Shimobaba band limit
        [Opt. Express 17, 19662 (2009)]: frequencies where the sampled
        transfer-function phase aliases are zeroed. Without it, long
        propagation distances produce spurious high-frequency interference.
    pad_factor : int
        Zero-pad the field to ``pad_factor`` times its size before
        propagating and crop afterwards (default 1, no padding). Padding
        suppresses the wrap-around artifacts of the circular FFT
        convolution when the field diffracts up to the window edges.

    Returns
    -------
    2D complex array
        Field on the same grid at distance ``z``.
    """
    _validate_medium(wavelength, n)
    if pad_factor < 1:
        raise ValueError("pad_factor must be at least 1.")

    dx, dy = grid_spacing(grid)
    ny, nx = U.shape

    if pad_factor > 1:
        pad_x = ((pad_factor - 1) * nx) // 2
        pad_y = ((pad_factor - 1) * ny) // 2
        U = np.pad(U, ((pad_y, pad_y), (pad_x, pad_x)))

    NY, NX = U.shape
    fx1d = np.fft.fftshift(np.fft.fftfreq(NX, d=dx))
    fy1d = np.fft.fftshift(np.fft.fftfreq(NY, d=dy))
    fx, fy = np.meshgrid(fx1d, fy1d)

    wavelength_medium = wavelength / n
    f2 = fx**2 + fy**2
    cutoff = (1.0 / wavelength_medium) ** 2

    if include_evanescent:
        kz = 2.0 * np.pi * np.sqrt(cutoff - f2 + 0.0j)
        H = np.exp(1.0j * kz * z)
    else:
        propagating = f2 <= cutoff
        kz = 2.0 * np.pi * np.sqrt(np.maximum(cutoff - f2, 0.0))
        H = np.where(propagating, np.exp(1.0j * kz * z), 0.0)

    if bandlimit and z != 0:
        # Zero the frequencies where exp(i kz z), sampled at (dfx, dfy),
        # is no longer Nyquist-sampled.
        dfx = 1.0 / (NX * dx)
        dfy = 1.0 / (NY * dy)
        fx_limit = 1.0 / (wavelength_medium * np.sqrt((2.0 * dfx * abs(z)) ** 2 + 1.0))
        fy_limit = 1.0 / (wavelength_medium * np.sqrt((2.0 * dfy * abs(z)) ** 2 + 1.0))
        H = np.where((np.abs(fx) <= fx_limit) & (np.abs(fy) <= fy_limit), H, 0.0)

    Uz = IFFT2(FFT2(U) * H)

    if pad_factor > 1:
        Uz = Uz[pad_y : pad_y + ny, pad_x : pad_x + nx]
    return Uz
