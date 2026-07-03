"""Scalar diffraction propagators.

Every propagator maps a :class:`~diffraction.field.Field` to a ``Field`` — the
sampling grid travels with the samples, so no separate ``grid`` argument is
threaded through the pipeline.

Implemented methods
-------------------
- :func:`fresnel_propagator` — single-FFT Fresnel (near-field, paraxial)
  diffraction integral. The output plane has its own coordinates, given by
  :func:`fresnel_output_grid` and carried by the returned field.
- :func:`fraunhofer_propagator` — far-field limit of the Fresnel integral.
- :func:`fresnel_zoom_propagator` — the same Fresnel integral evaluated by a
  direct (matrix) Fourier transform on a user-chosen output window.
- :func:`asm_propagator` — angular-spectrum method, expressed through the
  Fourier operators as ``IFT2(H · FT2(field))``. With an explicit
  ``output_grid`` the input and output sampling grids decouple (MPASM /
  scaled-ASM), which lets a small, finely-sampled input window map to an
  independently chosen output grid.

Conventions
-----------
All distances are in meters and ``wavelength`` is the vacuum wavelength; the
medium is characterized by its refractive index ``n``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .field import Field
from .fourier import FFT2, FT2, IFT2, frequency_grid
from .grids import Array, Grid

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
    """Output-plane grid for the single-FFT Fresnel/Fraunhofer methods.

    The single-FFT method evaluates the diffraction integral on a scaled
    grid ``x' = λ z fx``, ``y' = λ z fy``, so the physical window of the
    output plane grows linearly with the propagation distance.
    """
    _validate_medium(wavelength, n)
    if z <= 0:
        raise ValueError(
            "z must be positive for the single-FFT Fresnel/Fraunhofer methods "
            "(their output grid scales with z); use asm_propagator for "
            "back-propagation (z < 0)."
        )

    wavelength_medium = wavelength / n
    fgrid = frequency_grid(grid)
    return Grid(wavelength_medium * z * fgrid.x, wavelength_medium * z * fgrid.y)


def fresnel_propagator(field: Field, z: float, wavelength: float, n: float = 1.0) -> Field:
    """Propagate a field with the single-FFT Fresnel method.

    Parameters
    ----------
    field : Field
        Input field.
    z : float
        Propagation distance [m], positive (use :func:`asm_propagator` for
        back-propagation).
    wavelength : float
        Vacuum wavelength [m].
    n : float
        Refractive index of the propagation medium.

    Returns
    -------
    Field
        Field on the output plane (:func:`fresnel_output_grid`).
    """
    grid = field.grid
    x, y = grid
    dx, dy = grid.spacing
    wavelength_medium = wavelength / n
    k = 2 * np.pi / wavelength_medium
    out_grid = fresnel_output_grid(grid, z, wavelength, n=n)
    x_, y_ = out_grid

    T0 = np.exp(1.0j * k * z) / (1.0j * wavelength_medium * z)
    T1 = np.exp(1.0j * k * (x_**2 + y_**2) / (2.0 * z))
    T2 = np.exp(1.0j * k * (x**2 + y**2) / (2.0 * z))

    # dx*dy scales FFT2 into a Riemann-sum approximation of the continuous FT.
    values = T0 * T1 * (FFT2(T2 * field.values) * dx * dy)
    return Field(out_grid, values)


def fraunhofer_propagator(field: Field, z: float, wavelength: float, n: float = 1.0) -> Field:
    """Propagate a field to the far field (Fraunhofer regime).

    Identical to :func:`fresnel_propagator` but without the quadratic input
    phase, valid when ``z >> k * max(x² + y²) / 2``.
    """
    grid = field.grid
    dx, dy = grid.spacing
    wavelength_medium = wavelength / n
    k = 2 * np.pi / wavelength_medium
    out_grid = fresnel_output_grid(grid, z, wavelength, n=n)
    x_, y_ = out_grid

    T0 = np.exp(1.0j * k * z) / (1.0j * wavelength_medium * z)
    T1 = np.exp(1.0j * k * (x_**2 + y_**2) / (2.0 * z))

    values = T0 * T1 * (FFT2(field.values) * dx * dy)
    return Field(out_grid, values)


def fresnel_zoom_propagator(
    field: Field,
    z: float,
    wavelength: float,
    n: float = 1.0,
    *,
    output_grid: Optional[Grid] = None,
    output_half_width: Optional[float] = None,
    output_samples: int = 512,
) -> Field:
    """Fresnel propagation onto a freely chosen output window.

    The single-FFT method locks the output sampling to ``λ z / L``, which
    often under-resolves compact features such as focal spots. This variant
    evaluates the same Fresnel integral with a direct (matrix) Fourier
    transform on user-chosen output coordinates.

    Parameters
    ----------
    field : Field
        Input field.
    z : float
        Propagation distance [m], non-zero.
    wavelength : float
        Vacuum wavelength [m].
    n : float
        Refractive index of the propagation medium.
    output_grid : Grid, optional
        Explicit (square/rectangular, centered or not) output grid. If given,
        ``output_half_width``/``output_samples`` are ignored.
    output_half_width : float, optional
        Half-width of a square, centered output window [m]. Required if
        ``output_grid`` is not given.
    output_samples : int
        Samples per side of that window.

    Returns
    -------
    Field
        Field on the output window.

    Notes
    -----
    Cost scales as ``output_samples × N²`` (two matrix products) instead of
    ``N² log N``, so keep the output window modest (≲ 1024 samples).
    """
    _validate_medium(wavelength, n)
    if z <= 0:
        raise ValueError(
            "z must be positive for the single-FFT Fresnel/Fraunhofer methods "
            "(their output grid scales with z); use asm_propagator for "
            "back-propagation (z < 0)."
        )

    if output_grid is None:
        if output_half_width is None:
            raise ValueError("Provide either output_grid or output_half_width.")
        if output_half_width <= 0:
            raise ValueError("output_half_width must be positive.")
        if output_samples < 2:
            raise ValueError("output_samples must be at least 2.")
        xo = np.linspace(-output_half_width, output_half_width, output_samples)
        gx, gy = np.meshgrid(xo, xo)
        output_grid = Grid(gx, gy)

    grid = field.grid
    x, y = grid
    dx, dy = grid.spacing
    wavelength_medium = wavelength / n
    k = 2 * np.pi / wavelength_medium

    x_, y_ = output_grid
    xo1 = output_grid.x[0, :]
    yo1 = output_grid.y[:, 0]
    fx = xo1 / (wavelength_medium * z)  # output x-positions as spatial frequencies
    fy = yo1 / (wavelength_medium * z)

    # Direct Fourier transform of T2*U evaluated at (fx, fy), separably.
    T2 = np.exp(1.0j * k * (x**2 + y**2) / (2.0 * z))
    Ey = np.exp(-2.0j * np.pi * np.outer(fy, y[:, 0]))
    Ex = np.exp(-2.0j * np.pi * np.outer(x[0, :], fx))
    G = (Ey @ (T2 * field.values) @ Ex) * (dx * dy)

    T0 = np.exp(1.0j * k * z) / (1.0j * wavelength_medium * z)
    T1 = np.exp(1.0j * k * (x_**2 + y_**2) / (2.0 * z))
    return Field(output_grid, T0 * T1 * G)


def _transfer_function(field_grid: Grid, z: float, wavelength_medium: float,
                       include_evanescent: bool, bandlimit: bool) -> Array:
    """Angular-spectrum transfer function ``exp(i kz z)`` on the field's k-grid."""
    kgrid = frequency_grid(field_grid)
    fx, fy = kgrid
    dx, dy = field_grid.spacing
    NY, NX = field_grid.shape

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
        # Matsushima-Shimobaba band limit: zero frequencies where the sampled
        # transfer-function phase aliases at this distance.
        dfx = 1.0 / (NX * dx)
        dfy = 1.0 / (NY * dy)
        fx_limit = 1.0 / (wavelength_medium * np.sqrt((2.0 * dfx * abs(z)) ** 2 + 1.0))
        fy_limit = 1.0 / (wavelength_medium * np.sqrt((2.0 * dfy * abs(z)) ** 2 + 1.0))
        H = np.where((np.abs(fx) <= fx_limit) & (np.abs(fy) <= fy_limit), H, 0.0)
    return H


def asm_propagator(
    field: Field,
    z: float,
    wavelength: float,
    n: float = 1.0,
    *,
    output_grid: Optional[Grid] = None,
    include_evanescent: bool = False,
    bandlimit: bool = True,
    pad_factor: int = 1,
    engine: str = "auto",
) -> Field:
    """Propagate a field with the angular-spectrum method.

    Expressed through the Fourier operators as ``IFT2(H · FT2(field))``.

    Parameters
    ----------
    field : Field
        Input field.
    z : float
        Propagation distance [m]. Negative values back-propagate.
    wavelength : float
        Vacuum wavelength [m].
    n : float
        Refractive index of the propagation medium.
    output_grid : Grid, optional
        If given, the inverse transform is sampled on this grid, decoupling
        the output sampling/window from the input (MPASM / scaled-ASM). This
        can drastically cut the sample count when the input support is small,
        but does **not** relieve the input-``dx`` Nyquist requirement of the
        field itself (see :mod:`diffraction.sampling`). ``None`` (default)
        returns the field on its own grid.
    include_evanescent : bool
        Keep evanescent components (exponential decay) instead of filtering.
    bandlimit : bool
        Apply the Matsushima–Shimobaba band limit (default True).
    pad_factor : int
        Zero-pad the field by this factor before propagating (default 1),
        suppressing the circular-convolution wrap-around of the FFT.
    engine : {"auto", "matrix", "czt"}
        Engine for the inverse transform when ``output_grid`` is given
        (ignored otherwise; the native path always uses the FFT).

    Returns
    -------
    Field
        Propagated field, on ``field.grid`` (or ``output_grid`` if given).
    """
    _validate_medium(wavelength, n)
    if pad_factor < 1:
        raise ValueError("pad_factor must be at least 1.")

    grid = field.grid
    U = field.values
    dx, dy = grid.spacing
    ny, nx = U.shape
    wavelength_medium = wavelength / n

    if pad_factor > 1:
        pad_x = ((pad_factor - 1) * nx) // 2
        pad_y = ((pad_factor - 1) * ny) // 2
        U = np.pad(U, ((pad_y, pad_y), (pad_x, pad_x)))
        NY, NX = U.shape
        coords_x = (np.arange(NX) - NX / 2) * dx
        coords_y = (np.arange(NY) - NY / 2) * dy
        gx, gy = np.meshgrid(coords_x, coords_y)
        work_grid = Grid(gx, gy)
    else:
        pad_x = pad_y = 0
        work_grid = grid

    H = _transfer_function(work_grid, z, wavelength_medium, include_evanescent, bandlimit)
    spectrum = FT2(Field(work_grid, U))
    spectrum = spectrum.like(spectrum.values * H)

    if output_grid is None:
        propagated = IFT2(spectrum)  # native inverse, back on work_grid
        values = propagated.values
        if pad_factor > 1:
            values = values[pad_y : pad_y + ny, pad_x : pad_x + nx]
        return Field(grid, values)

    return IFT2(spectrum, grid=output_grid, engine=engine)
