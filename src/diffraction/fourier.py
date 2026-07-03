"""The Fourier operators.

Two levels live here:

- **Array operators** ``FFT2`` / ``IFFT2`` — centered 2D (inverse) FFTs that
  keep the zero-frequency component at the center of the array, so fields on
  symmetric grids (from :func:`diffraction.grids.make_grid`) transform without
  extra bookkeeping. These are the fast O(N log N) core on the FFT-native grid.
- **Field operators** ``FT2`` / ``IFT2`` — the Fourier transform understood as
  an *operator applied to a* :class:`~diffraction.field.Field`. On the native
  conjugate grid they defer to ``FFT2`` / ``IFFT2``; given an explicit target
  grid they evaluate the transform there via a direct matrix DFT (exact, no
  extra dependencies) or, for uniform grids, a chirp-Z transform (O(N log N),
  needs SciPy). This is what decouples the input and output sampling grids.

Fourier transforms are operators, not methods: write ``FT2(phi)``, never
``phi.FT2()``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .backend import array_module
from .field import Field
from .grids import Grid, grid_spacing

__all__ = ["FFT2", "IFFT2", "FT2", "IFT2", "frequency_grid"]

try:  # SciPy powers the (optional) chirp-Z fast path; see backend.py's pattern.
    from scipy.signal import czt as _scipy_czt

    _SCIPY_CZT = True
except Exception:  # pragma: no cover - exercised only without SciPy
    _scipy_czt = None
    _SCIPY_CZT = False

# Above this many samples, ``engine="auto"`` prefers the O(N log N) chirp-Z
# path over the O(N^2) matrix DFT for a uniform target grid (when SciPy is
# present and the field is on the CPU).
_CZT_AUTO_MIN = 512


def FFT2(g: np.ndarray) -> np.ndarray:
    """Centered 2D FFT: shift, transform, shift back (NumPy or CuPy input)."""
    xp = array_module(g)
    g_ = xp.fft.ifftshift(g)
    G_ = xp.fft.fft2(g_)
    return xp.fft.fftshift(G_)


def IFFT2(G: np.ndarray) -> np.ndarray:
    """Centered inverse 2D FFT: shift, inverse transform, shift back (NumPy or CuPy input)."""
    xp = array_module(G)
    G_ = xp.fft.ifftshift(G)
    g_ = xp.fft.ifft2(G_)
    return xp.fft.fftshift(g_)


def frequency_grid(grid: Grid) -> Grid:
    """Return centered spatial-frequency grids ``(fx, fy)`` for a spatial grid.

    Parameters
    ----------
    grid : (x, y)
        Spatial coordinate grids from :func:`numpy.meshgrid` (2D arrays of
        equal shape, uniformly sampled).

    Returns
    -------
    Grid
        Spatial-frequency coordinates in cycles per unit length (unpacks as
        ``fx, fy = frequency_grid(grid)``), centered so that ``fx = fy = 0``
        sits at the middle of the array.
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
    fxx, fyy = np.meshgrid(fx, fy)
    return Grid(fxx, fyy)


# --------------------------------------------------------------------------- #
# Field-level Fourier operators: FT2 / IFT2 and their engines.
#
# All grids in the package are separable meshgrids (x varies along axis 1,
# y along axis 0), so a 2D transform factors into a 1D transform per axis.
# The forward transform samples  G(f) = sum_r phi(r) exp(-2 pi i f . r)  at the
# requested output coordinates; the inverse divides by the source sample count.
# Both matrix and chirp-Z engines were validated to reproduce FFT2/IFFT2 on the
# native grid to machine precision.
# --------------------------------------------------------------------------- #


def _axes_1d(grid: Grid):
    """The separable 1D coordinate axes ``(x_row, y_col)`` of a meshgrid."""
    return grid.x[0, :], grid.y[:, 0]


def _is_uniform_linear(grid: Grid) -> bool:
    """True if ``grid`` is a separable meshgrid uniformly spaced on each axis."""
    x1, y1 = _axes_1d(grid)
    if x1.size < 2 or y1.size < 2:
        return False
    xp = array_module(grid.x)
    # rows/cols constant (a genuine meshgrid) and uniform steps along each axis
    if not (
        bool(xp.allclose(grid.x, grid.x[0, :][None, :]))
        and bool(xp.allclose(grid.y, grid.y[:, 0][:, None]))
    ):
        return False
    dx = xp.diff(x1)
    dy = xp.diff(y1)
    return bool(xp.allclose(dx, dx[0]) and xp.allclose(dy, dy[0]))


def _dft2_matrix(values, src_grid: Grid, dst_grid: Grid, sign: int):
    """Separable direct DFT: out[a] = sum_b values[b] exp(sign 2πi dst_a·src_b)."""
    xp = array_module(values)
    sx, sy = _axes_1d(src_grid)
    dx, dy = _axes_1d(dst_grid)
    two_pi_i = sign * 2.0j * np.pi
    Ex = xp.exp(two_pi_i * xp.outer(dx, sx))  # (Mx, Nx)
    Ey = xp.exp(two_pi_i * xp.outer(dy, sy))  # (My, Ny)
    return Ey @ values @ Ex.T


def _czt_axis(values, s0, ds, t0, dt, m, sign, axis):
    """1D chirp-Z of ``values`` along ``axis`` between two uniform grids."""
    n = values.shape[axis]
    b = np.arange(n)
    shape_in = [1] * values.ndim
    shape_in[axis] = n
    pre = np.exp(sign * 2.0j * np.pi * t0 * b * ds).reshape(shape_in)
    W = np.exp(sign * 2.0j * np.pi * dt * ds)
    core = _scipy_czt(values * pre, m=m, w=W, a=1.0, axis=axis)
    a = np.arange(m)
    shape_out = [1] * values.ndim
    shape_out[axis] = m
    post = (
        np.exp(sign * 2.0j * np.pi * t0 * s0)
        * np.exp(sign * 2.0j * np.pi * a * dt * s0)
    ).reshape(shape_out)
    return core * post


def _dft2_czt(values, src_grid: Grid, dst_grid: Grid, sign: int):
    """Separable chirp-Z DFT (uniform grids, CPU/NumPy, SciPy required)."""
    sx, sy = _axes_1d(src_grid)
    dx, dy = _axes_1d(dst_grid)
    out = _czt_axis(values, sx[0], sx[1] - sx[0], dx[0], dx[1] - dx[0], dx.size, sign, axis=1)
    out = _czt_axis(out, sy[0], sy[1] - sy[0], dy[0], dy[1] - dy[0], dy.size, sign, axis=0)
    return out


def _resolve_engine(engine: str, values, native: bool, dst_grid: Grid) -> str:
    if engine not in ("auto", "fft", "matrix", "czt"):
        raise ValueError(f"Unknown engine {engine!r}; use auto/fft/matrix/czt.")
    on_cpu = array_module(values) is np
    if engine == "auto":
        if native:
            return "fft"
        if _SCIPY_CZT and on_cpu and values.size >= _CZT_AUTO_MIN and _is_uniform_linear(dst_grid):
            return "czt"
        return "matrix"
    if engine == "fft" and not native:
        raise ValueError("engine='fft' only applies to the native (default) grid.")
    if engine == "czt":
        if not _SCIPY_CZT:
            raise ImportError("engine='czt' needs SciPy (pip install scipy).")
        if not on_cpu:
            raise ValueError("engine='czt' runs on CPU/NumPy arrays only.")
        if not _is_uniform_linear(dst_grid):
            raise ValueError("engine='czt' needs a uniform, linearly-spaced target grid.")
    return engine


def FT2(field: Field, *, kgrid: Optional[Grid] = None, engine: str = "auto") -> Field:
    """Forward Fourier operator: sample the transform of ``field`` on ``kgrid``.

    Parameters
    ----------
    field : Field
        Input field (spatial domain).
    kgrid : Grid, optional
        Target spatial-frequency grid. ``None`` (default) uses the FFT-native
        conjugate grid :func:`frequency_grid` and the fast ``FFT2`` path. An
        explicit grid decouples the output frequency sampling from the input.
    engine : {"auto", "fft", "matrix", "czt"}
        Transform engine. ``"auto"`` picks ``fft`` on the native grid, the
        chirp-Z fast path on a uniform target grid (SciPy, CPU), else the exact
        matrix DFT.

    Returns
    -------
    Field
        Transform sampled on ``kgrid`` (or the native grid), ``domain="frequency"``.
    """
    native = kgrid is None
    target = frequency_grid(field.grid) if native else kgrid
    eng = _resolve_engine(engine, field.values, native, target)
    if eng == "fft":
        values = FFT2(field.values)
    elif eng == "czt":
        values = _dft2_czt(field.values, field.grid, target, sign=-1)
    else:
        values = _dft2_matrix(field.values, field.grid, target, sign=-1)
    return Field(target, values, domain="frequency")


def IFT2(field: Field, *, grid: Optional[Grid] = None, engine: str = "auto") -> Field:
    """Inverse Fourier operator: sample the inverse transform on ``grid``.

    Parameters
    ----------
    field : Field
        Input field (frequency domain).
    grid : Grid, optional
        Target spatial grid. ``None`` (default) uses the FFT-native conjugate
        grid (``frequency_grid`` is its own inverse up to the centered
        convention) and the fast ``IFFT2`` path.
    engine : {"auto", "fft", "matrix", "czt"}
        As in :func:`FT2`.

    Returns
    -------
    Field
        Inverse transform sampled on ``grid`` (or the native grid),
        ``domain="space"``.
    """
    native = grid is None
    target = frequency_grid(field.grid) if native else grid
    eng = _resolve_engine(engine, field.values, native, target)
    if eng == "fft":
        values = IFFT2(field.values)
    else:
        n_src = field.grid.shape[0] * field.grid.shape[1]
        if eng == "czt":
            values = _dft2_czt(field.values, field.grid, target, sign=+1) / n_src
        else:
            values = _dft2_matrix(field.values, field.grid, target, sign=+1) / n_src
    return Field(target, values, domain="space")
