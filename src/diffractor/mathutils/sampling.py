"""Choosing an adequate sampling grid for near-field propagation.

The angular-spectrum method aliases when the input grid does not resolve the
field it is given: a hard-edged aperture in the deep near field (high Fresnel
number ``R²/(λz)``) develops boundary-wave ripples finer than a coarse ``dx``,
and the undersampled content folds back as a non-physical, axis-aligned
cross-hatch. This is an *input* sampling limit — ``k_max = π/dx`` is fixed by
``dx`` alone — so no choice of output grid or transform engine removes it; only
a finer ``dx`` does.

This module helps pick that grid:

- :func:`recommend_grid_convergence` runs the actual propagation at increasing
  ``N`` and stops when the result stops changing (a grid-convergence study,
  exactly what one does by hand). It is empirical and honest: it verifies
  adequacy rather than guessing it.
- :func:`fresnel_min_distance` / :func:`fresnel_max_spacing` expose the
  well-established single-FFT Fresnel sampling criterion ``z ≳ N dx²/λ``.
- :func:`next_fft_size` rounds a sample count up to an FFT-friendly size.

There is deliberately no closed-form "minimum N" for the ASM near-field case:
a naive Nyquist estimate misses the true requirement by more than tenfold, so
convergence testing is the trustworthy route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from ..physics.field import Field
from .grids import Grid, make_grid
from ..physics.propagation import asm_propagator

__all__ = [
    "GridRecommendation",
    "fresnel_max_spacing",
    "fresnel_min_distance",
    "next_fft_size",
    "recommend_grid_convergence",
]

try:  # optional 5-smooth FFT sizing, backend.py-style optional import
    from scipy.fft import next_fast_len as _next_fast_len

    _SCIPY_FFT = True
except Exception:  # pragma: no cover
    _next_fast_len = None
    _SCIPY_FFT = False


@dataclass(frozen=True)
class GridRecommendation:
    """Result of a grid-sizing search.

    Attributes
    ----------
    n : int
        Recommended samples per side.
    length : float
        Physical window [m] (unchanged from the request).
    dx : float
        Resulting sample spacing ``length / n`` [m].
    metric_name : str
        Which convergence metric was used.
    tol : float
        Tolerance the metric had to reach.
    converged : bool
        Whether the metric reached ``tol`` before ``n_max``.
    history : tuple of (int, float)
        ``(N, metric)`` for each grid tried, in order.
    """

    n: int
    length: float
    dx: float
    metric_name: str
    tol: float
    converged: bool
    history: Tuple[Tuple[int, float], ...]


def next_fft_size(n: int, *, prefer_fast_len: bool = True) -> int:
    """Round ``n`` up to an FFT-friendly size.

    Uses :func:`scipy.fft.next_fast_len` (a 5-smooth size) when SciPy is
    available, otherwise the next power of two.
    """
    n = max(int(n), 2)
    if prefer_fast_len and _SCIPY_FFT:
        return int(_next_fast_len(n))
    return int(2 ** np.ceil(np.log2(n)))


def fresnel_min_distance(n: int, dx: float, wavelength: float, n_medium: float = 1.0) -> float:
    """Minimum ``z`` for a well-sampled single-FFT Fresnel propagation.

    From ``z ≳ N dx² / λ_medium`` (the criterion in the README). Returns that
    bound in meters.
    """
    if n < 1 or dx <= 0 or wavelength <= 0 or n_medium <= 0:
        raise ValueError("n, dx, wavelength, n_medium must be positive.")
    return n * dx**2 / (wavelength / n_medium)


def fresnel_max_spacing(n: int, z: float, wavelength: float, n_medium: float = 1.0) -> float:
    """Largest ``dx`` keeping a single-FFT Fresnel propagation well sampled at ``z``.

    The inverse of :func:`fresnel_min_distance`: ``dx ≤ sqrt(λ_medium z / N)``.
    """
    if n < 1 or z <= 0 or wavelength <= 0 or n_medium <= 0:
        raise ValueError("n, z, wavelength, n_medium must be positive.")
    return float(np.sqrt((wavelength / n_medium) * z / n))


def _azimuthal_cv(
    field: Field,
    radial_window: Tuple[float, float],
    n_r: int = 24,
    n_theta: int = 64,
) -> float:
    """Max over radii of the intensity coefficient-of-variation *around* a ring.

    Intensity is sampled at fixed radius over many angles (nearest pixel on the
    uniform grid), so the metric isolates *azimuthal* variation — the radial
    ring structure of a real diffraction pattern is not counted. A rotationally
    symmetric field gives ~0; aliasing that breaks that symmetry raises it.
    """
    x, y = field.grid
    dx, dy = field.grid.spacing
    x0 = float(x[0, 0])
    y0 = float(y[0, 0])
    ny, nx = field.shape
    I = np.abs(field.values) ** 2

    rmin, rmax = radial_window
    radii = np.linspace(rmin, rmax, n_r)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(thetas), np.sin(thetas)

    worst = 0.0
    for r in radii:
        # Bilinear sample at the exact radius, so ring-crossing (radial) fringes
        # are not mistaken for azimuthal variation.
        fx = (r * ct - x0) / dx
        fy = (r * st - y0) / dy
        i0 = np.floor(fx).astype(int)
        j0 = np.floor(fy).astype(int)
        ok = (i0 >= 0) & (i0 < nx - 1) & (j0 >= 0) & (j0 < ny - 1)
        if ok.sum() < 8:
            continue
        i0, j0 = i0[ok], j0[ok]
        tx = fx[ok] - i0
        ty = fy[ok] - j0
        vals = (
            I[j0, i0] * (1 - tx) * (1 - ty)
            + I[j0, i0 + 1] * tx * (1 - ty)
            + I[j0 + 1, i0] * (1 - tx) * ty
            + I[j0 + 1, i0 + 1] * tx * ty
        )
        mean = vals.mean()
        if mean > 0:
            worst = max(worst, float(vals.std() / mean))
    return worst


def _successive_diff(fine: Field, coarse: Field) -> float:
    """Relative L2 difference of intensities between a grid and its 2x-coarser one."""
    I_fine = np.abs(fine.values) ** 2
    I_coarse = np.abs(coarse.values) ** 2
    I_fine_down = I_fine[::2, ::2]
    denom = np.sqrt(np.sum(I_coarse**2))
    if denom == 0:
        return 0.0
    return float(np.sqrt(np.sum((I_fine_down - I_coarse) ** 2)) / denom)


def recommend_grid_convergence(
    aperture: Callable[[Grid], Field],
    length: float,
    wavelength: float,
    z: Union[float, Sequence[float]],
    *,
    n0: int = 256,
    n_max: int = 8192,
    n_medium: float = 1.0,
    metric: str = "successive_diff",
    radial_window: Optional[Tuple[float, float]] = None,
    tol: float = 0.05,
    pad_factor: int = 1,
    bandlimit: bool = True,
    fast_len: bool = True,
) -> GridRecommendation:
    """Search for an ``N`` that resolves a near-field ASM propagation.

    Doubles ``N`` from ``n0`` (rounded FFT-friendly), rebuilds the aperture and
    propagates with :func:`asm_propagator` at each ``z``, and stops when a
    convergence metric reaches ``tol``.

    Parameters
    ----------
    aperture : callable
        ``aperture(grid) -> Field`` building the input field on a given grid
        (e.g. ``lambda g: antialiased(circular_aperture, g, R)``).
    length : float
        Physical window [m], held fixed across the search.
    wavelength : float
        Vacuum wavelength [m].
    z : float or sequence of float
        Propagation distance(s) [m]; the metric is maxed over them. Pass the
        smallest (worst-case, highest Fresnel number) distance to be safe.
    n0, n_max : int
        Starting and maximum samples per side.
    metric : {"successive_diff", "azimuthal_cv"}
        ``successive_diff`` (default, any aperture): relative change between a
        grid and its 2×-coarser predecessor. ``azimuthal_cv`` (circular
        apertures only): departure from rotational symmetry — a direct,
        ground-truth-like check. Requires ``radial_window``.
    radial_window : (rmin, rmax), optional
        Annulus [m] over which ``azimuthal_cv`` is measured.
    tol : float
        Convergence tolerance for the chosen metric.
    pad_factor, bandlimit : int, bool
        Passed through to :func:`asm_propagator`.
    fast_len : bool
        Round trial sizes with :func:`next_fft_size`.

    Returns
    -------
    GridRecommendation
        The recommended ``N`` (the coarser member of the converged pair for
        ``successive_diff``; the first converged grid for ``azimuthal_cv``),
        or the largest grid tried with ``converged=False``.
    """
    if length <= 0 or wavelength <= 0 or n_medium <= 0:
        raise ValueError("length, wavelength, n_medium must be positive.")
    if n0 < 2 or n_max < n0:
        raise ValueError("Require 2 <= n0 <= n_max.")
    if metric not in ("successive_diff", "azimuthal_cv"):
        raise ValueError("metric must be 'successive_diff' or 'azimuthal_cv'.")
    if metric == "azimuthal_cv":
        if radial_window is None:
            raise ValueError("azimuthal_cv requires radial_window=(rmin, rmax).")
        rmin, rmax = radial_window
        if not (0.0 <= rmin < rmax <= length / 2):
            raise ValueError("radial_window must satisfy 0 <= rmin < rmax <= length/2.")

    zs = [float(z)] if np.isscalar(z) else [float(v) for v in z]

    def propagate_all(n: int):
        grid = make_grid(n, length)
        field = aperture(grid)
        return [
            asm_propagator(
                field, z=zz, wavelength=wavelength, n=n_medium,
                pad_factor=pad_factor, bandlimit=bandlimit,
            )
            for zz in zs
        ]

    history: list[Tuple[int, float]] = []
    n = next_fft_size(n0) if fast_len else n0
    prev_n: Optional[int] = None
    prev_fields = None

    while n <= n_max:
        fields = propagate_all(n)

        if metric == "azimuthal_cv":
            value = max(_azimuthal_cv(f, radial_window) for f in fields)
            history.append((n, value))
            if value <= tol:
                return GridRecommendation(
                    n, length, length / n, metric, tol, True, tuple(history)
                )
        else:  # successive_diff
            if prev_fields is not None:
                value = max(_successive_diff(f, pf) for f, pf in zip(fields, prev_fields))
                history.append((n, value))
                if value <= tol:
                    return GridRecommendation(
                        prev_n, length, length / prev_n, metric, tol, True, tuple(history)
                    )
            prev_n, prev_fields = n, fields

        next_n = 2 * n
        n = next_fft_size(next_n) if fast_len else next_n

    last_n = history[-1][0] if history else n0
    return GridRecommendation(
        last_n, length, length / last_n, metric, tol, False, tuple(history)
    )
