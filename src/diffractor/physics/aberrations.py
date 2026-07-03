"""Wavefront aberration analysis with Zernike polynomials.

Zernike circle polynomials in the Noll convention: single index ``j``
starting at 1 (piston), even ``j`` for cosine modes and odd ``j`` for sine
modes, each polynomial normalized to unit RMS over the unit disk. With
this normalization a fitted coefficient *is* the RMS wavefront
contribution of that mode, and the total RMS is the quadrature sum of the
coefficients (excluding piston).

Typical use: sample the optical path difference of a system on a pupil,
fit it with :func:`fit_zernikes`, and read off the aberration content —
defocus (j=4), astigmatism (5, 6), coma (7, 8), primary spherical (11) —
plus summary metrics from :func:`pv`, :func:`rms` and
:func:`marechal_strehl`.
"""

from __future__ import annotations

from math import factorial
from typing import Optional, Tuple, Union

import numpy as np

from .field import Field
from ..mathutils.grids import Array, Grid

__all__ = [
    "fit_zernikes",
    "marechal_strehl",
    "noll_to_nm",
    "pv",
    "rms",
    "synthesize_zernikes",
    "zernike",
    "zernike_name",
]

_NAMES = {
    1: "piston",
    2: "tip (x tilt)",
    3: "tilt (y tilt)",
    4: "defocus",
    5: "oblique astigmatism",
    6: "vertical astigmatism",
    7: "vertical coma",
    8: "horizontal coma",
    9: "vertical trefoil",
    10: "oblique trefoil",
    11: "primary spherical",
    12: "secondary astigmatism (v)",
    13: "secondary astigmatism (o)",
    14: "vertical quadrafoil",
    15: "oblique quadrafoil",
    16: "secondary coma (h)",
    17: "secondary coma (v)",
    22: "secondary spherical",
    37: "tertiary spherical",
}


def zernike_name(j: int) -> str:
    """Common name of the Noll mode ``j`` (or ``Z<j>`` if unnamed)."""
    return _NAMES.get(j, f"Z{j}")


def noll_to_nm(j: int) -> Tuple[int, int]:
    """Map a Noll index ``j >= 1`` to radial degree n and azimuthal order m.

    The sign of ``m`` encodes the angular function: ``m >= 0`` means
    ``cos(m θ)`` (even j), ``m < 0`` means ``sin(|m| θ)`` (odd j).
    """
    if j < 1:
        raise ValueError("Noll index must be >= 1.")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def _radial(n: int, m: int, rho: Array) -> Array:
    """Zernike radial polynomial R_n^m (m >= 0)."""
    R = np.zeros_like(rho, dtype=float)
    for k in range((n - m) // 2 + 1):
        c = (
            (-1) ** k
            * factorial(n - k)
            / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
        )
        R += c * rho ** (n - 2 * k)
    return R


def zernike(j: int, rho: Array, theta: Array) -> Array:
    """Noll-indexed, RMS-normalized Zernike polynomial on the unit disk.

    Parameters
    ----------
    j : int
        Noll index (1 = piston).
    rho, theta : arrays
        Polar pupil coordinates, ``rho`` normalized to the pupil radius.
    """
    n, m = noll_to_nm(j)
    am = abs(m)
    norm = np.sqrt(n + 1.0) if m == 0 else np.sqrt(2.0 * (n + 1.0))
    R = _radial(n, am, np.asarray(rho, dtype=float))
    if m == 0:
        return norm * R
    if m > 0:
        return norm * R * np.cos(am * np.asarray(theta))
    return norm * R * np.sin(am * np.asarray(theta))


def _pupil_polar(grid: Grid, radius: float, center: Tuple[float, float]) -> Tuple[Array, Array, Array]:
    x, y = grid
    x0, y0 = center
    rho = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / radius
    theta = np.arctan2(y - y0, x - x0)
    return rho, theta, rho <= 1.0


def fit_zernikes(
    field: Field,
    radius: float,
    jmax: int = 15,
    *,
    center: Tuple[float, float] = (0.0, 0.0),
    max_points: int = 200_000,
) -> Array:
    """Least-squares Zernike decomposition of a wavefront map.

    Parameters
    ----------
    field : Field
        Wavefront / OPD map (``field.values`` in any units; the coefficients
        inherit them). Samples outside the pupil are ignored; NaNs inside are
        ignored too.
    radius : float
        Pupil radius defining the unit disk of the polynomials.
    jmax : int
        Highest Noll index fitted.
    center : (x0, y0)
        Pupil center.
    max_points : int
        The fit subsamples the pupil (uniform stride) down to at most this
        many points to bound memory.

    Returns
    -------
    1D array of length ``jmax``
        Coefficients ``c[j-1]`` for Noll modes ``1..jmax``. With the RMS
        normalization used here, each coefficient is the RMS contribution
        of its mode and ``sqrt(sum(c[1:]**2))`` is the total RMS about
        piston (for a complete fit).
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if jmax < 1:
        raise ValueError("jmax must be >= 1.")

    grid = field.grid
    W = field.values
    if np.iscomplexobj(W):
        raise TypeError(
            "fit_zernikes expects a real wavefront/OPD map, not a complex "
            "field; pass e.g. the (unwrapped) phase divided by k, not the "
            "field itself."
        )
    rho, theta, inside = _pupil_polar(grid, radius, center)
    valid = inside & np.isfinite(W)
    rho_v, theta_v, W_v = rho[valid], theta[valid], np.asarray(W, dtype=float)[valid]
    if W_v.size < jmax:
        raise ValueError("Not enough valid pupil samples for the requested jmax.")

    if W_v.size > max_points:
        stride = int(np.ceil(W_v.size / max_points))
        rho_v, theta_v, W_v = rho_v[::stride], theta_v[::stride], W_v[::stride]

    A = np.column_stack([zernike(j, rho_v, theta_v) for j in range(1, jmax + 1)])
    coeffs, *_ = np.linalg.lstsq(A, W_v, rcond=None)
    return coeffs


def synthesize_zernikes(
    coeffs: Array,
    grid: Grid,
    radius: float,
    *,
    center: Tuple[float, float] = (0.0, 0.0),
) -> Field:
    """Wavefront map from Noll coefficients ``coeffs[j-1]`` as a :class:`Field`.

    The field's ``values`` are the OPD, NaN outside the pupil.
    """
    rho, theta, inside = _pupil_polar(grid, radius, center)
    W = np.zeros_like(rho)
    for j, c in enumerate(np.asarray(coeffs, dtype=float), start=1):
        if c != 0.0:
            W += c * zernike(j, rho, theta)
    return Field(grid, np.where(inside, W, np.nan))


def _wavefront_values(W: Union[Field, Array]) -> Array:
    values = W.values if isinstance(W, Field) else W
    return np.asarray(values, dtype=float)


def pv(W: Union[Field, Array], mask: Optional[Array] = None) -> float:
    """Peak-to-valley of a wavefront map (NaNs and masked-out samples ignored)."""
    values = _wavefront_values(W)
    if mask is not None:
        values = values[mask]
    values = values[np.isfinite(values)]
    return float(values.max() - values.min())


def rms(W: Union[Field, Array], mask: Optional[Array] = None) -> float:
    """RMS of a wavefront map about its mean (NaNs and masked-out samples ignored)."""
    values = _wavefront_values(W)
    if mask is not None:
        values = values[mask]
    values = values[np.isfinite(values)]
    return float(np.std(values))


def marechal_strehl(rms_waves: float) -> float:
    """Maréchal estimate of the Strehl ratio, ``exp(-(2π σ)²)``.

    ``rms_waves`` is the RMS wavefront error in waves; the estimate is
    accurate for small aberrations (σ ≲ 0.1 waves) and indicative below
    ~0.15 waves.
    """
    return float(np.exp(-((2.0 * np.pi * rms_waves) ** 2)))
