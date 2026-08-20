"""Propagation: transport of a field from one plane to another.

Every propagator here maps a :class:`~diffractor.field.Field` on a transverse
plane to a Field on a parallel plane a distance z away, reading the
wavelengths and the medium off the field itself.  Three of them are one
statement each in terms of the Fourier operators — which is the point: the
*method* (angular spectrum, Fresnel, Fraunhofer) is physics, and the *basis*
of the grid (cartesian FFT, polar Hankel modes) is a representation the
Fourier layer already owns.  Nothing in this module knows what a Bessel
function is.

======================  =====================================================
:func:`angular_spectrum`  exact — ``IFT2( e^{i k_z z} · FT2(field) )`` with
                          ``k_z = √(k_m² − |k⊥|²)``; the only hypothesis is
                          homogeneity of the medium.
:func:`fresnel`           paraxial single-transform integral, *gated*: it
                          refuses outside its validity regime unless told
                          otherwise, because a wrong answer with a colorbar
                          is worse than an exception.
:func:`fraunhofer`        the far-field limit of the same integral (no input
                          chirp).
:func:`rayleigh_sommerfeld`  the exact real-space reference (RS-I), kept for
                          cross-checking the spectral methods; axisymmetric
                          fields only — in 2-D its cost is O(N²M²) and the
                          angular spectrum is already exact.
======================  =====================================================

Output grids.  ``angular_spectrum`` returns the field on its own grid (the
transfer function is diagonal in k).  The single-transform methods evaluate
the field on the scaled frequency grid ``x' = (λ_m z/2π)·k`` — which depends
on the wavelength, so for a broadband field they *require* an explicit
``output_grid``: every spectral line must land on the same physical screen or
compositing them is meaningless.  An explicit ``output_grid`` is always
allowed and is the zoom that resolves a focal spot: the transform is then
evaluated by matrix quadrature exactly on the window you asked for.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .basis import CARTESIAN, POLAR
from .field import Field, _over_spectrum
from .fourier import FT2, IFT2, PolarPlan
from .space import Grid
from .spectrum import Spectrum

__all__ = ["angular_spectrum", "fresnel", "fraunhofer", "rayleigh_sommerfeld",
           "fresnel_validity_distance", "spectral_budget",
           "PARAXIAL_MARGIN", "SPECTRAL_MARGIN", "ASM_SAMPLES_PER_PERIOD"]

#: Safety margin on Goodman's paraxial condition z³ ≫ (π/4λ)·r_max⁴.  At
#: margin 1 the Fresnel integral is already visibly wrong at high aperture
#: (the validation session measured a 16× violation at NA 0.75); 3 is where
#: the axisymmetric test cases stay within a percent of the exact operators.
PARAXIAL_MARGIN: float = 3.0

#: How far past the propagating cone k = n·k₀ a propagator's default k-band
#: extends: enough to represent the evanescent shoulder the transfer function
#: needs, no more (the legacy 1.02·n/λ idiom, named).
SPECTRAL_MARGIN: float = 1.02

#: Samples per oscillation period that the polar ASM's default k-axis keeps —
#: denser than the plain transform's rule, because the transfer phase k_z·z
#: turns arbitrarily fast at the band edge and no linear rate estimate covers
#: it.  Measured on a hard disc at Fresnel number ~4: the on-axis error
#: against the closed form is 1.4e-2 at 5 samples/period, 2e-3 at 10, and
#: saturates (band-edge fringe misregistration, ~7e-3 of the peak off axis)
#: beyond — 10 is where accuracy stops paying for density.
ASM_SAMPLES_PER_PERIOD: float = 10.0

_POLICIES = ("raise", "warn", "force")


# ══════════════════════════════════════════════════════════════════════════
#  Validity
# ══════════════════════════════════════════════════════════════════════════
def fresnel_validity_distance(r_max: float, wavelength: float, *,
                              margin: float = PARAXIAL_MARGIN) -> float:
    """Minimum |z| for the paraxial (Fresnel) integral to be trusted.

    Goodman's condition ``z³ ≫ (π/4λ)·r_max⁴`` with a safety ``margin``;
    ``wavelength`` is the wavelength *in the medium*.
    """
    return margin * ((np.pi / (4.0 * wavelength)) * r_max**4) ** (1.0 / 3.0)


def _gate(field: Field, z: float, policy: str, method: str) -> None:
    if policy not in _POLICIES:
        raise ValueError(f"policy must be one of {_POLICIES}, got {policy!r}")
    r_max = float(np.sqrt(field.grid.r2().max()))
    lam_medium = field.spectrum.wavelengths.min() / field.medium.n
    z_min = fresnel_validity_distance(r_max, lam_medium)
    if abs(z) < z_min:
        msg = (f"{method} outside its paraxial validity regime: |z| = "
               f"{abs(z):.3e} < required {z_min:.3e} (r_max = {r_max:.3e}). "
               f"Use angular_spectrum, which is exact, or pass "
               f"policy='force' if you know better.")
        if policy == "raise":
            raise ValueError(msg)
        if policy == "warn":
            warnings.warn(msg, stacklevel=3)


# ══════════════════════════════════════════════════════════════════════════
#  Angular spectrum — exact
# ══════════════════════════════════════════════════════════════════════════
def _default_asm_kgrid(field: Field, z: float) -> Optional[Grid]:
    """The propagating cone plus its shoulder, sampled densely enough for z.

    Two choices, both physical.  The band: the transfer function only *acts*
    below the cone k = n·k₀, so the default k_max is the cone times
    :data:`SPECTRAL_MARGIN` — propagating over the grid's full representable
    band would spend the quadrature on frequencies about to be masked.  The
    density: the inverse integrand carries the phase k·r + k_z·z, which
    oscillates in k at a rate up to ≈ r_max + |z|, so Δk must shrink as the
    field travels — ``n_k = ⌈β·k_max·(r_max + |z|)/2π⌉ + 1`` with β the grid
    :data:`ASM_SAMPLES_PER_PERIOD`.  (The legacy propagator's default ignored
    the z term; that is why every long propagation needed a hand-picked
    n_rho.)
    Cartesian grids keep their FFT reciprocal — the FFT is exactly invertible,
    so its density needs no z scaling; what aliases there is the transfer
    function's *phase*, which the Matsushima limit masks.
    """
    if field.grid.basis is not POLAR:
        return None
    k_cut = 2.0 * np.pi * field.medium.n / field.spectrum.wavelengths.min()
    k_max = SPECTRAL_MARGIN * k_cut
    r_max = float(field.grid.axes[0][-1])
    n_k = int(np.ceil(ASM_SAMPLES_PER_PERIOD * k_max * (r_max + abs(z))
                      / (2.0 * np.pi))) + 1
    return field.grid.reciprocal(k_max=k_max, n_k=n_k)


def _transfer(kgrid: Grid, field: Field, z: float,
              include_evanescent: bool) -> np.ndarray:
    """``exp(i k_z z)`` on ``kgrid`` per spectral line, evanescent gated."""
    K2 = kgrid.r2()[..., np.newaxis]
    k_m = _over_spectrum([field.medium.k(lam)
                          for lam in field.spectrum.wavelengths])
    kz2 = k_m**2 - K2
    if include_evanescent:
        kz = np.emath.sqrt(kz2)
        # sign convention: decay, never growth, whichever way z points
        return np.exp(1j * kz.real * z - np.abs(kz.imag * z))
    H = np.zeros(np.broadcast_shapes(K2.shape, k_m.shape), complex)
    prop = kz2 >= 0
    H[prop] = np.exp(1j * np.sqrt(np.broadcast_to(kz2, H.shape)[prop]) * z)
    return H


def _matsushima_limit(kgrid: Grid, space_grid: Grid, z: float,
                      field: Field) -> np.ndarray:
    """The Matsushima–Shimobaba band limit for the sampled cartesian transfer
    function: beyond it the phase of e^{i k_z z} aliases between k samples.
    Angular form of the cycles-convention original, per spectral line."""
    KX, KY = kgrid.meshes()
    nx, ny = space_grid.shape
    dx = space_grid.axes[0][1] - space_grid.axes[0][0]
    dy = space_grid.axes[1][1] - space_grid.axes[1][0]
    lam_m = _over_spectrum(field.spectrum.wavelengths / field.medium.n)
    dfx, dfy = 1.0 / (nx * dx), 1.0 / (ny * dy)
    kx_lim = 2 * np.pi / (lam_m * np.sqrt((2 * dfx * abs(z)) ** 2 + 1.0))
    ky_lim = 2 * np.pi / (lam_m * np.sqrt((2 * dfy * abs(z)) ** 2 + 1.0))
    return ((np.abs(KX)[..., None] <= kx_lim)
            & (np.abs(KY)[..., None] <= ky_lim))


def angular_spectrum(field: Field, z: float, *,
                     include_evanescent: bool = False,
                     bandlimit: Optional[bool] = None,
                     pad_factor: int = 1,
                     kgrid: Optional[Grid] = None,
                     output_grid: Optional[Grid] = None) -> Field:
    """Propagate by the exact angular-spectrum method: ``IFT2(H·FT2(field))``.

    ``z`` may be negative (back-propagation).  ``include_evanescent`` keeps
    the decaying components instead of masking them.

    ``bandlimit`` and ``pad_factor`` belong to the cartesian FFT path only:
    the FFT convolution is *circular*, so it wraps around the window (padding
    fixes that) and its sampled transfer-function phase aliases at high k for
    long z (the Matsushima–Shimobaba limit, default on, masks that).  The
    polar path is an open quadrature with neither failure mode — its accuracy
    knobs are the band and density of ``kgrid`` — so passing either option on
    a polar field is refused rather than ignored.

    ``kgrid`` overrides the spectral sampling (polar default: the propagating
    cone × :data:`SPECTRAL_MARGIN`).  ``output_grid`` evaluates the inverse
    transform on a grid of your choice, decoupling output from input sampling
    — it does not relax the *input* sampling requirement, which no downstream
    transform can.
    """
    if field.domain != "space":
        raise ValueError("angular_spectrum propagates space-domain fields")
    grid = field.grid
    is_polar = grid.basis is POLAR

    if is_polar and (bandlimit is not None or pad_factor != 1):
        raise ValueError(
            "bandlimit/pad_factor address the wrap-around and phase aliasing "
            "of the periodic cartesian FFT; the polar path is an open "
            "quadrature — control its accuracy with kgrid "
            "(reciprocal(k_max=, n_k=)) instead")
    if pad_factor < 1:
        raise ValueError("pad_factor must be at least 1")

    work = field
    pad = (0, 0)
    if not is_polar and pad_factor > 1 and grid.basis is CARTESIAN:
        nx, ny = grid.shape
        px, py = ((pad_factor - 1) * nx) // 2, ((pad_factor - 1) * ny) // 2
        vals = np.pad(field.values, ((px, px), (py, py), (0, 0)))
        dx = grid.axes[0][1] - grid.axes[0][0]
        dy = grid.axes[1][1] - grid.axes[1][0]
        x = grid.axes[0][0] + (np.arange(nx + 2 * px) - px) * dx
        y = grid.axes[1][0] + (np.arange(ny + 2 * py) - py) * dy
        work = field.like(vals, grid=Grid.cartesian(x, y))
        pad = (px, py)

    plan = None
    if is_polar:
        kgrid = kgrid if kgrid is not None else _default_asm_kgrid(field, z)
        plan = PolarPlan.build(work.grid, kgrid)

    spectrum = FT2(work, kgrid=kgrid, plan=plan)
    H = _transfer(spectrum.grid, field, z, include_evanescent)
    if not is_polar and (bandlimit or bandlimit is None) and z != 0:
        H = np.where(_matsushima_limit(spectrum.grid, work.grid, z, field),
                     H, 0.0)
    spectrum = spectrum.like(spectrum.values * H)

    if output_grid is not None:
        return IFT2(spectrum, grid=output_grid)
    out = IFT2(spectrum, grid=work.grid, plan=plan)
    if pad != (0, 0):
        px, py = pad
        nx, ny = grid.shape
        out = field.like(out.values[px:px + nx, py:py + ny, :], grid=grid)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Single-transform methods — paraxial, gated
# ══════════════════════════════════════════════════════════════════════════
def _single_transform(field: Field, z: float, *, chirp_in: bool,
                      policy: str, output_grid: Optional[Grid],
                      method: str) -> Field:
    """The Fresnel/Fraunhofer integral through one Fourier transform:

        U'(x') = e^{i k_m z}/(i λ_m z) · e^{i k_m |x'|²/2z}
                 · FT2[ U · e^{i k_m |x|²/2z} ]( k⊥ = (k_m/z)·x' )

    basis-agnostic because |x|² is a grid quantity (``grid.r2()``).
    """
    if field.domain != "space":
        raise ValueError(f"{method} propagates space-domain fields")
    if z <= 0:
        raise ValueError(
            f"{method}'s output plane scales with z, so z must be positive; "
            f"use angular_spectrum for back-propagation")
    _gate(field, z, policy, method)

    wavelengths = field.spectrum.wavelengths
    k_m = np.array([field.medium.k(lam) for lam in wavelengths])
    lam_m = wavelengths / field.medium.n
    r2 = field.grid.r2()[..., np.newaxis]

    values = field.values
    if chirp_in:
        values = values * np.exp(1j * _over_spectrum(k_m) * r2 / (2.0 * z))
    chirped = field.like(values)

    if output_grid is None:
        if field.spectrum.n != 1:
            raise ValueError(
                f"{method}'s natural output grid x' = (λ z/2π)·k differs per "
                f"wavelength; pass output_grid= to land the whole spectrum "
                f"on one physical screen")
        F = FT2(chirped)
        out_grid = F.grid.scaled(float(lam_m[0]) * z / (2.0 * np.pi))
        G = F.values
    else:
        # evaluate the transform at k⊥ = (k_m/z)·x', per spectral line
        out_grid = output_grid
        if (field.grid.basis is POLAR and output_grid.basis is POLAR
                and not np.array_equal(output_grid.axes[1],
                                       field.grid.axes[1])):
            raise ValueError("a polar output_grid must reuse the field's "
                             "theta axis verbatim")
        G = np.empty((*output_grid.shape, field.spectrum.n), complex)
        for l, lam in enumerate(wavelengths):
            kg = output_grid.scaled(float(k_m[l]) / z)
            line = Field(chirped.grid, chirped.values[..., l:l + 1],
                         Spectrum.line(lam), medium=chirped.medium,
                         domain=chirped.domain)
            G[..., l:l + 1] = FT2(line, kgrid=kg).values

    r2_out = out_grid.r2()[..., np.newaxis]
    pre = (np.exp(1j * _over_spectrum(k_m) * z)
           / (1j * _over_spectrum(lam_m) * z)
           * np.exp(1j * _over_spectrum(k_m) * r2_out / (2.0 * z)))
    return field.like(pre * G, grid=out_grid, domain="space")


def fresnel(field: Field, z: float, *, policy: str = "raise",
            output_grid: Optional[Grid] = None) -> Field:
    """Paraxial single-transform Fresnel propagation, gated.

    The gate checks Goodman's condition through
    :func:`fresnel_validity_distance` at the field's shortest medium
    wavelength; ``policy`` is ``'raise'`` (default), ``'warn'`` or
    ``'force'``.  See the module docstring for the output-grid contract.
    """
    return _single_transform(field, z, chirp_in=True, policy=policy,
                             output_grid=output_grid, method="fresnel")


def fraunhofer(field: Field, z: float, *, policy: str = "raise",
               output_grid: Optional[Grid] = None) -> Field:
    """Far-field (Fraunhofer) propagation: :func:`fresnel` without the input
    chirp, valid when additionally ``z ≫ k_m·r_max²/2``.  Shares the same
    gate — one validity mechanism in the package, not two."""
    return _single_transform(field, z, chirp_in=False, policy=policy,
                             output_grid=output_grid, method="fraunhofer")


# ══════════════════════════════════════════════════════════════════════════
#  Rayleigh–Sommerfeld I — the exact real-space reference
# ══════════════════════════════════════════════════════════════════════════
def rayleigh_sommerfeld(field: Field, z: float, *,
                        output_grid: Optional[Grid] = None,
                        chunk: int = 64) -> Field:
    """Rayleigh–Sommerfeld I, integrated in real space (exact; = the angular
    spectrum written as a surface integral).

        U(P) = (1/2π) ∬ U (z/R)(i k_m − 1/R) e^{i k_m R}/R dA

    Axisymmetric fields only (``n_θ = 1``): the azimuthal integral is then a
    Bessel kernel and the cost is O(n_r·n_out).  On any other grid the honest
    answer is :func:`angular_spectrum`, which is exact on every basis —
    a 2-D real-space quadrature would be O(N²M²) for no accuracy in return.
    """
    from scipy.special import j0 as _j0

    grid = field.grid
    if grid.basis is not POLAR or grid.axes[1].size != 1:
        raise ValueError(
            "rayleigh_sommerfeld is the axisymmetric real-space reference "
            f"(polar grid, n_theta = 1); got basis {grid.basis.name!r} with "
            f"shape {grid.shape} — use angular_spectrum, which is exact on "
            f"every basis")
    out_grid = grid if output_grid is None else output_grid
    if out_grid.basis is not POLAR or out_grid.axes[1].size != 1:
        raise ValueError("the output grid must be axisymmetric polar too")

    r = grid.axes[0]
    rho = out_grid.axes[0]
    k_m = np.array([field.medium.k(lam) for lam in field.spectrum.wavelengths])
    w = np.gradient(r) if r.size > 2 else np.ones_like(r)   # trapezoid-like
    # composite trapezoid weights, explicit:
    w = np.empty_like(r)
    if r.size > 1:
        w[0] = (r[1] - r[0]) / 2.0
        w[-1] = (r[-1] - r[-2]) / 2.0
        w[1:-1] = (r[2:] - r[:-2]) / 2.0
    else:
        w[:] = 1.0
    Urw = field.values[:, 0, :] * (r * w)[:, None]           # (n_r, n_λ)

    out = np.empty((rho.size, 1, k_m.size), complex)
    for i in range(0, rho.size, chunk):
        p = rho[i:i + chunk][:, None]
        R = np.sqrt(z**2 + r[None, :] ** 2 + p**2)
        geom = (z / R) / R                                    # (P, n_r)
        for l, k in enumerate(k_m):
            kern = (geom * (1j * k - 1.0 / R) * np.exp(1j * k * R)
                    * _j0(k * r[None, :] * p / R))
            out[i:i + chunk, 0, l] = kern @ Urw[:, l]
    return field.like(out, grid=out_grid, domain="space")


# ══════════════════════════════════════════════════════════════════════════
#  Spectral budget
# ══════════════════════════════════════════════════════════════════════════
def spectral_budget(field: Field, *, kgrid: Optional[Grid] = None) -> np.ndarray:
    """Fraction of each spectral line's power beyond the propagating cone.

    The part of the angular spectrum with |k⊥| > n·k₀ is evanescent: it is
    what a band-limited propagation silently discards, and the reason a
    thin-element boundary condition leaks energy.  Measure it before trusting
    a result near a sharp feature.  Returns shape ``(n_λ,)``.
    """
    if kgrid is None and field.grid.basis is POLAR:
        # the budget must see past the cone; 2.2 spans the evanescent side
        # far enough that what lies beyond is quadrature noise, not power
        k_cut = 2.0 * np.pi * field.medium.n / field.spectrum.wavelengths.min()
        default = field.grid.reciprocal()
        k_max = max(2.2 * k_cut, float(default.axes[0][-1]))
        kgrid = field.grid.reciprocal(k_max=k_max)
    F = FT2(field, kgrid=kgrid)
    K2 = F.grid.r2()[..., np.newaxis]
    w = F.grid.weights()[..., np.newaxis]
    k_cut = _over_spectrum([field.medium.k(lam)
                            for lam in field.spectrum.wavelengths])
    dens = F.spectral_intensity * w
    total = dens.sum(axis=(0, 1))
    beyond = np.where(K2 > k_cut**2, dens, 0.0).sum(axis=(0, 1))
    return beyond / total
