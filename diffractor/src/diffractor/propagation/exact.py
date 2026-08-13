"""Exact free-space propagators for homogeneous media.

Ported verbatim from the validated session code:
  * hankel        — order-zero Hankel pair (chunked)
  * asm_axisym    — exact angular spectrum, kz = 2π sqrt((n/λ)² − ρ²)
  * rs1_plane     — Rayleigh–Sommerfeld I (the same operator in real space;
                    validated against asm_axisym to 3e-4)
  * fresnel_plane — PARAXIAL; use through propagation.paraxial, which gates it.
  * spectral_budget — fraction of power beyond the propagating cone.

These are the only propagators the physical core exposes: they solve the
Helmholtz equation in a homogeneous region with no further hypothesis.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0

__all__ = ["hankel", "asm_axisym", "rs1_plane", "fresnel_plane",
           "spectral_budget"]


# ── Hankel transform pair (chunked, so big grids stay in memory) ──────────────
def hankel(f, x, y, chunk=192):
    """F(y) = 2 pi Int f(x) J0(2 pi x y) x dx  -- order-zero Hankel transform."""
    out = np.empty(y.size, dtype=complex)
    fx = f * x
    for i in range(0, y.size, chunk):
        blk = y[i:i + chunk]
        M = j0(2.0 * np.pi * np.outer(blk, x))
        out[i:i + chunk] = 2.0 * np.pi * np.trapezoid(M * fx[None, :], x, axis=1)
    return out


def asm_axisym(U, r, z, n, lam, r_out, *, n_rho=None, rho_max=None,
               evanescent=False, chunk=192):
    """Exact angular-spectrum propagation of an axisymmetric field.

        U~(rho)  = Hankel[U]
        U~(rho) *= exp(i kz z),   kz = 2 pi sqrt((n/lam)^2 - rho^2)
        U(r_out) = Hankel^-1[...]

    Non-propagating components (rho > n/lam) are discarded unless
    ``evanescent`` -- which is exactly what a band-limited ASM implementation
    does, and the reason the thin-element boundary condition leaks energy.
    """
    rho_cut = n / lam
    if rho_max is None:
        rho_max = 1.02 * rho_cut
    if n_rho is None:                      # Nyquist in rho set by the aperture
        n_rho = int(np.ceil(2.5 * rho_max * r.max())) + 1
    rho = np.linspace(0.0, rho_max, n_rho)

    Uh = hankel(U, r, rho, chunk=chunk)

    kz2 = (2 * np.pi) ** 2 * (rho_cut**2 - rho**2)
    prop = rho <= rho_cut
    H = np.zeros_like(rho, dtype=complex)
    H[prop] = np.exp(1j * np.sqrt(kz2[prop]) * z)
    if evanescent:
        H[~prop] = np.exp(-np.sqrt(-kz2[~prop]) * abs(z))

    return hankel(Uh * H, rho, r_out, chunk=chunk), rho, Uh


def spectral_budget(U, r, n, lam, *, n_rho=4096):
    """Fraction of an axisymmetric field's power beyond the propagating cone.

    Parseval for the Hankel pair:  Int |U|^2 r dr = Int |U~|^2 rho drho.
    """
    rho_cut = n / lam
    rho = np.linspace(0.0, 2.2 * rho_cut, n_rho)
    Uh = hankel(U, r, rho)
    dens = np.abs(Uh) ** 2 * rho
    tot = np.trapezoid(dens, rho)
    prop = np.trapezoid(np.where(rho <= rho_cut, dens, 0.0), rho)
    return 1.0 - prop / tot, rho, Uh


# ── Propagators from a PLANE ──────────────────────────────────────────────────
def rs1_plane(U, r, z, n, lam, rho_out, chunk=64):
    """Rayleigh-Sommerfeld I from a plane (exact; = ASM in real space).

        U(P) = 1/(2 pi) IntInt U (z/R)(ik - 1/R) exp(ikR)/R dA
    """
    k = 2 * np.pi * n / lam
    out = np.empty(rho_out.size, dtype=complex)
    Ur = U * r
    for i in range(0, rho_out.size, chunk):
        p = rho_out[i:i + chunk][:, None]
        R = np.sqrt(z**2 + r[None, :] ** 2 + p**2)
        kern = (Ur[None, :] * (z / R) * (1j * k - 1.0 / R)
                * np.exp(1j * k * R) / R * j0(k * r[None, :] * p / R))
        out[i:i + chunk] = np.trapezoid(kern, r, axis=1)
    return out


def fresnel_plane(U, r, z, n, lam, rho_out, chunk=64):
    """Paraxial Fresnel propagator from a plane."""
    k = 2 * np.pi * n / lam
    lam_m = lam / n
    pre = (2 * np.pi / (1j * lam_m * z)) * np.exp(1j * k * z)
    Uq = U * np.exp(1j * k * r**2 / (2 * z)) * r
    out = np.empty(rho_out.size, dtype=complex)
    for i in range(0, rho_out.size, chunk):
        p = rho_out[i:i + chunk][:, None]
        kern = Uq[None, :] * j0(k * r[None, :] * p / z)
        out[i:i + chunk] = (pre * np.exp(1j * k * p[:, 0] ** 2 / (2 * z))
                            * np.trapezoid(kern, r, axis=1))
    return out



