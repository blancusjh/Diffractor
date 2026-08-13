"""The exact planar-interface operator — the one interface with no approximation.

For a flat boundary z = const between media n1 and n2, the transmission
operator is DIAGONAL in the angular spectrum: each plane-wave component
refracts independently, conserving its transverse wavevector,

    ψ̃₂(k⊥) = t(k⊥) · ψ̃₁(k⊥),     t = 2 k1z / (k1z + k2z),

    k_jz = sqrt( (n_j k0)² − |k⊥|² )        (evanescent for |k⊥| > n_j k0).

This is rung 1 of the validation ladder and the primitive every curved-
interface scheme must reduce to in the flat limit.  Axisymmetric version
(Hankel) here; Cartesian FFT version to follow the same pattern.
"""

from __future__ import annotations

import numpy as np

__all__ = ["t_spectral", "transmit_axisym"]


def t_spectral(k_rho, n1, n2, k0):
    """Scalar transmission per spectral component (k_rho = |k⊥|)."""
    k1z = np.emath.sqrt((n1 * k0) ** 2 - k_rho**2)
    k2z = np.emath.sqrt((n2 * k0) ** 2 - k_rho**2)
    return 2.0 * k1z / (k1z + k2z)


def transmit_axisym(U, r, n1, n2, wavelength, r_out=None, *, n_rho=None,
                    keep_evanescent=False):
    """Push an axisymmetric field through a flat interface, exactly.

    Hankel-transform U(r) → Ũ(ρ), multiply by t(2πρ), transform back.
    Exact for the scalar transmission problem on a plane — no approximation.
    """
    from ..propagation.exact import hankel

    k0 = 2 * np.pi / wavelength
    if r_out is None:
        r_out = r
    rho_cut = max(n1, n2) / wavelength
    rho_max = 1.05 * rho_cut if keep_evanescent else n2 / wavelength
    if n_rho is None:
        n_rho = int(np.ceil(2.5 * rho_max * r.max())) + 1
    rho = np.linspace(0.0, rho_max, n_rho)

    Uh = hankel(U, r, rho)
    t = t_spectral(2 * np.pi * rho, n1, n2, k0)
    return hankel(Uh * t, rho, r_out), rho, t
