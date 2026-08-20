"""The exact planar-interface operator — the one interface with no approximation.

For a flat boundary z = const between media n₁ and n₂, the transmission
operator is DIAGONAL in the angular spectrum: each plane-wave component
refracts independently, conserving its transverse wavevector,

    ψ̃₂(k⊥) = t(k⊥) · ψ̃₁(k⊥),     t = 2 k₁z / (k₁z + k₂z),

    k_jz = √( (n_j k₀)² − |k⊥|² )        (evanescent for |k⊥| > n_j k₀).

This is rung 1 of the validation ladder and the primitive every curved-
interface scheme must reduce to in the flat limit.  :func:`transmit` applies
it to a :class:`~diffractor.field.Field` through the Fourier operators —
``IFT2( t(|k⊥|) · FT2(field) )`` — so it works on every basis: the interface
does not care how the plane is being addressed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["t_spectral", "transmit"]


def t_spectral(k_perp, n1, n2, k0):
    """Scalar transmission per spectral component (k_perp = |k⊥|, angular)."""
    k1z = np.emath.sqrt((n1 * k0) ** 2 - np.asarray(k_perp) ** 2)
    k2z = np.emath.sqrt((n2 * k0) ** 2 - np.asarray(k_perp) ** 2)
    return 2.0 * k1z / (k1z + k2z)


def transmit(field, medium2, *, keep_evanescent: bool = False,
             kgrid: Optional[object] = None):
    """Push a field through a flat interface into ``medium2``, exactly.

    ``FT2 → multiply by t(|k⊥|) per spectral line → IFT2``, on whatever basis
    the field's grid uses.  Components beyond ``medium2``'s propagating cone
    are dropped unless ``keep_evanescent`` — which is exactly what a
    band-limited implementation does, and the reason the thin-element boundary
    condition leaks energy.  The returned field lives in ``medium2``.
    """
    from ..basis import POLAR
    from ..fourier import FT2, IFT2
    from ..propagation import SPECTRAL_MARGIN

    n1 = field.medium.n
    n2 = medium2.n
    if kgrid is None and field.grid.basis is POLAR:
        # the operator acts below the wider cone; band-limit the quadrature
        # there instead of collecting noise from the grid's full band
        k_cut = (2.0 * np.pi * max(n1, n2)
                 / field.spectrum.wavelengths.min())
        kgrid = field.grid.reciprocal(k_max=SPECTRAL_MARGIN * k_cut)
    F = FT2(field, kgrid=kgrid)
    k_perp2 = F.grid.r2()[..., np.newaxis]

    values = np.empty_like(F.values)
    for l, lam in enumerate(field.spectrum.wavelengths):
        k0 = 2.0 * np.pi / lam
        t = t_spectral(np.sqrt(k_perp2[..., 0]), n1, n2, k0)
        if not keep_evanescent:
            t = np.where(k_perp2[..., 0] <= (n2 * k0) ** 2, t, 0.0)
        values[..., l] = F.values[..., l] * t
    out = IFT2(F.like(values), grid=field.grid)
    return out.like(out.values, medium=medium2)
