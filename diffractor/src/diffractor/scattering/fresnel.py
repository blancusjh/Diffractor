"""Scalar Fresnel coefficients — the local response of an interface.

These live in `scattering` because that is what they are: the transmission/
reflection response of a planar (or locally planar) index step.  An Interface
object does not know them; a scattering computation applies them to one.

These are what the SCALAR Helmholtz transmission conditions
([ψ] = 0, [∂ψ/∂n] = 0) actually give — the TE/s coefficient and nothing else.
Do not average with t_p: t_p belongs to Maxwell, not to this framework.
(Session evidence: the BEM measurement matched t_s absolutely to 0.25 %.)

Conventions
-----------
* ``t_s`` multiplies the FIELD amplitude of the transmitted wave.
* ``T_s`` is the POWER transmittance; the (n2 cos i2)/(n1 cos i1) factor is the
  flux normalisation (n cos θ is the normal component of the conserved
  Helmholtz current) — it is not optional.
* Energy: R_s + T_s = 1 identically below total internal reflection.
"""

from __future__ import annotations

import numpy as np

__all__ = ["snell_cos", "t_s", "r_s", "T_s", "R_s"]


def snell_cos(n1, n2, cos_i1):
    """cos i2 from Snell, given cos i1.  Returns 0 past the critical angle."""
    sin_i1_sq = np.clip(1.0 - np.asarray(cos_i1) ** 2, 0.0, 1.0)
    sin_i2_sq = (n1 / n2) ** 2 * sin_i1_sq
    return np.sqrt(np.clip(1.0 - sin_i2_sq, 0.0, 1.0))


def t_s(n1, n2, cos_i1, cos_i2=None):
    """Scalar (TE) amplitude transmission  2 n1 c1 / (n1 c1 + n2 c2)."""
    if cos_i2 is None:
        cos_i2 = snell_cos(n1, n2, cos_i1)
    return 2.0 * n1 * cos_i1 / (n1 * cos_i1 + n2 * cos_i2)


def r_s(n1, n2, cos_i1, cos_i2=None):
    """Scalar (TE) amplitude reflection  (n1 c1 − n2 c2)/(n1 c1 + n2 c2)."""
    if cos_i2 is None:
        cos_i2 = snell_cos(n1, n2, cos_i1)
    return (n1 * cos_i1 - n2 * cos_i2) / (n1 * cos_i1 + n2 * cos_i2)


def T_s(n1, n2, cos_i1, cos_i2=None):
    """Power transmittance  (n2 c2 / n1 c1) |t_s|²."""
    if cos_i2 is None:
        cos_i2 = snell_cos(n1, n2, cos_i1)
    t = t_s(n1, n2, cos_i1, cos_i2)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(n1 * cos_i1 > 0,
                       (n2 * cos_i2) / (n1 * cos_i1) * t**2, 0.0)
    return out


def R_s(n1, n2, cos_i1, cos_i2=None):
    """Power reflectance |r_s|²."""
    return r_s(n1, n2, cos_i1, cos_i2) ** 2
