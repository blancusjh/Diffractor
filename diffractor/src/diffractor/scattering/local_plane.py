"""Local plane-wave (tangent-plane) transmission through a smooth interface.

At every point of the interface the incident field is treated as a plane wave
hitting the local tangent plane.  Snell's law fixes the refracted direction and
the scalar Fresnel coefficient fixes the amplitude, which gives the Cauchy pair
(psi, dpsi/dn) of the transmitted field on the surface.  Feeding that pair to a
Helmholtz surface representation propagates it exactly.

This is the model Diffractor implements; the boundary-integral solver in
``groundtruth`` is what it is measured against.  Its two approximations are
explicit: the interface is replaced by its tangent plane locally, and the
transmitted field is written as if the incident wave were the only one present
(no multiple scattering, no edge currents).
"""

from __future__ import annotations

import numpy as np

from .fresnel import snell_cos, t_s

__all__ = ["point_source_transmission"]


def point_source_transmission(rho, z, n_rho, n_z, source_z, n1, n2, k1, k2):
    """Transmitted Cauchy data on an interface lit by an on-axis point source.

    Parameters
    ----------
    rho, z : array
        Interface points, in the meridian half-plane.
    n_rho, n_z : array
        Unit normal at those points, pointing OUT of medium 2 (the transmission
        side), which is the convention of the surface representation.
    source_z : float
        Axial position of the point source, which sits in medium 1.
    k1, k2 : complex
        Wavenumbers; ``k2`` may be complex for an absorbing medium 2.

    Returns
    -------
    u, v : complex arrays
        ``u`` = transmitted field, ``v`` = its normal derivative along
        ``(n_rho, n_z)``.
    """
    rho, z = np.asarray(rho, float), np.asarray(z, float)
    d1 = np.hypot(rho, z - source_z)
    cos_i1 = np.abs((rho * n_rho + (z - source_z) * n_z) / d1)
    cos_i2 = snell_cos(n1, n2, cos_i1)

    u_inc = np.exp(1j * k1 * d1) / (4 * np.pi * d1)
    u = t_s(n1, n2, cos_i1, cos_i2) * u_inc
    # the refracted ray goes INTO medium 2, i.e. against the outward normal
    v = -1j * k2 * cos_i2 * u
    return u, v
