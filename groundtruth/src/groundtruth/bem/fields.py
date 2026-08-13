"""Batched evaluation of the scalar field off the boundary.

Interior (x inside Omega2):
    psi2(x) = Int_S [ G2 v - u dG2/dn ] dS

Exterior (x in Omega1):
    psi1(x) = psi_inc(x) + Int_S [ -G1 (v-v_inc) + (u-u_inc) dG1/dn ] dS

Both are vectorised over field points and chunked for memory; the previous
point-by-point loop made field maps impractical.
"""

from __future__ import annotations

import numpy as np

__all__ = ["field_map"]


def _kernels(Rq, Zq, g, k, cphi, sphi, wphi):
    """G and dG/dn_y integrated over phi, for field points (Rq, Zq).

    Returns two arrays of shape (n_points, n_panels): the phi-integrated
    single- and double-layer kernels including the rho' dS factor.
    """
    rho, z, nr, nz, h = g.rho, g.z, g.n_rho, g.n_z, g.h
    # (points, panels, phi)
    dr = Rq[:, None, None] - rho[None, :, None] * cphi[None, None, :]
    di = rho[None, :, None] * sphi[None, None, :]
    dz = Zq[:, None, None] - z[None, :, None]
    d = np.sqrt(np.maximum(dr**2 + di**2 + dz**2, 1e-300))

    G = np.exp(1j * k * d) / (4 * np.pi * d)
    # (y - x).n_y  with y on the ring, x = (Rq, 0, Zq)
    Wn = (nr[None, :, None] * (rho[None, :, None] - Rq[:, None, None] * cphi[None, None, :])
          + nz[None, :, None] * (z[None, :, None] - Zq[:, None, None]))
    dGdn = (1j * k - 1.0 / d) * G / d * Wn

    meas = (rho * h)[None, :]
    S = np.einsum("pjq,q->pj", G, wphi) * meas
    D = np.einsum("pjq,q->pj", dGdn, wphi) * meas
    return S, D


def field_map(g, k, u, v, rho_pts, z_pts, *, n_phi=None, chunk=None,
              incident=None, exterior=False):
    """Evaluate psi at the given points.

    ``incident`` (only for ``exterior=True``) must be a callable
    ``f(rho, z) -> (psi_inc, dpsi_inc/dn_on_surface)``; the surface values are
    taken from ``g``.

    ``n_phi=None`` scales the azimuthal rule with the electrical size: the
    integrand oscillates through up to k sqrt(R_max rho_max) / pi cycles in
    phi, and 8 Gauss points per cycle resolve it.
    """
    rho_pts = np.asarray(rho_pts, float).ravel()
    z_pts = np.asarray(z_pts, float).ravel()
    if n_phi is None:
        rmax = float(rho_pts.max()) if rho_pts.size else 0.0
        scale = np.sqrt(max(rmax, 1e-12) * g.rho.max())
        n_phi = max(48, int(np.ceil(8.0 * abs(k) * scale / np.pi)))
    xg, wg = np.polynomial.legendre.leggauss(n_phi)
    phi = 0.5 * np.pi * (xg + 1.0)
    wphi = 0.5 * np.pi * wg * 2.0
    cphi, sphi = np.cos(phi), np.sin(phi)

    out = np.empty(rho_pts.size, dtype=complex)

    if chunk is None:
        chunk = max(1, int(5.0e6 / (g.N * n_phi)))

    if exterior:
        u_s, v_s = incident(g.rho, g.z)
        du, dv = u - u_s, v - v_s

    for a in range(0, rho_pts.size, chunk):
        b = min(a + chunk, rho_pts.size)
        S, D = _kernels(rho_pts[a:b], z_pts[a:b], g, k, cphi, sphi, wphi)
        if exterior:
            out[a:b] = -S @ dv + D @ du
        else:
            out[a:b] = S @ v - D @ u
    return out
