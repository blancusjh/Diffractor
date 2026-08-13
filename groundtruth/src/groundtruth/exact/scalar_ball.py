"""
EXACT solver for the SCALAR Helmholtz transmission problem on a dielectric ball.

This is not Mie.  Mie solves Maxwell's equations and its coefficients follow
from the continuity of tangential E and H.  Here the governing equation is the
scalar Helmholtz equation

    (nabla^2 + k^2) psi = 0 ,      k = k0 n  (piecewise constant)

and the only boundary conditions a scalar field possesses are

    [psi] = 0 ,      [d psi / dn] = 0        across the interface.

Those are exactly the conditions that produced t = 2 n1 cos i1/(n1 cos i1 +
n2 cos i2) for a plane interface -- the TE/s coefficient -- so this solver is
the correct reference for a scalar diffraction framework, and its plane-
interface limit is a check we can run.

Separation of variables (ball radius a, centred at the origin, on-axis
incidence, so only m = 0 survives):

    psi_inc = exp(i k1 z)        = sum_l (2l+1) i^l  j_l(k1 r) P_l(cos th)
    psi_sca = sum_l (2l+1) i^l A_l h_l(k1 r) P_l(cos th)      (r > a)
    psi_int = sum_l (2l+1) i^l B_l j_l(k2 r) P_l(cos th)      (r < a)

Matching psi and dpsi/dr at r = a gives a 2x2 system per order.  Solving it by
Cramer's rule and using the Wronskian j_l h_l' - h_l j_l' = i/x^2 keeps B_l
free of any division by j_l(k2 a), which would otherwise blow up wherever that
function has a zero.

    A_l = [ k2 j_l(x1) j_l'(x2) - k1 j_l(x2) j_l'(x1) ] / det
    B_l = i k1 / (x1^2 det)
    det =  k1 j_l(x2) h_l'(x1) - k2 h_l(x1) j_l'(x2)
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn


# ── Legendre P_l(x) for all l up to lmax, by stable upward recurrence ─────────
def legendre_all(lmax, x):
    """P[l, i] = P_l(x_i) for l = 0..lmax.  Upward recurrence is stable on |x|<=1."""
    x = np.atleast_1d(x)
    P = np.empty((lmax + 1, x.size))
    P[0] = 1.0
    if lmax >= 1:
        P[1] = x
    for l in range(1, lmax):
        P[l + 1] = ((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1)
    return P


class ScalarBall:
    """Exact scalar-Helmholtz solution for a plane wave on a dielectric ball."""

    def __init__(self, n1, n2, a, lam, lmax=None, lmax_pad=15):
        self.n1, self.n2, self.a, self.lam = n1, n2, a, lam
        self.k0 = 2 * np.pi / lam
        self.k1, self.k2 = self.k0 * n1, self.k0 * n2
        self.x1, self.x2 = self.k1 * a, self.k2 * a

        # The incident plane wave only excites multipoles up to l ~ x1 (impact
        # parameter <= a), so Wiscombe's criterion on x1 -- NOT x2 -- sets the
        # truncation.  Going past it would also overflow y_l(x1).
        if lmax is None:
            lmax = int(self.x1 + 4.05 * self.x1 ** (1 / 3) + lmax_pad)
        self.lmax = lmax
        self.l = np.arange(lmax + 1)
        self._coefficients()

    def _coefficients(self):
        l, x1, x2 = self.l, self.x1, self.x2
        k1, k2 = self.k1, self.k2

        j1 = spherical_jn(l, x1);  dj1 = spherical_jn(l, x1, derivative=True)
        y1 = spherical_yn(l, x1);  dy1 = spherical_yn(l, x1, derivative=True)
        j2 = spherical_jn(l, x2);  dj2 = spherical_jn(l, x2, derivative=True)

        h1 = j1 + 1j * y1
        dh1 = dj1 + 1j * dy1

        det = k1 * j2 * dh1 - k2 * h1 * dj2
        self.A = (k2 * j1 * dj2 - k1 * j2 * dj1) / det
        self.B = 1j * k1 / (x1**2 * det)      # Wronskian form: no j_l(x2) in a denominator
        self._cache = dict(j1=j1, dj1=dj1, y1=y1, dy1=dy1, j2=j2, dj2=dj2,
                           h1=h1, dh1=dh1)

    # -- field evaluation ------------------------------------------------------
    def _series(self, r, cos_th, radial, coeff):
        """sum_l (2l+1) i^l coeff_l radial_l(r) P_l(cos th), r and cos_th 1-D."""
        r = np.atleast_1d(r); cos_th = np.atleast_1d(cos_th)
        P = legendre_all(self.lmax, cos_th)                    # (L+1, N)
        w = (2 * self.l + 1) * (1j ** self.l) * coeff          # (L+1,)
        out = np.empty(r.size, dtype=complex)
        # radial functions depend on r, which differs per point
        for i, ri in enumerate(r):
            out[i] = np.sum(w * radial(ri) * P[:, i])
        return out

    def internal(self, r, cos_th):
        """psi inside the ball (r < a)."""
        return self._series(r, cos_th,
                            lambda ri: spherical_jn(self.l, self.k2 * ri),
                            self.B)

    def scattered(self, r, cos_th):
        """psi_scattered outside the ball (r > a)."""
        def rad(ri):
            x = self.k1 * ri
            return spherical_jn(self.l, x) + 1j * spherical_yn(self.l, x)
        return self._series(r, cos_th, rad, self.A)

    def incident(self, r, cos_th):
        return self._series(r, cos_th,
                            lambda ri: spherical_jn(self.l, self.k1 * ri),
                            np.ones_like(self.A))

    def surface_field(self, cos_th):
        """psi on r = a, taken from the exterior side (incident + scattered).

        Same value as the interior limit, since [psi] = 0 there; this is what
        a boundary-integral solver returns as its unknown u.
        """
        cos_th = np.atleast_1d(np.asarray(cos_th, float))
        w = (2 * self.l + 1) * (1j ** self.l)
        return (w * (self._cache["j1"] + self.A * self._cache["h1"])) \
            @ legendre_all(self.lmax, cos_th)

    def field_grid(self, rho_pts, z_pts):
        """Total field at arbitrary points: internal inside, incident+scattered
        outside.  Vectorised over all points at once."""
        rho_pts = np.asarray(rho_pts, float).ravel()
        z_pts = np.asarray(z_pts, float).ravel()
        r = np.hypot(rho_pts, z_pts)
        cth = np.where(r > 0, z_pts / np.where(r > 0, r, 1.0), 1.0)
        P = legendre_all(self.lmax, cth)                    # (L+1, Npts)
        w = (2 * self.l + 1) * (1j ** self.l)
        out = np.empty(r.size, dtype=complex)

        inside = r < self.a
        if inside.any():
            jn = spherical_jn(self.l[:, None], self.k2 * r[inside][None, :])
            out[inside] = np.einsum("l,lp,lp->p", w * self.B, jn, P[:, inside])
        if (~inside).any():
            x = self.k1 * r[~inside]
            jn = spherical_jn(self.l[:, None], x[None, :])
            yn = spherical_yn(self.l[:, None], x[None, :])
            tot = jn + self.A[:, None] * (jn + 1j * yn)
            out[~inside] = np.einsum("l,lp,lp->p", w, tot, P[:, ~inside])
        return out

    def internal_cartesian(self, z, rho):
        """psi inside, at cylindrical (rho, z) measured from the BALL CENTRE."""
        z = np.atleast_1d(z); rho = np.atleast_1d(rho)
        r = np.hypot(rho, z)
        cth = np.where(r > 0, z / np.where(r > 0, r, 1.0), 1.0)
        return self.internal(r, cth)

    # -- verification ----------------------------------------------------------
    def bc_residual(self, n_th=256):
        """Continuity of psi and dpsi/dr at r = a, evaluated from both sides."""
        c = self._cache
        th = np.linspace(0, np.pi, n_th)
        P = legendre_all(self.lmax, np.cos(th))
        w = (2 * self.l + 1) * (1j ** self.l)

        out_psi = (w * (c["j1"] + self.A * c["h1"])) @ P
        in_psi = (w * (self.B * c["j2"])) @ P
        out_d = (w * self.k1 * (c["dj1"] + self.A * c["dh1"])) @ P
        in_d = (w * self.k2 * (self.B * c["dj2"])) @ P

        s_psi = np.abs(out_psi).max()
        s_d = np.abs(out_d).max()
        return (np.abs(out_psi - in_psi).max() / s_psi,
                np.abs(out_d - in_d).max() / s_d)

    def flux_balance(self):
        """Net scalar Helmholtz flux through the sphere r = a must vanish.

        The conserved current of (nabla^2+k^2)psi = 0 is j ~ Im(psi* grad psi);
        for a lossless dielectric the net outward flux through the interface is
        zero.  Using orthogonality of P_l, Int P_l P_l' dOmega = 4pi/(2l+1)
        delta_ll', the total is a single sum over l.
        """
        c = self._cache
        tot_out = c["j1"] + self.A * c["h1"]
        dtot_out = self.k1 * (c["dj1"] + self.A * c["dh1"])
        w2 = (2 * self.l + 1)                     # |(2l+1) i^l|^2 / (2l+1) * ...
        # flux ~ sum_l (2l+1) * a^2 * Im( conj(psi_l) dpsi_l/dr )
        f = w2 * np.imag(np.conj(tot_out) * dtot_out)
        scale = np.sum(w2 * np.abs(tot_out) * np.abs(dtot_out))
        return np.abs(np.sum(f)) / scale


# ── the exact plane-interface limit, for cross-checking the BCs ──────────────
def scalar_plane_t(n1, n2, i1):
    """t and T for a plane interface from the SAME scalar BCs."""
    i2 = np.arcsin(np.clip(n1 * np.sin(i1) / n2, -1, 1))
    c1, c2 = np.cos(i1), np.cos(i2)
    t = 2 * n1 * c1 / (n1 * c1 + n2 * c2)
    r = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    T = (n2 * c2) / (n1 * c1) * t**2
    R = r**2
    return t, r, T, R


if __name__ == "__main__":
    LAM = 1.0                      # work in units of the vacuum wavelength
    N1, N2 = 1.0, 2.5

    print("=" * 74)
    print("EXACT SCALAR-HELMHOLTZ BALL  —  verification")
    print("=" * 74)

    # -- plane-interface limit: R + T = 1 from the scalar BCs -----------------
    i1 = np.radians([0, 20, 40, 60, 80])
    t, r, T, R = scalar_plane_t(N1, N2, i1)
    print("\n[a] plane-interface limit of the same boundary conditions")
    print(f"    max |R + T - 1| = {np.abs(R + T - 1).max():.3e}   (energy conservation)")

    # -- ball: BC residuals, flux, convergence --------------------------------
    print("\n[b] ball solver")
    print(f"    {'a/lam':>7} {'x1':>8} {'lmax':>6} {'|[psi]|':>11} "
          f"{'|[dpsi/dn]|':>13} {'flux':>11}")
    for a_over_lam in (5.0, 10.0, 20.0, 40.0, 80.0):
        b = ScalarBall(N1, N2, a_over_lam * LAM, LAM)
        e1, e2 = b.bc_residual()
        print(f"    {a_over_lam:7.1f} {b.x1:8.1f} {b.lmax:6d} "
              f"{e1:11.3e} {e2:13.3e} {b.flux_balance():11.3e}")

    # -- convergence in lmax ---------------------------------------------------
    print("\n[c] convergence in the truncation order (a = 20 lam)")
    ref = ScalarBall(N1, N2, 20 * LAM, LAM, lmax_pad=60)
    zf = (N2 / (N2 - N1)) * 20 * LAM - 20 * LAM      # paraxial focus from centre
    probe_z = np.linspace(zf - 3, zf + 3, 25)
    ref_v = ref.internal_cartesian(probe_z, np.zeros_like(probe_z))
    for pad in (0, 5, 15, 30):
        bb = ScalarBall(N1, N2, 20 * LAM, LAM, lmax_pad=pad)
        v = bb.internal_cartesian(probe_z, np.zeros_like(probe_z))
        print(f"    lmax_pad={pad:3d}  lmax={bb.lmax:5d}   "
              f"max rel diff vs pad=60 : {np.abs(v-ref_v).max()/np.abs(ref_v).max():.3e}")

    print(f"\n    paraxial internal focus at z = {zf:.3f} lam from the centre "
          f"(ball radius {20.0:.0f} lam)")
