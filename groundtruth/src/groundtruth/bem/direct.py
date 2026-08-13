"""
BOR-BEM: boundary elements for the SCALAR Helmholtz transmission problem
on a body of revolution.

Formulation
-----------
Exterior Omega1 (k1) with incident field; interior Omega2 (k2); interface S
with outward normal n (pointing out of Omega2).  Unknowns are the Cauchy data

    u = psi|_S ,     v = dpsi/dn|_S

which are the SAME from both sides -- that is the scalar transmission
condition [psi] = [dpsi/dn] = 0.  Green's third identity applied on each side
and taken to the boundary gives

    interior :   (1/2 I + D2) u - S2 v = 0
    exterior :   (1/2 I - D1) u + S1 v = u_inc

The exterior right-hand side collapses to u_inc alone because the incident
field is source-free inside Omega2, so (1/2 I + D1) u_inc = S1 v_inc.  That
identity is worth keeping in mind: it means v_inc never has to be assembled.

    (S_j f)(x) = Int_S G_j(x,y) f(y) dS_y ,   G_j = exp(i k_j R)/(4 pi R)
    (D_j f)(x) = Int_S dG_j(x,y)/dn_y f(y) dS_y                  (principal value)

Azimuthal reduction (m = 0)
---------------------------
With an axisymmetric incident field, u and v depend only on arclength.  The
azimuthal integral is done once, analytically where it is singular.  Writing
d^2 = A - B cos(phi) with

    A = (rho-rho')^2 + (z-z')^2 + 2 rho rho' ,   B = 2 rho rho'
    A + B = (rho+rho')^2 + (z-z')^2 ,            A - B = Delta^2 ,
    m = 2B/(A+B)

the three integrals that carry every singularity are exact:

    I1 = Int dphi / d      = 4 K(m) / sqrt(A+B)
    I3 = Int dphi / d^3    = 4 E(m) / (Delta^2 sqrt(A+B))
    Id = Int dphi  d       = 4 sqrt(A+B) E(m)

The single-layer kernel is split as  exp(ikd)/d = 1/d + [exp(ikd)-1]/d, the
second piece being bounded (-> ik).  The double layer needs one more term
subtracted, because [(ikd-1)exp(ikd)]/d^3 = -1/d^3 - k^2/(2d) + O(1); with both
static pieces removed the remainder tends to -i k^3/3 and integrates cleanly.

Self term
---------
On the diagonal I1 diverges logarithmically.  Integrating the leading
behaviour I1 ~ (2/rho) ln(8 rho/|s|) across an element of length h,

    Int_elem I1 ds = (2h/rho) [ ln(16 rho / h) + 1 ]

which is used in place of h*I1.  The double layer keeps a finite curvature
contribution there: C1 = O(Delta^2) cancels the Delta^2 in I3 and leaves
-(kappa_m - kappa_phi)/rho, with kappa_phi = n_rho/rho the azimuthal principal
curvature.  It vanishes identically on a sphere, which is convenient -- the
sphere validation is then insensitive to it, and mesh refinement tests it
separately on the ovoid.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipk, ellipe


# ── geometry ──────────────────────────────────────────────────────────────────
class Generator:
    """A meridian curve, discretised into straight elements with midpoint nodes."""

    def __init__(self, rho_nodes, z_nodes):
        self.rn = np.asarray(rho_nodes, float)
        self.zn = np.asarray(z_nodes, float)
        drho = np.diff(self.rn)
        dz = np.diff(self.zn)
        self.h = np.hypot(drho, dz)                    # element lengths
        self.rho = 0.5 * (self.rn[1:] + self.rn[:-1])  # collocation points
        self.z = 0.5 * (self.zn[1:] + self.zn[:-1])
        # unit tangent, then normal rotated -90 deg  ->  outward for a curve
        # traced with the interior on the left
        t_rho, t_z = drho / self.h, dz / self.h
        self.n_rho, self.n_z = t_z, -t_rho
        self.N = self.h.size

    def curvatures(self):
        """Meridional and azimuthal principal curvatures at the collocation points."""
        s = np.concatenate([[0.0], np.cumsum(self.h)])
        sc = 0.5 * (s[1:] + s[:-1])
        drho = np.gradient(self.rho, sc)
        dz = np.gradient(self.z, sc)
        d2rho = np.gradient(drho, sc)
        d2z = np.gradient(dz, sc)
        kappa_m = drho * d2z - dz * d2rho
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa_phi = np.where(self.rho > 1e-12, self.n_rho / self.rho, 0.0)
        return kappa_m, kappa_phi

    def flip_normal(self):
        self.n_rho, self.n_z = -self.n_rho, -self.n_z
        return self


def sphere_generator(a, n_elem):
    """Meridian of a sphere of radius a, traced from the +z pole to the -z pole."""
    th = np.linspace(0.0, np.pi, n_elem + 1)
    return Generator(a * np.sin(th), a * np.cos(th))


# ── kernel assembly ───────────────────────────────────────────────────────────
def assemble(g: Generator, k, n_phi=96, block=64):
    """Single- and double-layer matrices for wavenumber k.

    Returns S, D with  (S f)_i = sum_j S_ij f_j  ~  Int_S G f dS  and likewise
    for D with dG/dn_y.  The 1/2-jump is NOT included.
    """
    N = g.N
    S = np.zeros((N, N), dtype=complex)
    D = np.zeros((N, N), dtype=complex)

    # Gauss-Legendre on [0, pi]; the integrand is even in phi so double it.
    xg, wg = np.polynomial.legendre.leggauss(n_phi)
    phi = 0.5 * np.pi * (xg + 1.0)
    wphi = 0.5 * np.pi * wg * 2.0
    cphi = np.cos(phi)

    rho, z = g.rho, g.z
    nr, nz, h = g.n_rho, g.n_z, g.h
    kappa_m, kappa_phi = g.curvatures()

    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        R = rho[i0:i1, None]          # field points (block, 1)
        Z = z[i0:i1, None]

        A = (R - rho[None, :]) ** 2 + (Z - z[None, :]) ** 2 + 2 * R * rho[None, :]
        B = 2 * R * rho[None, :]
        Apb = (R + rho[None, :]) ** 2 + (Z - z[None, :]) ** 2
        D2 = A - B                                    # Delta^2
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(Apb > 0, 2 * B / Apb, 0.0)
        m = np.clip(m, 0.0, 1.0 - 1e-15)

        sq = np.sqrt(np.maximum(Apb, 1e-300))
        Km, Em = ellipk(m), ellipe(m)
        I1 = 4 * Km / sq
        with np.errstate(divide="ignore", invalid="ignore"):
            I3 = np.where(D2 > 0, 4 * Em / (np.maximum(D2, 1e-300) * sq), 0.0)
        Id = 4 * sq * Em

        # d(phi) for the smooth remainders: (block, N, n_phi)
        d = np.sqrt(np.maximum(A[..., None] - B[..., None] * cphi, 0.0))
        eikd = np.exp(1j * k * d)

        # -- single layer ------------------------------------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            rem_S = np.where(d > 0, (eikd - 1.0) / np.maximum(d, 1e-300), 1j * k)
        smoothS = np.einsum("ijp,p->ij", rem_S, wphi)
        Sblk = (rho[None, :] / (4 * np.pi)) * (I1 + smoothS)

        # -- double layer ------------------------------------------------------
        # W = C1 + C2 d^2 ,  C1 = (y-x).n' - n_rho' Delta^2/(2 rho')
        bracket = (nr[None, :] * (rho[None, :] - R)
                   + nz[None, :] * (z[None, :] - Z))
        with np.errstate(divide="ignore", invalid="ignore"):
            C1 = bracket - np.where(rho[None, :] > 0,
                                    nr[None, :] * D2 / (2 * np.maximum(rho[None, :], 1e-300)),
                                    0.0)
            C2 = np.where(B > 0, nr[None, :] * R / np.maximum(B, 1e-300), 0.0)

        W = C1[..., None] + C2[..., None] * d**2
        with np.errstate(divide="ignore", invalid="ignore"):
            rem_D = np.where(
                d > 0,
                ((1j * k * d - 1.0) * eikd + 1.0 + 0.5 * (k * d) ** 2)
                / np.maximum(d**3, 1e-300),
                -1j * k**3 / 3.0,
            )
        smoothD = np.einsum("ijp,p->ij", W * rem_D, wphi)
        static = -(C1 * I3 + C2 * I1) - 0.5 * k**2 * (C1 * I1 + C2 * Id)
        Dblk = (rho[None, :] / (4 * np.pi)) * (static + smoothD)

        # -- multiply by the element length (pulse basis, midpoint rule) -------
        S[i0:i1] = Sblk * h[None, :]
        D[i0:i1] = Dblk * h[None, :]

        # -- replace the diagonal with the log-integrated self term ------------
        idx = np.arange(i0, i1)
        loc = idx - i0
        rr, hh = rho[idx], h[idx]
        Llog = np.where(rr > 1e-12, (2 * hh / np.maximum(rr, 1e-300))
                        * (np.log(16 * np.maximum(rr, 1e-300) / hh) + 1.0), 0.0)
        S[idx, idx] = (rr / (4 * np.pi)) * (
            Llog + hh * smoothS[loc, idx])
        curv = -(kappa_m[idx] - kappa_phi[idx]) / np.maximum(rr, 1e-300)
        D[idx, idx] = (rr / (4 * np.pi)) * (
            -C2[loc, idx] * Llog
            - 0.5 * k**2 * (C2[loc, idx] * Id[loc, idx]) * hh
            + smoothD[loc, idx] * hh
            + curv * hh)
    return S, D


def solve_transmission(g: Generator, k1, k2, u_inc, n_phi=96):
    """Solve for the Cauchy data (u, v) on the interface."""
    S1, D1 = assemble(g, k1, n_phi=n_phi)
    S2, D2 = assemble(g, k2, n_phi=n_phi)
    N = g.N
    I = np.eye(N)
    M = np.zeros((2 * N, 2 * N), dtype=complex)
    rhs = np.zeros(2 * N, dtype=complex)
    M[:N, :N] = 0.5 * I + D2
    M[:N, N:] = -S2
    M[N:, :N] = 0.5 * I - D1
    M[N:, N:] = S1
    rhs[N:] = u_inc
    sol = np.linalg.solve(M, rhs)
    return sol[:N], sol[N:]


# ── field evaluation away from the boundary ───────────────────────────────────
def field_interior(g, k2, u, v, pts_rho, pts_z, n_phi=160):
    """psi(x) = Int_S [ G2 v - u dG2/dn ] dS   for x strictly inside Omega2."""
    xg, wg = np.polynomial.legendre.leggauss(n_phi)
    phi = 0.5 * np.pi * (xg + 1.0)
    wphi = 0.5 * np.pi * wg * 2.0
    cphi = np.cos(phi)
    out = np.empty(len(pts_rho), dtype=complex)
    for q, (R, Z) in enumerate(zip(pts_rho, pts_z)):
        d = np.sqrt(np.maximum(
            (R - g.rho[:, None] * cphi[None, :]) ** 2
            + (g.rho[:, None] * np.sin(phi)[None, :]) ** 2
            + (Z - g.z[:, None]) ** 2, 1e-300))
        G = np.exp(1j * k2 * d) / (4 * np.pi * d)
        # (y - x).n'   with x = (R,0,Z), y = (rho' cos, rho' sin, z')
        Wn = (g.n_rho[:, None] * (g.rho[:, None] - R * cphi[None, :])
              + g.n_z[:, None] * (g.z[:, None] - Z))
        dGdn = (1j * k2 - 1.0 / d) * G / d * Wn
        meas = (g.rho * g.h)[:, None]
        integ = (G * v[:, None] - dGdn * u[:, None]) * meas
        out[q] = np.einsum("jp,p->", integ, wphi)
    return out


# ── accurate assembly: proper source-element quadrature ──────────────────────
def _kern(Rf, Zf, rs, zs, nrs, nzs, k, cphi, wphi, flat_self=False):
    """m=0 reduced kernel DENSITIES (per unit source arclength), incl. the rho'
    factor.  Vectorised over an array of (field, source) pairs.

    ``flat_self`` asserts that field and source lie on the SAME straight panel,
    so (y-x).n' vanishes identically -- which removes the catastrophic
    cancellation in C1 that would otherwise wreck a graded quadrature.
    """
    A = (Rf - rs) ** 2 + (Zf - zs) ** 2 + 2 * Rf * rs
    B = 2 * Rf * rs
    Apb = (Rf + rs) ** 2 + (Zf - zs) ** 2
    D2 = np.maximum(A - B, 0.0)
    m = np.clip(np.where(Apb > 0, 2 * B / Apb, 0.0), 0.0, 1.0 - 1e-15)
    sq = np.sqrt(np.maximum(Apb, 1e-300))

    I1 = 4 * ellipk(m) / sq
    I3 = 4 * ellipe(m) / (np.maximum(D2, 1e-300) * sq)
    Id = 4 * sq * ellipe(m)

    d = np.sqrt(np.maximum(A[:, None] - B[:, None] * cphi[None, :], 0.0))
    eikd = np.exp(1j * k * d)

    rem_S = np.where(d > 0, (eikd - 1.0) / np.maximum(d, 1e-300), 1j * k)
    Ks = (rs / (4 * np.pi)) * (I1 + rem_S @ wphi)

    bracket = 0.0 if flat_self else (nrs * (rs - Rf) + nzs * (zs - Zf))
    C1 = bracket - nrs * D2 / (2 * np.maximum(rs, 1e-300))
    C2 = np.where(B > 0, nrs * Rf / np.maximum(B, 1e-300), 0.0)

    W = C1[:, None] + C2[:, None] * d**2
    rem_D = np.where(d > 0,
                     ((1j * k * d - 1.0) * eikd + 1.0 + 0.5 * (k * d) ** 2)
                     / np.maximum(d**3, 1e-300),
                     -1j * k**3 / 3.0)
    static = -(C1 * I3 + C2 * I1) - 0.5 * k**2 * (C1 * I1 + C2 * Id)
    Kd = (rs / (4 * np.pi)) * (static + (W * rem_D) @ wphi)
    return Ks, Kd


def _graded_nodes(h, levels=14, ng=3):
    """Symmetric geometrically graded quadrature on [-h/2, h/2], clustering at 0.

    The self-panel kernel behaves like -C ln|s|; geometric grading integrates
    that to near machine precision without needing a log-weighted rule.
    """
    xg, wg = np.polynomial.legendre.leggauss(ng)
    s, w = [], []
    hi = h / 2
    for _ in range(levels):
        lo = hi / 2
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        for sgn in (+1, -1):
            s.append(sgn * (mid + half * xg))
            w.append(half * wg)
        hi = lo
    # innermost sliver, integrated as if smooth (its weight is ~h*2^-levels)
    return np.concatenate(s), np.concatenate(w)


def assemble2(g: Generator, k, n_phi=96, n_near=3, ng_near=8, block=48):
    """Assemble S and D with the singular and near-singular panels integrated
    properly in the source variable, instead of a midpoint rule."""
    N = g.N
    xg, wg = np.polynomial.legendre.leggauss(n_phi)
    phi = 0.5 * np.pi * (xg + 1.0)
    wphi = 0.5 * np.pi * wg * 2.0
    cphi = np.cos(phi)

    rho, z, nr, nz, h = g.rho, g.z, g.n_rho, g.n_z, g.h
    # panel end points, to place quadrature nodes along each panel
    rn, zn = g.rn, g.zn
    trho = (rn[1:] - rn[:-1]) / h
    tz = (zn[1:] - zn[:-1]) / h

    S = np.zeros((N, N), dtype=complex)
    D = np.zeros((N, N), dtype=complex)

    # ---- far field: single midpoint node per panel --------------------------
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        nb = i1 - i0
        Rf = np.repeat(rho[i0:i1], N)
        Zf = np.repeat(z[i0:i1], N)
        rs = np.tile(rho, nb); zs = np.tile(z, nb)
        nrs = np.tile(nr, nb); nzs = np.tile(nz, nb)
        Ks, Kd = _kern(Rf, Zf, rs, zs, nrs, nzs, k, cphi, wphi)
        S[i0:i1] = (Ks.reshape(nb, N)) * h[None, :]
        D[i0:i1] = (Kd.reshape(nb, N)) * h[None, :]

    # ---- near band and self panel: proper quadrature along the source panel -
    xg2, wg2 = np.polynomial.legendre.leggauss(ng_near)
    for i in range(N):
        lo, hi = max(0, i - n_near), min(N, i + n_near + 1)
        for j in range(lo, hi):
            if j == i:
                sq_, wq_ = _graded_nodes(h[j])
                flat = True
            else:
                sq_ = 0.5 * h[j] * xg2
                wq_ = 0.5 * h[j] * wg2
                flat = False
            rs = rho[j] + sq_ * trho[j]
            zs = z[j] + sq_ * tz[j]
            P = rs.size
            Ks, Kd = _kern(np.full(P, rho[i]), np.full(P, z[i]), rs, zs,
                           np.full(P, nr[j]), np.full(P, nz[j]),
                           k, cphi, wphi, flat_self=flat)
            S[i, j] = Ks @ wq_
            D[i, j] = Kd @ wq_
    return S, D


def solve_transmission2(g: Generator, k1, k2, u_inc, v_inc, n_phi=96, **kw):
    """Transmission solve with the accurate assembly and a discretely
    consistent right-hand side (same operators on both sides)."""
    S1, D1 = assemble2(g, k1, n_phi=n_phi, **kw)
    S2, D2 = assemble2(g, k2, n_phi=n_phi, **kw)
    N = g.N
    I = np.eye(N)
    M = np.zeros((2 * N, 2 * N), dtype=complex)
    M[:N, :N] = 0.5 * I + D2
    M[:N, N:] = -S2
    M[N:, :N] = 0.5 * I - D1
    M[N:, N:] = S1
    rhs = np.concatenate([np.zeros(N, dtype=complex),
                          0.5 * u_inc - D1 @ u_inc + S1 @ v_inc])
    sol = np.linalg.solve(M, rhs)
    return sol[:N], sol[N:], M
