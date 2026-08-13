"""
Muller second-kind formulation of the scalar Helmholtz transmission problem.

Why
---
The direct formulation mixes a second-kind operator (1/2 I + K) with a
first-kind one (S).  S has eigenvalues decaying like 1/l, so cond(M) grows
like N, the O(h) consistency error gets amplified to O(1), and v stops
converging.

Muller's trick is to add the interior and exterior equations so that only
DIFFERENCES of the two Green functions survive.  Since

    G1 - G2 = [exp(i k1 R) - exp(i k2 R)] / (4 pi R)
            = i(k1-k2)/(4 pi) - (k1^2-k2^2) R/(8 pi) + O(R^2)

the 1/R blows away: S1-S2 has a bounded kernel.  The same cancellation makes
K1-K2 and K1'-K2' smooth, and reduces the hypersingular T1-T2 from 1/R^3 to a
mere 1/R.  The result is I + compact:

    (I - dK ) u  +      dS   v  = u_inc
      -dT   u  + (I + dK') v  = v_inc

with cond independent of N.

Kernels, with r = x - y, R = |r|, f(R) = exp(ikR)/(4 pi R):

    S  : f
    K  : -f'(r^.n_y) = +f' W/R   ;  W  = (y-x).n_y   (grad_y gives the sign)
    K' : +f'(r^.n_x) = +f' W'/R  ;  W' = (x-y).n_x
    T  :  f'' W W'/R^2 - (f'/R) n_x.n_y - (f'/R) W W'/R^2

Stability: the differences must not be formed by subtracting nearly equal
singular numbers.  With

    P(k,R) = expm1(ikR)(2-2ikR-k^2R^2) - 2ikR - k^2R^2
    Q(k,R) = expm1(ikR)(ikR-1) + ikR

one has f'' = [2+P]/(4 pi R^3) and f'/R = [-1+Q]/(4 pi R^3), so the
differences are [P1-P2] and [Q1-Q2] over 4 pi R^3 — accurate down to R -> 0.

Only  -d(f'/R) n_x.n_y  stays singular in dT; its azimuthal integral is
analytic in terms of the complete elliptic integrals I1 and Id.

Performance notes (this rewrite)
--------------------------------
* exp(i k R) is evaluated ONCE per wavenumber and shared by P, Q and dG
  (the previous version computed expm1 four times where two suffice).
* The near-band and self-panel corrections are assembled in batched calls
  instead of a Python double loop over ~7N element pairs.
* Defaults calibrated against the exact ball rather than guessed:
  - n_phi scales with the electrical size.  The DIFFERENCE kernels are smooth
    in phi (no singularity), but they still OSCILLATE like exp(ik d) with a
    phase swing of up to 2 k rho_max across phi in [0, pi] — that is
    k rho_max / pi full cycles.  A fixed rule is exact at small ka and
    catastrophically wrong at large ka: at k2 a ~ 47 the surface error is
    43 % with 24 points and 7.8e-4 with 96.  The default now allots 8 Gauss
    points per oscillation cycle, with a floor of 24.
  - ng_smooth = 1 (midpoint).  Counter-intuitively this is ~3.6x MORE accurate
    than 2-point Gauss on the source panel, because midpoint quadrature paired
    with midpoint collocation has an even-order error cancellation that Gauss
    breaks.  Faster and better.
  - n_near = 5.  Widening the refined band is what removes the convergence
    floor at fine meshes, and it costs O(N), not O(N^2).
Net effect ~5x, which is what makes the kR sweep affordable.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

from .direct import Generator, sphere_generator  # noqa: F401

__all__ = ["assemble_muller", "solve_muller", "graded_nodes"]

_GAUSS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss(n):
    if n not in _GAUSS_CACHE:
        _GAUSS_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GAUSS_CACHE[n]


def graded_nodes(h, levels=14, ng=3):
    """Symmetric geometrically graded rule on [-h/2, h/2], clustering at 0.

    The self-panel kernel behaves like -C ln|s|; dyadic grading integrates
    that accurately without a log-weighted rule.  Returned in units of h so
    the result can be cached and rescaled.
    """
    xg, wg = _gauss(ng)
    s, w = [], []
    hi = 0.5
    for _ in range(levels):
        lo = hi / 2
        mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
        for sgn in (+1, -1):
            s.append(sgn * (mid + half * xg))
            w.append(half * wg)
        hi = lo
    return np.concatenate(s) * h, np.concatenate(w) * h


def _diff_kernels(Rf, Zf, nrf, nzf, rs, zs, nrs, nzs, k1, k2, cphi, wphi):
    """m=0 reduced DIFFERENCE kernel densities (incl. the rho' factor).

    Vectorised over an array of (field, source) pairs; returns dS, dK, dKp, dT.
    """
    A = (Rf - rs) ** 2 + (Zf - zs) ** 2 + 2 * Rf * rs
    B = 2 * Rf * rs
    Apb = (Rf + rs) ** 2 + (Zf - zs) ** 2
    m = np.clip(np.where(Apb > 0, 2 * B / Apb, 0.0), 0.0, 1.0 - 1e-15)
    sq = np.sqrt(np.maximum(Apb, 1e-300))
    I1 = 4 * ellipk(m) / sq
    Id = 4 * sq * ellipe(m)

    d = np.sqrt(np.maximum(A[:, None] - B[:, None] * cphi[None, :], 1e-300))

    # the only transcendental work: one exponential per wavenumber
    x1 = 1j * k1 * d
    x2 = 1j * k2 * d
    e1 = np.expm1(x1)
    e2 = np.expm1(x2)

    kd1sq = (k1 * d) ** 2
    kd2sq = (k2 * d) ** 2
    P1 = e1 * (2.0 - 2.0 * x1 - kd1sq) - 2.0 * x1 - kd1sq
    P2 = e2 * (2.0 - 2.0 * x2 - kd2sq) - 2.0 * x2 - kd2sq
    Q1 = e1 * (x1 - 1.0) + x1
    Q2 = e2 * (x2 - 1.0) + x2

    inv4pi_d3 = 1.0 / (4 * np.pi * d**3)
    dG = (e1 - e2) / (4 * np.pi * d)
    dfpp = (P1 - P2) * inv4pi_d3          # f1'' - f2''   (bounded)
    dfp_R = (Q1 - Q2) * inv4pi_d3         # (f1'-f2')/R   (~ 1/d)
    dfp = dfp_R * d                       # f1' - f2'     (bounded)

    c = cphi[None, :]
    W = (nrs[:, None] * (rs[:, None] - Rf[:, None] * c)
         + nzs[:, None] * (zs[:, None] - Zf[:, None]))
    Wp = (nrf[:, None] * (Rf[:, None] - rs[:, None] * c)
          + nzf[:, None] * (Zf[:, None] - zs[:, None]))
    nxny = nrf[:, None] * nrs[:, None] * c + (nzf * nzs)[:, None]

    pre = rs                                     # the rho' surface factor
    dS = pre * (dG @ wphi)
    dK = pre * ((dfp * W / d) @ wphi)
    dKp = pre * ((dfp * Wp / d) @ wphi)

    # dT: split off the only singular piece, -d(f'/R) * n_x.n_y
    sing = (k1**2 - k2**2) / (8 * np.pi)          # -d(f'/R) ~ +sing/d
    WWp_d2 = W * Wp / d**2
    smooth = (dfpp - dfp_R) * WWp_d2 - (dfp_R + sing / d) * nxny
    invB = 1.0 / np.maximum(B, 1e-300)
    analytic = sing * (nrf * nrs * (A * invB * I1 - Id * invB) + nzf * nzs * I1)
    dT = pre * ((smooth @ wphi) + analytic)
    return dS, dK, dKp, dT


def _batched(field_idx, src_idx, s_nodes, w_nodes, g, k1, k2, cphi, wphi,
             out, max_pts=None):
    """Accumulate kernel blocks for a list of (i, j) pairs with per-pair
    quadrature nodes ``s_nodes`` (shape (P, Q)) along the source panel."""
    rho, z, nr, nz = g.rho, g.z, g.n_rho, g.n_z
    trho = (g.rn[1:] - g.rn[:-1]) / g.h
    tz = (g.zn[1:] - g.zn[:-1]) / g.h
    Pn, Qn = s_nodes.shape
    if max_pts is None:
        max_pts = max(1, int(2.0e6 / cphi.size))
    per = max(1, max_pts // Qn)

    for a in range(0, Pn, per):
        b = min(a + per, Pn)
        ii, jj = field_idx[a:b], src_idx[a:b]
        s = s_nodes[a:b]                                   # (p, Q)
        rsn = (rho[jj][:, None] + s * trho[jj][:, None]).ravel()
        zsn = (z[jj][:, None] + s * tz[jj][:, None]).ravel()
        rep = np.repeat(np.arange(b - a), Qn)
        o = _diff_kernels(
            np.repeat(rho[ii], Qn), np.repeat(z[ii], Qn),
            np.repeat(nr[ii], Qn), np.repeat(nz[ii], Qn),
            rsn, zsn, np.repeat(nr[jj], Qn), np.repeat(nz[jj], Qn),
            k1, k2, cphi, wphi)
        w = w_nodes[a:b]
        for mat, vals in zip(out, o):
            mat[ii, jj] = np.einsum("pq,pq->p", vals.reshape(b - a, Qn), w)


def assemble_muller(g: Generator, k1, k2, n_phi=None, n_near=5, ng_near=8,
                    ng_smooth=1, block=48):
    """Assemble dS, dK, dK', dT.

    dS, dK, dK' have smooth kernels, so two Gauss points per source panel
    already give O(h^2).  dT keeps a log singularity, so its self panel uses
    the graded rule and its near band a finer Gauss rule.

    ``n_phi=None`` scales the azimuthal rule with the electrical size:
    the phi-integrand oscillates through k rho_max / pi cycles, and 8 Gauss
    points per cycle resolve it (verified against the exact ball).
    """
    N = g.N
    if n_phi is None:
        kmax = max(abs(k1), abs(k2))
        n_phi = max(24, int(np.ceil(8.0 * kmax * g.rho.max() / np.pi)))
    xg, wg = _gauss(n_phi)
    phi = 0.5 * np.pi * (xg + 1.0)
    wphi = 0.5 * np.pi * wg * 2.0
    cphi = np.cos(phi)

    rho, z, nr, nz, h = g.rho, g.z, g.n_rho, g.n_z, g.h
    trho = (g.rn[1:] - g.rn[:-1]) / h
    tz = (g.zn[1:] - g.zn[:-1]) / h

    mats = [np.zeros((N, N), complex) for _ in range(4)]

    # ---- base pass: Gauss nodes along every source panel --------------------
    xs, ws = _gauss(ng_smooth)
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        nb = i1 - i0
        acc = [np.zeros((nb, N), complex) for _ in range(4)]
        Rf = np.repeat(rho[i0:i1], N); Zf = np.repeat(z[i0:i1], N)
        nrf = np.repeat(nr[i0:i1], N); nzf = np.repeat(nz[i0:i1], N)
        nrs_t = np.tile(nr, nb); nzs_t = np.tile(nz, nb)
        for xq, wq in zip(xs, ws):
            sq_ = 0.5 * h * xq
            out = _diff_kernels(Rf, Zf, nrf, nzf,
                                np.tile(rho + sq_ * trho, nb),
                                np.tile(z + sq_ * tz, nb),
                                nrs_t, nzs_t, k1, k2, cphi, wphi)
            wgt = (0.5 * h * wq)[None, :]
            for a, o in zip(acc, out):
                a += o.reshape(nb, N) * wgt
        for mat, a in zip(mats, acc):
            mat[i0:i1] = a

    # ---- near band (batched): redo those pairs with proper quadrature -------
    ii, jj = [], []
    for i in range(N):
        for j in range(max(0, i - n_near), min(N, i + n_near + 1)):
            if j != i:
                ii.append(i); jj.append(j)
    if ii:
        ii = np.array(ii); jj = np.array(jj)
        xg2, wg2 = _gauss(ng_near)
        s_nodes = 0.5 * h[jj][:, None] * xg2[None, :]
        w_nodes = 0.5 * h[jj][:, None] * wg2[None, :]
        _batched(ii, jj, s_nodes, w_nodes, g, k1, k2, cphi, wphi, mats)

    # ---- self panels (batched): graded rule ---------------------------------
    idx = np.arange(N)
    s_unit, w_unit = graded_nodes(1.0)
    s_self = h[:, None] * s_unit[None, :]
    w_self = h[:, None] * w_unit[None, :]
    _batched(idx, idx, s_self, w_self, g, k1, k2, cphi, wphi, mats)

    return tuple(mats)


def solve_muller(g: Generator, k1, k2, u_inc, v_inc, **kw):
    """Second-kind transmission solve.  Returns u, v and the system matrix."""
    dS, dK, dKp, dT = assemble_muller(g, k1, k2, **kw)
    N = g.N
    I = np.eye(N)
    M = np.zeros((2 * N, 2 * N), complex)
    M[:N, :N] = I - dK
    M[:N, N:] = dS
    M[N:, :N] = -dT
    M[N:, N:] = I + dKp
    sol = np.linalg.solve(M, np.concatenate([u_inc, v_inc]))
    return sol[:N], sol[N:], M
