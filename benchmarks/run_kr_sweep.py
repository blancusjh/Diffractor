"""Fase 1: the kR sweep.

Solve the stigmatic ovoid as a wave-equation BVP at four scales, measure the
pupil on the exit reference sphere, and test whether the residual against the
ray prediction  P = t_s d2/d1  falls like 1/(kR).

Scaling all lengths together holds NA fixed at 0.75 and changes only kR, so
any systematic trend is finite-wavelength physics, not geometry.
"""

import time

import numpy as np

from groundtruth.bem import field_interior, ovoid_body, point_source_cauchy
from groundtruth.bem import solve_muller

LAM = 1.0
N1, N2 = 1.0, 1.5
K0 = 2 * np.pi / LAM
K1, K2 = K0 * N1, K0 * N2
PPL = 32.0
R2_FRAC = 0.275


def run(zi, ppl=PPL, n_th=90):
    zo = -2.0 * zi
    g = ovoid_body(N1, N2, zo, zi, LAM, ppl=ppl)
    m = g.meta
    ov = m["ov"]
    g0 = m["g0"]
    R2 = R2_FRAC * zi

    u_inc, v_inc = point_source_cauchy(g, K1, zo)
    t0 = time.time()
    u, v, M = solve_muller(g, K1, K2, u_inc, v_inc)
    t_solve = time.time() - t0

    th_max = g0["th2"][-1]
    th = np.linspace(0.02 * th_max, 0.985 * th_max, n_th)
    psi = field_interior(g, K2, u, v, R2 * np.sin(th), zi - R2 * np.cos(th),
                         n_phi=200)
    A_meas = psi * R2 * np.exp(1j * K2 * R2)

    # absolute ray prediction: A = A0 t_s d2/d1, A0 = 1/4pi, no fitted scale
    ts = 2 * N1 * g0["cos_i1"] / (N1 * g0["cos_i1"] + N2 * g0["cos_i2"])
    P_ray = ts * g0["d2"] / g0["d1"]
    A_ray = (1.0 / (4 * np.pi)) * np.interp(th, g0["th2"], P_ray)

    ph = np.angle(A_meas * np.exp(-1j * K0 * ov.C))
    am = np.abs(A_meas)
    inner = th < 0.9 * th_max

    # ripple-limited error bar on the edge/axis factor
    r = am / A_ray
    trend = np.polyval(np.polyfit(th, r, 3), th)
    rip = r - trend
    sig_out = np.std(rip[th > 0.66 * th_max])
    sig_in = np.std(rip[th < 0.34 * th_max])
    meas = am[-1] / am[0]
    rel = np.hypot(sig_out / r[-1], sig_in / r[0])

    E = lambda a: 2 * np.pi * np.trapezoid(a**2 * np.sin(th), th)
    return dict(zi=zi, N=g.N, kR=K2 * R2, R2=R2, t=t_solve, th=th, thm=th_max,
                am=am, ar=A_ray, ph=ph, NA=m["NA"],
                edge_meas=meas, edge_err=meas * rel, edge_ray=A_ray[-1] / A_ray[0],
                ratio=np.mean(r[inner]), scatter=np.std(rip),
                phase=np.ptp(ph[inner]) / (2 * np.pi),
                E_meas=E(am), E_ray=E(A_ray), E_flat=E(np.full_like(A_ray, A_ray[0])))


if __name__ == "__main__":
    print("=" * 78)
    print("kR SWEEP — measured pupil vs  P = t_s d2/d1   (NA 0.75 held fixed)")
    print("=" * 78)
    print(f"  {'zi/λ':>5} {'N':>6} {'k2R2':>7} {'edge meas':>17} {'ray':>7} "
          f"{'ratio':>7} {'scat':>6} {'phase':>7} {'E/Eray':>7} {'t[s]':>6}")
    res = []
    for zi in (8.0, 12.0, 16.0, 24.0):
        r = run(zi)
        res.append(r)
        print(f"  {r['zi']:5.0f} {r['N']:6d} {r['kR']:7.1f} "
              f"{r['edge_meas']:8.4f} ± {r['edge_err']:6.4f} {r['edge_ray']:7.4f} "
              f"{r['ratio']:7.4f} {r['scatter']:6.3f} {r['phase']:7.4f} "
              f"{r['E_meas']/r['E_ray']:7.4f} {r['t']:6.0f}")

    print(f"\n  constant-amplitude assumption: edge/axis = 1.0000, "
          f"E/E_ray = {res[0]['E_flat']/res[0]['E_ray']:.3f}")

    # does the residual fall like 1/(kR)?
    kR = np.array([r["kR"] for r in res])
    dev = np.abs(np.array([r["edge_meas"] for r in res])
                 - np.array([r["edge_ray"] for r in res]))
    p = np.polyfit(np.log(kR), np.log(dev), 1)
    print(f"\n  |edge_meas − edge_ray| ~ (kR)^{p[0]:.2f}   "
          f"(geometrical optics predicts −1)")

    np.savez("/home/claude/monorepo/benchmarks/golden/kr_sweep.npz",
             **{f"{k}{int(r['zi'])}": r[k] for r in res
                for k in ("th", "am", "ar", "ph", "thm", "kR", "R2", "N")},
             zis=np.array([r["zi"] for r in res]),
             edge_meas=np.array([r["edge_meas"] for r in res]),
             edge_err=np.array([r["edge_err"] for r in res]),
             edge_ray=np.array([r["edge_ray"] for r in res]),
             ratio=np.array([r["ratio"] for r in res]),
             scatter=np.array([r["scatter"] for r in res]),
             phase=np.array([r["phase"] for r in res]),
             kRs=kR, E_meas=np.array([r["E_meas"] for r in res]),
             E_ray=np.array([r["E_ray"] for r in res]),
             E_flat=np.array([r["E_flat"] for r in res]),
             slope=p[0])
    print("\n  saved benchmarks/golden/kr_sweep.npz")
