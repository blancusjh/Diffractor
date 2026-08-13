"""Compute the scalar field maps used in the field figures.

The stigmatic body is solved once with a lossless interior (n2 real).
The body has to be closed for the boundary-integral formulation; a closed
lossless dielectric is a resonator, so the field contains both the direct
converging beam and trapped modes.  The tangent-plane model on the cap alone
isolates the direct beam without multiple scattering.
"""

import time
from pathlib import Path

import numpy as np
from scipy.special import j0

from groundtruth.bem import (field_map, ovoid_body, point_source_cauchy,
                             solve_muller)
from groundtruth.exact import ScalarBall

LAM = 1.0
OUT = str(Path(__file__).resolve().parent / "golden") + "/"


# ══ 1. stigmatic ovoid, solved by Muller BOR-BEM ══════════════════════════════
def ovoid_fields(zi=8.0, ppl=32.0, n1=1.0, n2=1.5):
    zo, k0 = -2.0 * zi, 2 * np.pi / LAM
    k1, k2 = k0 * n1, k0 * n2
    g = ovoid_body(n1, n2, zo, zi, LAM, ppl=ppl)
    m, g0, ov = g.meta, g.meta["g0"], g.meta["ov"]
    u_inc, v_inc = point_source_cauchy(g, k1, zo)
    t0 = time.time()
    u, v, _ = solve_muller(g, k1, k2, u_inc, v_inc)
    print(f"  ovoid zi={zi}λ: N={g.N}, solved in {time.time()-t0:.0f}s")

    r_rim, z_rim, z_end, z_apex = (m["r_rim"], m["z_rim"], m["z_end"], m["z_apex"])
    cap_r, cap_z = g0["r"], g0["z"]

    def is_inside(R, Z):
        sag = np.interp(np.abs(R), cap_r, cap_z, left=cap_z[0], right=np.inf)
        cone_r = np.where(Z <= z_end, r_rim,
                          r_rim * np.clip((z_apex - Z) / (z_apex - z_end), 0, 1))
        return (np.abs(R) <= cone_r) & (Z >= sag) & (Z <= z_apex)

    def inc_pair(rho_s, z_s):
        d = np.hypot(rho_s, z_s - zo)
        us = np.exp(1j * k1 * d) / (4 * np.pi * d)
        rn = (rho_s * g.n_rho + (z_s - zo) * g.n_z) / d
        return us, (1j * k1 - 1.0 / d) * us * rn

    def evaluate(R, Z):
        """Total field at arbitrary points, interior or exterior."""
        R, Z = np.asarray(R, float), np.asarray(Z, float)
        out = np.zeros(R.shape, complex)
        ins = is_inside(R, Z)
        if ins.any():
            out[ins] = field_map(g, k2, u, v, R[ins], Z[ins], n_phi=32)
        ext = ~ins
        if ext.any():
            d = np.hypot(R[ext], Z[ext] - zo)
            out[ext] = (np.exp(1j * k1 * d) / (4 * np.pi * d)
                        + field_map(g, k1, u, v, R[ext], Z[ext], n_phi=32,
                                    exterior=True, incident=inc_pair))
        return out, ins

    # -- longitudinal section --------------------------------------------------
    zg = np.linspace(-3.0, z_apex + 1.5, 300)
    rg = np.linspace(0.0, r_rim + 2.2, 110)
    RG, ZG = np.meshgrid(rg, zg, indexing="ij")
    t0 = time.time()
    psi, inside = evaluate(RG, ZG)
    print(f"    longitudinal map ({RG.size} pts) in {time.time()-t0:.0f}s")

    # -- transverse focal plane, kept strictly inside the body -----------------
    airy = 0.61 * LAM / m["NA"]
    r_tr = np.linspace(0.0, 0.85 * r_rim, 190)
    psi_tr = field_map(g, k2, u, v, r_tr, np.full_like(r_tr, zi), n_phi=64)

    # -- through focus, kept clear of the cap and of the closing cone ----------
    z_ax = np.linspace(z_rim + 0.35, min(zi + 6.0, z_end - 0.5), 320)
    psi_ax = field_map(g, k2, u, v, np.zeros_like(z_ax), z_ax, n_phi=64)

    # -- entrance sphere S1: total field (incident + reflected) ----------------
    R1 = 0.5 * abs(zo)
    th1 = np.linspace(0.0, 0.995, 140) * g0["th1"][-1]
    p1_rho, p1_z = R1 * np.sin(th1), zo + R1 * np.cos(th1)
    psi1, _ = evaluate(p1_rho, p1_z)
    A1 = psi1 * R1 * np.exp(-1j * k1 * R1)
    d1p = np.hypot(p1_rho, p1_z - zo)
    A1_inc = (np.exp(1j * k1 * d1p) / (4 * np.pi * d1p)) * R1 * np.exp(-1j * k1 * R1)

    # -- exit sphere S2 --------------------------------------------------------
    R2 = 1.5
    th2 = np.linspace(0.0, 1.30, 160) * g0["th2"][-1]
    p2_rho, p2_z = R2 * np.sin(th2), zi - R2 * np.cos(th2)
    psi2 = field_map(g, k2, u, v, p2_rho, p2_z, n_phi=96)
    A2 = psi2 * R2 * np.exp(1j * k2 * R2)

    # ray pupil, and the same pupil propagated diffractively (Debye-Wolf)
    ts = 2 * n1 * g0["cos_i1"] / (n1 * g0["cos_i1"] + n2 * g0["cos_i2"])
    P_ray = ts * g0["d2"] / g0["d1"] / (4 * np.pi)
    A2_ray = np.interp(th2, g0["th2"], P_ray, right=0.0)
    meas = P_ray * g0["w2"]

    def debye(rr, dz):
        return np.array([-1j * k2 * np.trapezoid(
            j0(k0 * n2 * r_ * g0["sin2"]) * np.exp(1j * k2 * z_ * g0["cos2"]) * meas,
            g0["d1"]) for r_, z_ in zip(np.atleast_1d(rr), np.atleast_1d(dz))])

    A2_dw = debye(p2_rho, p2_z - zi) * R2 * np.exp(1j * k2 * R2)
    dw_tr = debye(r_tr, np.zeros_like(r_tr))
    dw_ax = debye(np.zeros_like(z_ax), z_ax - zi)

    np.savez(OUT + "ovoid_fields.npz",
             rg=rg, zg=zg, psi=psi, inside=inside, cap_r=cap_r, cap_z=cap_z,
             r_rim=r_rim, z_rim=z_rim, z_end=z_end, z_apex=z_apex,
             zi=zi, zo=zo, NA=m["NA"], N=g.N, airy=airy, n1=n1, n2=n2,
             th1max=g0["th1"][-1], th2max=g0["th2"][-1],
             r_tr=r_tr, psi_tr=psi_tr, dw_tr=dw_tr,
             z_ax=z_ax, psi_ax=psi_ax, dw_ax=dw_ax,
             th1=th1, A1=A1, A1_inc=A1_inc, R1=R1,
             th2=th2, A2=A2, A2_ray=A2_ray, A2_dw=A2_dw, R2=R2,
             opl=ov.C, k0=k0, k2=k2, ppl=ppl)
    print("    -> ovoid_fields.npz")


# ══ 1b. the same body, tangent-plane model instead of a solver ═══════════════
def ovoid_po(zi=8.0, ppl=32.0, n1=1.0, n2=1.5):
    """Local plane-wave transmission on the cap, propagated by the exact
    Helmholtz surface representation.  No solver, no shroud: only the cap
    carries a field."""
    from diffractor.scattering import point_source_transmission

    zo, k0 = -2.0 * zi, 2 * np.pi / LAM
    k1, k2 = k0 * n1, k0 * n2
    g = ovoid_body(n1, n2, zo, zi, LAM, ppl=ppl)
    m, g0 = g.meta, g.meta["g0"]
    u, v = point_source_transmission(g.rho, g.z, g.n_rho, g.n_z, zo,
                                     n1, n2, k1, k2)
    cap = np.zeros(g.N, bool)
    cap[:m["n_cap"]] = True
    u, v = u * cap, v * cap

    D = np.load(OUT + "ovoid_fields.npz")      # reuse the BEM sampling grids
    rg, zg = D["rg"], D["zg"]
    RG, ZG = np.meshgrid(rg, zg, indexing="ij")
    inside = D["inside"]
    psi = np.zeros(RG.shape, complex)
    t0 = time.time()
    psi[inside] = field_map(g, k2, u, v, RG[inside], ZG[inside], n_phi=32)
    print(f"    PO longitudinal map in {time.time()-t0:.0f}s")

    r_tr, z_ax = D["r_tr"], D["z_ax"]
    psi_tr = field_map(g, k2, u, v, r_tr, np.full_like(r_tr, zi), n_phi=64)
    psi_ax = field_map(g, k2, u, v, np.zeros_like(z_ax), z_ax, n_phi=64)
    th2, R2 = D["th2"], float(D["R2"])
    A2 = field_map(g, k2, u, v, R2 * np.sin(th2), zi - R2 * np.cos(th2),
                   n_phi=96) * R2 * np.exp(1j * k2 * R2)

    np.savez(OUT + "ovoid_po.npz", psi=psi, psi_tr=psi_tr, psi_ax=psi_ax,
             A2=A2, th2=th2, R2=R2, N=g.N, n_cap=m["n_cap"])
    print("    -> ovoid_po.npz")


# ══ 2. dielectric ball: exact series and Muller BEM on the same body ═════════
def ball_fields(a=6.0, n1=1.0, n2=2.5, n_elem=900):
    b = ScalarBall(n1, n2, a, LAM, lmax_pad=25)
    zf = a * n1 / (n2 - n1)                       # paraxial internal focus
    print(f"  ball a={a}λ: x1={b.x1:.1f}, lmax={b.lmax}, focus z={zf:.2f}λ")

    zg = np.linspace(-1.6 * a, 1.6 * a, 340)
    rg = np.linspace(0.0, 1.6 * a, 170)
    RG, ZG = np.meshgrid(rg, zg, indexing="ij")
    psi = b.field_grid(RG.ravel(), ZG.ravel()).reshape(RG.shape)

    from groundtruth.bem import sphere_generator
    g = sphere_generator(a, n_elem)
    if (g.n_rho * g.rho + g.n_z * g.z).mean() < 0:
        g.flip_normal()
    ui = np.exp(1j * b.k1 * g.z)
    vi = 1j * b.k1 * g.n_z * ui
    t0 = time.time()
    u, v, _ = solve_muller(g, b.k1, b.k2, ui, vi)
    z_ax = np.linspace(-0.85 * a, 0.85 * a, 200)
    bem_ax = field_map(g, b.k2, u, v, np.zeros_like(z_ax), z_ax, n_phi=64)
    exact_ax = b.field_grid(np.zeros_like(z_ax), z_ax)
    print(f"    BEM N={g.N} in {time.time()-t0:.0f}s;  axial agreement "
          f"{np.abs(bem_ax-exact_ax).max()/np.abs(exact_ax).max():.2e}")

    np.savez(OUT + "ball_fields.npz", rg=rg, zg=zg, psi=psi, a=a, zf=zf,
             n1=n1, n2=n2, z_ax=z_ax, bem_ax=bem_ax, exact_ax=exact_ax,
             N=g.N, x1=b.x1)
    print("    -> ball_fields.npz")


if __name__ == "__main__":
    print("computing field maps")
    ovoid_fields()
    ovoid_po()
    ball_fields()
