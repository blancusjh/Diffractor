"""The stigmatic surface and the spherical one — what the difference costs.

A point source A in air is imaged into glass by a single refracting surface.
Two surfaces are compared, with the *same vertex curvature*, so they have the
same paraxial focus and differ only in how they bend the outer rays:

  * the **Cartesian ovoid** of Descartes — the exact solution of n₁d₁ + n₂d₂ = C
    (`diffractor.geometry.DescartesOvoid`).  Every ray from A arrives at A′ with
    the same optical path: the wavefront error is zero by construction, at any
    aperture, and `analysis.opd_waves` measures it back out of the coordinates
    as a residual, never as an assumption.
  * a **sphere** of the same vertex radius R = (n₂−n₁)/(n₂/z_i − n₁/z_o) —
    what a polishing machine makes.  Its outer rays cross the axis early: that
    is spherical aberration, and it grows as the fourth power of the aperture.

Both pupils are built from the same general transport law
(`propagation.ray_tube_amplitude`: n₁|U₁|²dΩ₁ · T = n₂|U₂|²dΩ₂), and the field
near focus is the Debye integral of that pupil.  For the ovoid the general law
collapses to `stigmatic_pupil` = t_s·d₂/d₁, which the script checks.

Run:  python examples/04_stigmatic_vs_spherical.py
"""
import numpy as np
from scipy.special import j0

from diffractor.analysis import opd_waves
from diffractor.optics import Medium, stigmatic_interface
from diffractor.transport import ray_tube_amplitude, stigmatic_pupil
from diffractor.scattering import T_s
from style import BLUE, EDGE, GREEN, INK2, MUTED, ORANGE, RED, VIOLET, save, ttl

import matplotlib.pyplot as plt

LAM = 1.0                                  # everything in wavelengths
N1, N2 = 1.0, 1.5
ZO, ZI = -800.0, 400.0                     # source, paraxial image
K0, K2 = 2 * np.pi / LAM, 2 * np.pi * N2 / LAM
A0 = 1.0 / (4 * np.pi)                     # point-source amplitude

iface = stigmatic_interface(Medium(N1), Medium(N2), ZO, ZI)
R_V = (N2 - N1) / (N2 / ZI - N1 / ZO)      # common vertex radius of curvature
g_full = iface.sample(n_pts=40001)
fit = np.polyfit(g_full["r"][g_full["r"] < 2.0] ** 2,
                 g_full["z"][g_full["r"] < 2.0], 1)
print(f"vertex curvature: ovoid {1/(2*fit[0]):.4f}λ, "
      f"paraxial formula {R_V:.4f}λ")
print(f"ovoid stigmatism: |OPD| ≤ {np.abs(opd_waves(iface, g_full, LAM)).max():.1e} "
      f"waves out to its rim at r = {g_full['r'][-1]:.1f}λ")


# ── the two surfaces, as pupils ──────────────────────────────────────────────
def ovoid_pupil(r_max, n=2400):
    """Exact stigmatic surface: amplitude t_s·d₂/d₁, phase identically flat."""
    d1 = np.interp(np.linspace(0.0, r_max, n), g_full["r"], g_full["d1"])
    g = iface.profile.geometry(d1)
    P = stigmatic_pupil(iface, g)
    gen = ray_tube_amplitude(N1, N2, g["w1"], g["w2"],
                             T_s(N1, N2, g["cos_i1"], g["cos_i2"]))
    return dict(param=d1, alpha=g["th2"], P=A0 * P, w2=g["w2"],
                W=opd_waves(iface, g, LAM), r=g["r"], z=g["z"],
                gen_check=np.abs(P[1:] / gen[1:] - 1).max())


def sphere_pupil(r_max, n=2400):
    """Sphere of the same vertex radius, traced exactly and refracted by Snell."""
    r = np.linspace(0.0, r_max, n)
    z = R_V - np.sqrt(R_V**2 - r**2)
    n_r, n_z = -r / R_V, np.sqrt(R_V**2 - r**2) / R_V     # normal into medium 2
    d1 = np.hypot(r, z - ZO)
    s1r, s1z = r / d1, (z - ZO) / d1
    cos_i1 = s1r * n_r + s1z * n_z
    mu = N1 / N2
    cos_i2 = np.sqrt(1.0 - mu**2 * (1.0 - cos_i1**2))
    s2r = mu * s1r + (cos_i2 - mu * cos_i1) * n_r         # vector Snell
    s2z = mu * s1z + (cos_i2 - mu * cos_i1) * n_z
    alpha = np.arctan2(-s2r, s2z)                          # convergence angle
    th1 = np.arctan2(r, z - ZO)

    # Wavefront error, in the form the Debye integral needs: the optical path
    # from A to the plane through A′ normal to the outgoing ray,
    #     W = n₁d₁ + n₂ (A′ − X)·ŝ₂,
    # which is `analysis.two_leg_opl` whenever the ray actually passes through
    # A′ — i.e. exactly for the stigmatic surface — and its generalisation when
    # it misses.  Referenced to the axial ray, so W(0) = 0.
    W = (N1 * d1 + N2 * (-r * s2r + (ZI - z) * s2z)
         - (N1 * abs(ZO) + N2 * ZI)) / LAM

    w1 = np.sin(th1) * np.gradient(th1, r, edge_order=2)
    w2 = np.sin(alpha) * np.gradient(alpha, r, edge_order=2)
    P = ray_tube_amplitude(N1, N2, w1, w2, T_s(N1, N2, cos_i1, cos_i2))
    z_cross = z - r * s2z / np.where(s2r == 0, -np.inf, s2r)   # axis crossing
    return dict(param=r, alpha=alpha, P=A0 * P, w2=w2, W=W, r=r, z=z,
                z_cross=z_cross)


# ── the Debye integral: a pupil, focused ─────────────────────────────────────
def focal_field(pup, r_out, dz):
    """U(r, z) near focus from the pupil (r_out and dz are 1-D; result 2-D)."""
    a = pup["alpha"]
    meas = pup["P"] * np.exp(2j * np.pi * pup["W"]) * pup["w2"]
    J = j0(K2 * np.outer(r_out, np.sin(a)))                 # (n_r, n_alpha)
    out = np.empty((r_out.size, dz.size), complex)
    for j, z in enumerate(dz):
        integ = meas * np.exp(1j * K2 * z * np.cos(a))
        out[:, j] = -1j * K2 * np.trapezoid(J * integ, pup["param"], axis=1)
    return out


# ══ 1. how the error grows with the aperture ═════════════════════════════════
R_APS = np.linspace(10.0, 92.0, 20)
pv, rms, strehl = [], [], []
r_peak = np.linspace(0.0, 0.6, 3)
for ra in R_APS:
    sp = sphere_pupil(ra)
    # area-weighted rms of the wavefront error, and the true Strehl ratio:
    # the same pupil with its aberration switched off is the perfect reference.
    w = sp["W"] - (np.trapezoid(sp["W"] * sp["r"], sp["r"])
                   / np.trapezoid(sp["r"], sp["r"]))
    pv.append(np.ptp(sp["W"]))
    rms.append(np.sqrt(np.trapezoid(w**2 * sp["r"], sp["r"])
                       / np.trapezoid(sp["r"], sp["r"])))
    zs = np.linspace(-1.25 * abs(sp["z_cross"][-1] - ZI) - 2.0, 4.0, 240)
    I_ab = np.abs(focal_field(sp, r_peak, zs)) ** 2
    perfect = dict(sp, W=np.zeros_like(sp["W"]))
    I_0 = np.abs(focal_field(perfect, np.zeros(1), np.zeros(1))) ** 2
    strehl.append(I_ab.max() / I_0.max())
pv, rms, strehl = map(np.array, (pv, rms, strehl))
marechal = np.exp(-(2 * np.pi * rms) ** 2)
print(f"aperture sweep: r_max {R_APS[0]:.0f} → {R_APS[-1]:.0f}λ, sphere PV "
      f"{pv[0]:.3f} → {pv[-1]:.1f} waves, Strehl {strehl[0]:.3f} → "
      f"{strehl[-1]:.3f}")


# ══ 2. the caustics, at the widest aperture ══════════════════════════════════
R_MAX = 85.0
ov, sp = ovoid_pupil(R_MAX, n=4000), sphere_pupil(R_MAX, n=4000)
NA_OV = N2 * np.sin(ov["alpha"][-1])
AIRY = 0.61 * LAM / NA_OV
print(f"stigmatic_pupil vs the general ray_tube_amplitude: "
      f"max relative difference {ov['gen_check']:.1e}")
print(f"aperture r = {R_MAX:g}λ → NA = {NA_OV:.3f}, Airy radius {AIRY:.2f}λ; "
      f"sphere OPD PV = {np.ptp(sp['W']):.2f} waves, "
      f"marginal ray crosses the axis at z = {sp['z_cross'][-1]:.1f}λ "
      f"({ZI - sp['z_cross'][-1]:.0f}λ short of the paraxial image)")

dz = np.linspace(-1.15 * (ZI - sp["z_cross"][-1]), 45.0, 340)
r_map = np.linspace(0.0, 9.0, 180)
U_ov = focal_field(ov, r_map, dz)
U_sp = focal_field(sp, r_map, dz)
peak = np.abs(U_ov).max()

r_psf = np.linspace(0.0, 10.0, 500)
psf_ov = np.abs(focal_field(ov, r_psf, np.zeros(1))[:, 0]) ** 2
psf_sp0 = np.abs(focal_field(sp, r_psf, np.zeros(1))[:, 0]) ** 2
best = dz[np.abs(U_sp[0]).argmax()]
psf_spb = np.abs(focal_field(sp, r_psf, np.array([best]))[:, 0]) ** 2
print(f"peak intensity: sphere / ovoid = {psf_spb.max()/psf_ov.max():.4f} "
      f"at its own best focus (z = z_i {best:+.1f}λ), "
      f"{psf_sp0.max()/psf_ov.max():.4f} at the paraxial image")
first_min = r_psf[1:-1][np.diff(np.sign(np.diff(psf_ov))) > 0][0]
print(f"ovoid PSF first zero at {first_min:.2f}λ "
      f"(0.61λ/NA = {AIRY:.2f}λ; the pupil is not uniform, so they differ "
      f"by {100*abs(first_min/AIRY-1):.0f} %)")


# ══ figure ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14.0, 13.2))
gs = fig.add_gridspec(3, 6, height_ratios=[0.80, 1.0, 0.80], hspace=0.46,
                      wspace=1.5, top=0.935, bottom=0.075, left=0.06, right=0.985)

# (a) the two surfaces
a = fig.add_subplot(gs[0, :3])
rr = np.linspace(0, R_MAX, 400)
z_ov = np.interp(rr, ov["r"], ov["z"])
z_sp = R_V - np.sqrt(R_V**2 - rr**2)
a.plot(z_ov, rr, color=EDGE, lw=2.6, label="Cartesian ovoid")
a.plot(z_sp, rr, color=VIOLET, lw=1.6, ls=(0, (5, 3)),
       label=f"sphere, R = {R_V:g}λ")
a.set_xlabel("z  [λ]"); a.set_ylabel("r  [λ]")
a.set_ylim(0, R_MAX * 1.02)
a.legend(fontsize=8.5, loc="upper left")
ttl(a, "The two surfaces", "same vertex curvature, same paraxial focus")
ins = a.inset_axes([0.56, 0.11, 0.41, 0.36])
ins.plot(rr, z_ov - z_sp, color=ORANGE, lw=1.8)
ins.set_xlabel("r  [λ]", fontsize=7.5, labelpad=1)
ins.set_ylabel("Δz  [λ]", fontsize=7.5, labelpad=1)
ins.tick_params(labelsize=7)
ins.set_title("ovoid − sphere", fontsize=8, color=INK2, pad=3)

# (b) the wavefront error
a = fig.add_subplot(gs[0, 3:])
a.axhspan(-0.25, 0.25, color=GREEN, alpha=0.10)
a.text(0.4, 0.30, "Rayleigh λ/4", fontsize=8, color=GREEN)
a.plot(sp["r"] / R_MAX, sp["W"] - sp["W"][0], color=VIOLET, lw=2.0,
       label="sphere")
a.plot(ov["r"] / R_MAX, ov["W"], color=EDGE, lw=2.4, label="Cartesian ovoid")
a.set_xlim(0, 1); a.set_xlabel("r / r_max"); a.set_ylabel("wavefront error  [waves]")
a.legend(fontsize=8.5, loc="lower left")
ttl(a, f"Wavefront error at r_max = {R_MAX:g}λ  (NA = {NA_OV:.2f})",
    f"measured from the coordinates by `analysis.opd_waves` · sphere PV = "
    f"{np.ptp(sp['W']):.1f} waves, ovoid {np.abs(ov['W']).max():.0e}")

# (c, d) the caustics
def caustic(ax, U, title, sub, mark_focus=None):
    rrm = np.concatenate([-r_map[::-1], r_map[1:]])
    MM = np.vstack([np.abs(U)[::-1], np.abs(U)[1:]]) / peak
    im = ax.imshow(np.log10(np.maximum(MM, 10**-2.2)),
                   extent=[ZI + dz[0], ZI + dz[-1], rrm[0], rrm[-1]],
                   origin="lower", aspect="auto", cmap="magma", vmin=-2.2,
                   vmax=0.02, interpolation="bilinear")
    ax.axvline(ZI, color=EDGE, lw=1.0, ls=(0, (4, 3)), alpha=.8)
    ax.text(ZI + 0.6, rrm[-1] * 0.80, "paraxial\nimage A′", color=EDGE,
            fontsize=8.5)
    if mark_focus is not None:
        ax.plot([ZI + mark_focus], [0], "*", ms=13, color="white", mec="black",
                mew=.6)
    ax.set_xlabel("z  [λ]"); ax.set_ylabel("r  [λ]"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.010)
    cb.set_label("log₁₀ ( |U| / peak )", fontsize=8.5)
    ttl(ax, title, sub)


caustic(fig.add_subplot(gs[1, :3]), U_ov, "Cartesian ovoid — the caustic is a point",
        "one wavelength-sized focus, at the design position, at full aperture")
caustic(fig.add_subplot(gs[1, 3:]), U_sp, "Sphere — the same rays, spread along the axis",
        "spherical aberration: the edge focuses short, and nothing focuses well",
        mark_focus=best)

# (e) PSFs
a = fig.add_subplot(gs[2, :2])
a.semilogy(r_psf / AIRY, psf_ov / psf_ov.max(), color=EDGE, lw=2.4,
           label="ovoid, at A′")
a.semilogy(r_psf / AIRY, psf_spb / psf_ov.max(), color=VIOLET, lw=1.8,
           label="sphere, best focus")
a.semilogy(r_psf / AIRY, psf_sp0 / psf_ov.max(), color=ORANGE, lw=1.5,
           ls=(0, (4, 3)), label="sphere, at A′")
a.set_xlim(0, r_psf[-1] / AIRY); a.set_ylim(1e-4, 2)
a.set_xlabel("r  [Airy radii]"); a.set_ylabel("I / I_ovoid(0)")
a.legend(fontsize=8, loc="upper right")
ttl(a, "Point-spread functions", f"Airy radius 0.61λ/NA = {AIRY:.2f}λ")

# (f) axial
a = fig.add_subplot(gs[2, 2:4])
a.plot(ZI + dz, np.abs(U_ov[0]) ** 2 / peak**2, color=EDGE, lw=2.4, label="ovoid")
a.plot(ZI + dz, np.abs(U_sp[0]) ** 2 / peak**2, color=VIOLET, lw=1.8,
       label="sphere")
a.axvline(ZI, color=MUTED, lw=1.0, ls=(0, (4, 3)))
a.set_xlabel("z  [λ]"); a.set_ylabel("I / I_ovoid,max  on the axis")
a.legend(fontsize=8.5, loc="upper left")
ttl(a, "Along the axis", "the sphere's light is spread over tens of λ")

# (g) the aperture sweep
a = fig.add_subplot(gs[2, 4:])
a.semilogy(R_APS, np.maximum(strehl, 1e-3), color=VIOLET, lw=2.0, marker="o",
           ms=3.5, label="Strehl, from the diffraction integral")
a.semilogy(R_APS, np.maximum(marechal, 1e-3), color=GREEN, lw=1.6,
           ls=(0, (4, 3)), label="Maréchal  exp[−(2πσ)²]")
a.axhline(0.8, color=MUTED, lw=1.0)
a.text(R_APS[-1] - 1, 0.86, "diffraction-limited (0.8)", fontsize=7.5,
       color=MUTED, ha="right")
a.set_xlim(R_APS[0], R_APS[-1]); a.set_ylim(1e-3, 1.8)
a.set_xlabel("aperture radius  [λ]"); a.set_ylabel("Strehl ratio")
a.legend(fontsize=8, loc="lower left")
ttl(a, "Where the sphere stops working",
    "peak intensity against the same pupil with its aberration switched off · "
    "PV grows as r⁴ · Maréchal is only the small-error limit")

fig.suptitle("A stigmatic surface against a spherical one — same glass, same "
             "vertex, same paraxial focus", fontsize=12.5, fontweight="bold",
             x=0.006, ha="left", y=0.988)
fig.text(0.006, 0.004,
         f"Point source at z = {ZO:g}λ in air, image at z = {ZI:g}λ inside "
         f"n₂ = {N2}; both surfaces share the vertex radius R = {R_V:g}λ.  "
         "Pupils from the general transport law\n"
         "`ray_tube_amplitude` (for the ovoid it agrees with `stigmatic_pupil` "
         f"= t_s·d₂/d₁ to {ov['gen_check']:.0e}), phases from the exact ray "
         "trace, focal fields by the Debye integral of that pupil.\n"
         "Both caustics share one colour scale, normalised to the ovoid's peak: "
         f"at r_max = {R_MAX:g}λ the sphere keeps "
         f"{100*psf_spb.max()/psf_ov.max():.0f} % of it, and only by moving the "
         f"screen {abs(best):.0f}λ; at A′ itself, "
         f"{100*psf_sp0.max()/psf_ov.max():.0f} %.\n"
         "The two are not the same cone: bending the marginal rays too hard is "
         "what aberration IS, so the sphere works at a wider image-side angle\n"
         f"(NA {N2*np.sin(sp['alpha'][-1]):.2f} against {NA_OV:.2f}) — which is "
         "why its Strehl is measured against its own unaberrated pupil, not "
         "against the ovoid.",
         fontsize=8.5, color=INK2)
save(fig, "stigmatic_vs_spherical.png")
