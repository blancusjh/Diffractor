"""Scalar-field figures: meridional sections, focal region, the two reference
spheres, and the solver check against the exact ball."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
G = str(_HERE / "golden") + "/"
OV = np.load(G + "ovoid_fields.npz")
BL = np.load(G + "ball_fields.npz")
MC = np.load(G + "method_check.npz")
PO = np.load(G + "ovoid_po.npz")
OUT = str(_HERE) + "/"

BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#008300", "#e34948"
YELLOW, VIOLET, AQUA = "#eda100", "#4a3aa7", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
EDGE = "#7fd4ff"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF})


def ttl(a, t, s=None):
    a.set_title(t, fontsize=10.5, fontweight="bold", pad=9 if s is None else 17,
                loc="left")
    if s:
        a.text(0, 1.025, s, transform=a.transAxes, fontsize=8.5, color=MUTED)


def mirror(rg, F):
    return np.concatenate([-rg[::-1], rg[1:]]), np.vstack([F[::-1], F[1:]])


def focal_value(D, grid=None):
    """Peak |psi| within 1.5 lambda of the design focus, on the axis."""
    S = grid if grid is not None else D
    rg, zg = S["rg"], S["zg"]
    w = (np.abs(zg - float(S["zi"])) < 1.5)[None, :] & (rg < 0.4)[:, None]
    return np.abs(D["psi"])[w].max()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — meridional sections of the stigmatic ovoid
# ══════════════════════════════════════════════════════════════════════════════
def fig_longitudinal():
    fig, ax = plt.subplots(2, 1, figsize=(15.0, 10.5))
    rows = [(ax[0], OV, "Boundary-integral solution, lossless interior  n₂ = 1.5"),
            (ax[1], None, "Tangent-plane model on the cap alone,  n₂ = 1.5")]

    for a, D, head in rows:
        model = D is None
        S = OV if model else D
        rg, zg = S["rg"], S["zg"]
        psi = (PO["psi"] if model else D["psi"]).copy()
        if model:
            psi[~S["inside"]] = np.nan
        rr, PP = mirror(rg, psi)
        A = np.abs(PP) / (focal_value(PO, OV) if model else focal_value(D))
        im = a.imshow(np.log10(np.maximum(A, 10**-2.5)),
                      extent=[zg[0], zg[-1], rr[0], rr[-1]], origin="lower",
                      aspect="equal", cmap="magma", vmin=-2.5, vmax=0.2,
                      interpolation="bilinear")
        cap_r, cap_z = S["cap_r"], S["cap_z"]
        r_rim = float(S["r_rim"]); z_end = float(S["z_end"])
        z_apex = float(S["z_apex"]); zi = float(S["zi"])
        for sgn in (1, -1):
            a.plot(cap_z, sgn * cap_r, color=EDGE, lw=2.0)
            if not model:
                a.plot([cap_z[-1], z_end], [sgn * r_rim] * 2, color="white",
                       lw=1.1, alpha=.75)
                a.plot([z_end, z_apex], [sgn * r_rim, 0], color="white", lw=1.1,
                       alpha=.75)
            a.plot([cap_z[-1], zi], [sgn * r_rim, 0], color=EDGE, lw=0.9,
                   ls=(0, (5, 4)), alpha=.8)
        a.plot(zi, 0, "*", ms=14, color=EDGE, mec="black", mew=.6)
        a.annotate("A′", (zi, 0), textcoords="offset points", xytext=(7, 7),
                   color=EDGE, fontsize=11, fontweight="bold")
        a.set_xlabel("z  [λ]"); a.set_ylabel("r  [λ]")
        a.set_xlim(zg[0], zg[-1]); a.set_ylim(-5.2, 5.2)
        a.grid(False)
        cb = fig.colorbar(im, ax=a, fraction=0.019, pad=0.010)
        cb.set_label("log₁₀ ( |ψ| / peak )", fontsize=8.5)

        if model:
            sub = (f"Fresnel-transmitted field on the {int(PO['n_cap'])} cap "
                   f"panels, propagated by the Helmholtz surface "
                   f"representation · no shroud, no multiple scattering · "
                   f"evaluated inside the body only")
        else:
            sub = (f"Müller BOR-BEM, N = {int(D['N'])} panels, "
                   f"{float(D['ppl']):.0f} panels/λ₂ · whole closed body")
        ttl(a, head, sub)

    zi = float(OV["zi"])
    fig.suptitle("Meridional |ψ| of the stigmatic ovoid   ·   source A at "
                 f"z₀ = {float(OV['zo']):.0f}λ, design focus A′ at z = {zi:.0f}λ "
                 f"·   n₁ = 1.0, NA = {float(OV['NA']):.3f} "
                 f"(θ₂,max = {np.degrees(float(OV['th2max'])):.1f}°)   ·   "
                 f"cap rim r = {float(OV['r_rim']):.2f}λ at z = "
                 f"{float(OV['z_rim']):.2f}λ",
                 fontsize=12.5, fontweight="bold", x=0.006, ha="left", y=0.998)
    fig.text(0.008, 0.004,
             "Cyan: the Cartesian oval, stigmatic for A → A′; dashed cyan: the "
             "marginal ray. White: the cylinder and cone that close the body so "
             "the boundary-integral problem is well posed. "
             "Common colour scale; each panel is normalised to its own peak |ψ| "
             "within 1.5λ of A′ on the axis.", fontsize=8.5, color=INK2)

    fig.tight_layout(rect=[0, 0.016, 1, 0.982])
    fig.savefig(OUT + "f1_longitudinal.png", dpi=150, bbox_inches="tight",
                facecolor=SURF)
    plt.close(fig)
    print("f1_longitudinal.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — focal region
# ══════════════════════════════════════════════════════════════════════════════
def fig_focal():
    fig, ax = plt.subplots(1, 3, figsize=(16.8, 5.4))
    airy = float(OV["airy"]); zi = float(OV["zi"])

    # (a) focal plane, revolved from the axisymmetric radial profile
    a = ax[0]
    r_tr, I = OV["r_tr"], np.abs(OV["psi_tr"]) ** 2
    xs = np.linspace(-r_tr[-1], r_tr[-1], 320)
    X, Y = np.meshgrid(xs, xs)
    img = np.interp(np.hypot(X, Y).ravel(), r_tr, I / I.max(),
                    right=np.nan).reshape(X.shape)
    im = a.imshow(np.log10(np.maximum(img, 1e-4)),
                  extent=[-r_tr[-1] / airy, r_tr[-1] / airy] * 2,
                  origin="lower", cmap="magma", vmin=-4, vmax=0,
                  interpolation="bilinear")
    for m in (1, 2, 3):
        a.add_patch(plt.Circle((0, 0), m, fill=False, color="white", lw=0.6,
                               alpha=0.35))
    a.set_xlabel("x  [Airy radii]"); a.set_ylabel("y  [Airy radii]")
    a.grid(False)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    cb.set_label("log₁₀ ( I / I_max )", fontsize=8.5)
    ttl(a, "Focal plane",
        f"z = {zi:.0f}λ · Airy radius 0.61λ/NA = {airy:.3f}λ · "
        f"circles at 1, 2, 3 Airy radii")

    CURVES = [("BEM, whole closed body", RED, "-", 2.2, OV, "psi"),
              ("tangent-plane model on the cap", BLUE, (0, (5, 2)), 2.0,
               PO, "psi"),
              ("Debye–Wolf from the ray pupil P = t_s d₂/d₁", GREEN,
               (0, (1.6, 2)), 2.0, OV, "dw")]

    # (b) radial profile
    a = ax[1]
    for lab, col, ls, lw, D, key in CURVES:
        a.semilogy(OV["r_tr"] / airy, np.abs(D[key + "_tr"]), color=col, lw=lw,
                   ls=ls, label=lab)
    a.set_xlim(0, r_tr[-1] / airy); a.set_ylim(6e-6, 6e-2)
    a.set_xlabel("r  [Airy radii]"); a.set_ylabel("|ψ|")
    a.legend(fontsize=8.0, loc="upper right")
    ttl(a, "Radial profile in the plane z = z_i", "absolute amplitude, no rescaling")

    # (c) axial profile
    a = ax[2]
    for lab, col, ls, lw, D, key in CURVES:
        a.plot(OV["z_ax"], np.abs(D[key + "_ax"]), color=col, lw=lw, ls=ls,
               label=lab)
    a.axvline(zi, color=INK2, lw=1.0, ls=(0, (4, 3)))
    a.text(zi + 0.12, a.get_ylim()[1] * 0.97, "A′", color=INK2, fontsize=9,
           va="top")
    a.set_xlabel("z  [λ]"); a.set_ylabel("|ψ| on the axis")
    a.set_ylim(0, 0.052)
    a.legend(fontsize=8.0, loc="upper right")
    zb, ab_ = OV["z_ax"], np.abs(OV["psi_ax"])
    zp, ap = OV["z_ax"], np.abs(PO["psi_ax"])
    ttl(a, "Axial profile through the focus",
        f"peak at z = {zb[ab_.argmax()]:.2f}λ (BEM) and {zp[ap.argmax()]:.2f}λ "
        f"(tangent-plane), against the design z = {zi:.0f}λ; "
        f"aperture Fresnel number a²n₂/(λz_i) = "
        f"{float(OV['r_rim'])**2*1.5/zi:.2f}")

    fig.suptitle("Focal region of the stigmatic ovoid  "
                 f"(z_i = {zi:.0f}λ, NA = {float(OV['NA']):.3f})",
                 fontsize=12.5, fontweight="bold", x=0.006, ha="left", y=1.02)
    fig.text(0.006, -0.035,
             "The three curves differ in what they include: the BEM curve "
             "solves the Helmholtz transmission problem for the whole closed "
             "body (cap + cylinder + cone); the tangent-plane curve carries "
             "a field on the cap only; the Debye–Wolf curve replaces the "
             "diffraction integral by its large-Fresnel-number limit.",
             fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT + "f2_focal.png", dpi=170, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("f2_focal.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — the field on the two reference spheres
# ══════════════════════════════════════════════════════════════════════════════
def fig_spheres():
    fig, ax = plt.subplots(2, 2, figsize=(14.0, 9.4))
    k0, opl = float(OV["k0"]), float(OV["opl"])

    # ---- S1 : amplitude -----------------------------------------------------
    a = ax[0, 0]
    t1 = np.degrees(OV["th1"]); ref = np.abs(OV["A1_inc"])
    a.plot(t1, np.abs(OV["A1"]) / ref, color=RED, lw=1.8, label="total field")
    a.axhline(1.0, color=GREEN, lw=2.0, label="incident point source alone")
    a.set_xlim(0, np.degrees(float(OV["th1"][-1])))
    a.set_xlabel(r"$\theta_1$  [deg]")
    a.set_ylabel("|A| / |A_incident|")
    a.legend(fontsize=8.5, loc="lower right")
    ttl(a, "Entrance sphere $S_1$ — amplitude",
        f"R₁ = {float(OV['R1']):.0f}λ about the source A; total field = "
        f"incident + wave reflected by the cap")

    # ---- S1 : phase ---------------------------------------------------------
    a = ax[0, 1]
    t1 = np.degrees(OV["th1"])
    ph = np.angle(OV["A1"] / OV["A1_inc"]) / (2 * np.pi)
    a.plot(t1, ph, color=RED, lw=1.8, label="total field")
    a.axhline(0.0, color=GREEN, lw=2.0, label="incident point source alone")
    a.set_xlim(0, np.degrees(float(OV["th1"][-1])))
    a.set_xlabel(r"$\theta_1$  [deg]"); a.set_ylabel("phase  [waves]")
    a.legend(fontsize=8.5, loc="center right")
    ttl(a, "Entrance sphere $S_1$ — phase",
        "referenced to the incident spherical wave centred on A")

    # ---- S2 : amplitude -----------------------------------------------------
    a = ax[1, 0]
    thm = np.degrees(float(OV["th2max"]))
    n0 = float(np.abs(OV["A2_ray"][0]))
    t2 = np.degrees(OV["th2"])
    a.plot(t2, np.abs(OV["A2"]) / n0, color=RED, lw=2.2,
           label="BEM, whole closed body")
    a.plot(np.degrees(PO["th2"]), np.abs(PO["A2"]) / n0, color=BLUE, lw=2.0,
           ls=(0, (5, 2)), label="tangent-plane model on the cap")
    a.plot(t2, np.abs(OV["A2_dw"]) / n0, color=GREEN, lw=2.0, ls=(0, (1.6, 2)),
           label="Debye–Wolf from P")
    a.plot(t2, OV["A2_ray"] / n0, color=INK2, lw=1.6, ls=(0, (4, 2, 1, 2)),
           label="ray pupil  P = t_s d₂/d₁")
    a.axvline(thm, color=MUTED, lw=1.0)
    a.text(thm - 0.4, 1.28, "rim", color=MUTED, fontsize=8.5, ha="right")
    a.set_xlim(0, t2[-1]); a.set_ylim(0, 1.35)
    a.set_xlabel(r"$\theta_2$  [deg]")
    a.set_ylabel("|A| / P(0)")
    a.legend(fontsize=8.5, loc="lower left")
    inn = np.degrees(OV["th2"]) <= thm
    rms = np.sqrt(np.mean((np.abs(PO["A2"][inn]) / np.abs(OV["A2"][inn]) - 1) ** 2))
    ttl(a, "Exit sphere $S_2$ — amplitude",
        f"R₂ = {float(OV['R2']):.1f}λ about A′; A ≡ ψ·R₂·exp(i k₂R₂), so a purely "
        f"geometric converging wave would give A = P · "
        f"tangent-plane vs BEM inside the rim: {100*rms:.1f} % rms")

    # ---- S2 : phase ---------------------------------------------------------
    a = ax[1, 1]
    def rel_phase(A):
        p = np.unwrap(np.angle(np.asarray(A) * np.exp(-1j * k0 * opl)))
        return (p - p[0]) / (2 * np.pi)

    for A, th, col, ls, lab in (
            (OV["A2"], OV["th2"], RED, "-", "BEM, whole closed body"),
            (PO["A2"], PO["th2"], BLUE, (0, (5, 2)),
             "tangent-plane model on the cap"),
            (OV["A2_dw"], OV["th2"], GREEN, (0, (1.6, 2)),
             "Debye–Wolf from P")):
        a.plot(np.degrees(th), rel_phase(A), color=col, lw=2.0, ls=ls, label=lab)
    a.axvline(thm, color=MUTED, lw=1.0)
    a.set_xlim(0, t2[-1])
    a.set_xlabel(r"$\theta_2$  [deg]"); a.set_ylabel("phase  [waves]")
    a.legend(fontsize=8.5, loc="upper left")

    ph_ov = np.unwrap(np.angle(OV["A2"] * np.exp(-1j * k0 * opl))) / (2 * np.pi)
    inn = np.degrees(OV["th2"]) <= thm
    ttl(a, "Exit sphere $S_2$ — phase",
        f"carrier exp(−i k₂R₂) and the stigmatic OPL k₀C removed; "
        f"peak-to-valley inside the rim = "
        f"{np.ptp(ph_ov[inn] - ph_ov[0]):.3f} waves")

    fig.suptitle("Scalar field on the reference spheres of the stigmatic ovoid  "
                 f"(z_i = {float(OV['zi']):.0f}λ, NA = {float(OV['NA']):.3f}, "
                 f"θ₂,max = {thm:.1f}°)",
                 fontsize=12.5, fontweight="bold", x=0.006, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUT + "f3_spheres.png", dpi=170, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("f3_spheres.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — solver check against the exact scalar solution
# ══════════════════════════════════════════════════════════════════════════════
def fig_methods():
    fig, ax = plt.subplots(1, 4, figsize=(20.5, 5.2))
    a0, n2, k2a = float(MC["a"]), float(MC["n2"]), float(MC["k2a"])

    a = ax[0]
    rg2, zg2, psi2 = BL["rg"], BL["zg"], BL["psi"]
    rr2, PP2 = mirror(rg2, psi2)
    A2 = np.abs(PP2)
    im = a.imshow(A2, extent=[zg2[0], zg2[-1], rr2[0], rr2[-1]], origin="lower",
                  aspect="equal", cmap="magma", vmin=0, vmax=np.percentile(A2, 99.5),
                  interpolation="bilinear")
    aa = float(BL["a"]); t = np.linspace(0, 2 * np.pi, 400)
    a.plot(aa * np.cos(t), aa * np.sin(t), color=EDGE, lw=1.6)
    a.set_xlabel("z  [λ]"); a.set_ylabel("r  [λ]"); a.grid(False)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    cb.set_label("|ψ|", fontsize=8.5)
    ttl(a, "Dielectric ball, exact series",
        f"a = {aa:.0f}λ, n₂ = {float(BL['n2'])}, k₁a = {float(BL['x1']):.1f} · "
        f"plane wave from −z")

    a = ax[1]
    z, be, ex = MC["z_ax"], MC["bem"], MC["exact"]
    a.plot(z, np.abs(ex), color=GREEN, lw=3.0, label="exact series")
    a.plot(z, np.abs(be), color=RED, lw=1.6, ls=(0, (5, 3)), label="Müller BOR-BEM")
    a.set_xlabel("z  [λ]"); a.set_ylabel("|ψ| on the axis")
    a.legend(fontsize=9)
    ttl(a, "Axial field inside a smaller ball",
        f"a = {a0}λ, n₂ = {n2}, k₂a = {k2a:.1f}, N = {int(MC['N'])} · "
        f"max rel. difference {np.abs(be-ex).max()/np.abs(ex).max():.1e}")

    a = ax[2]
    r, bt, et = MC["r_tr"], MC["bem_t"], MC["exact_t"]
    a.plot(r, np.abs(et), color=GREEN, lw=3.0, label="exact series")
    a.plot(r, np.abs(bt), color=RED, lw=1.6, ls=(0, (5, 3)), label="Müller BOR-BEM")
    a.set_xlabel("r  [λ]"); a.set_ylabel("|ψ|")
    a.legend(fontsize=9)
    ttl(a, "Transverse field inside the same ball",
        f"z = {float(MC['ztr']):.2f}λ · max rel. difference "
        f"{np.abs(bt-et).max()/np.abs(et).max():.1e}")

    a = ax[3]
    k2a_e, err, nn = MC["env_k2a"], MC["env_err"], MC["env_n2"]
    for val, col, mk in ((1.5, BLUE, "o"), (2.5, VIOLET, "s")):
        m = nn == val
        a.semilogy(k2a_e[m], err[m], mk + "-", color=col, lw=2.2, ms=7,
                   label=f"n₂ = {val}")
    a.axhline(1e-2, color=MUTED, lw=1.4, ls=(0, (6, 4)))
    a.text(11, 1.3e-2, "1 %", fontsize=8.5, color=MUTED)
    a.set_xlabel("k₂a")
    a.set_ylabel("max rel. error in u on the surface")
    a.legend(fontsize=9, loc="lower right")
    ttl(a, "BEM error versus electrical size",
        "64–100 panels per interior wavelength throughout")

    fig.suptitle("Müller BOR-BEM checked against the exact scalar solution for "
                 "a homogeneous ball",
                 fontsize=12.5, fontweight="bold", x=0.006, ha="left", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT + "f4_methods.png", dpi=170, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("f4_methods.png")


if __name__ == "__main__":
    fig_longitudinal()
    fig_focal()
    fig_spheres()
    fig_methods()
