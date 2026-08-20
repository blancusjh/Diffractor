"""Diffraction by circular and annular apertures — exact, and checked.

A unit-amplitude plane wave illuminates a hard aperture in an opaque screen.
The field is a `MonochromaticField` on an axisymmetric polar `Grid` (n_θ = 1),
and everything below is the exact angular spectrum

    U(z) = IFT2[ e^{i k_z z} · FT2 U ],   k_z = √(k² − k⊥²),

evaluated — because the grid is polar — through the order-zero Hankel
transform.  The meridional maps use the operator layer directly (`FT2` once,
one transfer factor per plane, `IFT2` back) with a shared `PolarPlan`, so the
Bessel kernels are computed once for 260 planes.  Two closed forms check the
result, and `rayleigh_sommerfeld` — the same exact operator written as a
real-space integral — cross-checks it independently:

  * on the axis,  U(0, z) = e^{ikz} − (z/√(a²+z²)) e^{ik√(a²+z²)},
  * in the far field, the Airy pattern [2 J₁(x)/x]², x = k a sinθ.

Everything is in units of the wavelength (λ = 1).

Run:  python examples/01_apertures.py
"""
import numpy as np
from scipy.special import j1

from diffractor import (FT2, IFT2, Grid, MonochromaticField, PolarPlan,
                        fraunhofer, rayleigh_sommerfeld)
from style import BLUE, EDGE, GREEN, INK2, MUTED, RED, save, ttl

import matplotlib.pyplot as plt

LAM = 1.0                       # everything in units of the wavelength
K = 2 * np.pi / LAM
A_OUT, A_IN = 15.0, 11.0        # disc radius; inner radius of the annulus

# The aperture is the integration domain itself: r runs 0 → a, so the disc has
# a mathematically sharp edge and no sampling error at all.
disc = MonochromaticField(Grid.polar(np.linspace(0.0, A_OUT, 4001)),
                          np.ones((4001, 1)), LAM)
annulus = MonochromaticField(Grid.polar(np.linspace(A_IN, A_OUT, 2001)),
                             np.ones((2001, 1)), LAM)


def axial_exact(z, a=A_OUT):
    """Closed-form on-axis field of a uniformly illuminated disc."""
    R = np.hypot(a, z)
    return np.exp(1j * K * z) - (z / R) * np.exp(1j * K * R)


def airy(theta, a=A_OUT):
    x = K * a * np.sin(theta)
    return np.where(x == 0, 1.0, (2 * j1(np.where(x == 0, 1, x)) /
                                  np.where(x == 0, 1, x)) ** 2)


# ── the z-sweep, through the operator layer ──────────────────────────────────
zg = np.linspace(4.0, 620.0, 260)
z_ax = np.linspace(2.0, 620.0, 900)
rg = Grid.polar(np.linspace(0.0, 34.0, 190))


def sweep(field, zs, out_grid, density=1.0):
    """|U| on out_grid at each z: FT2 once, one transfer per plane.

    `density` multiplies the k sampling: the transfer phase k_z·z turns
    arbitrarily fast at the band edge k → k₀, so quantitative curves at long
    z deserve a denser quadrature than a map for the eye.
    """
    kgrid = field.grid.reciprocal(
        k_max=1.02 * K,
        n_k=int(density * (np.ceil(5.0 * 1.02 * K
                                   * (field.grid.axes[0][-1] + zs.max())
                                   / (2 * np.pi)) + 1)))
    plan_in = PolarPlan.build(field.grid, kgrid)
    plan_out = PolarPlan.build(out_grid, kgrid)
    F = FT2(field, kgrid=kgrid, plan=plan_in)
    k = kgrid.axes[0]
    kz = np.sqrt(np.maximum(K**2 - k**2, 0.0))[:, None, None]
    live = (k**2 <= K**2)[:, None, None]
    out = np.empty((out_grid.axes[0].size, zs.size))
    for j, z in enumerate(zs):
        Fz = F.like(np.where(live, F.values * np.exp(1j * kz * z), 0.0))
        out[:, j] = np.abs(IFT2(Fz, grid=out_grid, plan=plan_out).u[:, 0])
    return out


print(f"meridional maps ({zg.size} planes each)")
M_disc = sweep(disc, zg, rg)
M_ann = sweep(annulus, zg, rg)

axis_grid = Grid.polar(np.array([0.0, 1e-6]))    # the axis (2 pts: quadrature)
ax_disc = sweep(disc, z_ax, axis_grid, density=4.0)[0]
ax_ann = sweep(annulus, z_ax, axis_grid, density=4.0)[0]
ax_ref = axial_exact(z_ax)
err = np.abs(ax_disc - np.abs(ax_ref)).max() / np.abs(ax_ref).max()
print(f"on-axis: max deviation from the closed form = {err:.2e}")

# The two exact propagators, compared where both are comfortably sampled.
Z_X = 300.0
out_x = Grid.polar(np.linspace(0.0, 34.0, 120))
x_asm = sweep(disc, np.array([Z_X]), out_x)[:, 0]
x_rs1 = np.abs(rayleigh_sommerfeld(disc, Z_X, output_grid=out_x).u[:, 0])
d_x = np.abs(x_asm - x_rs1).max() / x_rs1.max()
print(f"ASM vs Rayleigh–Sommerfeld at z = {Z_X:g}λ: "
      f"max relative difference = {d_x:.2e}")

# Far field: the Fraunhofer operator against the closed-form Airy pattern.
Z_FF = 3000.0
out_ff = Grid.polar(np.linspace(0.0, 900.0, 260))
ff_disc = fraunhofer(disc, Z_FF, output_grid=out_ff)
ff_ann = fraunhofer(annulus, Z_FF, output_grid=out_ff)
theta = np.arctan2(out_ff.axes[0], Z_FF)
airy_r = airy(theta)
i_disc = np.abs(ff_disc.u[:, 0]) ** 2 / np.abs(ff_disc.u[0, 0]) ** 2
i_ann = np.abs(ff_ann.u[:, 0]) ** 2 / np.abs(ff_ann.u[0, 0]) ** 2
d_ff = np.abs(i_disc - airy_r).max()
print(f"far field: max |fraunhofer − Airy| = {d_ff:.2e}")


# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.0, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.40,
                      wspace=0.21, top=0.90, bottom=0.14, left=0.055,
                      right=0.985)


def meridian(a, M, title, sub, aperture):
    r_ax = rg.axes[0]
    rr = np.concatenate([-r_ax[::-1], r_ax[1:]])
    MM = np.vstack([M[::-1], M[1:]])
    im = a.imshow(np.log10(np.maximum(MM, 1e-2)),
                  extent=[zg[0], zg[-1], rr[0], rr[-1]], origin="lower",
                  aspect="auto", cmap="magma", vmin=-2, vmax=0.45,
                  interpolation="bilinear")
    for sgn in (1, -1):
        a.plot([zg[0], zg[-1]], [sgn * aperture] * 2, color=EDGE, lw=0.9,
               ls=(0, (5, 4)), alpha=.75)
    a.set_xlabel("z  [λ]"); a.set_ylabel("r  [λ]"); a.grid(False)
    cb = fig.colorbar(im, ax=a, fraction=0.030, pad=0.012)
    cb.set_label("log₁₀ |U|", fontsize=8.5)
    ttl(a, title, sub)
    for nf in (2, 1):
        z_nf = A_OUT**2 / (LAM * nf)
        a.annotate(f"a²/λz = {nf}", (z_nf, rr[-1]), xytext=(0, -13),
                   textcoords="offset points", ha="center", fontsize=7.5,
                   color="white", alpha=.85)
        a.plot([z_nf, z_nf], [rr[-1], 0.93 * rr[-1]], color="white", lw=0.8,
               alpha=.6)


meridian(fig.add_subplot(gs[0, 0]), M_disc,
         f"Circular aperture, a = {A_OUT:g}λ",
         "exact angular spectrum · dashed: the geometric shadow edge", A_OUT)
meridian(fig.add_subplot(gs[0, 1]), M_ann,
         f"Annulus, {A_IN:g}λ < r < {A_OUT:g}λ",
         "the same aperture area concentrated at the rim", A_OUT)

a = fig.add_subplot(gs[1, 0])
a.plot(z_ax, np.abs(ax_ref) ** 2, color=GREEN, lw=3.0,
       label="closed form, disc:  |e^{ikz} − (z/R)e^{ikR}|²")
a.plot(z_ax, ax_disc ** 2, color=RED, lw=1.5, ls=(0, (5, 3)),
       label="angular spectrum, disc")
a.plot(z_ax, ax_ann ** 2, color=BLUE, lw=1.5, label="angular spectrum, annulus")
a.set_xlim(z_ax[0], z_ax[-1]); a.set_ylim(0, 4.6)
a.set_xlabel("z  [λ]"); a.set_ylabel("|U|²  on the axis")
a.legend(fontsize=8.5, loc="upper right", ncol=1)
ttl(a, "On-axis intensity, and what it should be",
    f"disc: bright and dark points where a²/λz is odd and even · "
    f"max deviation from the closed form {err:.1e}")

a = fig.add_subplot(gs[1, 1])
xa = out_ff.axes[0] / (0.61 * LAM / (A_OUT / Z_FF))
a.semilogy(xa, airy_r, color=GREEN, lw=3.0, label="Airy pattern [2J₁(x)/x]²")
a.semilogy(xa, i_disc, color=RED, lw=1.5, ls=(0, (5, 3)),
           label="fraunhofer, disc")
a.semilogy(xa, i_ann, color=BLUE, lw=1.5, label="fraunhofer, annulus")
a.axvline(1.0, color=MUTED, lw=1.0)
a.text(1.06, 3e-5, "first Airy zero\n0.61 λ/NA", fontsize=8, color=MUTED)
a.set_xlim(0, xa[-1]); a.set_ylim(1e-5, 1.6)
a.set_xlabel("r  [Airy radii]"); a.set_ylabel("I / I(0)")
a.legend(fontsize=8.5, loc="upper right")
ttl(a, f"Far field at z = {Z_FF:.0f}λ",
    f"the far-field operator against the closed-form Airy pattern · "
    f"max difference {d_ff:.1e} · the annulus narrows the core and pumps "
    f"the rings")

fig.suptitle("Diffraction by a hard aperture — computed exactly, then checked "
             "against closed form", fontsize=12.5, fontweight="bold",
             x=0.006, ha="left", y=0.985)
fig.text(0.006, 0.005,
         "Plane wave, λ = 1, n = 1.  Both maps share the colour scale; |U| = 1 is the undisturbed incident amplitude.  The on-axis field\n"
         "oscillates between 0 and 4× the incident intensity as a²/λz runs through the integers — the same interference that puts the\n"
         "Poisson–Arago spot behind the complementary disc.  In their common regime the two exact propagators agree: at z = "
         f"{Z_X:.0f}λ the\nangular-spectrum and Rayleigh–Sommerfeld profiles differ by {100*d_x:.2f} % of the peak.",
         fontsize=8.5, color=INK2)
save(fig, "apertures.png")
