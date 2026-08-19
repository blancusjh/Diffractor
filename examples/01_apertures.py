"""Diffraction by circular and annular apertures — exact, and checked.

A unit-amplitude plane wave illuminates a hard aperture in an opaque screen.
The field behind it is computed with `propagation.exact`:

    asm_axisym / hankel  — the exact angular spectrum, kz = 2π sqrt((n/λ)² − ρ²)
    rs1_plane            — the same operator written as a Rayleigh–Sommerfeld
                           surface integral, used here as an independent check

Both are exact solutions of the Helmholtz equation in the half-space; nothing
here is paraxial.  Two things are verified against closed form:

  * the on-axis field of a uniformly lit disc,
        U(0, z) = e^{ikz} − (z/√(a²+z²)) e^{ik√(a²+z²)},
    which is what Rayleigh–Sommerfeld I integrates to exactly, and
  * the far field, the Airy pattern [2 J₁(x)/x]², x = k a sinθ.

Run:  python examples/01_apertures.py
"""
import numpy as np
from scipy.special import j1

from diffractor.propagation import hankel, rs1_plane
from style import BLUE, EDGE, GREEN, INK2, MUTED, ORANGE, RED, save, ttl

import matplotlib.pyplot as plt

LAM, N_MED = 1.0, 1.0            # everything in units of the wavelength
K = 2 * np.pi * N_MED / LAM
A_OUT, A_IN = 15.0, 11.0         # disc radius; inner radius of the annulus


# ── the propagator, written once ─────────────────────────────────────────────
def spectrum(U, r, n_rho=12000):
    """Forward Hankel transform onto a propagating-cone ρ grid."""
    rho = np.linspace(0.0, N_MED / LAM, n_rho)
    return rho, hankel(U, r, rho)


def replay(rho, Uh, z, r_out):
    """Inverse transform of the propagated spectrum — one plane at z."""
    kz = 2 * np.pi * np.sqrt(np.maximum((N_MED / LAM) ** 2 - rho**2, 0.0))
    return hankel(Uh * np.exp(1j * kz * z), rho, r_out)


def axial_exact(z, a=A_OUT):
    """Closed-form on-axis field of a uniformly illuminated disc."""
    R = np.hypot(a, z)
    return np.exp(1j * K * z) - (z / R) * np.exp(1j * K * R)


def airy(theta, a=A_OUT):
    x = K * a * np.sin(theta)
    return np.where(x == 0, 1.0, (2 * j1(x) / np.where(x == 0, 1, x)) ** 2)


# ── fields ───────────────────────────────────────────────────────────────────
# The aperture is the integration domain itself: r runs 0 → a, so the disc has
# a mathematically sharp edge and no sampling error at all.
r_disc = np.linspace(0.0, A_OUT, 4001)
U_disc = np.ones_like(r_disc, dtype=complex)

r_ann = np.linspace(A_IN, A_OUT, 2001)
U_ann = np.ones_like(r_ann, dtype=complex)

print("forward transforms")
rho, Uh_disc = spectrum(U_disc, r_disc)
_, Uh_ann = spectrum(U_ann, r_ann)

zg = np.linspace(4.0, 620.0, 260)
rg = np.linspace(0.0, 34.0, 190)
print(f"meridional maps ({zg.size} planes)")
M_disc = np.array([np.abs(replay(rho, Uh_disc, z, rg)) for z in zg]).T
M_ann = np.array([np.abs(replay(rho, Uh_ann, z, rg)) for z in zg]).T

z_ax = np.linspace(2.0, 620.0, 900)
ax_disc = np.array([replay(rho, Uh_disc, z, np.zeros(1))[0] for z in z_ax])
ax_ann = np.array([replay(rho, Uh_ann, z, np.zeros(1))[0] for z in z_ax])
ax_ref = axial_exact(z_ax)
err = np.abs(np.abs(ax_disc) - np.abs(ax_ref)).max() / np.abs(ax_ref).max()
print(f"on-axis: max deviation from the closed form = {err:.2e}")

# The two exact propagators, compared where both are comfortably sampled.
Z_X = 300.0
r_x = np.linspace(0.0, 34.0, 120)
x_asm = replay(rho, Uh_disc, Z_X, r_x)
x_rs1 = rs1_plane(U_disc, r_disc, Z_X, N_MED, LAM, r_x)
d_x = np.abs(np.abs(x_asm) - np.abs(x_rs1)).max() / np.abs(x_rs1).max()
print(f"ASM vs RS1 at z = {Z_X:g}λ: max relative difference = {d_x:.2e}")

# Far field.  RS1 integrates in real space and stays accurate at any distance;
# the ASM would need a far finer ρ grid here, because exp(i kz z) oscillates
# ever faster in ρ as z grows.  Regimes, not favourites.
Z_FF = 3000.0
r_ff = np.linspace(0.0, 900.0, 260)
ff_rs1 = rs1_plane(U_disc, r_disc, Z_FF, N_MED, LAM, r_ff)
ff_ann = rs1_plane(U_ann, r_ann, Z_FF, N_MED, LAM, r_ff)
theta = np.arctan2(r_ff, Z_FF)
airy_r = airy(theta)
i_rs1 = np.abs(ff_rs1) ** 2 / np.abs(ff_rs1[0]) ** 2
d_ff = np.abs(i_rs1 - airy_r).max()
print(f"far field: max |RS1 − Airy| = {d_ff:.2e}")


# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.0, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.40,
                      wspace=0.21, top=0.90, bottom=0.14, left=0.055, right=0.985)


def meridian(a, M, title, sub, aperture):
    rr = np.concatenate([-rg[::-1], rg[1:]])
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
a.plot(z_ax, np.abs(ax_disc) ** 2, color=RED, lw=1.5, ls=(0, (5, 3)),
       label="angular spectrum, disc")
a.plot(z_ax, np.abs(ax_ann) ** 2, color=BLUE, lw=1.5, label="angular spectrum, annulus")
a.set_xlim(z_ax[0], z_ax[-1]); a.set_ylim(0, 4.6)
a.set_xlabel("z  [λ]"); a.set_ylabel("|U|²  on the axis")
a.legend(fontsize=8.5, loc="upper right", ncol=1)
ttl(a, "On-axis intensity, and what it should be",
    f"disc: bright and dark points where a²/λz is odd and even · "
    f"max deviation from the closed form {err:.1e}")

a = fig.add_subplot(gs[1, 1])
xa = r_ff / (0.61 * LAM / (A_OUT / Z_FF))
a.semilogy(xa, airy_r, color=GREEN, lw=3.0, label="Airy pattern [2J₁(x)/x]²")
a.semilogy(xa, i_rs1, color=RED, lw=1.5, ls=(0, (5, 3)),
           label="Rayleigh–Sommerfeld I, disc")
a.semilogy(xa, np.abs(ff_ann) ** 2 / np.abs(ff_ann[0]) ** 2, color=BLUE, lw=1.5,
           label="Rayleigh–Sommerfeld I, annulus")
a.axvline(1.0, color=MUTED, lw=1.0)
a.text(1.06, 3e-5, "first Airy zero\n0.61 λ/NA", fontsize=8, color=MUTED)
a.set_xlim(0, xa[-1]); a.set_ylim(1e-5, 1.6)
a.set_xlabel("r  [Airy radii]"); a.set_ylabel("I / I(0)")
a.legend(fontsize=8.5, loc="upper right")
ttl(a, f"Far field at z = {Z_FF:.0f}λ",
    f"the exact integral against the Fraunhofer limit · max difference "
    f"{d_ff:.1e} · the annulus narrows the core and pumps the rings")

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
