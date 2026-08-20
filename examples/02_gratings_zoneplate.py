"""Circular gratings and the Fresnel zone plate.

Two axisymmetric periodic screens, both `MonochromaticField`s on polar grids,
propagated by the package's exact operators:

  * a **ring grating** — transmission modulated in r with period d.  It sends
    light into rings at sinθ = m λ/d.  A sinusoidal modulation makes only
    m = 0, ±1; a binary (Ronchi) one makes every odd order as well.
  * a **Fresnel zone plate** — the same binary screen, but chirped so that
    every clear zone arrives in phase at one point: edges at r_m = sqrt(mλf).
    It is a lens made of diffraction alone, and it keeps the odd orders of the
    grating it came from: they focus at f/3, f/5, …

Far-field orders by `rayleigh_sommerfeld` (the real-space exact reference,
comfortable at any distance); the meridional map by the angular spectrum
through the operator layer (`FT2` once, one transfer per plane).

Everything is in units of the wavelength (λ = 1).

Run:  python examples/02_gratings_zoneplate.py
"""
import numpy as np
from scipy.special import j1

from diffractor import (FT2, IFT2, Grid, MonochromaticField, PolarPlan,
                        rayleigh_sommerfeld)
from style import EDGE, GREEN, INK2, MUTED, ORANGE, RED, VIOLET, save, ttl

import matplotlib.pyplot as plt

LAM = 1.0
K = 2 * np.pi / LAM


def sweep(field, zs, out_grid, density=1.0):
    """|U| on out_grid at each z: FT2 once, one transfer factor per plane."""
    r_max = field.grid.axes[0][-1]
    kgrid = field.grid.reciprocal(
        k_max=1.02 * K,
        n_k=int(density * (np.ceil(5.0 * 1.02 * K * (r_max + zs.max())
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


# ══ 1. ring gratings ═════════════════════════════════════════════════════════
D, A_G = 8.0, 100.0                    # period, aperture radius
grating_grid = Grid.polar(np.linspace(0.0, A_G, 16001))
r_g = grating_grid.axes[0]
t_sin = 0.5 * (1.0 + np.cos(2 * np.pi * r_g / D))
t_bin = (np.cos(2 * np.pi * r_g / D) >= 0).astype(float)
f_sin = MonochromaticField(grating_grid, t_sin[:, None], LAM)
f_bin = MonochromaticField(grating_grid, t_bin[:, None], LAM)

# Fraunhofer needs z >> a²/λ; put the screen 30 Rayleigh ranges away and read
# the pattern off in angle.
Z_G = 30.0 * A_G**2 / LAM
sin_th = np.linspace(0.0, 0.52, 1100)
screen = Grid.polar(Z_G * sin_th / np.sqrt(1.0 - sin_th**2))
print(f"ring gratings: far field at z = {Z_G:.0f}λ  (a²/λz = {A_G**2/Z_G:.3f})")
I_sin = np.abs(rayleigh_sommerfeld(f_sin, Z_G, output_grid=screen).u[:, 0]) ** 2
I_bin = np.abs(rayleigh_sommerfeld(f_bin, Z_G, output_grid=screen).u[:, 0]) ** 2
norm = I_bin.max()

half_width = LAM / (2 * A_G)           # half-width of one order
shifts = []
for m in (1, 3):
    s0 = m * LAM / D
    w = np.abs(sin_th - s0) < 0.4 * LAM / D
    peak = sin_th[w][I_bin[w].argmax()]
    shifts.append((peak - s0) / half_width)
    print(f"  order {m}: grating equation {s0:.4f}, measured {peak:.4f} "
          f"({shifts[-1]:+.2f} order half-widths)")


# ══ 2. Fresnel zone plate ════════════════════════════════════════════════════
F_ZP, M_ZONES = 400.0, 30
R_ZP = np.sqrt(M_ZONES * LAM * F_ZP)
NA_ZP = R_ZP / np.hypot(R_ZP, F_ZP)
zp_grid = Grid.polar(np.linspace(0.0, R_ZP, 9001))
r_z = zp_grid.axes[0]
t_zp = (np.floor(r_z**2 / (LAM * F_ZP)) % 2 == 0).astype(complex)
f_zp = MonochromaticField(zp_grid, t_zp[:, None], LAM)
print(f"zone plate: {M_ZONES} zones, R = {R_ZP:.1f}λ, f = {F_ZP:g}λ, "
      f"NA = {NA_ZP:.3f}, outer zone width {LAM*F_ZP/(2*R_ZP):.2f}λ")

zg = np.linspace(45.0, 620.0, 300)
map_grid = Grid.polar(np.linspace(0.0, 22.0, 150))
print(f"  meridional map ({zg.size} planes)")
MAP = sweep(f_zp, zg, map_grid)

z_ax = np.linspace(45.0, 620.0, 1400)
I_ax = sweep(f_zp, z_ax, Grid.polar(np.array([0.0, 1e-6])), density=3.0)[0] ** 2
peaks = {}
for m in (1, 3, 5):
    w = np.abs(z_ax - F_ZP / m) < 0.12 * F_ZP / m
    peaks[m] = z_ax[w][I_ax[w].argmax()]
    print(f"  order {m}: focus expected at f/{m} = {F_ZP/m:6.1f}λ, "
          f"peak at {peaks[m]:6.1f}λ  ({100*(peaks[m]-F_ZP/m)/(F_ZP/m):+.1f} %)")

foc_grid = Grid.polar(np.linspace(0.0, 9.0, 400))
I_foc = sweep(f_zp, np.array([peaks[1]]), foc_grid, density=3.0)[:, 0] ** 2
I_foc /= I_foc[0]
r_foc = foc_grid.axes[0]
x = K * R_ZP * (r_foc / np.hypot(r_foc, F_ZP))
airy_foc = np.where(x == 0, 1.0,
                    (2 * j1(np.where(x == 0, 1, x)) / np.where(x == 0, 1, x)) ** 2)
airy_radius = 0.61 * LAM / NA_ZP
print(f"  focal spot: Airy radius 0.61λ/NA = {airy_radius:.2f}λ")


# ══ figure ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14.0, 11.4))
gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.5],
                      height_ratios=[0.78, 1.0, 0.68], hspace=0.44, wspace=0.20,
                      top=0.925, bottom=0.10, left=0.055, right=0.985)

a = fig.add_subplot(gs[0, 0])
a.plot(r_g, t_bin, color=VIOLET, lw=1.6, label="binary (Ronchi) rings")
a.plot(r_g, t_sin, color=ORANGE, lw=1.6, label="sinusoidal rings")
a.set_xlim(0, 5 * D); a.set_ylim(-0.08, 1.3)
a.set_xlabel("r  [λ]"); a.set_ylabel("amplitude transmission  t(r)")
a.legend(fontsize=8.5, loc="upper right")
ttl(a, "Two ring gratings", f"period d = {D:g}λ, aperture radius {A_G:g}λ")

a = fig.add_subplot(gs[0, 1])
a.semilogy(sin_th, I_bin / norm, color=VIOLET, lw=1.4, label="binary rings")
a.semilogy(sin_th, I_sin / norm, color=ORANGE, lw=1.4, label="sinusoidal rings")
for m in (1, 2, 3, 4):
    s0 = m * LAM / D
    a.axvline(s0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    a.text(s0, 2.0, f"m = {m}", fontsize=8, color=MUTED, ha="center")
a.set_xlim(0, sin_th[-1]); a.set_ylim(1e-6, 6.0)
a.set_xlabel("sin θ"); a.set_ylabel("I / I_max")
a.legend(fontsize=8.5, loc="lower left")
ttl(a, f"Diffracted orders in the far field",
    "dashed: the grating equation sinθ = mλ/d · the sinusoidal screen carries "
    "only m = 0, ±1 · each ring order is a doublet: the two sides of the ring "
    "interfere")

a = fig.add_subplot(gs[1, :])
rg = map_grid.axes[0]
rr = np.concatenate([-rg[::-1], rg[1:]])
MM = np.vstack([MAP[::-1], MAP[1:]])
im = a.imshow(np.log10(np.maximum(MM, 10**-1.6)),
              extent=[zg[0], zg[-1], rr[0], rr[-1]], origin="lower",
              aspect="auto", cmap="magma", vmin=-1.6, vmax=np.log10(MM.max()),
              interpolation="bilinear")
for m, lab in ((5, "f/5"), (3, "f/3"), (1, "f")):
    a.plot([F_ZP / m, F_ZP / m], [-22, -16.5], color=EDGE, lw=1.3)
    a.text(F_ZP / m, -15.0, lab, color=EDGE, fontsize=10, ha="center",
           fontweight="bold")
a.set_xlabel("z  [λ]"); a.set_ylabel("r  [λ]"); a.grid(False)
cb = fig.colorbar(im, ax=a, fraction=0.020, pad=0.008)
cb.set_label("log₁₀ |U|", fontsize=8.5)
ttl(a, f"Fresnel zone plate — {M_ZONES} zones, R = {R_ZP:.0f}λ, f = {F_ZP:g}λ "
       f"(NA = {NA_ZP:.2f})",
    "one screen, three foci: the odd diffraction orders converge at f, f/3 "
    "and f/5 — the price of a binary lens")

a = fig.add_subplot(gs[2, 0])
a.plot(r_foc / airy_radius, I_foc, color=RED, lw=1.8, label="zone plate, z = focus")
a.plot(r_foc / airy_radius, airy_foc, color=GREEN, lw=1.6, ls=(0, (4, 3)),
       label="Airy pattern, same NA")
a.set_xlim(0, 3); a.set_ylim(0, 1.15)
a.set_xlabel("r  [Airy radii]"); a.set_ylabel("I / I(0)")
a.legend(fontsize=8.5, loc="upper right")
ttl(a, "The focal spot", f"Airy radius 0.61λ/NA = {airy_radius:.2f}λ")

a = fig.add_subplot(gs[2, 1])
a.semilogy(z_ax, I_ax / I_ax.max(), color=RED, lw=1.5)
for m in (1, 3, 5):
    a.axvline(F_ZP / m, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    a.text(F_ZP / m, 1.6, f"f/{m}" if m > 1 else "f", fontsize=8.5, color=MUTED,
           ha="center")
a.set_xlim(zg[0], zg[-1]); a.set_ylim(3e-4, 3.2)
a.set_xlabel("z  [λ]"); a.set_ylabel("I / I_max  on the axis")
ttl(a, "On-axis intensity through the focal series",
    f"the m = 1 peak sits at {peaks[1]:.0f}λ, "
    f"{100*(F_ZP-peaks[1])/F_ZP:.0f} % inside the geometric f — the focal shift of a "
    f"finite-aperture beam (Fresnel number R²/λf = {R_ZP**2/(LAM*F_ZP):.0f})")

fig.suptitle("Gratings that are rings — and the zone plate, which is a lens "
             "made of them", fontsize=12.5, fontweight="bold", x=0.006,
             ha="left", y=0.988)
fig.text(0.006, 0.005,
         "Plane wave, λ = 1, n = 1.  Far-field orders by Rayleigh–Sommerfeld, "
         "maps by the exact angular spectrum — the package operators throughout.\n"
         "The measured ring orders sit within "
         f"{max(abs(s) for s in shifts):.1f} half-widths of sinθ = mλ/d (a ring "
         "grating is not a plane one: the r dr weight of the transform pulls the "
         "peak).\nThe zone plate focuses to the Airy pattern of its own "
         "aperture — a binary screen still reaches the diffraction limit; it "
         "just sends most of the light elsewhere.",
         fontsize=8.5, color=INK2)
save(fig, "gratings_zoneplate.png")
