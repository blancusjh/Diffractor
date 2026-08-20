"""White light: what a broadband source does to a diffraction pattern.

Broadband is not a different physics but a different bookkeeping, and the
`Field` does the bookkeeping: its values carry a trailing spectral axis, its
`Spectrum` carries the source's lines and weights, and every propagator
vectorises over that axis — so a white-light computation is ONE call, not a
hand-written loop over wavelengths.  The per-line intensities come back from
`Field.spectral_intensity`, ready for the CIE compositing in
`examples/colorimetry.py`.

Two consequences of λ appearing where it does:

  * a diffraction pattern scales as λ, so the rings of an aperture fan out
    into spectra — blue inside, red outside;
  * a zone plate's focal length is f(λ) = f₀λ₀/λ, so white light is not
    focused at all but smeared along the axis.  Diffractive optics has
    chromatic aberration of the opposite sign, and a hundred times worse,
    than a glass lens.

Run:  python examples/03_polychromatic.py
"""
import numpy as np

from diffractor import (FT2, IFT2, Field, Grid, PolarPlan, Spectrum, mm,
                        nm, rayleigh_sommerfeld, um)
from colorimetry import spectrum_to_srgb, wavelength_to_rgb
from style import EDGE, INK2, save, ttl

import matplotlib.pyplot as plt

SPECTRUM = Spectrum.blackbody(400 * nm, 700 * nm, 31, temperature=6500.0)
NM = SPECTRUM.wavelengths / nm          # for the colorimetry, in nanometres
ILLUM = SPECTRUM.weights


def unit_white(grid, mask=None):
    """A white plane wave (every line at unit amplitude) through a mask."""
    values = np.ones((*grid.shape, SPECTRUM.n), complex)
    if mask is not None:
        values *= mask[..., None]
    return Field(grid, values, SPECTRUM)


def sweep(field, zs, out_grid, density=1.0):
    """Spectral intensity on out_grid at each z — every wavelength at once."""
    k_cut = 2 * np.pi / field.spectrum.wavelengths.min()
    r_max = field.grid.axes[0][-1]
    kgrid = field.grid.reciprocal(
        k_max=1.02 * k_cut,
        n_k=int(density * (np.ceil(5.0 * 1.02 * k_cut * (r_max + zs.max())
                                   / (2 * np.pi)) + 1)))
    plan_in = PolarPlan.build(field.grid, kgrid)
    plan_out = PolarPlan.build(out_grid, kgrid)
    F = FT2(field, kgrid=kgrid, plan=plan_in)
    k = kgrid.axes[0][:, None, None]
    k_m = (2 * np.pi / field.spectrum.wavelengths)[None, None, :]
    kz = np.sqrt(np.maximum(k_m**2 - k**2, 0.0))
    live = k**2 <= k_m**2
    out = np.empty((out_grid.axes[0].size, zs.size, SPECTRUM.n))
    for j, z in enumerate(zs):
        Fz = F.like(np.where(live, F.values * np.exp(1j * kz * z), 0.0))
        out[:, j, :] = IFT2(Fz, grid=out_grid,
                            plan=plan_out).spectral_intensity[:, 0, :]
    return out


# ══ 1. white-light Airy pattern of a circular aperture ═══════════════════════
A_AP, Z_AP = 60 * um, 150 * mm
aperture = unit_white(Grid.polar(np.linspace(0.0, A_AP, 2500)))
screen = Grid.polar(np.linspace(0.0, 3200 * um, 420))
print(f"aperture: {SPECTRUM.n} wavelengths at z = {Z_AP/mm:.0f} mm — one call")
S_ap = rayleigh_sommerfeld(aperture, Z_AP, output_grid=screen)
S = S_ap.spectral_intensity[:, 0, :]                     # (n_r, n_λ)
S /= S.max()
r_scr = screen.axes[0]

# revolve the radial profile into an image (inscribed square: no empty corners)
half = r_scr[-1] / np.sqrt(2.0)
side = np.linspace(-half, half, 420)
X, Y = np.meshgrid(side, side)
RR = np.hypot(X, Y)
cube = np.stack([np.interp(RR, r_scr, S[:, i], right=0.0)
                 for i in range(SPECTRUM.n)], axis=-1)
IMG_AP = spectrum_to_srgb(NM, cube, illuminant=ILLUM, stretch=0.42,
                          saturation=1.35, brightness=1.15)

airy0 = 0.61 * (550 * nm) / (A_AP / Z_AP)   # Airy radius at 550 nm


# ══ 2. zone plate in white light ═════════════════════════════════════════════
LAM0, F0, M_ZONES = 550 * nm, 200 * um, 30
R_ZP = np.sqrt(M_ZONES * LAM0 * F0)
NA_ZP = R_ZP / np.hypot(R_ZP, F0)
zp_grid = Grid.polar(np.linspace(0.0, R_ZP, 2500))
r_zp = zp_grid.axes[0]
zone_plate = unit_white(
    zp_grid,
    (np.floor(r_zp**2 / (LAM0 * F0)) % 2 == 0).astype(complex)[:, None])
print(f"zone plate: R = {R_ZP/um:.1f} µm, f(550 nm) = {F0/um:g} µm, "
      f"NA = {NA_ZP:.2f}")

zg = np.linspace(120 * um, 330 * um, 150)
map_grid = Grid.polar(np.linspace(0.0, 6 * um, 65))
print(f"  meridional map: {zg.size} planes × {SPECTRUM.n} wavelengths")
MAP = sweep(zone_plate, zg, map_grid)                    # (n_r, n_z, n_λ)

z_ax = np.linspace(120 * um, 330 * um, 260)
AXI = sweep(zone_plate, z_ax, Grid.polar(np.array([0.0, 1e-9])))[0]

focus = np.array([z_ax[AXI[:, i].argmax()] for i in range(SPECTRUM.n)])
pred = F0 * LAM0 / SPECTRUM.wavelengths
sel = (NM >= 430) & (NM <= 680)
print(f"focal law f = f₀λ₀/λ: max deviation of the measured peak "
      f"{np.abs(focus - pred)[sel].max()/um:.1f} µm over 430–680 nm")

IMG_ZP = spectrum_to_srgb(NM, MAP / MAP.max(), illuminant=ILLUM, stretch=0.45,
                          saturation=1.3, brightness=1.25)
IMG_ZP = np.concatenate([IMG_ZP[::-1], IMG_ZP[1:]], axis=0)


# ══ figure ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14.0, 9.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], hspace=0.42, wspace=0.20,
                      top=0.915, bottom=0.10, left=0.055, right=0.985)

a = fig.add_subplot(gs[0, 0])
ext = np.array([-1, 1, -1, 1]) * half / airy0
a.imshow(IMG_AP, extent=ext, origin="lower", interpolation="bilinear")
for m_ring in (1, 2, 3):
    a.add_patch(plt.Circle((0, 0), m_ring, fill=False, color="white", lw=0.6,
                           alpha=0.30))
a.set_xlabel("x  [Airy radii at 550 nm]"); a.set_ylabel("y  [Airy radii]")
a.grid(False)
ttl(a, "A circular aperture in white light",
    f"a = {A_AP/um:g} µm, screen at {Z_AP/mm:.0f} mm · 6500 K source · "
    f"circles at 1, 2, 3 Airy radii")

a = fig.add_subplot(gs[0, 1])
im = a.imshow(np.log10(np.maximum(S / S.max(axis=0), 1e-4)).T,
              extent=[0, r_scr[-1] / airy0, NM[0], NM[-1]], origin="lower",
              aspect="auto", cmap="magma", vmin=-4, vmax=0,
              interpolation="bilinear")
for zero in (1.0, 1.63, 2.23):
    a.plot(zero * NM / 550.0, NM, color=EDGE, lw=1.4, ls=(0, (5, 3)))
a.text(2.55, 470, "dark rings ∝ λ", color=EDGE, fontsize=9, fontweight="bold")
a.set_xlim(0, r_scr[-1] / airy0)
a.set_xlabel("r  [Airy radii at 550 nm]"); a.set_ylabel("wavelength  [nm]")
a.grid(False)
cb = fig.colorbar(im, ax=a, fraction=0.030, pad=0.012)
cb.set_label("log₁₀ ( I / I_max at that λ )", fontsize=8.5)
ttl(a, "The same pattern, wavelength by wavelength",
    "one Field, spectral axis unpacked · cyan: the Airy zeros scaled as λ — "
    "the whole of the colour to the left")

a = fig.add_subplot(gs[1, :])
r_map = map_grid.axes[0]
rr = np.concatenate([-r_map[::-1], r_map[1:]])
a.imshow(IMG_ZP, extent=[zg[0] / um, zg[-1] / um, rr[0] / um, rr[-1] / um],
         origin="lower", aspect="auto", interpolation="bilinear")
for i in range(0, SPECTRUM.n, 3):
    a.plot([pred[i] / um], [-5.3], "v", ms=6,
           color=wavelength_to_rgb(NM[i], brightness=1.0)[0], mec="white",
           mew=0.4)
a.text(pred[0] / um + 3, -5.3, "  f = f₀λ₀/λ", color="white", fontsize=9,
       va="center")
a.set_xlabel("z  [µm]"); a.set_ylabel("r  [µm]"); a.grid(False)
ttl(a, f"A Fresnel zone plate in white light  "
       f"(R = {R_ZP/um:.0f} µm, f = {F0/um:g} µm at 550 nm, NA = {NA_ZP:.2f})",
    "true-colour meridional section · every wavelength focuses somewhere "
    "else, so there is no white focus at all")

fig.suptitle("Polychromatic diffraction — one broadband Field, one call per "
             "screen, composited through the CIE 1931 functions",
             fontsize=12.5, fontweight="bold", x=0.006, ha="left", y=0.985)
fig.text(0.006, 0.005,
         f"{SPECTRUM.n} wavelengths, 400–700 nm, weighted by a 6500 K blackbody "
         "(`Spectrum.blackbody`); each panel is normalised so that the "
         "undiffracted source renders white, so hue\nmeans \"how this pattern "
         "differs, spectrally, from the light that made it\".  The measured "
         f"focus follows f₀λ₀/λ to {np.abs(focus - pred)[sel].max()/um:.1f} µm "
         f"across 430–680 nm — a "
         f"{100*(pred[0]-pred[-1])/F0:.0f} % focal spread over\nthe visible, "
         "against under 1 % for a glass singlet.  Intensities are shown with "
         "a 0.4-power stretch so the outer structure survives the print.",
         fontsize=8.5, color=INK2)
save(fig, "polychromatic.png")
