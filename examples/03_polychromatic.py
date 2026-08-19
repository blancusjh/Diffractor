"""White light: what a broadband source does to a diffraction pattern.

`diffractor` is monochromatic by construction — a wavelength is an argument to
a propagator, not a state of the field.  Broadband is therefore not a different
physics but a different *bookkeeping*: solve at each wavelength on a shared
output grid, then integrate the per-wavelength intensities against the CIE
colour-matching functions (`examples/colorimetry.py`).

Two consequences of λ appearing where it does:

  * a diffraction pattern scales as λ, so the rings of an aperture fan out into
    spectra — blue inside, red outside;
  * a zone plate's focal length is f(λ) = f₀ λ₀/λ, so white light is not
    focused at all but smeared along the axis.  Diffractive optics has
    chromatic aberration of the opposite sign, and a hundred times worse, than
    a glass lens.

Runtime a couple of minutes: it is ~20 independent monochromatic solves.

Run:  python examples/03_polychromatic.py
"""
import numpy as np

from diffractor.propagation import hankel, rs1_plane
from colorimetry import planck, spectrum_to_srgb, wavelength_to_rgb
from style import EDGE, INK2, MUTED, save, ttl

import matplotlib.pyplot as plt

# Everything in micrometres.
NM = np.linspace(400.0, 700.0, 31)
LAM = NM * 1e-3
ILLUM = planck(NM, 6500.0)            # a 6500 K source, i.e. daylight-ish
N_MED = 1.0


# ══ 1. white-light Airy pattern of a circular aperture ═══════════════════════
A_AP, Z_AP = 60.0, 1.5e5              # 60 µm radius, 150 mm away
r_ap = np.linspace(0.0, A_AP, 2500)
U_ap = np.ones_like(r_ap, dtype=complex)
r_scr = np.linspace(0.0, 3200.0, 420)
print(f"aperture: {NM.size} wavelengths at z = {Z_AP*1e-3:.0f} mm")
S_ap = np.stack([np.abs(rs1_plane(U_ap, r_ap, Z_AP, N_MED, lam, r_scr)) ** 2
                 for lam in LAM], axis=-1)
S_ap /= S_ap.max()

# revolve the radial profile into an image
half = r_scr[-1] / np.sqrt(2.0)          # inscribed square: no empty corners
side = np.linspace(-half, half, 420)
X, Y = np.meshgrid(side, side)
RR = np.hypot(X, Y)
cube = np.stack([np.interp(RR, r_scr, S_ap[:, i], right=0.0)
                 for i in range(NM.size)], axis=-1)
IMG_AP = spectrum_to_srgb(NM, cube, illuminant=ILLUM, stretch=0.42,
                          saturation=1.35, brightness=1.15)

airy0 = 0.61 * 0.55 / (A_AP / Z_AP)   # Airy radius at 550 nm


# ══ 2. zone plate in white light ═════════════════════════════════════════════
LAM0, F0, M_ZONES = 0.55, 200.0, 30
R_ZP = np.sqrt(M_ZONES * LAM0 * F0)
NA_ZP = R_ZP / np.hypot(R_ZP, F0)
r_zp = np.linspace(0.0, R_ZP, 2500)
t_zp = (np.floor(r_zp**2 / (LAM0 * F0)) % 2 == 0).astype(complex)
print(f"zone plate: R = {R_ZP:.1f} µm, f(550 nm) = {F0:g} µm, NA = {NA_ZP:.2f}")

zg = np.linspace(120.0, 330.0, 150)
rg = np.linspace(0.0, 6.0, 65)
z_ax = np.linspace(120.0, 330.0, 260)

MAP = np.empty((rg.size, zg.size, NM.size))
AXI = np.empty((z_ax.size, NM.size))
for i, lam in enumerate(LAM):
    rho = np.linspace(0.0, N_MED / lam, 6000)
    Uh = hankel(t_zp, r_zp, rho)
    kz = 2 * np.pi * np.sqrt(np.maximum((N_MED / lam) ** 2 - rho**2, 0.0))
    for j, z in enumerate(zg):
        MAP[:, j, i] = np.abs(hankel(Uh * np.exp(1j * kz * z), rho, rg)) ** 2
    AXI[:, i] = np.abs(np.array([hankel(Uh * np.exp(1j * kz * z), rho,
                                        np.zeros(1))[0] for z in z_ax])) ** 2
    print(f"  {NM[i]:.0f} nm done", end="\r", flush=True)
print(" " * 30, end="\r")

focus = np.array([z_ax[AXI[:, i].argmax()] for i in range(NM.size)])
pred = F0 * LAM0 / LAM
sel = (NM >= 430) & (NM <= 680)
print(f"focal law f = f₀λ₀/λ: max deviation of the measured peak "
      f"{np.abs(focus - pred)[sel].max():.1f} µm over 430–680 nm")

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
for m in (1, 2, 3):
    a.add_patch(plt.Circle((0, 0), m, fill=False, color="white", lw=0.6,
                           alpha=0.30))
a.set_xlabel("x  [Airy radii at 550 nm]"); a.set_ylabel("y  [Airy radii]")
a.grid(False)
ttl(a, "A circular aperture in white light",
    f"a = {A_AP:g} µm, screen at {Z_AP*1e-3:.0f} mm · 6500 K source · "
    f"circles at 1, 2, 3 Airy radii")

a = fig.add_subplot(gs[0, 1])
im = a.imshow(np.log10(np.maximum(S_ap / S_ap.max(axis=0), 1e-4)).T,
              extent=[0, r_scr[-1] / airy0, NM[0], NM[-1]], origin="lower",
              aspect="auto", cmap="magma", vmin=-4, vmax=0,
              interpolation="bilinear")
for zero, lab in ((1.0, "1st"), (1.63, "2nd"), (2.23, "3rd")):
    a.plot(zero * NM / 550.0, NM, color=EDGE, lw=1.4, ls=(0, (5, 3)))
a.text(2.55, 470, "dark rings ∝ λ", color=EDGE, fontsize=9, fontweight="bold")
a.set_xlim(0, r_scr[-1] / airy0)
a.set_xlabel("r  [Airy radii at 550 nm]"); a.set_ylabel("wavelength  [nm]")
a.grid(False)
cb = fig.colorbar(im, ax=a, fraction=0.030, pad=0.012)
cb.set_label("log₁₀ ( I / I_max at that λ )", fontsize=8.5)
ttl(a, "The same pattern, wavelength by wavelength",
    "cyan: the Airy zeros scaled as λ — this is the whole of the colour above")

a = fig.add_subplot(gs[1, :])
rr = np.concatenate([-rg[::-1], rg[1:]])
a.imshow(IMG_ZP, extent=[zg[0], zg[-1], rr[0], rr[-1]], origin="lower",
         aspect="auto", interpolation="bilinear")
a.plot(pred, np.zeros_like(pred) - 5.2, ".", ms=0)      # keep limits
for i in range(0, NM.size, 3):
    a.plot([pred[i]], [-5.3], "v", ms=6,
           color=wavelength_to_rgb(NM[i], brightness=1.0)[0], mec="white",
           mew=0.4)
a.text(pred[0] + 3, -5.3, "  f = f₀λ₀/λ", color="white", fontsize=9,
       va="center")
a.set_xlabel("z  [µm]"); a.set_ylabel("r  [µm]"); a.grid(False)
ttl(a, f"A Fresnel zone plate in white light  "
       f"(R = {R_ZP:.0f} µm, f = {F0:g} µm at 550 nm, NA = {NA_ZP:.2f})",
    "true-colour meridional section · every wavelength focuses somewhere else, "
    "so there is no white focus at all")

fig.suptitle("Polychromatic diffraction — one monochromatic solve per "
             "wavelength, composited through the CIE 1931 functions",
             fontsize=12.5, fontweight="bold", x=0.006, ha="left", y=0.985)
fig.text(0.006, 0.005,
         f"{NM.size} wavelengths, 400–700 nm, weighted by a 6500 K blackbody; "
         "each panel is normalised so that the undiffracted source renders "
         "white, so hue means\n\"how this pattern differs, spectrally, from the "
         "light that made it\".  The measured focus follows f₀λ₀/λ to "
         f"{np.abs(focus - pred)[sel].max():.1f} µm across 430–680 nm — a "
         f"{100*(pred[0]-pred[-1])/F0:.0f} % focal spread over the visible, "
         "against\nunder 1 % for a glass singlet.  Intensities are shown with a "
         "0.4-power stretch so the outer structure survives the print.",
         fontsize=8.5, color=INK2)
save(fig, "polychromatic.png")
