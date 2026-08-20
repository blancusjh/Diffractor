"""White light through a hexagonal aperture — the six-pointed star, in colour.

The one figure in the gallery with no rotational symmetry, so the field lives
on a cartesian `Grid` and the propagators take their FFT/matrix-DFT path
instead of the Hankel one — same operators, different representation.  Two
package facts carry the whole example:

  * `fresnel(field, z, output_grid=…)` evaluates the paraxial integral on a
    window YOU choose (a matrix transform, not an FFT), which is what lets
    every wavelength of a broadband `Field` land on the SAME physical screen —
    the package *refuses* the natural per-λ output grid for broadband input,
    because compositing fields on different screens is meaningless;
  * the trailing spectral axis: one `Field`, 31 wavelengths, one `fresnel`
    call per screen, and `spectral_intensity` comes out ready for the CIE
    compositing of `examples/colorimetry.py`.

What the figure shows: near the aperture the pattern is a hexagon with fringes
and almost no colour (at Fresnel number 6 every wavelength does roughly the
same thing); far from it the six edges throw six spikes, perpendicular to
themselves — a straight edge radiates along its own normal, and the measured
far field falls as r⁻² there against ~r⁻³·⁶ towards the corners.  Each spike
is a channelled spectrum: its fringes sit at radii ∝ λ.  A camera iris does
exactly this to a bright light.

Run:  python examples/05_hexagon_polychromatic.py
"""
import numpy as np
from scipy.ndimage import map_coordinates

from diffractor import (Field, Grid, MonochromaticField, Spectrum, fresnel,
                        fresnel_validity_distance, mm, nm,
                        rayleigh_sommerfeld, um)
from colorimetry import planck, spectrum_to_srgb
from masks import disc_mask, regular_polygon_mask
from style import EDGE, INK2, save, ttl

import matplotlib.pyplot as plt

A_HEX = 50 * um                                # circumradius of the hexagon
SPECTRUM = Spectrum.blackbody(400 * nm, 700 * nm, 31, temperature=6500.0)
NM = SPECTRUM.wavelengths / nm
ILLUM = SPECTRUM.weights
LAM0 = 550 * nm
N_IN, N_OUT = 768, 420

# ── cross-check the cartesian path against the exact axisymmetric one ────────
x_chk = np.linspace(-52.5 * um, 52.5 * um, N_IN)
g_chk = Grid.cartesian(x_chk, x_chk)
disc = MonochromaticField(g_chk, disc_mask(g_chk, A_HEX), LAM0)
z_chk = 100 * mm
w_chk = np.linspace(0.0, 8.0 * LAM0 * z_chk / (2 * A_HEX), 241)
line = fresnel(disc, z_chk,
               output_grid=Grid.cartesian(w_chk, np.zeros(1)))
ref = rayleigh_sommerfeld(
    MonochromaticField(Grid.polar(np.linspace(0.0, A_HEX, 3001)),
                       np.ones((3001, 1)), LAM0),
    z_chk, output_grid=Grid.polar(w_chk))
I_line = np.abs(line.u[:, 0]) ** 2
I_ref = np.abs(ref.u[:, 0]) ** 2
err = np.abs(I_line - I_ref).max() / I_ref.max()
print(f"cartesian fresnel-zoom vs the exact axisymmetric RS1 on a disc: "
      f"{err:.1e} of the peak")

# ── the aperture ─────────────────────────────────────────────────────────────
x_in = np.linspace(-1.05 * A_HEX, 1.05 * A_HEX, N_IN)
grid_in = Grid.cartesian(x_in, x_in)
MASK, NORMALS = regular_polygon_mask(grid_in, 6, A_HEX, orientation=np.pi / 2)
white = Field(grid_in,
              np.repeat(MASK.astype(complex)[..., None], SPECTRUM.n, axis=-1),
              SPECTRUM)

# ── the three screens ────────────────────────────────────────────────────────
Z_FAR = 100 * mm                                         # Fraunhofer regime
NF_NEAR = (3.5, 1.5)                                     # Fresnel numbers a²/λ₀z
Z_NEAR = tuple(A_HEX**2 / (nf * LAM0) for nf in NF_NEAR)
# the gate measures the grid's corner (it cannot know the mask's support)
# at the shortest wavelength (the strictest of the spectrum)
z_gate = fresnel_validity_distance(A_HEX * np.sqrt(2) * 1.05,
                                   SPECTRUM.wavelengths.min())
print(f"paraxial gate: z > {z_gate/um:.0f} µm; screens at "
      + ", ".join(f"{z/um:.0f}" for z in Z_NEAR) + f" and {Z_FAR/um:.0f} µm")
assert min(Z_NEAR) > z_gate, "a screen sits inside the paraxial gate"

WIDTH0 = LAM0 * Z_FAR / (2 * A_HEX)                      # diffraction width
HALF_FAR = 16.0 * WIDTH0
HALF_NEAR = (1.7 * A_HEX, 3.0 * A_HEX)

screens = [("near", Z_NEAR[0], HALF_NEAR[0]),
           ("mid", Z_NEAR[1], HALF_NEAR[1]),
           ("far", Z_FAR, HALF_FAR)]
cubes, grids = {}, {}
for tag, z, half in screens:
    xo = np.linspace(-half, half, N_OUT)
    out = fresnel(white, z, output_grid=Grid.cartesian(xo, xo))
    cubes[tag] = out.spectral_intensity.astype(np.float32)
    grids[tag] = xo
    print(f"  {tag}: z = {z/um:8.0f} µm, a²/λ₀z = {A_HEX**2/(LAM0*z):5.2f}, "
          f"screen ±{half/um:.0f} µm — one broadband fresnel call")

DISPLAY = {"near": dict(stretch=0.45, floor=2e-3),
           "mid": dict(stretch=0.45, floor=1e-3),
           "far": dict(stretch=0.28, floor=4e-5)}
RGB = {t: spectrum_to_srgb(NM, cubes[t] / cubes[t].max(), illuminant=ILLUM,
                           saturation=1.45, brightness=1.25, **DISPLAY[t])
       for t in cubes}

# ── where do the spikes point? ───────────────────────────────────────────────
far = cubes["far"][:, :, np.argmin(np.abs(NM - 550.0))]
x_far = grids["far"]
r_probe = 0.62 * x_far[-1]
th = np.linspace(0.0, 2 * np.pi, 2161)[:-1]


def on_ring(radius, angles, image=far):
    """Sample an image on a circle; values are (x, y) 'ij'-indexed."""
    pix = (radius * np.array([np.cos(angles), np.sin(angles)])
           / (x_far[1] - x_far[0]) + (N_OUT - 1) / 2)
    return map_coordinates(image, pix, order=1)


ring = on_ring(r_probe, th)
want = np.mod(NORMALS, 2 * np.pi)
got = np.array([th[np.abs(np.mod(th - t + np.pi, 2 * np.pi) - np.pi) < np.pi / 7]
                [ring[np.abs(np.mod(th - t + np.pi, 2 * np.pi) - np.pi)
                      < np.pi / 7].argmax()] for t in want])
off = np.degrees(np.abs(np.mod(got - want + np.pi, 2 * np.pi) - np.pi))
contrast = ring.max() / on_ring(r_probe, want + np.pi / 6).max()
print(f"spike directions: {np.degrees(np.sort(got)).round(2)}°")
print(f"edge normals:     {np.degrees(np.sort(want)).round(2)}°  "
      f"(each spike within {off.max():.2f}° of its normal)")
print(f"at r = {r_probe/WIDTH0:.0f} diffraction widths the normals are "
      f"{contrast:.0f}× brighter than the vertex directions")

# Why there is a star at all: the far field falls as r⁻² along an edge normal
# (a straight edge is a line of sources) and faster elsewhere (only the
# corners contribute).  Rotating the aperture 30° puts a vertex direction on
# the x axis, so both cuts are one monochromatic zoomed fresnel each.
r_cut = np.linspace(0.6, 30.0, 700) * WIDTH0
cut_grid = Grid.cartesian(r_cut, np.zeros(1))
mask_v, _ = regular_polygon_mask(grid_in, 6, A_HEX,
                                 orientation=np.pi / 2 + np.pi / 6)
cut, slope = {}, {}
for tag, m in (("edge normal", MASK), ("vertex direction", mask_v)):
    mono = MonochromaticField(grid_in, m.astype(complex), LAM0)
    cut[tag] = np.abs(fresnel(mono, Z_FAR, output_grid=cut_grid).u[:, 0]) ** 2
for tag, I in cut.items():
    env = np.maximum.accumulate(I[::-1])[::-1]
    keep = r_cut > 6 * WIDTH0
    slope[tag] = np.polyfit(np.log(r_cut[keep]), np.log(env[keep]), 1)[0]
    print(f"far field along the {tag}: envelope ∝ r^{slope[tag]:.2f}")

# ── one spike, wavelength by wavelength ──────────────────────────────────────
th_spike = NORMALS[0]
r_line = np.linspace(0.0, x_far[-1], 320)
pix = (r_line[None, :] * np.array([np.cos(th_spike), np.sin(th_spike)])[:, None]
       / (x_far[1] - x_far[0]) + (N_OUT - 1) / 2)
SPIKE = np.array([map_coordinates(cubes["far"][:, :, i], pix, order=1)
                  for i in range(SPECTRUM.n)]).T
SPIKE /= SPIKE.max(axis=0)


# ══ figure ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14.6, 9.9))
gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.30, top=0.905, bottom=0.115,
                      left=0.045, right=0.975)


def show(ax, img, half, title, sub):
    # values are (x, y): transpose for imshow's (row=y, col=x) convention
    ax.imshow(np.transpose(img, (1, 0, 2)),
              extent=np.array([-half, half, -half, half]) / um,
              origin="lower", interpolation="bilinear")
    ax.set_xlabel("x  [µm]"); ax.set_ylabel("y  [µm]")
    ax.grid(False)
    ttl(ax, title, sub)


a = fig.add_subplot(gs[0, 0])
a.imshow(MASK.T, extent=np.array([x_in[0], x_in[-1]] * 2) / um,
         origin="lower", cmap="magma", interpolation="bilinear")
for t in NORMALS:
    a.annotate("", xy=(A_HEX * np.cos(t) / um, A_HEX * np.sin(t) / um),
               xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=EDGE,
                                              lw=1.1))
a.set_xlabel("x  [µm]"); a.set_ylabel("y  [µm]"); a.grid(False)
ttl(a, f"The aperture — hexagon, r = {A_HEX/um:g} µm",
    "cyan: the six edge normals · grey edge pixels by area coverage")

show(fig.add_subplot(gs[0, 1]), RGB["near"], HALF_NEAR[0],
     f"White light, a²/λ₀z = {NF_NEAR[0]:g}",
     f"z = {Z_NEAR[0]/um:.0f} µm · still a hexagon, and barely coloured")
show(fig.add_subplot(gs[0, 2]), RGB["mid"], HALF_NEAR[1],
     f"White light, a²/λ₀z = {NF_NEAR[1]:g}",
     f"z = {Z_NEAR[1]/um:.0f} µm · the shape dissolves, the rays begin")

a = fig.add_subplot(gs[1, 0])
im = a.imshow(np.log10(np.maximum(far / far.max(), 1e-5)).T,
              extent=np.array([-HALF_FAR, HALF_FAR] * 2) / um,
              origin="lower", cmap="magma", vmin=-5, vmax=0,
              interpolation="bilinear")
a.set_xlabel("x  [µm]"); a.set_ylabel("y  [µm]"); a.grid(False)
cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
cb.set_label("log₁₀ ( I / I_max )", fontsize=8.5)
ttl(a, "Far field at 550 nm", f"z = {Z_FAR/mm:.0f} mm · six spikes, "
    "perpendicular to the six edges, and little in between")
ins = a.inset_axes([0.055, 0.60, 0.35, 0.31])
for tag, col in (("edge normal", EDGE), ("vertex direction", "#eb6834")):
    ins.loglog(r_cut / WIDTH0, cut[tag] / cut["edge normal"][0], color=col,
               lw=1.1, label=f"{tag}  ∝ r^{slope[tag]:.1f}")
ins.set_xlabel("r  [diffraction widths]", fontsize=7, labelpad=1)
ins.tick_params(labelsize=6.5)
ins.set_ylim(1e-9, 3)
ins.legend(fontsize=6.5, loc="upper right")
ins.set_title("why there is a star at all", fontsize=7.5, color=INK2, pad=3)

show(fig.add_subplot(gs[1, 1]), RGB["far"], HALF_FAR,
     "The same far field in white light",
     "every spike is a channelled spectrum, and the fringes ride out with λ")

a = fig.add_subplot(gs[1, 2])
im = a.imshow(np.log10(np.maximum(SPIKE, 1e-4)).T,
              extent=[0, r_line[-1] / WIDTH0, NM[0], NM[-1]], origin="lower",
              aspect="auto", cmap="magma", vmin=-4, vmax=0,
              interpolation="bilinear")
for m_line in (2, 5, 9):
    a.plot(m_line * NM / 550.0, NM, color=EDGE, lw=1.3, ls=(0, (5, 3)))
a.text(11.0, 428, "structure ∝ λ", color=EDGE, fontsize=9, fontweight="bold")
a.set_xlim(0, r_line[-1] / WIDTH0)
a.set_xlabel("distance along the spike  [λ₀z/2a at 550 nm]")
a.set_ylabel("wavelength  [nm]")
a.grid(False)
cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
cb.set_label("log₁₀ ( I / I_max at that λ )", fontsize=8.5)
ttl(a, "One spike, wavelength by wavelength",
    "cyan: nodes scaled as λ — the whole of the colour, left")

fig.suptitle("A hexagonal aperture in white light — from its own shadow to a "
             "six-pointed star", fontsize=12.5, fontweight="bold", x=0.006,
             ha="left", y=0.982)
fig.text(0.006, 0.005,
         f"{SPECTRUM.n} wavelengths, 400–700 nm, 6500 K source; one broadband "
         "`Field` on a cartesian grid, one `fresnel(…, output_grid=…)` call "
         "per screen — the package requires the\nexplicit window for broadband "
         "input, because every wavelength must land on the same physical "
         "screen.  Measured spike directions agree with the edge normals to "
         f"{off.max():.1f}°,\nand at {r_probe/WIDTH0:.0f} diffraction widths "
         f"they are {contrast:.0f}× brighter than the vertex directions.  All "
         f"three screens clear the paraxial gate ({z_gate/um:.0f} µm), and "
         "the same zoomed transform\nreproduces the exact axisymmetric "
         f"Rayleigh–Sommerfeld result for a circular aperture to {err:.0e} of "
         "the peak.  Colour panels carry a power stretch over a black floor.",
         fontsize=8.5, color=INK2)
save(fig, "hexagon_polychromatic.png")
