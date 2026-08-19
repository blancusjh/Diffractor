"""White light through a hexagonal aperture — the six-pointed star, in colour.

The classic demonstration, and the one figure the axisymmetric core cannot
produce on its own: a hexagon has no rotational symmetry, so it needs the 2-D
Fresnel integral in `examples/cartesian.py` (checked there against the
package's exact Rayleigh–Sommerfeld propagator to 1e-5 of the peak).

Everything else follows the same recipe as `03_polychromatic.py`: one
monochromatic solve per wavelength, all of them onto the SAME physical screen —
which is what the freely-chosen output window of `fresnel_zoom` is for — then
composited through the CIE 1931 functions.

What the figure shows:

  * near the aperture the pattern is a hexagon with fringes, and the colours are
    weak: at high Fresnel number every wavelength does roughly the same thing;
  * far from it the six edges throw six spikes, perpendicular to themselves —
    a straight edge radiates along its own normal, and the measured far field
    falls as r⁻² there against r⁻³·⁶ towards the corners.  Each spike is a
    channelled spectrum: its fringes sit at radii ∝ λ, so their colours change
    along it.  A camera iris does exactly this to a bright light.

Runtime about a minute and a half.

Run:  python examples/05_hexagon_polychromatic.py
"""
import numpy as np
from scipy.ndimage import map_coordinates

from diffractor.propagation.paraxial import fresnel_validity_distance
from cartesian import check_against_core, fresnel_zoom, regular_polygon_mask
from colorimetry import planck, spectrum_to_srgb
from style import EDGE, INK2, MUTED, save, ttl

import matplotlib.pyplot as plt

# micrometres throughout
A_HEX = 50.0                                   # circumradius of the hexagon
NM = np.linspace(400.0, 700.0, 31)
LAM = NM * 1e-3
LAM0 = 0.55
ILLUM = planck(NM, 6500.0)
N_IN, N_OUT = 768, 420

err = check_against_core()[-1]
print(f"2-D Fresnel integral vs the core's exact RS1: {err:.1e} of the peak")

# ── the aperture ─────────────────────────────────────────────────────────────
x_in = np.linspace(-1.05 * A_HEX, 1.05 * A_HEX, N_IN)
MASK, NORMALS = regular_polygon_mask(x_in, x_in, 6, A_HEX, orientation=np.pi / 2,
                                     supersample=4)
U0 = MASK.astype(complex)          # unit plane wave, area-coverage grey edges

# ── the three screens ────────────────────────────────────────────────────────
Z_FAR = 1.0e5                                            # 100 mm: Fraunhofer
NF_NEAR = (6.0, 1.5)                                     # Fresnel numbers a²/λz
Z_NEAR = tuple(A_HEX**2 / (nf * LAM0) for nf in NF_NEAR)
z_gate = fresnel_validity_distance(A_HEX, LAM0)
print(f"paraxial gate: z > {z_gate:.0f} µm; screens at "
      + ", ".join(f"{z:.0f}" for z in Z_NEAR) + f" and {Z_FAR:.0f} µm")
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
    cube = np.empty((N_OUT, N_OUT, NM.size), np.float32)
    for i, lam in enumerate(LAM):
        cube[:, :, i] = np.abs(fresnel_zoom(U0, x_in, x_in, z, lam, xo, xo)) ** 2
    cubes[tag], grids[tag] = cube, xo
    print(f"  {tag}: z = {z:8.0f} µm, a²/λ₀z = {A_HEX**2/(LAM0*z):5.2f}, "
          f"screen ±{half:.0f} µm")

DISPLAY = {"near": dict(stretch=0.45, floor=2e-3), "mid": dict(stretch=0.45, floor=1e-3),
           "far": dict(stretch=0.28, floor=4e-5)}
RGB = {t: spectrum_to_srgb(NM, cubes[t] / cubes[t].max(), illuminant=ILLUM,
                           saturation=1.45, brightness=1.25, **DISPLAY[t])
       for t in cubes}

# ── where do the spikes point? ───────────────────────────────────────────────
far, x_far = cubes["far"][:, :, np.argmin(np.abs(NM - 550.0))], grids["far"]
r_probe = 0.62 * x_far[-1]
th = np.linspace(0.0, 2 * np.pi, 2161)[:-1]


def on_ring(radius, angles, image=far):
    """Sample an image on a circle: rows are y, columns are x."""
    pix = (radius * np.array([np.sin(angles), np.cos(angles)])
           / (x_far[1] - x_far[0]) + (N_OUT - 1) / 2)
    return map_coordinates(image, pix, order=1)


ring = on_ring(r_probe, th)
want = np.mod(NORMALS, 2 * np.pi)
got = np.array([th[np.abs(np.mod(th - t + np.pi, 2 * np.pi) - np.pi) < np.pi / 7]
                [ring[np.abs(np.mod(th - t + np.pi, 2 * np.pi) - np.pi)
                      < np.pi / 7].argmax()] for t in want])
off = np.degrees(np.abs(np.mod(got - want + np.pi, 2 * np.pi) - np.pi))
vertex_dirs = want + np.pi / 6
contrast = ring.max() / on_ring(r_probe, vertex_dirs).max()
print(f"spike directions: {np.degrees(np.sort(got)).round(2)}°")
print(f"edge normals:     {np.degrees(np.sort(want)).round(2)}°  "
      f"(each spike within {off.max():.2f}° of its normal)")
print(f"at r = {r_probe/WIDTH0:.0f} diffraction widths the normals are "
      f"{contrast:.0f}× brighter than the vertex directions")

# Why there is a star at all: the far field falls as r⁻² along an edge normal
# (a straight edge is a line of sources) and as r⁻⁴ elsewhere (only the corners
# contribute).  Rotating the aperture by 30° puts a vertex direction on the x
# axis, so both cuts are separable and cost one transform each.
r_cut = np.linspace(0.6, 30.0, 700) * WIDTH0
mask_v, _ = regular_polygon_mask(x_in, x_in, 6, A_HEX,
                                 orientation=np.pi / 2 + np.pi / 6, supersample=4)
cut = {}
for tag, m in (("edge normal", MASK), ("vertex direction", mask_v)):
    cut[tag] = np.abs(fresnel_zoom(m.astype(complex), x_in, x_in, Z_FAR, LAM0,
                                   r_cut, np.zeros(1))[0]) ** 2
slope = {}
for tag, I in cut.items():
    env = np.maximum.accumulate(I[::-1])[::-1]
    keep = r_cut > 6 * WIDTH0
    slope[tag] = np.polyfit(np.log(r_cut[keep]), np.log(env[keep]), 1)[0]
    print(f"far field along the {tag}: envelope ∝ r^{slope[tag]:.2f}")

# ── one spike, wavelength by wavelength ──────────────────────────────────────
th_spike = NORMALS[0]
r_line = np.linspace(0.0, x_far[-1], 320)
pix = (r_line[None, :] * np.array([np.sin(th_spike), np.cos(th_spike)])[:, None]
       / (x_far[1] - x_far[0]) + (N_OUT - 1) / 2)
SPIKE = np.array([map_coordinates(cubes["far"][:, :, i], pix, order=1)
                  for i in range(NM.size)]).T
SPIKE /= SPIKE.max(axis=0)


# ══ figure ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14.6, 9.9))
gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.30, top=0.905, bottom=0.115,
                      left=0.045, right=0.975)


def show(ax, img, half, title, sub, unit="µm"):
    ax.imshow(img, extent=[-half, half, -half, half], origin="lower",
              interpolation="bilinear")
    ax.set_xlabel(f"x  [{unit}]"); ax.set_ylabel(f"y  [{unit}]")
    ax.grid(False)
    ttl(ax, title, sub)


a = fig.add_subplot(gs[0, 0])
a.imshow(MASK, extent=[x_in[0], x_in[-1]] * 2, origin="lower", cmap="magma",
         interpolation="bilinear")
for t in NORMALS:
    a.annotate("", xy=(1.0 * A_HEX * np.cos(t), 1.0 * A_HEX * np.sin(t)),
               xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=EDGE, lw=1.1))
a.set_xlabel("x  [µm]"); a.set_ylabel("y  [µm]"); a.grid(False)
ttl(a, f"The aperture — hexagon, r = {A_HEX:g} µm",
    "cyan: the six edge normals · grey edge pixels by area coverage")

show(fig.add_subplot(gs[0, 1]), RGB["near"], HALF_NEAR[0],
     f"White light, a²/λ₀z = {NF_NEAR[0]:g}",
     f"z = {Z_NEAR[0]:.0f} µm · still a hexagon, and barely coloured")
show(fig.add_subplot(gs[0, 2]), RGB["mid"], HALF_NEAR[1],
     f"White light, a²/λ₀z = {NF_NEAR[1]:g}",
     f"z = {Z_NEAR[1]:.0f} µm · the shape dissolves, the rays begin")

a = fig.add_subplot(gs[1, 0])
im = a.imshow(np.log10(np.maximum(far / far.max(), 1e-5)),
              extent=[-HALF_FAR, HALF_FAR] * 2, origin="lower", cmap="magma",
              vmin=-5, vmax=0, interpolation="bilinear")
a.set_xlabel("x  [µm]"); a.set_ylabel("y  [µm]"); a.grid(False)
cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
cb.set_label("log₁₀ ( I / I_max )", fontsize=8.5)
ttl(a, "Far field at 550 nm", f"z = {Z_FAR*1e-3:.0f} mm · six spikes, "
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
for m in (2, 5, 9):
    a.plot(m * NM / 550.0, NM, color=EDGE, lw=1.3, ls=(0, (5, 3)))
a.text(11.0, 428, "structure ∝ λ", color=EDGE, fontsize=9,
       fontweight="bold")
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
         f"{NM.size} wavelengths, 400–700 nm, 6500 K source; every wavelength "
         "propagated onto the same physical screen by the 2-D Fresnel integral "
         "of `examples/cartesian.py`, then\ncomposited through the CIE 1931 "
         "colour-matching functions.  Measured spike directions agree with the "
         f"edge normals to {off.max():.1f}°, and at {r_probe/WIDTH0:.0f} "
         f"diffraction widths they are {contrast:.0f}× brighter than the vertex "
         "directions.  All "
         "three screens clear the paraxial gate\n"
         f"(`fresnel_validity_distance` = {z_gate:.0f} µm), and the propagator "
         f"itself reproduces the core's exact Rayleigh–Sommerfeld result for a "
         f"circular aperture to {err:.0e} of the peak.  The colour panels carry "
         "a power stretch (0.45 near, 0.28 far) over a black floor, so that "
         "structure four decades down still prints.",
         fontsize=8.5, color=INK2)
save(fig, "hexagon_polychromatic.png")
