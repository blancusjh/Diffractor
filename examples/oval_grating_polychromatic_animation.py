"""Square grating imaged by a real refracting singlet — an animated white-light lattice.

Combines two features from opposite ends of the toolkit: a periodic
`cross_grating` (a 2D diffraction grating) and an exact stigmatic
`CartesianSurface` — a real single refracting surface, not the idealized
`thin_lens` phase — acting as the focusing element. The surface's conjugates
(``zo`` far away, ``zi`` at the screen) are the physical design of a singlet
lens; :meth:`~diffractor.physics.surfaces.Surface.phase_mask` imprints its exact
thin-element phase on the grating-diffracted field, and `asm_propagator`
carries it to the screen.

At the design plane the far field of the grating — a lattice of diffraction
orders — comes to a focus, one dispersed spot per order under white light
(each off-axis order spreads radially into its own little spectrum, just like
`cross_grating_lens.py`). The animation sweeps the screen through focus: the
lattice sharpens into points at the design plane and blurs again on either
side, all in true color via `PolychromaticField`.

Produces an animated GIF (`oval_grating_animation.gif` next to this script)
and a static PNG of the in-focus frame.

Run with:  python examples/oval_grating_polychromatic_animation.py
"""

import os

import numpy as np

from diffractor import (
    CartesianSurface,
    PolychromaticField,
    cross_grating,
    d65_weights,
    make_grid,
    plot_rgb,
    square_aperture,
)

N = 512
L = 6e-3  # input window [m]
N1, N2 = 1.0, 1.5  # air / glass
ZO = -1e5  # object distance [m] -- effectively collimated illumination
ZI = 0.12  # image (screen) distance [m] -- the singlet's focal plane
PERIOD = 70e-6  # grating period d [m]
BEAM_HALF = 1.6e-3  # illuminated half-width [m]
N_WAVELENGTHS = 20
N_FRAMES = 28
OUTPUT_SAMPLES = 320

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    grid = make_grid(N, L)
    x, y = grid
    surface = CartesianSurface(n1=N1, n2=N2, zo=ZO, zi=ZI)

    grating_mask = cross_grating(x, y, PERIOD, kind="ronchi", duty=0.5) * square_aperture(
        x, y, BEAM_HALF
    )

    wavelengths = np.linspace(430e-9, 660e-9, N_WAVELENGTHS)
    weights = d65_weights(wavelengths * 1e9)

    # The surface is chromatic, so the field is rebuilt per wavelength; n=N2
    # because the field travels in the glass beyond the surface. Build once —
    # .propagate does not mutate it, so it is reused for every frame.
    pf = PolychromaticField(
        grid, grating_mask, wavelengths=wavelengths, weights=weights, n=N2
    ).add_surface(surface, n1=N1, n2=N2)

    x_order = 550e-9 * ZI / PERIOD  # order spacing at the design plane
    screen_half = 2.6 * x_order

    zs = np.linspace(0.82 * ZI, 1.18 * ZI, N_FRAMES)
    disp = dict(
        method="fresnel_zoom", output_half_width=screen_half, output_samples=OUTPUT_SAMPLES,
        gamut="clip", stretch=0.55, saturation=1.4,
    )

    print(f"Rendering {N_FRAMES} polychromatic frames ({N_WAVELENGTHS} wavelengths each)...")
    frames = []
    out_grid = None
    for i, z in enumerate(zs):
        img = pf.propagate(z, **disp)
        frames.append(img.rgb)
        out_grid = img.grid
        print(f"  frame {i+1}/{N_FRAMES}  z = {z*1e3:.2f} mm", flush=True)

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    extent = [out_grid.x.min(), out_grid.x.max(), out_grid.y.min(), out_grid.y.max()]
    im = ax.imshow(
        np.clip(frames[0], 0.0, 1.0), extent=extent, origin="lower",
        interpolation="antialiased", interpolation_stage="rgba",
    )
    title = ax.set_title("")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    def update(i):
        im.set_data(np.clip(frames[i], 0.0, 1.0))
        title.set_text(f"Square grating through a real singlet (z = {zs[i]*1e3:.2f} mm)")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=120, blit=False)
    gif_path = os.path.join(HERE, "oval_grating_animation.gif")
    anim.save(gif_path, writer=animation.PillowWriter(fps=10))
    print(f"Saved animation to {gif_path}")

    # A static hero frame at the design (in-focus) plane.
    i_focus = int(np.argmin(np.abs(zs - ZI)))
    fig2, ax2 = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    plot_rgb(ax2, frames[i_focus], out_grid, title=f"In focus at z = zi = {ZI*1e3:.0f} mm")
    plt.show()


if __name__ == "__main__":
    main()
