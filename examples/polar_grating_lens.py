"""A polar grating at a lens focus — a polar lattice of orders.

The rotational counterpart of `cross_grating_lens.py`. A polar grating is
periodic in polar coordinates: concentric rings of radial period ``d_r``
(a circular grating) crossed with ``n_spokes`` angular sectors (a Siemens-star
pattern). A converging lens brings its far field to the back focal plane, where

* the radial rings diffract into ring-shaped orders at radii ``m λ f / d_r``,
* the angular spokes spread each ring into ``2·n_spokes`` azimuthal lobes,

so the focus shows a centered *polar* lattice of orders — arcs and spots laid
out on concentric rings instead of a square grid.

Three panels: the polar grating, its monochromatic focal-plane pattern, and —
since the lens phase is chromatic — the white-light version, where every
off-axis order disperses radially into a little spectrum
(a `PolychromaticField` replaying the lens phase per wavelength).

Run with:  python examples/polar_grating_lens.py
"""

import numpy as np

from diffractor import (
    MonochromaticField,
    PolychromaticField,
    circular_aperture,
    d65_weights,
    make_grid,
    polar_grating,
)

N = 1024
L = 8e-3  # input window [m]
WAVELENGTH = 550e-9
FOCAL_LENGTH = 0.25  # lens focal length [m]
RADIAL_PERIOD = 80e-6  # ring period d_r [m]
N_SPOKES = 16  # angular sectors
BEAM_HALF = 1.7e-3  # illuminated radius [m]


def grating_field(grid):
    # a circular stop bounds the beam and hides the singular spoke center
    return MonochromaticField(
        grid,
        lambda x, y: polar_grating(x, y, radial_period=RADIAL_PERIOD, n_spokes=N_SPOKES)
        * circular_aperture(x, y, BEAM_HALF)
        * (1.0 - circular_aperture(x, y, 60e-6)),
    )


def main() -> None:
    grid = make_grid(N, L)
    U0 = grating_field(grid)

    ring = WAVELENGTH * FOCAL_LENGTH / RADIAL_PERIOD  # radial order spacing
    screen_half = 2.6 * ring

    # Monochromatic: the polar grating through the lens, to the focal plane.
    spots = (
        MonochromaticField(grid, U0.to_field(), wavelength=WAVELENGTH)
        .add_lens(FOCAL_LENGTH)
        .propagate(
            FOCAL_LENGTH, method="fresnel_zoom",
            output_half_width=screen_half, output_samples=640,
        )
    )

    # White light: the lens phase is chromatic, so rebuild the field per λ.
    wavelengths = np.linspace(430e-9, 660e-9, 32)

    img = (
        PolychromaticField(
            grid, U0.to_field(), wavelengths=wavelengths, weights=d65_weights(wavelengths * 1e9)
        )
        .add_lens(FOCAL_LENGTH)
        .propagate(
            FOCAL_LENGTH, method="fresnel_zoom",
            output_half_width=screen_half, output_samples=640,
            gamut="clip", stretch=0.55, saturation=1.4,
        )
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    U0.plot(ax[0], title=f"Polar grating (d_r = {RADIAL_PERIOD*1e6:.0f} µm, {N_SPOKES} spokes)")
    ax[0].set_xlim(-BEAM_HALF, BEAM_HALF)
    ax[0].set_ylim(-BEAM_HALF, BEAM_HALF)
    spots.plot(ax[1], title=f"Focal-plane orders ({int(WAVELENGTH*1e9)} nm)", vmin=-4.0)
    img.plot(ax[2], title="White-light orders (each dispersed)")
    fig.suptitle("Polar grating at a lens focal plane: a centered polar lattice of orders")
    plt.show()


if __name__ == "__main__":
    main()
