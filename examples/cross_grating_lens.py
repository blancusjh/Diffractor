"""A square (2D) grating at a lens focus — a lattice of focused orders.

A cross grating is the product of two 1D gratings, so its far field is the
*outer product* of two 1D order combs: a square lattice of diffraction orders.
A converging lens brings that far field to its back focal plane, where order
``(m, n)`` of wavelength ``λ`` lands at ``(x, y) = (m λ f / d, n λ f / d)`` — a
clean, centered grid of sharp spots instead of the broad far-field pattern.

Three panels: the square grating, its monochromatic focal-plane lattice of
spots, and — since the lens phase is chromatic — the same under white light,
where each off-axis spot disperses radially into a little spectrum
(a `PolychromaticField` replaying the lens phase per wavelength).

A 50 % duty cross grating suppresses the even orders on each axis, so the
lattice is the characteristic sparse grid (missing every other row/column).

Run with:  python examples/cross_grating_lens.py
"""

import numpy as np

from diffractor import (
    MonochromaticField,
    PolychromaticField,
    cross_grating,
    d65_weights,
    make_grid,
    square_aperture,
)

N = 1024
L = 8e-3  # input window [m]
WAVELENGTH = 550e-9
FOCAL_LENGTH = 0.25  # lens focal length [m]
PERIOD = 60e-6  # grating period d [m]
BEAM_HALF = 1.6e-3  # illuminated half-width [m]


def grating_field(grid):
    return MonochromaticField(
        grid,
        lambda x, y: cross_grating(x, y, PERIOD, kind="ronchi", duty=0.5)
        * square_aperture(x, y, BEAM_HALF),
    )


def main() -> None:
    grid = make_grid(N, L)
    U0 = grating_field(grid)

    x_order = WAVELENGTH * FOCAL_LENGTH / PERIOD  # spot spacing at the focal plane
    screen_half = 3.6 * x_order

    # Monochromatic: the grating through the lens, to the back focal plane.
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
    U0.plot(ax[0], title=f"Square (cross) grating, d = {PERIOD*1e6:.0f} µm")
    ax[0].set_xlim(-5 * PERIOD, 5 * PERIOD)
    ax[0].set_ylim(-5 * PERIOD, 5 * PERIOD)
    spots.plot(ax[1], title=f"Focal-plane orders ({int(WAVELENGTH*1e9)} nm)", vmin=-4.0)
    img.plot(ax[2], title="White-light orders (each dispersed)")
    fig.suptitle("Square grating at a lens focal plane: a centered lattice of orders (x_mn = m λ f / d)")
    plt.show()


if __name__ == "__main__":
    main()
