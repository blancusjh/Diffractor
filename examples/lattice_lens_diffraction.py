"""Lattice aperture imaged by a lens — reciprocal-lattice spots.

A lattice of small holes has a far field that is the *reciprocal* lattice: a
grid of sharp diffraction spots whose brightness follows the single-hole form
factor. A converging lens brings that far field to its back focal plane, so
``lattice × thin_lens`` propagated to ``z = f`` gives clean, centered spots at
``x_mn = m λ f / a``.

Three panels: the hexagonal hole array, its monochromatic focal-plane spots,
and — since the lens phase is chromatic — the same under white light, where
each spot disperses radially into a little spectrum (a `PolychromaticField`
replaying the lens phase per wavelength). Everything is centered on the optical axis.

Run with:  python examples/lattice_lens_diffraction.py
"""

import numpy as np

from diffractor import (
    MonochromaticField,
    PolychromaticField,
    circular_aperture,
    d65_weights,
    lattice_aperture,
    make_grid,
)

N = 1024
L = 8e-3  # input window [m]
WAVELENGTH = 550e-9
FOCAL_LENGTH = 0.2  # lens focal length [m]
SPACING = 120e-6  # lattice spacing a [m]
HOLE_R = 22e-6  # hole radius [m]
LATTICE = "hexagonal"
SIZE = (11, 11)


def lattice_field(grid):
    return MonochromaticField(
        grid,
        lambda x, y: lattice_aperture(
            x, y, circular_aperture, SPACING, lattice=LATTICE, size=SIZE, R=HOLE_R
        ),
    )


def main() -> None:
    grid = make_grid(N, L)
    U0 = lattice_field(grid)

    x_spot = WAVELENGTH * FOCAL_LENGTH / SPACING
    screen_half = 5.0 * x_spot

    # Monochromatic: lattice through the lens, to the back focal plane.
    spots = (
        MonochromaticField(grid, U0.to_field(), wavelength=WAVELENGTH)
        .add_lens(FOCAL_LENGTH)
        .propagate(
            FOCAL_LENGTH, method="fresnel_zoom",
            output_half_width=screen_half, output_samples=640,
        )
    )

    # White light: the lens phase is chromatic, so build the field per λ.
    wavelengths = np.linspace(430e-9, 660e-9, 32)

    img = (
        PolychromaticField(
            grid, U0.to_field(), wavelengths=wavelengths, weights=d65_weights(wavelengths * 1e9)
        )
        .add_lens(FOCAL_LENGTH)
        .propagate(
            FOCAL_LENGTH, method="fresnel_zoom",
            output_half_width=screen_half, output_samples=640,
            gamut="clip", stretch=0.6, saturation=1.4,
        )
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    U0.plot(ax[0], title=f"{LATTICE.capitalize()} hole array")
    ax[0].set_xlim(-6 * SPACING, 6 * SPACING)
    ax[0].set_ylim(-6 * SPACING, 6 * SPACING)
    spots.plot(ax[1], title=f"Focal-plane spots ({int(WAVELENGTH*1e9)} nm)", vmin=-4.0)
    img.plot(ax[2], title="White-light focal spots (dispersed)")
    fig.suptitle("Lattice aperture at a lens focal plane: the reciprocal lattice, centered")
    plt.show()


if __name__ == "__main__":
    main()
