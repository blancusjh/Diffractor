"""Lattice aperture imaged by a lens — reciprocal-lattice spots.

A lattice of small holes has a far field that is the *reciprocal* lattice: a
grid of sharp diffraction spots whose brightness follows the single-hole form
factor. A converging lens brings that far field to its back focal plane, so
``lattice × thin_lens`` propagated to ``z = f`` gives clean, centered spots at
``x_mn = m λ f / a``.

Three panels: the hexagonal hole array, its monochromatic focal-plane spots,
and — since the lens phase is chromatic — the same under white light, where
each spot disperses radially into a little spectrum (`propagate_polychromatic`
with a per-wavelength `field_of`). Everything is centered on the optical axis.

Run with:  python examples/lattice_lens_diffraction.py
"""

import numpy as np

from diffraction import (
    Field,
    circular_aperture,
    d65_weights,
    fresnel_zoom_propagator,
    lattice_aperture,
    make_grid,
    plot_intensity,
    plot_rgb,
    propagate_polychromatic,
    thin_lens,
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
    x, y = grid
    mask = lattice_aperture(
        x, y, circular_aperture, SPACING, lattice=LATTICE, size=SIZE, R=HOLE_R
    )
    return Field(grid, mask.astype(complex))


def main() -> None:
    grid = make_grid(N, L)
    U0 = lattice_field(grid)

    x_spot = WAVELENGTH * FOCAL_LENGTH / SPACING
    screen_half = 5.0 * x_spot

    # Monochromatic: lattice through the lens, to the back focal plane.
    focused = U0 * thin_lens(grid, FOCAL_LENGTH, WAVELENGTH)
    spots = fresnel_zoom_propagator(
        focused, z=FOCAL_LENGTH, wavelength=WAVELENGTH,
        output_half_width=screen_half, output_samples=640,
    )

    # White light: the lens phase is chromatic, so build the field per λ.
    wavelengths = np.linspace(430e-9, 660e-9, 32)

    def field_of(lam):
        return U0 * thin_lens(grid, FOCAL_LENGTH, lam)

    rgb, rgb_grid = propagate_polychromatic(
        field_of, wavelengths, z=FOCAL_LENGTH,
        weights=d65_weights(wavelengths * 1e9),
        output_half_width=screen_half, output_samples=640,
        gamut="clip", stretch=0.6, saturation=1.4,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    plot_intensity(ax[0], U0, title=f"{LATTICE.capitalize()} hole array")
    ax[0].set_xlim(-6 * SPACING, 6 * SPACING)
    ax[0].set_ylim(-6 * SPACING, 6 * SPACING)
    plot_intensity(ax[1], spots, title=f"Focal-plane spots ({int(WAVELENGTH*1e9)} nm)", vmin=-4.0)
    plot_rgb(ax[2], rgb, rgb_grid, title="White-light focal spots (dispersed)")
    fig.suptitle("Lattice aperture at a lens focal plane: the reciprocal lattice, centered")
    plt.show()


if __name__ == "__main__":
    main()
