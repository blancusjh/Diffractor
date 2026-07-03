"""White-light hexagon at a lens focus — a six-pointed colored star.

A lens brings the Fraunhofer pattern of an aperture to its back focal plane. A
hexagon's Fraunhofer pattern is a six-pointed star (spikes perpendicular to the
edges), so a white-light hexagon through a lens forms a colored six-pointed
star — the same effect that gives hexagonal camera irises their starburst
flare. The lens phase is chromatic, so the source is rebuilt per wavelength
(`field_of(λ) = hexagon × thin_lens(λ)`).

Combines `polygon_aperture` + `thin_lens` + `propagate_polychromatic`. A mild
display stretch (`rgb ** 0.5`) reveals the faint spikes without changing the
physics.

Run with:  python examples/hexagon_lens_star.py
"""

import numpy as np

from diffraction import (
    antialiased,
    d65_weights,
    make_grid,
    plot_rgb,
    polygon_aperture,
    propagate_polychromatic,
    thin_lens,
)

N = 1024
L = 6e-3
FOCAL_LENGTH = 0.3  # lens focal length [m]
HEX_RADIUS = 0.45e-3  # hexagon circumradius [m]
N_WAVELENGTHS = 34


def main() -> None:
    grid = make_grid(N, L)
    hexagon = antialiased(polygon_aperture, grid, 6, HEX_RADIUS)

    def field_of(lam):
        return hexagon * thin_lens(grid, FOCAL_LENGTH, lam)

    wavelengths = np.linspace(420e-9, 680e-9, N_WAVELENGTHS)
    screen_half = 9.0 * 550e-9 * FOCAL_LENGTH / (2 * HEX_RADIUS)

    rgb, out_grid = propagate_polychromatic(
        field_of,
        wavelengths,
        z=FOCAL_LENGTH,
        weights=d65_weights(wavelengths * 1e9),
        output_half_width=screen_half,
        output_samples=700,
        # stretch reveals the faint star spikes; clip + saturation keep vivid hues.
        gamut="clip",
        stretch=0.5,
        saturation=1.4,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    plot_rgb(ax, rgb, out_grid, title="White-light hexagon at a lens focus")
    plt.show()


if __name__ == "__main__":
    main()
