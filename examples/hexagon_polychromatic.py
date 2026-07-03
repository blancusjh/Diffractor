"""Hexagonal aperture in white light — a faithful diffractsim reproduction.

This mirrors diffractsim's flagship `hexagon_polychromatic.py`: a 5.6 mm
hexagonal aperture in an 18 mm window, illuminated by a D65 (daylight)
spectrum, propagated 0.8 m and viewed over a ±7 mm screen. The result is the
iconic colored hexagonal Fresnel pattern — a hexagonal bright core with
chromatic interference fringes and six star spikes from the edges.

`polygon_aperture` builds the hexagon exactly (intersection of six
half-planes), and `propagate_polychromatic` runs the per-wavelength
angular-spectrum propagation onto the shared 18 mm grid before compositing to
sRGB through the CIE 1931 color-matching functions.

Run with:  python examples/hexagon_polychromatic.py
"""

import numpy as np

from diffraction import (
    antialiased,
    d65_weights,
    make_grid,
    plot_rgb,
    polygon_aperture,
    propagate_polychromatic,
)

N = 1024
L = 18e-3  # window [m]
Z = 0.8  # propagation distance [m]
FLAT_TO_FLAT = 5.6e-3  # hexagon width across flats [m]
N_WAVELENGTHS = 36
SCREEN_HALF = 7e-3  # plot half-width [m]


def main() -> None:
    grid = make_grid(N, L)
    # across-flats width = 2 · apothem = 2 · radius · cos(π/6)
    radius = FLAT_TO_FLAT / (2.0 * np.cos(np.pi / 6.0))
    hexagon = antialiased(polygon_aperture, grid, 6, radius)

    wavelengths = np.linspace(420e-9, 680e-9, N_WAVELENGTHS)
    rgb, out_grid = propagate_polychromatic(
        hexagon,
        wavelengths,
        z=Z,
        weights=d65_weights(wavelengths * 1e9),
        propagator="asm",
        pad_factor=2,
        # display controls for a vivid render: clip gamut keeps saturated hues,
        # the stretch compresses the dynamic range so the colored fringes *and*
        # the faint star spikes both show, and a touch of extra saturation.
        gamut="clip",
        stretch=0.6,
        saturation=1.4,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    plot_rgb(ax, rgb, out_grid, title=f"Hexagon in white light (D65, z = {Z} m)")
    ax.set_xlim(-SCREEN_HALF, SCREEN_HALF)
    ax.set_ylim(-SCREEN_HALF, SCREEN_HALF)
    plt.show()


if __name__ == "__main__":
    main()
