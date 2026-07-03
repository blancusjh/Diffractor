"""White light through a diffraction grating — a dispersed spectrum.

The showcase for the polychromatic + grating features together. A grating
disperses each wavelength to x_m = m λ z / d, so under white light every
non-zero order spreads into a spectrum: the 0th order stays white, the ±1
orders fan out into rainbows (red furthest from center), and so on. This is
exactly how a grating spectrometer works.

`PolychromaticField` propagates the (wavelength-independent) grating
transmission at each wavelength onto one shared screen and composites the
result through the CIE 1931 color-matching functions.

Run with:  python examples/white_light_grating.py
"""

import numpy as np

from diffractor import (
    PolychromaticField,
    d65_weights,
    make_grid,
    ronchi_grating,
    square_aperture,
)

N = 1024
L = 4e-3
Z = 0.5
PERIOD = 60e-6  # grating period [m]
BEAM_HALF = 1.2e-3
N_WAVELENGTHS = 40


def main() -> None:
    grid = make_grid(N, L)

    wavelengths = np.linspace(410e-9, 680e-9, N_WAVELENGTHS)
    weights = d65_weights(wavelengths * 1e9)

    # Screen wide enough to hold the dispersed ±1 (and part of ±2) orders.
    x1_red = 680e-9 * Z / PERIOD
    screen_half = 2.4 * x1_red

    # A wavelength-independent amplitude grating (the same mask replayed at
    # every wavelength); dispersion comes from propagation, not the mask.
    img = (
        PolychromaticField(
            grid,
            lambda x, y: ronchi_grating(x, y, PERIOD, duty=0.5) * square_aperture(x, y, BEAM_HALF),
            wavelengths=wavelengths,
            weights=weights,
        ).propagate(
            Z,
            method="fresnel_zoom",
            output_half_width=screen_half,
            output_samples=768,
            gamut="clip",
            stretch=0.7,
            saturation=1.4,
        )
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    img.plot(ax, title=f"White light through a grating (d = {PERIOD*1e6:.0f} µm, z = {Z} m)")
    plt.show()


if __name__ == "__main__":
    main()
