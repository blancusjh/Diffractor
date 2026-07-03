"""White-light diffraction of a circular aperture — colored rings.

A monochromatic circular aperture gives a grayscale Airy/Fresnel pattern whose
ring radii scale with wavelength. Illuminate it with a continuous (D65
daylight) spectrum and each wavelength's pattern lands, slightly rescaled, on
the same screen; composited through the CIE 1931 color-matching functions the
rings come out colored — red rings sit outside blue ones.

`PolychromaticField` does the per-wavelength propagation onto a shared
output grid and the spectrum→sRGB conversion; here we sample the visible band
weighted by the D65 illuminant.

Run with:  python examples/polychromatic_aperture.py
"""

import numpy as np

from diffractor import (
    MonochromaticField,
    PolychromaticField,
    circular_aperture,
    d65_weights,
    make_grid,
)

N = 512
L = 4e-3  # input window [m]
RADIUS = 0.18e-3  # aperture radius [m]
Z = 0.35  # propagation distance [m]
N_WAVELENGTHS = 28
SCREEN_HALF = 2.4e-3  # output half-width [m]


def main() -> None:
    grid = make_grid(N, L)
    U0 = MonochromaticField(grid, 1.0).add_aperture(circular_aperture, RADIUS, antialiased=True)

    wavelengths = np.linspace(410e-9, 680e-9, N_WAVELENGTHS)
    weights = d65_weights(wavelengths * 1e9)

    img = (
        PolychromaticField(grid, 1.0, wavelengths=wavelengths, weights=weights)
        .add_aperture(circular_aperture, RADIUS, antialiased=True)
        .propagate(
            Z,
            method="fresnel_zoom",
            output_half_width=SCREEN_HALF,
            output_samples=512,
            gamut="clip",
            stretch=0.6,
            saturation=1.4,
        )
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    U0.plot(ax[0], title="Aperture")
    img.plot(ax[1], title=f"White-light pattern (z = {Z} m)")
    fig.suptitle("Chromatic diffraction rings under a D65 (daylight) spectrum")
    plt.show()


if __name__ == "__main__":
    main()
