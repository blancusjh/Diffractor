"""Young's double slit — and an N-slit grating — in white light.

The classic experiment in color: white light through two (or several) slits
gives interference fringes whose spacing ``λ z / d`` grows with wavelength, so
the central fringe stays white while higher orders fan out into colored
fringes that wash toward white further out. Adding more slits (an N-slit
grating) sharpens the maxima into cleaner colored lines.

`nslit_aperture` builds the slits; `PolychromaticField` propagates every
wavelength onto a shared screen and composites to sRGB.

Run with:  python examples/double_slit_white_light.py
"""

import numpy as np

from diffractor import (
    PolychromaticField,
    d65_weights,
    make_grid,
    nslit_aperture,
    square_aperture,
)

N = 1024
L = 6e-3
Z = 1.0
SLIT_WIDTH = 30e-6
SPACING = 180e-6  # center-to-center [m]
BEAM_HALF = 1.5e-3
SCREEN_HALF = 12e-3
N_WAVELENGTHS = 40


def slit_mask(n_slits):
    return lambda x, y: nslit_aperture(x, y, n_slits, SLIT_WIDTH, SPACING) * square_aperture(
        x, y, BEAM_HALF
    )


def main() -> None:
    grid = make_grid(N, L)
    wavelengths = np.linspace(420e-9, 680e-9, N_WAVELENGTHS)
    weights = d65_weights(wavelengths * 1e9)

    panels = [(2, "Double slit (N = 2)"), (5, "Five-slit grating (N = 5)")]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(panels), 1, figsize=(10, 5.2), constrained_layout=True)
    for a, (n_slits, title) in zip(ax, panels):
        img = PolychromaticField(
            grid, slit_mask(n_slits), wavelengths=wavelengths, weights=weights
        ).propagate(
            Z, method="fresnel_zoom",
            output_half_width=SCREEN_HALF, output_samples=768,
            gamut="clip", saturation=1.3, stretch=0.8,
        )
        img.plot(a, title=title)
        a.set_ylim(-2e-3, 2e-3)
    fig.suptitle(f"White-light interference (slit spacing {SPACING*1e6:.0f} µm, z = {Z} m)")
    plt.show()


if __name__ == "__main__":
    main()
