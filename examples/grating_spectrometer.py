"""A grating spectrometer: grating + lens → a focused spectrum.

A diffraction grating disperses white light, and a lens focuses each
diffracted direction to a point — together they are a spectrometer. At the
lens's back focal plane, order ``m`` of wavelength ``λ`` lands at
``x_m = m λ f / d``, so every order is a sharp, spread-out spectrum instead of
the broad far-field pattern.

Two gratings, side by side:

* a **Ronchi** (binary amplitude) grating sends light symmetrically into the
  0th (white) and ±1, ±2 orders — a spectrum on each side;
* a **blazed** (sawtooth phase) grating steers almost all the light into a
  single order — one bright spectrum, which is what real spectrometers use.

The grating and the lens are both chromatic, so the field is rebuilt per
wavelength (`field_of(λ)`); `propagate_polychromatic` composites the focal
plane to sRGB.

Run with:  python examples/grating_spectrometer.py
"""

import numpy as np

from diffraction import (
    Field,
    d65_weights,
    make_grid,
    phase_grating,
    plot_rgb,
    propagate_polychromatic,
    ronchi_grating,
    square_aperture,
    thin_lens,
)

N = 1024
L = 8e-3
FOCAL_LENGTH = 0.25  # focusing lens [m]
PERIOD = 40e-6  # grating period d [m]
BEAM_HALF = 2e-3  # illuminated half-width [m]
N_WAVELENGTHS = 44


def main() -> None:
    grid = make_grid(N, L)
    x, y = grid
    aperture = Field(grid, square_aperture(x, y, BEAM_HALF).astype(complex))
    wavelengths = np.linspace(410e-9, 680e-9, N_WAVELENGTHS)
    weights = d65_weights(wavelengths * 1e9)

    x1 = 550e-9 * FOCAL_LENGTH / PERIOD  # first-order position at 550 nm
    screen_half = 3.4 * x1

    # Ronchi amplitude grating: a fixed (achromatic) transmission mask.
    ronchi = Field(grid, ronchi_grating(x, y, PERIOD).astype(complex))

    def ronchi_field(lam):
        return ronchi * thin_lens(grid, FOCAL_LENGTH, lam)

    # Blazed phase grating: chromatic, built per wavelength (2π depth at 530 nm).
    blaze_height = 530e-9 / (1.5 - 1.0)

    def blazed_field(lam):
        pg = phase_grating(grid, lam, period=PERIOD, height=blaze_height, profile="blazed")
        return pg * aperture * thin_lens(grid, FOCAL_LENGTH, lam)

    disp = dict(
        z=FOCAL_LENGTH, weights=weights, output_half_width=screen_half,
        output_samples=768, gamut="clip", stretch=0.7, saturation=1.4,
    )
    rgb_ronchi, grid_r = propagate_polychromatic(ronchi_field, wavelengths, **disp)
    rgb_blazed, grid_b = propagate_polychromatic(blazed_field, wavelengths, **disp)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 5.4), constrained_layout=True)
    plot_rgb(ax[0], rgb_ronchi, grid_r, title="Ronchi grating + lens — symmetric spectra")
    plot_rgb(ax[1], rgb_blazed, grid_b, title="Blazed grating + lens — energy in one order")
    for a in ax:
        a.set_ylim(-0.7e-3, 0.7e-3)
    fig.suptitle(f"Grating spectrometer (d = {PERIOD*1e6:.0f} µm, f = {FOCAL_LENGTH} m)")
    plt.show()


if __name__ == "__main__":
    main()
