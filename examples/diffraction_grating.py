"""Diffraction orders of a monochromatic grating.

A grating of period d sends order m to sin θ_m = m λ / d; propagated to a
screen at distance z the orders land at x_m = m λ z / d. This propagates a
Ronchi (binary amplitude) grating, illuminated over a finite aperture, to the
far field and overlays the predicted order positions.

Run with:  python examples/diffraction_grating.py
"""

import numpy as np

from diffraction import (
    Field,
    fresnel_zoom_propagator,
    make_grid,
    plot_intensity,
    ronchi_grating,
    square_aperture,
)

N = 1024
L = 4e-3
WAVELENGTH = 550e-9
Z = 0.5
PERIOD = 50e-6  # grating period d [m]
BEAM_HALF = 1e-3  # illuminated half-width [m]


def main() -> None:
    grid = make_grid(N, L)
    x, y = grid

    grating = ronchi_grating(x, y, PERIOD, duty=0.5)
    beam = square_aperture(x, y, BEAM_HALF)
    U0 = Field(grid, (grating * beam).astype(complex))

    x_order = WAVELENGTH * Z / PERIOD
    screen_half = 3.4 * x_order
    Uz = fresnel_zoom_propagator(
        U0, z=Z, wavelength=WAVELENGTH, output_half_width=screen_half, output_samples=2048
    )

    # Report measured vs predicted order positions.
    row = np.abs(Uz.values[Uz.shape[0] // 2]) ** 2
    xo = Uz.grid.x[0, :]
    print(f"Predicted order spacing  x_1 = m λ z / d = {x_order * 1e3:.3f} mm")
    for m in (0, 1, 2, 3):
        window = np.abs(xo - m * x_order) < 0.4 * x_order
        if window.any():
            xm = xo[window][np.argmax(row[window])]
            print(f"  order {m}: measured {xm * 1e3:+.3f} mm   predicted {m * x_order * 1e3:+.3f} mm")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    plot_intensity(ax[0], U0, title="Ronchi grating × aperture")
    ax[0].set_xlim(-3 * PERIOD, 3 * PERIOD)
    ax[0].set_ylim(-3 * PERIOD, 3 * PERIOD)
    plot_intensity(ax[1], Uz, title=f"Far-field orders (z = {Z} m, d = {PERIOD*1e6:.0f} µm)", vmin=-4.0)
    for m in range(-3, 4):
        ax[1].axvline(m * x_order, color="cyan", lw=0.6, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
