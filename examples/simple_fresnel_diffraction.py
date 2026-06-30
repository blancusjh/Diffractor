"""Fresnel diffraction of a circular aperture.

Propagates a uniformly illuminated circular aperture with the single-FFT
Fresnel method and shows the input and output intensity patterns.

Run with:  python examples/simple_fresnel_diffraction.py
"""

import matplotlib.pyplot as plt

from diffraction import (
    antialiased,
    circular_aperture,
    fresnel_output_grid,
    fresnel_propagator,
    make_grid,
    plot_intensity,
)

N = 2048  # samples per side
L = 6e-3  # grid side length [m]
WAVELENGTH = 532e-9  # vacuum wavelength [m]
Z = 1.15  # propagation distance [m]
RADIUS = 0.3e-3  # aperture radius [m]


def main() -> None:
    grid = make_grid(N, L)

    U0 = antialiased(circular_aperture, grid, RADIUS).astype(complex)
    Uz = fresnel_propagator(U0, grid, z=Z, wavelength=WAVELENGTH)
    grid_out = fresnel_output_grid(grid, z=Z, wavelength=WAVELENGTH)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    plot_intensity(ax[0], U0, grid, title="Input intensity")
    plot_intensity(ax[1], Uz, grid_out, title=f"Propagated intensity (z = {Z} m)")
    plt.show()


if __name__ == "__main__":
    main()
