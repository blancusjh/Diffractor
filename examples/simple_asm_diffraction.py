"""Angular-spectrum diffraction of a circular aperture.

Propagates a uniformly illuminated circular aperture with the angular
spectrum method. Unlike the single-FFT Fresnel method, the output field
lives on the same grid as the input.

Run with:  python examples/simple_asm_diffraction.py
"""

import matplotlib.pyplot as plt

from diffraction import (
    antialiased,
    asm_propagator,
    circular_aperture,
    make_grid,
    plot_intensity,
)

N = 1024  # samples per side
L = 6e-3  # grid side length [m]
WAVELENGTH = 532e-9  # vacuum wavelength [m]
Z = 0.05  # propagation distance [m] (near field, where the ASM shines)
RADIUS = 0.3e-3  # aperture radius [m]


def main() -> None:
    grid = make_grid(N, L)

    U0 = antialiased(circular_aperture, grid, RADIUS).astype(complex)
    Uz = asm_propagator(U0, grid, z=Z, wavelength=WAVELENGTH, pad_factor=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    plot_intensity(ax[0], U0, grid, title="Input intensity")
    plot_intensity(ax[1], Uz, grid, title=f"Propagated intensity (z = {Z} m)")
    plt.show()


if __name__ == "__main__":
    main()
