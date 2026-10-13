"""Fresnel diffraction of a circular aperture.

Propagates a uniformly illuminated circular aperture with the single-FFT
Fresnel method and shows the input and output intensity patterns.

Run with:  python examples/simple_fresnel_diffraction.py
"""

import matplotlib.pyplot as plt

from diffractor import (
    MonochromaticField,
    circular_aperture,
    make_grid,
)

N = 2048  # samples per side
L = 6e-3  # grid side length [m]
WAVELENGTH = 532e-9  # vacuum wavelength [m]
Z = 1.15  # propagation distance [m]
RADIUS = 0.3e-3  # aperture radius [m]


def main() -> None:
    grid = make_grid(N, L)

    U0 = MonochromaticField(grid, 1.0, wavelength=WAVELENGTH).add_aperture(
        circular_aperture, RADIUS, antialiased=True
    )
    Uz = U0.propagate(Z, method="fresnel")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    U0.plot(ax[0], title="Input intensity")
    Uz.plot(ax[1], title=f"Propagated intensity (z = {Z} m)")

    # The single-FFT method's output window scales as wavelength*z/dx (see
    # fresnel_output_grid), which at z=1.15 m spans +/-104 mm — far wider than
    # the Airy pattern it contains (first null at ~1.24 mm here). Zoom to the
    # pattern itself; otherwise imshow has to bin hundreds of fringes into a
    # handful of display pixels, and the resulting moire looks like noise
    # even though the underlying field is a clean, correctly-sampled Airy
    # pattern (verified against the analytic first-null radius in the tests).
    zoom = 3.0e-3
    ax[1].set_xlim(-zoom, zoom)
    ax[1].set_ylim(-zoom, zoom)

    plt.show()


if __name__ == "__main__":
    main()
