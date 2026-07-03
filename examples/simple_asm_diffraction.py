"""Angular-spectrum diffraction of a circular aperture.

Propagates a uniformly illuminated circular aperture with the angular
spectrum method. Unlike the single-FFT Fresnel method, the output field
lives on the same grid as the input.

Run with:  python examples/simple_asm_diffraction.py
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
Z = 0.08  # propagation distance [m] (near field, where the ASM shines)
RADIUS = 0.3e-3  # aperture radius [m]

# Sampling note: this aperture/distance pair has a high Fresnel number
# (R^2 / (lambda z) ~ 2), i.e. deep near field. The field's boundary-wave
# ripples near the geometric shadow edge have a local spatial frequency that
# scales with distance; too coarse a grid (dx too large relative to that
# frequency) under-samples them, and the resulting alias is *not* isotropic
# like the true field. It shows up as spurious axis-aligned streaks, because
# the Cartesian FFT grid — unlike the physics — is not rotationally
# symmetric. N=2048 keeps that alias well below the visible dynamic range
# here; padding/band-limiting do not help (they address wrap-around and
# transfer-function aliasing, not this input-sampling limit).


def main() -> None:
    grid = make_grid(N, L)

    U0 = MonochromaticField(grid, 1.0, wavelength=WAVELENGTH).add_aperture(
        circular_aperture, RADIUS, antialiased=True
    )
    Uz = U0.propagate(Z, method="asm", pad_factor=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    U0.plot(ax[0], title="Input intensity")
    Uz.plot(ax[1], title=f"Propagated intensity (z = {Z} m)")
    plt.show()


if __name__ == "__main__":
    main()
