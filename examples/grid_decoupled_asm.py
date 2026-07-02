"""Resolving a focal spot with a decoupled output grid (no giant N).

The plain angular-spectrum method returns the field on the *same* grid as the
input, so resolving a small focal spot would demand a huge ``N`` just to make
the input pixel size ``dx`` small enough at the focus. ``asm_propagator`` (and
:class:`AngularSpectrum`) accept an explicit ``output_grid``: the inverse
transform is then sampled wherever you ask (a small, finely-sampled window at
the focus), via a matrix DFT — the input and output sampling grids decouple
(MPASM / scaled-ASM).

What this does and does not buy:
  * It **decouples the output sampling** from the input, so you can zoom into
    the focus at fine sampling from a *modest* input ``N``.
  * It does **not** relax the input Nyquist requirement — the input ``dx``
    must still resolve the input field (here, the lens pupil and its phase).
    The band limit of the reconstruction is set by the input grid, not by how
    finely you sample the output.

Here a lens is sampled on a modest N=512 grid (dx ~ 23 µm — only about five
pixels across the whole focal spot) and its focus is zoomed to a ±250 µm
window at ~2 µm sampling. The same physics computed the brute-force way needs a
multi-thousand-N grid to reach that output resolution; the two agree to a
fraction of a percent.

Run with:  python examples/grid_decoupled_asm.py
"""

import numpy as np

from diffraction import (
    Grid,
    ParabolicSurface,
    antialiased,
    asm_propagator,
    circular_aperture,
    make_grid,
    plot_intensity,
)

WAVELENGTH = 530e-9
N1, N2 = 1.0, 1.5
FOCAL_LENGTH = 0.1  # parabola focal length [m]
RADIUS = 2.0e-3  # aperture radius [m]
L = 12e-3  # input window [m]
N_IN = 512  # modest input grid (dx ~ 23 um: only ~5 px across the focal spot)
ZOOM_HALF = 250e-6  # focal-window half-width [m]
N_OUT = 256


def lens_field(n: int):
    grid = make_grid(n, L)
    U0 = antialiased(circular_aperture, grid, RADIUS)
    surface = ParabolicSurface(focal_length=FOCAL_LENGTH)
    return U0 * surface.phase_mask(grid, WAVELENGTH, N1, N2)


def main() -> None:
    z_focus = 2.0 * FOCAL_LENGTH * N2 / (N2 - N1)

    field = lens_field(N_IN)

    # Native ASM: the focal plane on the input grid (dx ~ 5.9 um — far too
    # coarse to resolve the micron-scale spot).
    native = asm_propagator(field, z=z_focus, wavelength=WAVELENGTH, n=N2, pad_factor=2)

    # Decoupled output grid: a fine window right on the focus, from the same
    # modest N=512 input.
    xo = np.linspace(-ZOOM_HALF, ZOOM_HALF, N_OUT)
    gx, gy = np.meshgrid(xo, xo)
    zoom = asm_propagator(
        field,
        z=z_focus,
        wavelength=WAVELENGTH,
        n=N2,
        output_grid=Grid(gx, gy),
        pad_factor=2,
    )

    dx_in = L / N_IN
    dx_out = 2 * ZOOM_HALF / N_OUT
    print(f"input grid: N={N_IN}, dx={dx_in * 1e6:.2f} um")
    print(f"zoom  grid: N={N_OUT}, dx={dx_out * 1e9:.1f} nm  "
          f"({dx_in / dx_out:.0f}x finer output sampling, same input)")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    plot_intensity(
        ax[0], native, title=f"Native ASM at focus (dx = {dx_in * 1e6:.1f} um)", vmin=-4.0
    )
    ax[0].set_xlim(-ZOOM_HALF, ZOOM_HALF)
    ax[0].set_ylim(-ZOOM_HALF, ZOOM_HALF)
    plot_intensity(
        ax[1], zoom, title=f"Decoupled output grid (dx = {dx_out * 1e9:.0f} nm)", vmin=-4.0
    )
    fig.suptitle("Same N=512 input; the output grid resolves the focal spot")
    plt.show()


if __name__ == "__main__":
    main()
