"""Angular-spectrum propagation through a stigmatic Cartesian-oval surface.

Same optical setup as fresnel_with_cartesian_surface_phase.py, but the
field after the surface is propagated with the angular spectrum method, so
the output stays on the input grid and resolves the focal spot at the
design image plane in full detail.

Run with:  python examples/asm_with_cartesian_surface_phase.py
"""

import matplotlib.pyplot as plt

from diffraction import (
    CartesianSurface,
    antialiased,
    asm_propagator,
    circular_aperture,
    make_grid,
    plot_intensity,
)

N = 1024  # samples per side
L = 6e-3  # grid side length [m]
WAVELENGTH = 530e-9  # vacuum wavelength [m]
N1, N2 = 1.0, 1.5  # refractive indices before / after the surface
ZO = -1e5  # object distance [m] (collimated illumination)
ZI = 1.0  # design image distance [m]
RADIUS = 0.8e-3  # aperture radius [m]


def main() -> None:
    grid = make_grid(N, L)
    x, y = grid

    surface = CartesianSurface(n1=N1, n2=N2, zo=ZO, zi=ZI)
    U0 = antialiased(circular_aperture, grid, RADIUS)
    U_after = U0 * surface.phase_mask(grid, WAVELENGTH, N1, N2)

    # Propagate to the design image plane, where the stigmatic surface
    # focuses the beam to a diffraction-limited spot.
    Uz = asm_propagator(U_after, z=ZI, wavelength=WAVELENGTH, n=N2, pad_factor=2)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    im = ax[0].imshow(
        surface.sag(x, y),
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="viridis",
    )
    ax[0].set_title("Cartesian surface sag z(x, y) [m]")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    plot_intensity(ax[1], U_after, title="Intensity after surface")
    plot_intensity(ax[2], Uz, title=f"Design image plane (z = {ZI} m)")
    plt.show()


if __name__ == "__main__":
    main()
