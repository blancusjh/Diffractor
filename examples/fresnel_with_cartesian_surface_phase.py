"""Fresnel propagation through a stigmatic Cartesian-oval surface.

Collimated light (object at zo -> -inf) crosses the Cartesian refracting
surface designed to image it perfectly at zi, and is propagated to that
design image plane, where it forms a diffraction-limited Airy spot. The
focal region is evaluated with the zoom (matrix-DFT) Fresnel propagator,
whose output sampling is independent of the input grid.

Run with:  python examples/fresnel_with_cartesian_surface_phase.py
"""

import matplotlib.pyplot as plt

from diffraction import (
    CartesianSurface,
    antialiased,
    circular_aperture,
    fresnel_zoom_propagator,
    make_grid,
    plot_intensity,
)

N = 2048  # samples per side
L = 6e-3  # grid side length [m]
WAVELENGTH = 530e-9  # vacuum wavelength [m]
N1, N2 = 1.0, 1.5  # refractive indices before / after the surface
ZO = -1e5  # object distance [m] (collimated illumination)
ZI = 1.0  # design image distance [m]
RADIUS = 0.4e-3  # aperture radius [m] (sets the Airy spot scale at focus)
ZOOM = 2.0e-3  # half-width of the output focal window [m]


def main() -> None:
    grid = make_grid(N, L)
    x, y = grid

    surface = CartesianSurface(n1=N1, n2=N2, zo=ZO, zi=ZI)
    U0 = antialiased(circular_aperture, grid, RADIUS)
    U_after = U0 * surface.phase_mask(grid, WAVELENGTH, N1, N2)

    # Propagate to the design image plane: the surface is stigmatic, so
    # the spot there is limited only by aperture diffraction.
    Uz = fresnel_zoom_propagator(
        U_after,
        z=ZI,
        wavelength=WAVELENGTH,
        n=N2,
        output_half_width=ZOOM,
    )

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
    plot_intensity(ax[2], Uz, title=f"Design image plane (z = {ZI} m)", vmin=-4.0)
    plt.show()


if __name__ == "__main__":
    main()
