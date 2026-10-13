"""Fresnel propagation through a parabolic refracting surface.

A collimated beam crosses a parabolic interface between media n1 and n2
(thin-element phase approximation) and is propagated to the paraxial focus
at F = 2 f n2 / (n2 - n1), where it forms a diffraction-limited Airy spot.
The focal region is evaluated with the zoom (matrix-DFT) Fresnel
propagator, whose output sampling is independent of the input grid.

Run with:  python examples/fresnel_with_parabolic_surface_phase.py
"""

import matplotlib.pyplot as plt

from diffractor import (
    MonochromaticField,
    ParabolicSurface,
    circular_aperture,
    make_grid,
)

N = 2048  # samples per side
L = 6e-3  # grid side length [m]
WAVELENGTH = 530e-9  # vacuum wavelength [m]
N1, N2 = 1.0, 1.5  # refractive indices before / after the surface
FOCAL_LENGTH = 0.12  # parabola focal length [m]
RADIUS = 0.4e-3  # aperture radius [m] (sets the Airy spot scale at focus)
ZOOM = 1.5e-3  # half-width of the output focal window [m]


def main() -> None:
    grid = make_grid(N, L)
    x, y = grid

    surface = ParabolicSurface(focal_length=FOCAL_LENGTH)
    U_after = MonochromaticField(grid, 1.0, wavelength=WAVELENGTH).add_aperture(
        circular_aperture, RADIUS, antialiased=True
    ).add_surface(surface, n1=N1, n2=N2)

    # Paraxial focus of the refracting parabola in medium n2.
    z_focus = 2.0 * FOCAL_LENGTH * N2 / (N2 - N1)
    Uz = U_after.propagate(
        z_focus,
        method="fresnel_zoom",
        output_half_width=ZOOM,
    )

    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    im = ax[0].imshow(
        surface.sag(x, y),
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="viridis",
    )
    ax[0].set_title("Parabolic surface sag z(x, y) [m]")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    U_after.plot(ax[1], title="Intensity after surface")
    Uz.plot(ax[2], title=f"Focal plane (z = {z_focus:.2f} m)", vmin=-4.0)
    plt.show()


if __name__ == "__main__":
    main()
