"""The focusing cone of a lens, seen edge-on (a longitudinal field).

A circular aperture behind a thin lens is propagated through a range of
distances straddling the focal length, and the on-axis (y = 0) transverse line
is taken at every plane. Stacking those lines gives an ``x–z`` cross-section:
the converging cone, the bright focal waist at ``z = f``, and the diverging
cone beyond it — the field along the optical axis, drawn directly.

Uses `longitudinal_field` (an `AngularSpectrum` z-sweep sliced on axis) and
`plot_longitudinal`.

Run with:  python examples/lens_longitudinal_focus.py
"""

import numpy as np

from diffraction import (
    antialiased,
    circular_aperture,
    longitudinal_field,
    make_grid,
    plot_longitudinal,
    thin_lens,
)

N = 1024
L = 4e-3  # window [m]
WAVELENGTH = 550e-9
APERTURE_R = 1.4e-3  # lens/aperture radius [m]
FOCAL_LENGTH = 0.10  # focal length [m]
N_PLANES = 180


def main() -> None:
    grid = make_grid(N, L)
    # A uniformly illuminated circular aperture immediately behind the lens.
    aperture = antialiased(circular_aperture, grid, APERTURE_R)
    lens = thin_lens(grid, FOCAL_LENGTH, WAVELENGTH)
    U0 = aperture * lens

    # Sweep from well before to well past the focus. The converging beam stays
    # well inside the window over this range, so no zero-padding is needed.
    zs = np.linspace(0.55 * FOCAL_LENGTH, 1.45 * FOCAL_LENGTH, N_PLANES)
    section = longitudinal_field(U0, WAVELENGTH, zs, axis="x", pad_factor=1)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    plot_longitudinal(
        ax, section, vmin=-3.0,
        title=f"Lens focusing cone (f = {FOCAL_LENGTH} m, D = {2*APERTURE_R*1e3:.1f} mm)",
    )
    ax.set_ylim(-0.35e-3, 0.35e-3)  # zoom onto the cone near the axis
    ax.axvline(FOCAL_LENGTH, color="cyan", lw=0.8, ls="--", alpha=0.7)
    ax.text(FOCAL_LENGTH, 0.30e-3, "  focus (z = f)", color="cyan", fontsize=9)
    plt.show()


if __name__ == "__main__":
    main()
