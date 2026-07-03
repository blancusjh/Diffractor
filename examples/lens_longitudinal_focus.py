"""The focusing cone of a lens, seen edge-on (a longitudinal field).

A circular aperture behind a thin lens is propagated through a range of
distances straddling the focal length, and the on-axis (y = 0) transverse line
is taken at every plane. Stacking those lines gives an ``x–z`` cross-section:
the converging cone, the bright focal waist at ``z = f``, and the diverging
cone beyond it — the field along the optical axis, drawn directly.

Both panels use a *decoupled* transverse output line (`output_half_width`):
each plane is sampled on a fine matrix-DFT line instead of the native grid.
For the waist this resolves Airy-scale structure far below the input grid
spacing; for the wide cone view it instead picks a transverse pixel size
matched to the z-resolution, so the diagonal cone edges stay smooth — sampled
on the native grid instead, the same smoothness would need roughly 4x more
z-planes (the per-plane transverse shift, set by the cone's geometric slope
``APERTURE_R / FOCAL_LENGTH``, must stay under one output pixel or the edges
staircase).

Uses `longitudinal_field` (an `AngularSpectrum` z-sweep sliced on axis) and
`plot_longitudinal`.

Run with:  python examples/lens_longitudinal_focus.py
"""

import numpy as np

from diffractor import (
    MonochromaticField,
    circular_aperture,
    make_grid,
    plot_longitudinal,
)

N = 1024
L = 4e-3  # window [m]
WAVELENGTH = 550e-9
APERTURE_R = 1.4e-3  # lens/aperture radius [m]
FOCAL_LENGTH = 0.10  # focal length [m]
CONE_HALF_WIDTH = 0.4e-3  # cone panel's displayed transverse half-width [m]
CONE_SAMPLES = 140
N_PLANES_CONE = 230
N_PLANES_WAIST = 300


def main() -> None:
    grid = make_grid(N, L)
    # A uniformly illuminated circular aperture immediately behind the lens.
    U0 = MonochromaticField(grid, 1.0, wavelength=WAVELENGTH).add_aperture(
        circular_aperture, APERTURE_R, antialiased=True
    ).add_lens(FOCAL_LENGTH)

    # Sweep from well before to well past the focus. The converging beam stays
    # well inside the window over this range, so no zero-padding is needed.
    zs = np.linspace(0.55 * FOCAL_LENGTH, 1.45 * FOCAL_LENGTH, N_PLANES_CONE)
    section = U0.longitudinal(
        zs, axis="x", pad_factor=1,
        output_half_width=CONE_HALF_WIDTH, output_samples=CONE_SAMPLES,
    )

    # Zoom onto the waist itself with a decoupled fine transverse line: the
    # Airy radius here is ~24 µm — only a few input pixels at dx ≈ 3.9 µm.
    airy_radius = 0.61 * WAVELENGTH * FOCAL_LENGTH / APERTURE_R
    zw = np.linspace(0.94 * FOCAL_LENGTH, 1.06 * FOCAL_LENGTH, N_PLANES_WAIST)
    waist = U0.longitudinal(
        zw, axis="x", pad_factor=1,
        output_half_width=5.0 * airy_radius, output_samples=401,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 8.2), constrained_layout=True)
    plot_longitudinal(
        ax[0], section, vmin=-3.0,
        title=f"Lens focusing cone (f = {FOCAL_LENGTH} m, D = {2*APERTURE_R*1e3:.1f} mm)",
    )
    ax[0].set_ylim(-0.35e-3, 0.35e-3)  # zoom onto the cone near the axis
    ax[0].axvline(FOCAL_LENGTH, color="cyan", lw=0.8, ls="--", alpha=0.7)
    ax[0].text(FOCAL_LENGTH, 0.30e-3, "  focus (z = f)", color="cyan", fontsize=9)

    plot_longitudinal(
        ax[1], waist, vmin=-3.5,
        title="The waist, resolved on a decoupled fine line (output_half_width)",
    )
    ax[1].axvline(FOCAL_LENGTH, color="cyan", lw=0.8, ls="--", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
