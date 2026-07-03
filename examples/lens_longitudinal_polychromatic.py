"""The focusing cone of a lens in white light, seen edge-on — in true color.

The polychromatic companion to `lens_longitudinal_focus.py`: the same circular
aperture and thin lens, but propagated at ~24 wavelengths across the visible
band and composited to sRGB at every point of the ``x–z`` cross-section with
`PolychromaticField.longitudinal`.

The lens here is *achromatic by construction* (`thin_lens` targets the same
physical focal length regardless of wavelength — there is no dispersive
material model), so the geometric cone itself — its opening angle, and the
focal plane at ``z = f`` — is identical for every color. What differs is the
*diffraction* structure superimposed on that cone: the Airy scale (side-lobe
spacing, ring radii) scales with wavelength, so the fine fringe structure
along the cone's edges and around the waist shows up as color fringing —
blue rings sit slightly inside red ones — exactly like the colored fringes of
a white-light aperture diffraction pattern, just viewed edge-on instead of on
a single screen.

Run with:  python examples/lens_longitudinal_polychromatic.py
"""

import numpy as np

from diffractor import (
    MonochromaticField,
    PolychromaticField,
    circular_aperture,
    d65_weights,
    make_grid,
    plot_rgb_longitudinal,
)

N = 1024
L = 4e-3  # window [m]
APERTURE_R = 1.4e-3  # lens/aperture radius [m]
FOCAL_LENGTH = 0.10  # focal length [m]
N_WAVELENGTHS = 24
CONE_HALF_WIDTH = 0.4e-3  # cone panel's displayed transverse half-width [m]
CONE_SAMPLES = 140  # transverse samples for the cone panel
N_PLANES_CONE = 230  # z-planes: keeps the per-plane transverse shift (set by
# the cone's geometric slope APERTURE_R/FOCAL_LENGTH) under one CONE_SAMPLES
# pixel, so the diagonal cone edges stay smooth instead of staircasing.

# A decoupled *transverse* output line -- via `output_half_width` -- is used
# for both panels below, not just to zoom in (as in the waist), but also,
# for the wide cone view, to deliberately pick a coarser transverse pixel
# size matched to what N_PLANES_CONE can render alias-free: computing the
# cone panel on the full native (3.9 um) grid would need roughly 4x more
# z-planes for the same smoothness, at proportionally more cost.


def main() -> None:
    grid = make_grid(N, L)
    aperture = MonochromaticField(grid, 1.0).add_aperture(
        circular_aperture, APERTURE_R, antialiased=True
    ).to_field()

    wavelengths = np.linspace(430e-9, 680e-9, N_WAVELENGTHS)
    weights = d65_weights(wavelengths * 1e9)

    # The lens phase is chromatic, so the field is rebuilt per wavelength.
    pf = PolychromaticField(
        grid, aperture, wavelengths=wavelengths, weights=weights
    ).add_lens(FOCAL_LENGTH)

    # The wide view of the cone (matches lens_longitudinal_focus.py's framing).
    zs = np.linspace(0.55 * FOCAL_LENGTH, 1.45 * FOCAL_LENGTH, N_PLANES_CONE)
    cone = pf.longitudinal(
        zs, axis="x", pad_factor=1,
        output_half_width=CONE_HALF_WIDTH, output_samples=CONE_SAMPLES,
        gamut="clip", stretch=0.6, saturation=1.3,
    )

    # Zoom onto the waist with a decoupled fine transverse line, in color.
    airy_550 = 0.61 * 550e-9 * FOCAL_LENGTH / APERTURE_R
    zw = np.linspace(0.94 * FOCAL_LENGTH, 1.06 * FOCAL_LENGTH, 185)
    waist = pf.longitudinal(
        zw, axis="x", pad_factor=1,
        output_half_width=5.0 * airy_550, output_samples=251,
        gamut="clip", stretch=0.6, saturation=1.3,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 8.2), constrained_layout=True)
    plot_rgb_longitudinal(
        ax[0], cone,
        title=f"White-light focusing cone (f = {FOCAL_LENGTH} m, D = {2*APERTURE_R*1e3:.1f} mm)",
    )
    ax[0].set_ylim(-0.35e-3, 0.35e-3)
    ax[0].axvline(FOCAL_LENGTH, color="cyan", lw=0.8, ls="--", alpha=0.7)
    ax[0].text(FOCAL_LENGTH, 0.30e-3, "  focus (z = f)", color="cyan", fontsize=9)

    plot_rgb_longitudinal(
        ax[1], waist,
        title="The waist in color — Airy scale ∝ λ gives colored ring fringes",
    )
    ax[1].axvline(FOCAL_LENGTH, color="cyan", lw=0.8, ls="--", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
