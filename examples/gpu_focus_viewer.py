"""GPU-batched focus scan viewed with the VisPy scalar-field viewer.

Ties together the two "extra" pieces of the toolkit that aren't exercised by
the core examples:

  * :class:`diffraction.AngularSpectrum` — the reusable, batch-friendly
    angular-spectrum propagator, run on an NVIDIA GPU through CuPy when one
    is available (falls back to NumPy on the CPU otherwise).
  * :mod:`diffraction.viz` — the VisPy-based interactive viewers
    (:func:`plot_scalar_field`, :func:`plot_scalar_field_3d`), which render
    large fields with a GPU-accelerated, pannable/zoomable canvas instead of
    matplotlib's raster path.

A collimated beam is focused by a parabolic refracting surface (thin-element
approximation) and then swept through a stack of planes around the paraxial
focus in a single batched call. The in-focus plane is shown as a 2D image and
as a shaded 3D surface (the Airy-pattern "sombrero"); a defocused plane is
shown alongside for comparison.

If VisPy (the ``viz`` extra) is not installed, this falls back to matplotlib
so the example still runs and produces a figure — install it for the
interactive, GPU-rendered version:

    pip install "diffraction[viz]"     # interactive viewers
    pip install "diffraction[gpu]"     # CUDA propagation (needs an NVIDIA GPU)

Run with:  python examples/gpu_focus_viewer.py
"""

import numpy as np

from diffraction import (
    AngularSpectrum,
    CUPY_AVAILABLE,
    ParabolicSurface,
    antialiased,
    circular_aperture,
    make_grid,
    to_device,
)

N = 1024  # samples per side
L = 3e-3  # grid side length [m]
WAVELENGTH = 530e-9  # vacuum wavelength [m]
N1, N2 = 1.0, 1.5  # refractive indices before / after the surface
FOCAL_LENGTH = 0.06  # parabola focal length [m]
RADIUS = 0.4e-3  # aperture radius [m]
DEFOCUS_SPAN = 4e-3  # scan +/- this far from the paraxial focus [m]
N_PLANES = 25


def main() -> None:
    grid = make_grid(N, L)

    surface = ParabolicSurface(focal_length=FOCAL_LENGTH)
    U0 = antialiased(circular_aperture, grid, RADIUS).astype(complex)
    U_after = U0 * surface.phase_mask(grid, WAVELENGTH, N1, N2)

    z_focus = 2.0 * FOCAL_LENGTH * N2 / (N2 - N1)

    device = "gpu" if CUPY_AVAILABLE else "cpu"
    if device == "gpu":
        U_after = to_device(U_after, "gpu")
    tag = "[GPU/CuPy]" if device == "gpu" else "[CPU/NumPy]"
    print(f"Batched focus scan of {N_PLANES} planes on {device.upper()} {tag}")

    # One propagator, reused for every plane in the scan: the transfer-
    # function machinery and the input FFT are built once (see
    # AngularSpectrum's docstring), so each extra plane costs one IFFT.
    # pad_factor=2 matters here: the beam converges and re-diverges over the
    # ~360 mm scan, and without padding it wraps around the small (3 mm)
    # window through the FFT's periodic boundary, aliasing back in as a
    # spurious grid-aligned cross-hatch pattern.
    prop = AngularSpectrum(U_after, grid, wavelength=WAVELENGTH, n=N2, pad_factor=2)

    zs = z_focus + np.linspace(-DEFOCUS_SPAN, DEFOCUS_SPAN, N_PLANES)
    stack = prop.intensity_stack(zs, normalize=False)  # host arrays, physical units

    peak_idx = int(np.argmax([I.max() for I in stack]))
    focus_plane = stack[peak_idx]
    defocused_plane = stack[0]

    focus_plane = focus_plane / focus_plane.max()
    defocused_plane = defocused_plane / defocused_plane.max()

    x, _ = grid
    extent_um = [x.min() * 1e6, x.max() * 1e6, x.min() * 1e6, x.max() * 1e6]
    print(f"Best focus in the scan: z = {zs[peak_idx] * 1e3:.3f} mm "
          f"(paraxial estimate {z_focus * 1e3:.3f} mm)")

    try:
        from diffraction import plot_scalar_field, plot_scalar_field_3d

        plot_scalar_field(
            focus_plane,
            title=f"In-focus intensity {tag}  (z = {zs[peak_idx] * 1e3:.2f} mm)",
            cmap="inferno",
            log_scale=True,
            extent=extent_um,
            xlabel="x [um]",
            ylabel="y [um]",
            show_center_cross=True,
        )
        plot_scalar_field_3d(
            focus_plane,
            title=f"In-focus intensity, 3D {tag}",
            cmap="inferno",
            log_scale=False,
            extent=extent_um,
        )
        plot_scalar_field(
            defocused_plane,
            title=f"Defocused plane {tag}  (z - z_focus = {-DEFOCUS_SPAN * 1e3:.2f} mm)",
            cmap="inferno",
            log_scale=True,
            extent=extent_um,
            xlabel="x [um]",
            ylabel="y [um]",
        )
    except ImportError:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        for a, (frame, title) in zip(
            ax,
            [
                (focus_plane, f"In focus (z = {zs[peak_idx] * 1e3:.2f} mm)"),
                (defocused_plane, f"Defocused (z - z_focus = {-DEFOCUS_SPAN * 1e3:.2f} mm)"),
            ],
        ):
            a.imshow(
                np.log10(frame + 1e-6),
                extent=extent_um,
                origin="lower",
                cmap="inferno",
                vmin=-6,
                vmax=0,
            )
            a.set_title(title)
            a.set_xlabel("x [um]")
        ax[0].set_ylabel("y [um]")
        fig.suptitle(f"GPU focus scan {tag} (install the 'viz' extra for the VisPy viewer)")
        plt.show()


if __name__ == "__main__":
    main()
