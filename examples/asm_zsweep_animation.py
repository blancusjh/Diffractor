"""Animated angular-spectrum z-sweep, GPU-accelerated when available.

Builds a circular aperture once, wraps it in a reusable :class:`AngularSpectrum`
propagator (which precomputes the transfer-function machinery and the input
FFT), then evaluates a whole stack of propagation distances. Each extra plane
costs a single inverse FFT, so sweeping dozens of distances is cheap — and if
CuPy and a CUDA GPU are present the entire batch runs on the device.

Two viewing paths:
  * VisPy (the ``viz`` extra) → an interactive movie with play/pause and
    frame-stepping (diffraction.viz.animate).
  * matplotlib fallback → a static montage of a few planes.

Run with:  python examples/asm_zsweep_animation.py
"""

import numpy as np

from diffraction import (
    AngularSpectrum,
    CUPY_AVAILABLE,
    antialiased,
    circular_aperture,
    make_grid,
    to_device,
)

N = 2048  # samples per side — see the sampling note in simple_asm_diffraction.py:
# a coarser grid under-resolves the near-field boundary-wave ripples of this
# small aperture and leaves an axis-aligned aliasing artifact in the
# mid-sweep frames.
L = 6e-3  # grid side length [m]
WAVELENGTH = 532e-9  # vacuum wavelength [m]
RADIUS = 0.3e-3  # aperture radius [m]
# The boundary-wave ripple frequency grows as z shrinks (Fresnel number
# R^2/(lambda z)), so N=2048 only resolves it cleanly from z >~ 45 mm on;
# starting the sweep there keeps every frame alias-free.
Z_MIN, Z_MAX, N_FRAMES = 0.05, 0.16, 40


def main() -> None:
    grid = make_grid(N, L)
    U0 = antialiased(circular_aperture, grid, RADIUS).astype(complex)

    device = "gpu" if CUPY_AVAILABLE else "cpu"
    if device == "gpu":
        U0 = to_device(U0, "gpu")
    tag = "[GPU]" if device == "gpu" else "[CPU]"
    print(f"Propagating a {N_FRAMES}-plane z-sweep on the {device.upper()} {tag}")

    prop = AngularSpectrum(U0, grid, wavelength=WAVELENGTH, pad_factor=2)
    zs = np.linspace(Z_MIN, Z_MAX, N_FRAMES)

    # log-intensity frames on the host, ready for display.
    frames = [
        np.log10(I + 1e-6) for I in prop.intensity_stack(zs, normalize=True)
    ]

    x, _ = grid
    extent_mm = [x.min() * 1e3, x.max() * 1e3, x.min() * 1e3, x.max() * 1e3]

    try:
        from diffraction import animate

        animate(
            frames,
            list(zs),
            title="Circular aperture — ASM z-sweep",
            extent=extent_mm,
            log_scale=True,
            annotation=tag,
        )
    except ImportError:
        import matplotlib.pyplot as plt

        picks = np.linspace(0, N_FRAMES - 1, 4).astype(int)
        fig, ax = plt.subplots(1, len(picks), figsize=(16, 4), constrained_layout=True)
        for a, idx in zip(ax, picks):
            a.imshow(
                frames[idx],
                extent=extent_mm,
                origin="lower",
                cmap="hot",
                vmin=-6,
                vmax=0,
            )
            a.set_title(f"z = {zs[idx] * 1e3:.1f} mm")
            a.set_xlabel("x [mm]")
        ax[0].set_ylabel("y [mm]")
        fig.suptitle(f"ASM z-sweep montage {tag} (install the 'viz' extra to animate)")
        plt.show()


if __name__ == "__main__":
    main()
