"""A 4f imaging system: two thin lenses and a filter in their common focal plane.

Object -- f -- lens -- f -- filter -- f -- lens -- f -- image.
The unfiltered image has unit magnification and is inverted. A Gaussian
amplitude filter suppresses fine detail; its complement emphasizes edges.
All four propagation steps use the existing single-FFT Fresnel method.

The window L=sqrt(N*lambda*f) makes successive FFT sampling grids equal.
Increasing N refines the sampling and enlarges the window at fixed physical
object size. Ideal lenses have no aperture stop here; this is a scalar,
paraxial, coherent system. The Fourier filter is the only lossy element.

Run: python examples/four_f_imaging.py
Saves a figure and field arrays; add --show to open the figure.
"""

import argparse
from pathlib import Path

import numpy as np

from diffractor import FFT2, MonochromaticField, make_grid, sample_intensity

WAVELENGTH = 532e-9
FOCAL_LENGTH = 0.2
FILTER_RADIUS = 0.22e-3


def object_amplitude(x, y):
    """An asymmetric F and three bars, with smooth 35 µm edges."""
    def rectangle(cx, cy, wx, wy):
        edge = 35e-6
        return (0.5 * (np.tanh((x - cx + wx / 2) / edge) - np.tanh((x - cx - wx / 2) / edge))
                * 0.5 * (np.tanh((y - cy + wy / 2) / edge) - np.tanh((y - cy - wy / 2) / edge)))
    masks = [rectangle(-0.7e-3, 0, 0.17e-3, 1.25e-3),
             rectangle(-0.43e-3, 0.53e-3, 0.7e-3, 0.17e-3),
             rectangle(-0.48e-3, 0.03e-3, 0.6e-3, 0.17e-3)]
    masks += [rectangle(cx, -0.15e-3, 0.09e-3, 0.85e-3) for cx in (0.2e-3, 0.43e-3, 0.66e-3)]
    return 1 - np.prod([1 - mask for mask in masks], axis=0)


def simulate(samples=512):
    if samples < 64 or samples % 2:
        raise ValueError("Use an even sample count of at least 64.")
    grid = make_grid(samples, np.sqrt(samples * WAVELENGTH * FOCAL_LENGTH))
    source = MonochromaticField(grid, object_amplitude, wavelength=WAVELENGTH)
    fourier = (source.propagate(FOCAL_LENGTH, method="fresnel")
               .add_lens(FOCAL_LENGTH).propagate(FOCAL_LENGTH, method="fresnel"))
    x, y = fourier.grid
    lowpass = np.exp(-(x**2 + y**2) / (2 * FILTER_RADIUS**2))
    masks = {"Unfiltered": np.ones(grid.shape), "Low-pass": lowpass, "High-pass": 1 - lowpass}
    images, errors, transmission = {}, {}, {}
    input_energy = np.sum(source.intensity)
    for name, mask in masks.items():
        # A fresh field keeps the shared Fourier plane unchanged between runs.
        filtered = MonochromaticField(fourier.grid, fourier.values * mask, wavelength=WAVELENGTH)
        result = (filtered.propagate(FOCAL_LENGTH, method="fresnel")
                  .add_lens(FOCAL_LENGTH).propagate(FOCAL_LENGTH, method="fresnel"))
        images[name] = result
        # Independent ideal 4f transfer relation: two forward transforms give
        # inversion. Global phase cancels in the intensity comparison.
        reference = FFT2(FFT2(source.values) * mask) / samples**2
        errors[name] = float(np.max(np.abs(result.intensity - np.abs(reference)**2)))
        transmission[name] = float(np.sum(result.intensity) / input_energy)
    if max(errors.values()) > 1e-8:
        raise RuntimeError(f"4f system disagrees with its Fourier transfer relation: {errors}")
    return source, fourier, images, transmission, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("output/examples/four_f_imaging.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, fourier, images, transmission, errors = simulate(args.samples)
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.3), constrained_layout=True)
    x = source.grid.x[0] * 1e3
    dx = x[1] - x[0]
    bounds = [x[0] - dx / 2, x[-1] + dx / 2]
    extent = bounds + bounds
    for a, u, title in [(ax[0, 0], source, "Object"),
                        (ax[0, 1], images["Unfiltered"], "Image · inverted 1:1"),
                        (ax[0, 2], images["Low-pass"], "Low-pass · softened detail")]:
        im = a.imshow(u.intensity, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=1)
        a.set(title=title, xlim=(-1.1, 1.1), ylim=(-0.9, 0.9), xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax[0], label="Intensity / unit incident intensity", shrink=0.8)
    normalized_spectrum = fourier.intensity / fourier.intensity.max()
    im = ax[1, 0].imshow(np.log10(np.maximum(normalized_spectrum, 1e-6)),
                         extent=extent, origin="lower", cmap="magma", vmin=-6, vmax=0)
    ax[1, 0].add_patch(plt.Circle((0, 0), FILTER_RADIUS * 1e3, fill=False, color="cyan", lw=1))
    ax[1, 0].set(title="Fourier plane · filter radius", xlim=(-1, 1), ylim=(-1, 1),
                 xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax[1, 0], label="log₁₀(I / Fourier-plane peak)", shrink=0.8)
    im = ax[1, 1].imshow(np.log10(np.maximum(images["High-pass"].intensity, 1e-5)),
                         extent=extent, origin="lower", cmap="inferno", vmin=-5, vmax=0)
    ax[1, 1].set(title="High-pass · log intensity", xlim=(-1.1, 1.1), ylim=(-0.9, 0.9),
                 xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax[1, 1], label="log₁₀(I / unit incident intensity)", shrink=0.8)
    probes = np.linspace(-1.1e-3, 1.1e-3, 501)
    for name, u in images.items():
        ax[1, 2].plot(probes * 1e3, sample_intensity(u, probes, 0.15e-3), label=name)
    ax[1, 2].set(title="Image probe · y = 0.15 mm", xlabel="x [mm]", ylabel="Intensity")
    ax[1, 2].legend(fontsize=9)
    fig.suptitle("4f imaging · object → lens → Fourier filter → lens → image", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    np.savez(args.output.with_suffix(".npz"), x=source.grid.x[0], object_field=source.values,
             fourier_field=fourier.values, image_field=images["Unfiltered"].values,
             lowpass_field=images["Low-pass"].values, highpass_field=images["High-pass"].values,
             wavelength=WAVELENGTH, focal_length=FOCAL_LENGTH)
    for name in images:
        print(f"{name}: transmitted power {transmission[name]:.6f}; max reference error {errors[name]:.3e}")
    print(f"Saved {args.output} and {args.output.with_suffix('.npz')}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
