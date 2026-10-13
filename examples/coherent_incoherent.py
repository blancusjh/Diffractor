"""The same two emitters, with locked or uncorrelated relative phases.

Two Gaussian emitters illuminate an ideal thin lens. Propagating each once
gives U1 and U2 at its focal plane. Locked emitters produce |U1+U2|²; mutually
incoherent emitters produce |U1|²+|U2|². A seeded ensemble with uniform random
relative phase approaches that incoherent limit without changing wavelength,
source power, aperture or propagation method.

The finite ensemble retains residual fringes proportional to
|mean(exp(i*phase))|. Random-phase convergence is statistical, not monotonic.
This is an ensemble measurement of monochromatic realizations, not a
time-domain calculation or a claim of general partial-coherence support.

Run: python examples/coherent_incoherent.py
Saves a figure and numerical profiles; add --show to open the figure.
"""

import argparse
from pathlib import Path

import numpy as np

from diffractor import (
    MonochromaticField, coherent_sum, gaussian_beam, incoherent_intensity,
    make_grid, sample_intensity,
)

WAVELENGTH = 532e-9
FOCAL_LENGTH = 0.25
SEPARATION = 0.6e-3
WAIST = 0.09e-3
WIDTH = 3.0e-3
SCREEN_WIDTH = 2.0e-3


def simulate(samples=256, realizations=256, seed=7):
    if samples < 64 or samples % 2 or realizations < 1:
        raise ValueError("Use an even sample count >=64 and at least one realization.")
    grid = make_grid(samples, WIDTH)
    screen = make_grid(samples, SCREEN_WIDTH)
    screens = [
        MonochromaticField(grid, gaussian_beam(grid, WAIST, center=(cx, 0)), wavelength=WAVELENGTH)
        .add_lens(FOCAL_LENGTH).propagate(FOCAL_LENGTH, method="fresnel_zoom", output_grid=screen)
        for cx in (-SEPARATION / 2, SEPARATION / 2)
    ]
    locked = coherent_sum(screens)
    incoherent = incoherent_intensity(screens)
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, realizations)
    averaged = np.zeros_like(incoherent)
    for phase in phases:
        averaged += coherent_sum(screens, coefficients=[1, np.exp(1j * phase)]).intensity
    averaged /= realizations
    gamma = np.mean(np.exp(1j * phases))
    reference = incoherent + 2 * np.real(gamma * screens[1].values * screens[0].values.conj())
    error = float(np.max(np.abs(averaged - reference)) / incoherent.max())
    if error > 1e-10:
        raise RuntimeError(f"Ensemble disagrees with its coherence reference: {error:.3e}")
    x = locked.grid.x[0]
    line = sample_intensity(locked, x, 0.0)
    return locked, incoherent, averaged, phases, gamma, line, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--realizations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("output/examples/coherent_incoherent.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    locked, incoherent, averaged, phases, gamma, line, error = simulate(
        args.samples, args.realizations, args.seed)
    x = locked.grid.x[0]
    scale = incoherent.max()
    fig, ax = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    dx = x[1] - x[0]
    bounds = [(x[0] - dx / 2) * 1e3, (x[-1] + dx / 2) * 1e3]
    extent = bounds + bounds
    panels = [(locked.intensity, "Locked phases"),
              (averaged, f"Random phases · {args.realizations} realizations"),
              (incoherent, "Incoherent limit · intensities added")]
    for a, (values, title) in zip(ax[0], panels):
        im = a.imshow(values / scale, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=2)
        a.set(title=title, xlim=(-0.9, 0.9), ylim=(-0.65, 0.65), xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax[0], label="Intensity / incoherent peak (shared scale)", shrink=0.8)
    for values, label in [(line, "Locked"), (averaged[len(x) // 2], "Finite ensemble"),
                          (incoherent[len(x) // 2], "Incoherent limit")]:
        ax[1, 0].plot(x * 1e3, values / scale, label=label)
    ax[1, 0].set(title="Line probe at y = 0", xlim=(-0.9, 0.9), xlabel="x [mm]", ylabel="Intensity / incoherent peak")
    ax[1, 0].legend(fontsize=9)
    counts = np.arange(1, len(phases) + 1)
    coherence = np.abs(np.cumsum(np.exp(1j * phases)) / counts)
    ax[1, 1].loglog(counts, coherence, label="This seeded ensemble")
    ax[1, 1].loglog(counts, 1 / np.sqrt(counts), "--", label="RMS reference: 1/√M")
    ax[1, 1].set(title="Residual phase coherence", xlabel="Number of realizations M", ylabel="|mean(exp(iφ))|")
    ax[1, 1].legend(fontsize=9)
    ax[1, 2].hist(phases / np.pi, bins=np.linspace(0, 2, 17), color="#355f8d", edgecolor="white")
    ax[1, 2].set(title=f"Relative phase distribution · seed {args.seed}", xlabel="Relative phase / π", ylabel="Count")
    fig.suptitle("Coherence changes interference · same wavelength, emitters and optical system", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    np.savez(args.output.with_suffix(".npz"), x=x, locked_intensity=locked.intensity,
             ensemble_intensity=averaged, incoherent_intensity=incoherent,
             relative_phases=phases, seed=args.seed, wavelength=WAVELENGTH)
    print(f"Residual phase coherence: {abs(gamma):.6f}; ensemble reference error: {error:.3e}")
    print(f"Saved {args.output} and {args.output.with_suffix('.npz')}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
