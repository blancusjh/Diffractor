"""Two crossing plane waves: phase-controlled fringes and numerical probes.

Each source has unit intensity. For equal, opposite angles theta, the exact
screen intensity is 2 + 2*cos(2*k*sin(theta)*x - relative_phase). The tilt is
on a native FFT frequency and the boundary is periodic, so this example
isolates interference and probe accuracy from aperture-edge diffraction.

Run: python examples/two_source_interference.py
Saves a figure and probe arrays; add --show to open the figure.
"""

import argparse
from pathlib import Path

import numpy as np

from diffractor import (
    MonochromaticField, coherent_sum, make_grid, plane_wave,
    sample_field, sample_intensity,
)

WAVELENGTH = 532e-9
WIDTH = 3.2e-3
DISTANCE = 0.2
FRINGES = 12


def simulate(samples=256):
    if samples < 64 or samples % 2:
        raise ValueError("Use an even sample count of at least 64.")
    grid = make_grid(samples, WIDTH)
    theta = np.arcsin(WAVELENGTH * FRINGES / (2 * WIDTH))
    sources = [
        MonochromaticField(grid, plane_wave(grid, WAVELENGTH, theta_x=angle),
                           wavelength=WAVELENGTH)
        for angle in (theta, -theta)
    ]
    # Both waves acquire the same longitudinal phase during propagation.
    screens = [u.propagate(DISTANCE, pad_factor=1, bandlimit=False) for u in sources]
    phases = np.array([0.0, np.pi / 2, np.pi])
    fields = [coherent_sum(screens, coefficients=[1, np.exp(1j * p)]) for p in phases]
    x = grid.x[0]
    profiles = np.array([sample_intensity(u, x, 0.0) for u in fields])
    expected = 2 + 2 * np.cos(2 * np.pi * FRINGES * x[None, :] / WIDTH - phases[:, None])
    sweep = np.linspace(0, 2 * np.pi, 65)
    point = np.array([
        sample_field(coherent_sum(screens, coefficients=[1, np.exp(1j * p)]), 0.0, 0.0)
        for p in sweep
    ])
    error = float(np.max(np.abs(profiles - expected)))
    if error > 1e-9:
        raise RuntimeError(f"Interference disagrees with the analytic result: {error:.3e}")
    return fields, x, profiles, expected, sweep, point, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("output/examples/two_source_interference.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields, x, profiles, expected, sweep, point, error = simulate(args.samples)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    dx = x[1] - x[0]
    bounds = [(x[0] - dx / 2) * 1e3, (x[-1] + dx / 2) * 1e3]
    extent = bounds + bounds
    for a, u, title in zip(ax[0], fields, ("Relative phase 0", "Relative phase π/2")):
        im = a.imshow(u.intensity, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=4)
        a.axhline(0, color="white", lw=1, ls="--")
        a.plot(0, 0, "o", color="cyan", ms=5)
        a.set(title=title, xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax[0], label="Intensity / single-source intensity", shrink=0.85)
    for values, reference, label in zip(profiles, expected, ("0", "π/2", "π")):
        line, = ax[1, 0].plot(x * 1e3, values, label=f"Phase {label}")
        ax[1, 0].plot(x[::8] * 1e3, reference[::8], ".", color=line.get_color(), ms=4)
    ax[1, 0].set(xlim=(-0.55, 0.55), xlabel="x [mm]", ylabel="Intensity",
                 title="Line probe at y = 0 · dots: analytic reference")
    ax[1, 0].legend(ncol=3, fontsize=9)
    ax[1, 1].plot(sweep / np.pi, 2 + 2 * np.cos(sweep), label="2 + 2 cos(phase)")
    ax[1, 1].plot(sweep[::4] / np.pi, np.abs(point[::4]) ** 2, "o", ms=4, label="Point probe")
    ax[1, 1].set(xlabel="Relative phase / π", ylabel="Intensity", title="Point probe at the origin")
    ax[1, 1].legend()
    fig.suptitle("Two coherent sources · move the fringes by changing phase", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    np.savez(args.output.with_suffix(".npz"), x=x, intensity=profiles, reference=expected,
             relative_phase=sweep, point_field=point, wavelength=WAVELENGTH)
    print(f"Fringe pitch: {WIDTH / FRINGES * 1e6:.3f} µm; max analytic error: {error:.3e}")
    print(f"Saved {args.output} and {args.output.with_suffix('.npz')}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
