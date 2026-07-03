"""Letting the toolkit choose the grid for a near-field ASM propagation.

A small circular aperture in the deep near field (high Fresnel number) aliases
on a coarse grid: the undersampled boundary-wave ripples fold back as a
non-physical, axis-aligned cross-hatch. The fix is a finer ``dx`` — but how
fine? :func:`recommend_grid_convergence` answers empirically, by propagating at
increasing ``N`` and watching a rotational-symmetry metric (the intensity
coefficient-of-variation around each ring, which *must* be ~0 for a circular
aperture) fall to tolerance.

This reproduces the manual sweep from the debugging session: the metric drops
across N = 512 → 1024 → 2048 → …, and the before/after panels show the
cross-hatch at the old hand-picked N=1024 giving way to clean rings at the
recommended grid.

Run with:  python examples/adaptive_grid_selection.py
"""

from diffractor import (
    MonochromaticField,
    circular_aperture,
    make_grid,
    recommend_grid_convergence,
)

WAVELENGTH = 532e-9
L = 6e-3
RADIUS = 0.3e-3
Z = 0.05
N_OLD = 1024  # the coarse grid that aliases


def aperture(grid):
    return MonochromaticField(grid, 1.0, wavelength=WAVELENGTH).add_aperture(
        circular_aperture, RADIUS, antialiased=True
    ).to_field()


def propagate(n):
    grid = make_grid(n, L)
    U0 = MonochromaticField(grid, 1.0, wavelength=WAVELENGTH).add_aperture(
        circular_aperture, RADIUS, antialiased=True
    )
    return U0.propagate(Z, method="asm", pad_factor=2)


def main() -> None:
    rec = recommend_grid_convergence(
        aperture,
        length=L,
        wavelength=WAVELENGTH,
        z=Z,
        n0=512,
        n_max=4096,
        metric="azimuthal_cv",
        radial_window=(0.3e-3, 1.2e-3),  # where the pattern carries real signal
        tol=0.05,
    )

    print("Convergence history (azimuthal coefficient-of-variation):")
    for n, cv in rec.history:
        print(f"  N = {n:5d}   CV = {cv:.3f}")
    print(f"\nConverged: {rec.converged}")
    print(f"Recommended grid: N = {rec.n}, dx = {rec.dx * 1e6:.2f} um")
    print(f"(the old hand-picked N={N_OLD} was chosen by trial and error)")

    old = propagate(N_OLD)
    new = propagate(rec.n)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    old.plot(ax[0], title=f"N = {N_OLD} (aliased cross-hatch)")
    ax[0].set_xlim(-3e-3, 3e-3)
    ax[0].set_ylim(-3e-3, 3e-3)
    new.plot(ax[1], title=f"Recommended N = {rec.n} (clean rings)")
    ax[1].set_xlim(-3e-3, 3e-3)
    ax[1].set_ylim(-3e-3, 3e-3)
    fig.suptitle("recommend_grid_convergence picks an adequate near-field grid")
    plt.show()


if __name__ == "__main__":
    main()
