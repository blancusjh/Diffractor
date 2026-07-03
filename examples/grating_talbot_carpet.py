"""A grating's Talbot carpet — the longitudinal near field of self-imaging.

A periodic grating illuminated by a plane wave reconstructs its own image at
multiples of the Talbot distance ``z_T = 2 d² / λ``, a shifted image at
``z_T / 2``, and reduced-period copies at rational fractions of ``z_T``. Drawn
as an ``x–z`` cross-section, this web of revivals is the famous *Talbot carpet*.

The grating fills an integer number of periods across the window, so the FFT's
periodic wrap-around is exactly the infinite-grating boundary condition
(`pad_factor=1`) — the carpet is physically correct, not an artifact.

Uses `longitudinal_field` + `plot_longitudinal`.

Run with:  python examples/grating_talbot_carpet.py
"""

import numpy as np

from diffraction import (
    Field,
    longitudinal_field,
    make_grid,
    plot_longitudinal,
    ronchi_grating,
)

N = 2048
PERIOD = 40e-6  # grating period d [m]
L = 50 * PERIOD  # exactly 50 periods across the window -> clean periodic wrap
WAVELENGTH = 550e-9
N_PLANES = 280


def main() -> None:
    grid = make_grid(N, L)
    x, y = grid
    # A plane wave through a Ronchi grating (a full-window periodic mask).
    U0 = Field(grid, ronchi_grating(x, y, PERIOD, duty=0.5).astype(complex))

    z_talbot = 2.0 * PERIOD**2 / WAVELENGTH
    zs = np.linspace(0.0, 2.0 * z_talbot, N_PLANES)
    section = longitudinal_field(U0, WAVELENGTH, zs, axis="x", pad_factor=1)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    plot_longitudinal(
        ax, section, log=False, vmax=1.0, cmap="inferno",
        title=f"Talbot carpet (d = {PERIOD*1e6:.0f} µm, z_T = 2 d²/λ = {z_talbot*1e3:.2f} mm)",
    )
    for k in (0.5, 1.0, 1.5, 2.0):
        ax.axvline(k * z_talbot, color="cyan", lw=0.6, ls="--", alpha=0.5)
    ax.set_ylim(-3 * PERIOD, 3 * PERIOD)  # a few central periods
    plt.show()


if __name__ == "__main__":
    main()
