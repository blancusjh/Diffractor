"""Measuring the aberrations of a Cartesian oval vs its parabolic approximation.

The exact optical path difference of each surface (spherical wave from the
design object point, exact ray paths through the curved interface — no
thin-element approximation) is sampled over the pupil and decomposed into
Noll Zernike modes with the diffractor.physics.aberrations module.

Expected physics: the stigmatic oval measures zero in every mode; the
osculating paraboloid carries pure rotationally-symmetric aberration —
defocus + primary (and a little secondary) spherical. The balanced RMS
(defocus removed, i.e. at best focus) plugged into the Maréchal formula
reproduces the Strehl ratio measured independently by the Huygens
propagation in cartesian_vs_parabolic_aberration.py.

Run with:  python examples/aberration_measurement.py
"""

import matplotlib.pyplot as plt
import numpy as np

from diffractor import (
    CartesianSurface,
    Field,
    ParabolicSurface,
    fit_zernikes,
    make_grid,
    marechal_strehl,
    pv,
    rms,
    synthesize_zernikes,
    zernike_name,
)

# Same design as cartesian_vs_parabolic_aberration.py.
N1, N2 = 1.0, 1.5
ZO, ZI = -0.05, 0.1  # object / image distances [m]
WAVELENGTH = 530e-9
R_AP = 3.5e-3  # pupil radius [m]
JMAX = 22  # fit up to secondary spherical

N = 512
GRID = make_grid(N, 2.2 * R_AP)


def exact_opd_waves(surface):
    """Exact geometric OPD (waves) vs the axial ray, on the pupil grid."""
    x, y = GRID
    z_s = surface.sag(x, y)
    r2 = x**2 + y**2
    d1 = np.sqrt((z_s - ZO) ** 2 + r2)
    d2 = np.sqrt((ZI - z_s) ** 2 + r2)
    W = (N1 * d1 + N2 * d2 - (N1 * abs(ZO) + N2 * ZI)) / WAVELENGTH
    return Field(GRID, np.where(np.sqrt(r2) <= R_AP, W, np.nan))


def report(name, W):
    c = fit_zernikes(W, R_AP, jmax=JMAX)
    total_rms = np.sqrt(np.sum(c[1:] ** 2))  # about piston
    balanced = c.copy()
    balanced[[0, 1, 2, 3]] = 0.0  # remove piston, tilts and defocus
    balanced_rms = np.sqrt(np.sum(balanced**2))

    print(f"\n{name}")
    print(f"  PV  = {pv(W):.3e} waves    RMS = {rms(W):.3e} waves")
    for j in (4, 11, 22):
        print(f"  Z{j:<2d} {zernike_name(j):<20s} = {c[j-1]:+.4f} waves RMS")
    print(f"  RMS about piston (from fit)      = {total_rms:.4f} waves")
    print(f"  balanced RMS (defocus removed)   = {balanced_rms:.4f} waves")
    print(f"  Marechal Strehl at best focus    = {marechal_strehl(balanced_rms):.3f}")
    return c


def main() -> None:
    oval = CartesianSurface(n1=N1, n2=N2, zo=ZO, zi=ZI)
    parab = ParabolicSurface(focal_length=1.0 / (2.0 * oval.paraxial_curvature()))

    W_oval = exact_opd_waves(oval)
    W_parab = exact_opd_waves(parab)
    c_oval = report("Cartesian oval (stigmatic)", W_oval)
    c_parab = report("Osculating paraboloid", W_parab)

    # Radial profiles: measured OPD, Zernike reconstruction, residual.
    x, y = GRID
    row = N // 2
    r_mm = x[row, N // 2 :] * 1e3
    W_fit = synthesize_zernikes(c_parab, GRID, R_AP)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)

    im = ax[0].imshow(
        W_parab.values,
        extent=[x.min() * 1e3, x.max() * 1e3, y.min() * 1e3, y.max() * 1e3],
        origin="lower",
        cmap="RdBu_r",
    )
    ax[0].set_title("Paraboloid OPD over the pupil [waves]")
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    js = np.arange(1, JMAX + 1)
    ax[1].bar(js - 0.2, np.abs(c_oval), width=0.4, label="Cartesian oval", color="C2")
    ax[1].bar(js + 0.2, np.abs(c_parab), width=0.4, label="paraboloid", color="C3")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-12, 1.0)
    ax[1].set_xticks(js[::2])
    ax[1].set_xlabel("Noll index j")
    ax[1].set_ylabel("|coefficient| [waves RMS]")
    ax[1].set_title("Zernike spectrum (log scale)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")

    ax[2].plot(r_mm, W_parab.values[row, N // 2 :], "C3", lw=2, label="paraboloid OPD (exact)")
    ax[2].plot(r_mm, W_fit.values[row, N // 2 :], "k--", lw=1, label=f"Zernike fit (j ≤ {JMAX})")
    ax[2].plot(r_mm, W_oval.values[row, N // 2 :], "C2", label="Cartesian oval (≡ 0)")
    ax[2].set_xlabel("r [mm]")
    ax[2].set_ylabel("OPD [waves]")
    ax[2].set_title("Radial profile and fit")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
