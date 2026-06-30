"""Cartesian oval vs its paraxial (parabolic) approximation.

A spherical wave from the design object point is focused by (a) the exact
stigmatic Cartesian-oval surface and (b) the osculating paraboloid with
the same vertex curvature. The comparison uses the exact geometric OPD
and an axisymmetric Huygens integral evaluated from the *curved* surface,
because the thin-element phase-screen model has errors of the same r^4
order as the oval-parabola difference under study.

Propagating from the curved surface with exact distances is equivalent to
applying the sag in the angular spectrum (each plane-wave component
referenced from the local surface height with its own exp(i kz z_s)):
cross-validated, both give identical PSFs and Strehl ratios here, while
the thin-element screen exp(i k0 (n1 - n2) z_s) - which implicitly sets
kz = k for every component - misassigns ~1.2 waves of spurious OPD to
the stigmatic oval at this aperture (NA_obj ~ 0.07).

The oval focuses to a textbook Airy pattern; the paraboloid shows pure
primary spherical aberration: a halo at the design plane, a best focus
shifted toward the surface, and a Strehl ratio that refocusing only
partially recovers.

Requires scipy (pip install "diffraction[examples]").

Run with:  python examples/cartesian_vs_parabolic_aberration.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0, j1

from diffraction import CartesianSurface, ParabolicSurface

# Design: point source imaged by a single refracting surface.
N1, N2 = 1.0, 1.5
ZO, ZI = -0.05, 0.1  # object / image distances [m]
WAVELENGTH = 530e-9
R_AP = 3.5e-3  # aperture radius [m] (NA_obj ~ 0.07)

K1 = 2 * np.pi * N1 / WAVELENGTH
K2 = 2 * np.pi * N2 / WAVELENGTH


def geometric_opd_waves(z_s, r):
    """Exact optical path error vs the axial ray, in waves."""
    d1 = np.sqrt((z_s - ZO) ** 2 + r**2)
    d2 = np.sqrt((ZI - z_s) ** 2 + r**2)
    return (N1 * d1 + N2 * d2 - (N1 * abs(ZO) + N2 * ZI)) / WAVELENGTH


def huygens_field(z_s, r, z_img, r_out):
    """Axisymmetric Huygens integral from the curved surface.

    The azimuthal integral reduces to a Bessel J0 factor; the quadratic
    term dropped in that reduction contributes ~1e-4 rad here.
    """
    d1 = np.sqrt((z_s - ZO) ** 2 + r**2)
    source = np.exp(1j * K1 * d1) / d1 * r
    out = np.empty(len(r_out), dtype=complex)
    for i, rho in enumerate(r_out):
        d2 = np.sqrt((z_img - z_s) ** 2 + r**2 + rho**2)
        out[i] = np.trapezoid(source * np.exp(1j * K2 * d2) / d2 * j0(K2 * r * rho / d2), r)
    return out


def main() -> None:
    oval = CartesianSurface(n1=N1, n2=N2, zo=ZO, zi=ZI)
    parab = ParabolicSurface(focal_length=1.0 / (2.0 * oval.paraxial_curvature()))

    r = np.linspace(0.0, R_AP, 6000)
    zeros = np.zeros_like(r)
    z_o, z_p = oval.sag(r, zeros), parab.sag(r, zeros)

    W_p = geometric_opd_waves(z_p, r)
    print(f"parabola OPD: PV = {np.ptp(W_p):.3f} waves, RMS = {np.std(W_p):.3f} waves")

    # Through-focus axial response.
    zs = np.linspace(0.97 * ZI, 1.03 * ZI, 241)
    F_o = np.abs([huygens_field(z_o, r, z, np.array([0.0]))[0] for z in zs]) ** 2
    F_p = np.abs([huygens_field(z_p, r, z, np.array([0.0]))[0] for z in zs]) ** 2
    I0 = F_o.max()
    z_best = zs[np.argmax(F_p)]
    print(f"parabola best focus: z = {z_best*1e3:.2f} mm (design {ZI*1e3:.0f} mm)")

    # PSF radial profiles.
    r_out = np.linspace(0.0, 30e-6, 400)
    P_o = np.abs(huygens_field(z_o, r, ZI, r_out)) ** 2 / I0
    P_p = np.abs(huygens_field(z_p, r, ZI, r_out)) ** 2 / I0
    P_pb = np.abs(huygens_field(z_p, r, z_best, r_out)) ** 2 / I0
    print(f"parabola Strehl @ design plane: {P_p[0]:.3f}, @ best focus: {P_pb[0]:.3f}")

    na = R_AP / np.sqrt(R_AP**2 + ZI**2)
    v = np.where(r_out == 0, 1e-12, K2 * na * r_out)
    airy = (2 * j1(v) / v) ** 2

    def encircled(P):
        c = np.cumsum(P * r_out)
        return c / c[-1]

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    ax[0, 0].plot(r * 1e3, (z_o - z_p) * 1e6, "C0")
    ax[0, 0].set_xlabel("r [mm]")
    ax[0, 0].set_ylabel("z_oval − z_parabola [µm]")
    ax[0, 0].set_title("Shape difference (∝ r⁴)")
    ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(r * 1e3, W_p, "C3", label="parabola")
    ax[0, 1].plot(r * 1e3, geometric_opd_waves(z_o, r), "C2", label="Cartesian oval (≡ 0)")
    ax[0, 1].set_xlabel("r [mm]")
    ax[0, 1].set_ylabel("OPD [waves]")
    ax[0, 1].set_title("Exact geometric spherical aberration")
    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)

    ax[0, 2].plot(zs * 1e3, F_o / I0, "C2", label="Cartesian oval")
    ax[0, 2].plot(zs * 1e3, F_p / I0, "C3", label="parabola")
    ax[0, 2].axvline(ZI * 1e3, color="gray", ls="--", lw=0.8)
    ax[0, 2].set_xlabel("z [mm]")
    ax[0, 2].set_ylabel("I(0, z) / oval peak")
    ax[0, 2].set_title("Through-focus axial response")
    ax[0, 2].legend()
    ax[0, 2].grid(alpha=0.3)

    ax[1, 0].semilogy(r_out * 1e6, P_o, "C2", label="oval @ design plane")
    ax[1, 0].semilogy(r_out * 1e6, airy, "k--", lw=1, label="analytic Airy")
    ax[1, 0].semilogy(r_out * 1e6, P_p, "C3", label="parabola @ design plane")
    ax[1, 0].set_ylim(1e-7, 2)
    ax[1, 0].set_xlabel("r [µm]")
    ax[1, 0].set_ylabel("I / oval peak")
    ax[1, 0].set_title("PSF at the design image plane")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].semilogy(r_out * 1e6, P_o, "C2", label="oval @ design plane")
    ax[1, 1].semilogy(r_out * 1e6, P_pb, "C1", label=f"parabola @ best focus ({z_best*1e3:.2f} mm)")
    ax[1, 1].set_ylim(1e-7, 2)
    ax[1, 1].set_xlabel("r [µm]")
    ax[1, 1].set_ylabel("I / oval peak")
    ax[1, 1].set_title("Refocused parabola vs oval")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)

    ax[1, 2].plot(r_out * 1e6, encircled(P_o), "C2", label="oval @ design")
    ax[1, 2].plot(r_out * 1e6, encircled(P_p), "C3", label="parabola @ design")
    ax[1, 2].plot(r_out * 1e6, encircled(P_pb), "C1", label="parabola @ best focus")
    ax[1, 2].axhline(0.838, color="gray", lw=0.8, ls=":")
    ax[1, 2].set_xlabel("r [µm]")
    ax[1, 2].set_ylabel("encircled energy")
    ax[1, 2].set_title("Encircled energy (83.8% = Airy disc)")
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].grid(alpha=0.3)

    fig.suptitle(
        f"Spherical wave (zo = {ZO*1e3:.0f} mm) focused to zi = {ZI*1e3:.0f} mm, "
        f"n1 = {N1}, n2 = {N2}, aperture {R_AP*1e3:.1f} mm, λ = {WAVELENGTH*1e9:.0f} nm"
    )
    plt.show()


if __name__ == "__main__":
    main()
