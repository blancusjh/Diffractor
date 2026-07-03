"""Oval vs paraboloid, longitudinally, across three apertures.

The same stigmatic Cartesian oval and its osculating paraboloid as
`cartesian_vs_parabolic_aberration.py` (same conjugates: a near-collimated
source at ``zo``, imaged to ``zi``), swept across three aperture radii —
low, medium and very high NA — to show spherical aberration *emerging* as the
aperture opens: at low NA the two system caustics are visually
indistinguishable (both a clean cone to a sharp point); at medium NA the
paraboloid starts smearing; at high NA (the hero comparison) it collapses
into a wide, badly shifted caustic while the oval stays perfect at every
aperture (it is exact by construction, at any radius its own sag solver
supports).

Six panels (3 rows × 2 columns): each row is one aperture, each column one
surface, drawn with the same "beam → surface → cone → focus → beyond"
construction as the flagship example (see its docstring for the physics and
the honest caveats about the J0-reduced integral).

Requires scipy (pip install "diffraction[examples]").

Run with:  python examples/oval_vs_parabola_apertures.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0

from diffraction import CartesianSurface, LongitudinalSection, ParabolicSurface, plot_longitudinal

N1, N2 = 1.0, 1.5
ZO, ZI = -0.1, 12.0e-3  # object / image distances [m] (object ~collimated)
WAVELENGTH = 530e-9
APERTURES = (0.6e-3, 1.5e-3, 3.0e-3)  # aperture radii [m]: low / medium / high NA

K1 = 2 * np.pi * N1 / WAVELENGTH
K2 = 2 * np.pi * N2 / WAVELENGTH


def geometric_opd_waves(z_s, r):
    d1 = np.sqrt((z_s - ZO) ** 2 + r**2)
    d2 = np.sqrt((ZI - z_s) ** 2 + r**2)
    return (N1 * d1 + N2 * d2 - (N1 * abs(ZO) + N2 * ZI)) / WAVELENGTH


def huygens_field(z_s, r, z_img, r_out):
    r_out = np.atleast_1d(r_out)
    d1 = np.sqrt((z_s - ZO) ** 2 + r**2)
    source = np.exp(1j * K1 * d1) / d1 * r
    d2 = np.sqrt((z_img - z_s)[None, :] ** 2 + r[None, :] ** 2 + r_out[:, None] ** 2)
    kernel = source[None, :] * np.exp(1j * K2 * d2) / d2 * j0(K2 * r[None, :] * r_out[:, None] / d2)
    return np.trapezoid(kernel, r, axis=1)


def supported_aperture(oval, r_target, *, shrink=0.98, tries=60):
    r = float(r_target)
    for _ in range(tries):
        try:
            oval.sag(np.array([r]), np.array([0.0]))
            return r
        except Exception:
            r *= shrink
    raise RuntimeError("no supported aperture found for this oval")


def render_row(oval, parab, O, R_target):
    """Compute the oval/parabola system caustics for one aperture radius."""
    R = supported_aperture(oval, R_target)

    # Integration density scaled to this row's own aperture (matching the
    # flagship example's proven density at R = 3 mm: ~1.5e6 points/m) --
    # a shared fixed point count would either under-resolve the high-NA row
    # (speckle) or waste effort on the low-NA ones.
    n_r = max(1200, int(1.5e6 * R))
    r = np.linspace(0.0, R, n_r)
    zeros = np.zeros_like(r)
    z_o, z_p = oval.sag(r, zeros), parab.sag(r, zeros)

    na = R / np.sqrt(R**2 + ZI**2)
    W_p = geometric_opd_waves(z_p, r)

    # Locate the parabola's best on-axis plane (shift grows with aperture).
    zs = np.linspace(0.5 * ZI, 1.03 * ZI, 150)
    F_o = np.abs([huygens_field(z_o, r, z, 0.0)[0] for z in zs]) ** 2
    F_p = np.abs([huygens_field(z_p, r, z, 0.0)[0] for z in zs]) ** 2
    I0 = F_o.max()
    z_best = zs[np.argmax(F_p)]

    # Beam -> surface -> cone -> focus -> beyond, one continuous propagation
    # (see cartesian_vs_parabolic_aberration.py for the incident-wave-units note).
    sag_max = max(z_o.max(), z_p.max())
    z_join = 1.15 * sag_max
    zc = np.linspace(-0.15 * ZI, 1.7 * ZI, 170)
    rc = np.linspace(0.0, 1.05 * R, 170)
    tc = np.concatenate([-rc[::-1], rc[1:]])

    n_master, n_base, z_ref = 10 * r.size, r.size, 8e-3
    r_m = np.linspace(0.0, R, n_master)
    zeros_m = np.zeros_like(r_m)
    sag_master = {"oval": oval.sag(r_m, zeros_m), "parab": parab.sag(r_m, zeros_m)}

    def system_caustic(which):
        sag = sag_master[which]
        rows = np.empty((zc.size, rc.size))
        for i, z in enumerate(zc):
            if z <= z_join:
                d1 = np.sqrt((z - ZO) ** 2 + rc**2)
                rows[i] = 1.0 / (K2 * d1) ** 2
            else:
                boost = abs(1.0 / z - 1.0 / ZI) / abs(1.0 / z_ref - 1.0 / ZI)
                # Floor of 3 (not 1): near focus, adjacent z-planes can land in
                # different integer k bins, and the resulting quadrature-error
                # jump reads as a jagged sawtooth along the shadow-side
                # diffraction fringes (verified to vanish at this floor).
                k = max(1, int(n_master // (n_base * min(max(boost, 3.0), 10.0))))
                rows[i] = np.abs(huygens_field(sag[::k], r_m[::k], z, rc)) ** 2
        full = np.concatenate([rows[:, ::-1], rows[:, 1:]], axis=1)
        return LongitudinalSection(intensity=full / I0, z=zc, t=tc, axis="x")

    sec_o = system_caustic("oval")
    sec_p = system_caustic("parab")
    return dict(
        R=R, na=na, W_p=W_p, z_best=z_best, z_o=z_o, z_p=z_p, r=r,
        sec_o=sec_o, sec_p=sec_p, zc=zc, I0=I0,
    )


def main() -> None:
    oval = CartesianSurface(n1=N1, n2=N2, zo=ZO, zi=ZI)
    O = oval.paraxial_curvature()
    parab = ParabolicSurface(focal_length=1.0 / (2.0 * O))

    print(f"shared conjugates: zo = {ZO*1e3:.0f} mm, zi = {ZI*1e3:.1f} mm, O = {O:.0f} m⁻¹")
    rows = [render_row(oval, parab, O, R) for R in APERTURES]
    for row in rows:
        print(
            f"  R = {row['R']*1e3:5.2f} mm   NA = {row['na']:.3f}   "
            f"parabola OPD PV = {np.ptp(row['W_p']):7.2f} waves   "
            f"best plane = {row['z_best']*1e3:.2f} mm"
        )

    fig, ax = plt.subplots(3, 2, figsize=(11, 12.6), constrained_layout=True)
    for i, row in enumerate(rows):
        # incident-beam level in this row's own I0-normalized units (I0 grows
        # with aperture, so the floor must track it row by row).
        v_lo = np.log10(1.0 / ((K2 * ZO) ** 2 * row["I0"])) - 0.7
        plot_longitudinal(
            ax[i, 0], row["sec_o"], normalize=False, vmin=v_lo,
            title=f"Oval — NA {row['na']:.2f} (R = {row['R']*1e3:.2f} mm)",
        )
        plot_longitudinal(
            ax[i, 1], row["sec_p"], normalize=False, vmin=v_lo,
            title=f"Parabola — PV {np.ptp(row['W_p']):.1f} waves",
        )
        for j, zsag in ((0, row["z_o"]), (1, row["z_p"])):
            a = ax[i, j]
            a.plot(zsag, row["r"], color="white", lw=1.0)
            a.plot(zsag, -row["r"], color="white", lw=1.0)
            a.set_xlim(row["zc"].min(), row["zc"].max())
            a.axvline(ZI, color="cyan", ls="--", lw=0.8, alpha=0.8)
        ax[i, 1].axvline(row["z_best"], color="w", ls=":", lw=0.8, alpha=0.6)

    fig.suptitle(
        "Spherical aberration emerging with aperture: oval (exact, always sharp) "
        f"vs paraboloid (same vertex curvature) — zo = {ZO*1e3:.0f} mm, zi = {ZI*1e3:.1f} mm",
        fontsize=12,
    )
    plt.show()


if __name__ == "__main__":
    main()
