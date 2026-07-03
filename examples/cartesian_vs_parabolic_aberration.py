"""Cartesian oval vs its paraxial (parabolic) approximation — at high NA.

A spherical wave from the design object point is focused by (a) the exact
stigmatic Cartesian-oval surface and (b) the osculating paraboloid with the
same vertex curvature. At a **high numerical aperture** the two diverge
strongly: the oval still focuses to a textbook Airy pattern, while the
paraboloid pours several waves of primary spherical aberration into a smeared
caustic.

The comparison uses the exact geometric OPD and an axisymmetric Huygens
integral evaluated from the *curved* surface, because the thin-element
phase-screen model (and hence the package's `Surface.phase_mask` / ASM path)
has errors of the same r^4 order as the oval-parabola difference — it would
misassign spurious OPD to the stigmatic oval. Propagating from the curved
surface with exact distances is instead equivalent to referencing each
plane-wave component from the local surface height with its own exp(i kz z_s).

Two families of views:

* **Edge-on through-focus caustics** (top row): |U(x, z)|^2 near focus for each
  surface, on a *shared* oval-peak scale — the oval a tight cone pinching to a
  bright waist at the design plane, the parabola a flared caustic with a best
  focus shifted toward the surface and a much lower peak.
* **Focal-spot images** (second row): the 2-D PSF at the design plane (oval
  Airy disc vs parabola halo) and the parabola refocused to its best plane.

Below them, the quantitative panels: shape difference, exact OPD, on-axis
through-focus, PSF profiles vs the analytic Airy, and encircled energy.

At high NA the J0-reduced azimuthal integral drops a small quadratic term
(printed below) and the scalar model omits vector/polarization effects, so the
renders are illustrative and quantitative to the stated integral.

Requires scipy (pip install "diffraction[examples]").

Run with:  python examples/cartesian_vs_parabolic_aberration.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0, j1

from diffraction import CartesianSurface, LongitudinalSection, ParabolicSurface, plot_longitudinal

# Design: a point source imaged by a single refracting surface, at high NA so
# the oval and its osculating paraboloid separate clearly.
N1, N2 = 1.0, 1.5
ZO, ZI = -6.0e-3, 12.0e-3  # object / image distances [m]
WAVELENGTH = 530e-9
R_AP = 1.5e-3  # aperture radius [m] — image-side NA ~ 0.12
# A single refracting surface at high NA gives its paraboloid *many* waves of
# spherical aberration (∝ R_AP⁴): here ~20 waves PV, a dramatic caustic while
# the stigmatic oval stays diffraction-limited. Raising R_AP raises the NA and
# the aberration together; the through-focus framing below adapts to it.

K1 = 2 * np.pi * N1 / WAVELENGTH
K2 = 2 * np.pi * N2 / WAVELENGTH


def geometric_opd_waves(z_s, r):
    """Exact optical path error vs the axial ray, in waves."""
    d1 = np.sqrt((z_s - ZO) ** 2 + r**2)
    d2 = np.sqrt((ZI - z_s) ** 2 + r**2)
    return (N1 * d1 + N2 * d2 - (N1 * abs(ZO) + N2 * ZI)) / WAVELENGTH


def huygens_field(z_s, r, z_img, r_out):
    """Axisymmetric Huygens integral from the curved surface (vectorized).

    Evaluates the field at every output radius in ``r_out`` at once. The
    azimuthal integral reduces to a Bessel J0 factor; the quadratic term
    dropped in that reduction is estimated by :func:`dropped_phase_term`.
    """
    r_out = np.atleast_1d(r_out)
    d1 = np.sqrt((z_s - ZO) ** 2 + r**2)
    source = np.exp(1j * K1 * d1) / d1 * r  # (n_r,)
    d2 = np.sqrt((z_img - z_s)[None, :] ** 2 + r[None, :] ** 2 + r_out[:, None] ** 2)
    kernel = source[None, :] * np.exp(1j * K2 * d2) / d2 * j0(K2 * r[None, :] * r_out[:, None] / d2)
    return np.trapezoid(kernel, r, axis=1)  # (n_out,)


def dropped_phase_term(z_s, r, z_img, r_out_max):
    """Max magnitude [rad] of the azimuthal-quadratic term dropped by the J0 form."""
    d2 = np.sqrt((z_img - z_s) ** 2 + r**2 + r_out_max**2)
    return float(np.max(K2 * (r * r_out_max) ** 2 / (2.0 * d2**3)))


def radial_to_image(profile, r_out, half, n=221):
    """Rotate a radial intensity profile into a centered 2-D image."""
    xs = np.linspace(-half, half, n)
    xg, yg = np.meshgrid(xs, xs)
    rg = np.sqrt(xg**2 + yg**2)
    img = np.interp(rg.ravel(), r_out, profile, right=0.0).reshape(rg.shape)
    return img, xs


def main() -> None:
    oval = CartesianSurface(n1=N1, n2=N2, zo=ZO, zi=ZI)
    parab = ParabolicSurface(focal_length=1.0 / (2.0 * oval.paraxial_curvature()))

    r = np.linspace(0.0, R_AP, 3000)
    zeros = np.zeros_like(r)
    z_o, z_p = oval.sag(r, zeros), parab.sag(r, zeros)

    na = R_AP / np.sqrt(R_AP**2 + ZI**2)
    W_p = geometric_opd_waves(z_p, r)
    print(f"image-side NA = {na:.3f}")
    print(f"parabola OPD: PV = {np.ptp(W_p):.3f} waves, RMS = {np.std(W_p):.3f} waves")

    # Locate the parabola's best (on-axis) focus over a generous window (the
    # marginal focus shifts well toward the surface at this much aberration).
    zs = np.linspace(0.72 * ZI, 1.03 * ZI, 321)
    F_o = np.abs([huygens_field(z_o, r, z, 0.0)[0] for z in zs]) ** 2
    F_p = np.abs([huygens_field(z_p, r, z, 0.0)[0] for z in zs]) ** 2
    I0 = F_o.max()
    z_best = zs[np.argmax(F_p)]
    print(f"parabola best focus: z = {z_best*1e3:.3f} mm (design {ZI*1e3:.1f} mm)")

    # Transverse windows: an Airy radius ~ 0.61 lambda / NA sets the scale.
    airy_radius = 0.61 * WAVELENGTH / na
    r_out = np.linspace(0.0, 22e-6, 400)
    print(f"Airy radius ~ {airy_radius*1e6:.2f} um; J0 dropped term "
          f"<= {dropped_phase_term(z_p, r, ZI, r_out.max()):.3f} rad")

    # PSF radial profiles (design plane, and parabola at best focus).
    P_o = np.abs(huygens_field(z_o, r, ZI, r_out)) ** 2 / I0
    P_p = np.abs(huygens_field(z_p, r, ZI, r_out)) ** 2 / I0
    P_pb = np.abs(huygens_field(z_p, r, z_best, r_out)) ** 2 / I0
    print(f"parabola Strehl @ design plane: {P_p[0]:.3f}, @ best focus: {P_pb[0]:.3f}")

    v = np.where(r_out == 0, 1e-12, K2 * na * r_out)
    airy = (2 * j1(v) / v) ** 2

    def encircled(P):
        c = np.cumsum(P * r_out)
        return c / c[-1]

    # --- Edge-on through-focus caustics, framed around the caustic ------------
    # The caustic runs from the marginal focus (z_best) to the paraxial design
    # plane; its transverse reach is ~ the marginal-ray height there. Frame both
    # axes off the measured focus shift so it adapts to the aberration size.
    dz = ZI - z_best
    rc_max = max(1.5 * R_AP * dz / ZI, 12.0 * airy_radius)
    zc = np.linspace(z_best - 0.35 * dz, ZI + 0.5 * dz, 160)
    rc = np.linspace(0.0, rc_max, 160)
    tc = np.concatenate([-rc[::-1], rc[1:]])  # mirror to a symmetric x axis

    def caustic(z_s):
        rows = [np.abs(huygens_field(z_s, r, z, rc)) ** 2 for z in zc]
        prof = np.asarray(rows)  # (n_z, n_r)
        full = np.concatenate([prof[:, ::-1], prof[:, 1:]], axis=1)  # (n_z, n_t)
        return LongitudinalSection(intensity=full / I0, z=zc, t=tc, axis="x")

    sec_o, sec_p = caustic(z_o), caustic(z_p)

    # --- 2-D focal-spot images (shared scale) --------------------------------
    spot_half = r_out.max()
    img_o, xs = radial_to_image(P_o, r_out, spot_half)
    img_pd, _ = radial_to_image(P_p, r_out, spot_half)
    img_pb, _ = radial_to_image(P_pb, r_out, spot_half)

    def show_spot(ax, img, title):
        im = ax.imshow(
            np.log10(img + 1e-4),
            extent=[-spot_half * 1e6, spot_half * 1e6, -spot_half * 1e6, spot_half * 1e6],
            origin="lower", cmap="inferno", vmin=-4, vmax=0,
            interpolation="antialiased", interpolation_stage="rgba",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        return im

    fig, ax = plt.subplots(4, 3, figsize=(15, 15.5), constrained_layout=True)

    # Row 0 — edge-on caustics on a shared oval-peak scale.
    im_c = plot_longitudinal(ax[0, 0], sec_o, normalize=False, vmin=-4,
                             title="Oval — through-focus caustic")
    plot_longitudinal(ax[0, 1], sec_p, normalize=False, vmin=-4,
                      title="Parabola — through-focus caustic")
    ax[0, 0].axvline(ZI, color="cyan", ls="--", lw=0.8, alpha=0.8)  # oval focus = design
    ax[0, 1].axvline(ZI, color="cyan", ls="--", lw=0.8, alpha=0.8)
    ax[0, 1].axvline(z_best, color="w", ls=":", lw=0.8, alpha=0.7)  # shifted best focus
    ax[0, 2].axis("off")
    cb = fig.colorbar(im_c, ax=ax[0, 2], fraction=0.5, aspect=18)
    cb.set_label("log₁₀(I / oval peak)")
    ax[0, 2].text(0.5, 0.5, "cyan — — design plane\nwhite ···· parabola best focus",
                  transform=ax[0, 2].transAxes, ha="center", va="center", fontsize=9)

    # Row 1 — 2-D focal spots on a shared scale.
    show_spot(ax[1, 0], img_o, "Oval PSF @ design plane")
    im_s = show_spot(ax[1, 1], img_pd, "Parabola PSF @ design plane")
    show_spot(ax[1, 2], img_pb, f"Parabola PSF @ best focus ({z_best*1e3:.2f} mm)")
    fig.colorbar(im_s, ax=list(ax[1, :]), location="right", fraction=0.03, aspect=30,
                 label="log₁₀(I / oval peak)")

    # Row 2 — shape difference, OPD, on-axis through-focus.
    ax[2, 0].plot(r * 1e3, (z_o - z_p) * 1e6, "C0")
    ax[2, 0].set_xlabel("r [mm]")
    ax[2, 0].set_ylabel("z_oval − z_parabola [µm]")
    ax[2, 0].set_title("Shape difference (∝ r⁴)")
    ax[2, 0].grid(alpha=0.3)

    ax[2, 1].plot(r * 1e3, W_p, "C3", label="parabola")
    ax[2, 1].plot(r * 1e3, geometric_opd_waves(z_o, r), "C2", label="Cartesian oval (≡ 0)")
    ax[2, 1].set_xlabel("r [mm]")
    ax[2, 1].set_ylabel("OPD [waves]")
    ax[2, 1].set_title("Exact geometric spherical aberration")
    ax[2, 1].legend()
    ax[2, 1].grid(alpha=0.3)

    ax[2, 2].plot(zs * 1e3, F_o / I0, "C2", label="Cartesian oval")
    ax[2, 2].plot(zs * 1e3, F_p / I0, "C3", label="parabola")
    ax[2, 2].axvline(ZI * 1e3, color="gray", ls="--", lw=0.8)
    ax[2, 2].set_xlabel("z [mm]")
    ax[2, 2].set_ylabel("I(0, z) / oval peak")
    ax[2, 2].set_title("Through-focus axial response")
    ax[2, 2].legend()
    ax[2, 2].grid(alpha=0.3)

    # Row 3 — PSF profiles and encircled energy.
    ax[3, 0].semilogy(r_out * 1e6, P_o, "C2", label="oval @ design plane")
    ax[3, 0].semilogy(r_out * 1e6, airy, "k--", lw=1, label="analytic Airy")
    ax[3, 0].semilogy(r_out * 1e6, P_p, "C3", label="parabola @ design plane")
    ax[3, 0].set_ylim(1e-7, 2)
    ax[3, 0].set_xlabel("r [µm]")
    ax[3, 0].set_ylabel("I / oval peak")
    ax[3, 0].set_title("PSF at the design image plane")
    ax[3, 0].legend(fontsize=8)
    ax[3, 0].grid(alpha=0.3)

    ax[3, 1].semilogy(r_out * 1e6, P_o, "C2", label="oval @ design plane")
    ax[3, 1].semilogy(r_out * 1e6, P_pb, "C1", label=f"parabola @ best focus ({z_best*1e3:.2f} mm)")
    ax[3, 1].set_ylim(1e-7, 2)
    ax[3, 1].set_xlabel("r [µm]")
    ax[3, 1].set_ylabel("I / oval peak")
    ax[3, 1].set_title("Refocused parabola vs oval")
    ax[3, 1].legend(fontsize=8)
    ax[3, 1].grid(alpha=0.3)

    ax[3, 2].plot(r_out * 1e6, encircled(P_o), "C2", label="oval @ design")
    ax[3, 2].plot(r_out * 1e6, encircled(P_p), "C3", label="parabola @ design")
    ax[3, 2].plot(r_out * 1e6, encircled(P_pb), "C1", label="parabola @ best focus")
    ax[3, 2].axhline(0.838, color="gray", lw=0.8, ls=":")
    ax[3, 2].set_xlabel("r [µm]")
    ax[3, 2].set_ylabel("encircled energy")
    ax[3, 2].set_title("Encircled energy (83.8% = Airy disc)")
    ax[3, 2].legend(fontsize=8)
    ax[3, 2].grid(alpha=0.3)

    fig.suptitle(
        f"Cartesian oval vs paraboloid at NA = {na:.2f}  "
        f"(zo = {ZO*1e3:.0f} mm, zi = {ZI*1e3:.0f} mm, n1 = {N1}, n2 = {N2}, "
        f"aperture {R_AP*1e3:.1f} mm, λ = {WAVELENGTH*1e9:.0f} nm)",
        fontsize=12,
    )
    plt.show()


if __name__ == "__main__":
    main()
