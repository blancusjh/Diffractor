"""Cartesian oval vs its paraxial (parabolic) approximation — at very high NA.

A near-collimated source is focused by (a) the exact stigmatic Cartesian-oval
surface and (b) the osculating paraboloid with the same vertex curvature. At a
**very high numerical aperture** the two diverge violently: the oval still
focuses to a textbook Airy pattern, while the paraboloid pours ~a hundred waves
of primary spherical aberration into a wide caustic with no recognizable focus.

The comparison uses the exact geometric OPD and an axisymmetric Huygens
integral evaluated from the *curved* surface, because the thin-element
phase-screen model (and hence the package's `Surface.phase_mask` / ASM path)
has errors of the same r^4 order as the oval-parabola difference — it would
misassign spurious OPD to the stigmatic oval. Propagating from the curved
surface with exact distances is instead equivalent to referencing each
plane-wave component from the local surface height with its own exp(i kz z_s).

The object is placed far away (near-collimated): for a *close* conjugate the
oval's own radial extent caps the aperture near its vertex radius of curvature
(NA ≲ 0.14 here), so opening up to NA ~ 0.25 needs a distant source — the
classic "fast single refractor focusing a distant point" spherical-aberration
scenario.

**Figure 1 (images):**

* **Edge-on system caustics** (top): |U(x, z)|^2 across the whole system for each
  surface, on a *shared* oval-peak scale — the curved refractor drawn at left,
  then the converging cone, the focus, and the field diverging beyond it. The
  oval pinches to a bright diffraction-limited waist at the design plane and
  reopens symmetrically; the parabola cone crosses into a wide flared caustic.
* **Focal-spot images** (bottom): the 2-D PSF at the design plane (oval Airy
  disc vs parabola halo) and the parabola at its best on-axis plane.

**Figure 2 (metrics):** the r⁴ shape difference, the exact OPD, the on-axis
through-focus response, PSF profiles vs the analytic Airy, encircled energy, and
a numeric summary. The equivalent paraboloid has sag ``z = O r² / 2`` with
vertex curvature ``O = (n2·zo − n1·zi) / (zi·zo·(n2 − n1))``.

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

# Design: a distant (near-collimated) point source imaged by a single refracting
# surface, opened to very high NA so the oval and its paraboloid separate hugely.
N1, N2 = 1.0, 1.5
ZO, ZI = -0.1, 12.0e-3  # object / image distances [m] (object ~collimated)
WAVELENGTH = 530e-9
R_AP = 3.0e-3  # aperture radius [m] — target image-side NA ~ 0.25 (clamped below)
# A single refracting surface at very high NA gives its paraboloid ~a hundred
# waves of spherical aberration; the stigmatic oval stays diffraction-limited.
# The oval's sag solver raises past the surface's radial extent, so R_AP is
# clamped to what the oval supports before use.

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


def supported_aperture(oval, r_target, *, shrink=0.98, tries=60):
    """Largest radius <= r_target for which the oval's sag solver converges."""
    r = float(r_target)
    for _ in range(tries):
        try:
            oval.sag(np.array([r]), np.array([0.0]))
            return r
        except Exception:
            r *= shrink
    raise RuntimeError("no supported aperture found for this oval")


def main() -> None:
    oval = CartesianSurface(n1=N1, n2=N2, zo=ZO, zi=ZI)
    # The equivalent (osculating) paraboloid has sag  z = O r² / 2  with O the
    # oval's paraxial curvature,  O = (n2·zo − n1·zi) / (zi·zo·(n2 − n1)).
    # ParabolicSurface.sag = ρ²/(4f), so f = 1/(2O) reproduces exactly O r²/2.
    O = oval.paraxial_curvature()
    parab = ParabolicSurface(focal_length=1.0 / (2.0 * O))

    R = supported_aperture(oval, R_AP)  # clamp to the oval's physical extent
    if R < R_AP:
        print(f"aperture clamped to oval extent: {R*1e3:.3f} mm (asked {R_AP*1e3:.2f} mm)")

    r = np.linspace(0.0, R, 4500)  # fine enough that K2·NA·dr < π at the focus
    zeros = np.zeros_like(r)
    z_o, z_p = oval.sag(r, zeros), parab.sag(r, zeros)

    na = R / np.sqrt(R**2 + ZI**2)
    W_p = geometric_opd_waves(z_p, r)
    print(f"image-side NA = {na:.3f}")
    print(f"parabola OPD: PV = {np.ptp(W_p):.2f} waves, RMS = {np.std(W_p):.2f} waves")

    # Locate the parabola's best on-axis plane over a generous window (at this
    # much aberration it is broad and only weakly peaked).
    zs = np.linspace(0.85 * ZI, 1.03 * ZI, 301)
    F_o = np.abs([huygens_field(z_o, r, z, 0.0)[0] for z in zs]) ** 2
    F_p = np.abs([huygens_field(z_p, r, z, 0.0)[0] for z in zs]) ** 2
    I0 = F_o.max()
    z_best = zs[np.argmax(F_p)]
    print(f"parabola best on-axis plane: z = {z_best*1e3:.3f} mm (design {ZI*1e3:.1f} mm)")

    airy_radius = 0.61 * WAVELENGTH / na
    r_out = np.linspace(0.0, max(16e-6, 8 * airy_radius), 400)
    print(f"Airy radius ~ {airy_radius*1e6:.2f} um; J0 dropped term "
          f"<= {dropped_phase_term(z_p, r, ZI, r_out.max()):.4f} rad")

    P_o = np.abs(huygens_field(z_o, r, ZI, r_out)) ** 2 / I0
    P_p = np.abs(huygens_field(z_p, r, ZI, r_out)) ** 2 / I0
    P_pb = np.abs(huygens_field(z_p, r, z_best, r_out)) ** 2 / I0
    print(f"parabola Strehl @ design plane: {P_p[0]:.4f}, @ best plane: {P_pb[0]:.4f}")

    v = np.where(r_out == 0, 1e-12, K2 * na * r_out)
    airy = (2 * j1(v) / v) ** 2

    def encircled(P):
        c = np.cumsum(P * r_out)
        return c / c[-1]

    # --- Full-system longitudinal maps (surface -> focus), surfaces overlaid ---
    z_start = 1.4 * max(z_o.max(), z_p.max())  # begin just past the surface
    zc = np.linspace(z_start, 1.5 * ZI, 200)  # through the focus and well beyond
    rc = np.linspace(0.0, R, 220)
    tc = np.concatenate([-rc[::-1], rc[1:]])  # mirror to a symmetric x axis

    def system_caustic(z_s):
        rows = [np.abs(huygens_field(z_s, r, z, rc)) ** 2 for z in zc]
        prof = np.asarray(rows)  # (n_z, n_r)
        full = np.concatenate([prof[:, ::-1], prof[:, 1:]], axis=1)  # (n_z, n_t)
        return LongitudinalSection(intensity=full / I0, z=zc, t=tc, axis="x")

    sec_o = system_caustic(z_o)
    sec_p = system_caustic(z_p)

    # --- 2-D focal-spot images (shared scale) --------------------------------
    spot_half = r_out.max()
    img_o, _ = radial_to_image(P_o, r_out, spot_half)
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

    # ======================= Figure 1 — images ===============================
    fig1, ax1 = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)

    im_c = plot_longitudinal(ax1[0, 0], sec_o, normalize=False, vmin=-6.5,
                             title="Oval — system caustic (surface → focus)")
    plot_longitudinal(ax1[0, 1], sec_p, normalize=False, vmin=-6.5,
                      title="Parabola — system caustic (surface → focus)")
    for a, zsag in ((ax1[0, 0], z_o), (ax1[0, 1], z_p)):
        a.set_facecolor("black")  # blend the pre-surface region with the map
        a.plot(zsag, r, color="white", lw=1.1)   # the refracting surface, drawn
        a.plot(zsag, -r, color="white", lw=1.1)  # in the same x–z axes
        a.set_xlim(0.0, zc.max())
        a.axvline(ZI, color="cyan", ls="--", lw=0.8, alpha=0.8)
    ax1[0, 1].axvline(z_best, color="w", ls=":", lw=0.8, alpha=0.6)

    ax1[0, 2].axis("off")
    cb = fig1.colorbar(im_c, ax=ax1[0, 2], fraction=0.5, aspect=18)
    cb.set_label("log₁₀(I / oval peak)")
    ax1[0, 2].text(
        0.5, 0.30,
        "white — refracting surface\ncyan — — design plane\nwhite ···· parabola best plane",
        transform=ax1[0, 2].transAxes, ha="center", va="center", fontsize=8,
    )

    show_spot(ax1[1, 0], img_o, "Oval PSF @ design plane")
    im_s = show_spot(ax1[1, 1], img_pd, "Parabola PSF @ design plane")
    show_spot(ax1[1, 2], img_pb, f"Parabola PSF @ best plane ({z_best*1e3:.2f} mm)")
    fig1.colorbar(im_s, ax=list(ax1[1, :]), location="right", fraction=0.03, aspect=30,
                  label="log₁₀(I / oval peak)")

    fig1.suptitle(
        f"Cartesian oval vs paraboloid at NA = {na:.2f}  "
        f"(zo = {ZO*1e3:.0f} mm, zi = {ZI*1e3:.0f} mm, n1 = {N1}, n2 = {N2}, "
        f"aperture {R*1e3:.2f} mm, λ = {WAVELENGTH*1e9:.0f} nm)",
        fontsize=12,
    )

    # ======================= Figure 2 — metrics ==============================
    fig2, ax2 = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)

    ax2[0, 0].plot(r * 1e3, (z_o - z_p) * 1e6, "C0")
    ax2[0, 0].set_xlabel("r [mm]")
    ax2[0, 0].set_ylabel("z_oval − z_parabola [µm]")
    ax2[0, 0].set_title("Shape difference (∝ r⁴)")
    ax2[0, 0].grid(alpha=0.3)

    ax2[0, 1].plot(r * 1e3, W_p, "C3", label="parabola")
    ax2[0, 1].plot(r * 1e3, geometric_opd_waves(z_o, r), "C2", label="Cartesian oval (≡ 0)")
    ax2[0, 1].set_xlabel("r [mm]")
    ax2[0, 1].set_ylabel("OPD [waves]")
    ax2[0, 1].set_title("Exact geometric spherical aberration")
    ax2[0, 1].legend()
    ax2[0, 1].grid(alpha=0.3)

    ax2[0, 2].plot(zs * 1e3, F_o / I0, "C2", label="Cartesian oval")
    ax2[0, 2].plot(zs * 1e3, F_p / I0, "C3", label="parabola")
    ax2[0, 2].axvline(ZI * 1e3, color="gray", ls="--", lw=0.8)
    ax2[0, 2].set_xlabel("z [mm]")
    ax2[0, 2].set_ylabel("I(0, z) / oval peak")
    ax2[0, 2].set_title("Through-focus axial response")
    ax2[0, 2].legend()
    ax2[0, 2].grid(alpha=0.3)

    ax2[1, 0].semilogy(r_out * 1e6, P_o, "C2", label="oval @ design")
    ax2[1, 0].semilogy(r_out * 1e6, airy, "k--", lw=1, label="analytic Airy")
    ax2[1, 0].semilogy(r_out * 1e6, P_p, "C3", label="parabola @ design")
    ax2[1, 0].semilogy(r_out * 1e6, P_pb, "C1", lw=1, label=f"parabola @ best ({z_best*1e3:.2f} mm)")
    ax2[1, 0].set_ylim(1e-7, 2)
    ax2[1, 0].set_xlabel("r [µm]")
    ax2[1, 0].set_ylabel("I / oval peak")
    ax2[1, 0].set_title("PSF profiles")
    ax2[1, 0].legend(fontsize=8)
    ax2[1, 0].grid(alpha=0.3)

    ax2[1, 1].plot(r_out * 1e6, encircled(P_o), "C2", label="oval @ design")
    ax2[1, 1].plot(r_out * 1e6, encircled(P_p), "C3", label="parabola @ design")
    ax2[1, 1].plot(r_out * 1e6, encircled(P_pb), "C1", label="parabola @ best")
    ax2[1, 1].axhline(0.838, color="gray", lw=0.8, ls=":")
    ax2[1, 1].set_xlabel("r [µm]")
    ax2[1, 1].set_ylabel("encircled energy")
    ax2[1, 1].set_title("Encircled energy (83.8% = Airy disc)")
    ax2[1, 1].legend(fontsize=8)
    ax2[1, 1].grid(alpha=0.3)

    ax2[1, 2].axis("off")
    summary = (
        f"NA (image)   = {na:.3f}\n"
        f"aperture R   = {R*1e3:.2f} mm\n"
        f"O            = {O:.0f} m⁻¹\n"
        f"R_c = 1/O    = {1e3/O:.2f} mm\n"
        f"\n"
        f"parabola spherical aberration\n"
        f"  PV  = {np.ptp(W_p):.1f} waves\n"
        f"  RMS = {np.std(W_p):.1f} waves\n"
        f"\n"
        f"Strehl (oval)            ≈ 1.00\n"
        f"Strehl (parab @ design)  = {P_p[0]:.3f}\n"
        f"Strehl (parab @ best)    = {P_pb[0]:.3f}\n"
        f"\n"
        f"best on-axis plane = {z_best*1e3:.2f} mm\n"
        f"design plane       = {ZI*1e3:.1f} mm"
    )
    ax2[1, 2].text(0.0, 0.98, summary, transform=ax2[1, 2].transAxes,
                   va="top", ha="left", fontsize=10, family="monospace")

    fig2.suptitle("Cartesian oval vs paraboloid — quantitative metrics", fontsize=12)

    plt.show()


if __name__ == "__main__":
    main()
